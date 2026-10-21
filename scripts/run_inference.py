#!/usr/bin/env python3
"""End-to-end inference: AOI -> gate -> daily -> harmonized -> calendar -> forecast.

Loads trained artifacts (integrity-verified), rebuilds the daily AOI series
from raw source CSVs, produces the harmonized calendar, anomaly scores, and a
probabilistic 14-day regional forecast, then exports the app-consumable
calendar/forecast records plus a provenance manifest. Every failure path is
explicit (empty AOI, missing artifacts, unavailable forecast) - never silent.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from src.artifacts.manifest import build_manifest
from src.artifacts.serialization import (
    ArtifactIntegrityError,
    load_forecaster_bundle,
    load_harmonizer_bundle,
    new_artifact_dir,
    save_json,
)
from src.calendar.anomalies import score_anomalies
from src.calendar.forecast import torch_available
from src.calendar.records import (
    build_calendar_forecast_records,
    validate_calendar_forecast_frame,
)
from src.calendar.seasonal import SeasonalClimatology
from src.common.config import (
    CalendarConfig,
    HarmonizationConfig,
    resolve_code_commit,
    seed_everything,
)
from src.common.logging import configure_logging, get_logger
from src.data.aggregation import EmptyAOIError, build_daily_frame, summarize_daily
from src.data.gate import DataGateError, FivePassDataGate, assert_gate_passed
from src.data.schema import PRODUCT_MODIS, PRODUCT_VIIRS
from src.geo.aoi import parse_aoi
from src.harmonization.inference import build_harmonized_daily

LOGGER = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--harmonizer-dir", required=True, help="Trained harmonizer artifact directory")
    parser.add_argument("--forecaster-dir", default=None, help="Optional forecaster artifact directory")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--aoi", required=True, help="AOI spec, e.g. bbox:-123.5,39.5,-120.5,42.0")
    parser.add_argument("--config", default=None, help="Harmonization YAML")
    parser.add_argument("--calendar-config", default=None, help="Calendar YAML")
    parser.add_argument("--as-of", default=None, help="Forecast issue date (YYYY-MM-DD; default last observed)")
    parser.add_argument("--seed", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    args = build_parser().parse_args(argv)
    config = HarmonizationConfig.from_yaml(args.config) if args.config else HarmonizationConfig()
    calendar_config = (
        CalendarConfig.from_yaml(args.calendar_config)
        if args.calendar_config
        else CalendarConfig()
    )
    if args.seed is not None:
        config = HarmonizationConfig.from_dict({**config.to_dict(), "seed": int(args.seed)})
    seed_everything(config.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    aoi = parse_aoi(args.aoi)

    # 1. Gate (fail closed) - reports written even on failure.
    gate = FivePassDataGate(config)
    outcome = gate.run(args.source_dir)
    save_json(output_dir / "inventory_manifest.json", outcome.inventory_manifest)
    save_json(output_dir / "dataset_audit_report.json", outcome.audit_report)
    try:
        assert_gate_passed(outcome)
    except DataGateError as exc:
        print(f"Five-pass data gate failed; inference aborted: {exc}", file=sys.stderr)
        return 1

    # 2. AOI filtering + daily aggregation.
    from src.geo.aoi import filter_records

    frames = {
        product: filter_records(frame, aoi)
        for product, frame in outcome.frames.items()
    }
    try:
        daily = build_daily_frame(frames, aoi_id=aoi.aoi_id)
    except EmptyAOIError as exc:
        # Explicit empty-AOI state (docs: "Empty AOIs ... must have explicit states").
        empty_state = {
            "status": "empty_aoi",
            "aoi_id": aoi.aoi_id,
            "message": str(exc),
            "records": [],
        }
        save_json(output_dir / "calendar_forecast_records.json", empty_state)
        print(f"AOI contains no validated observations: {exc}", file=sys.stderr)
        return 2

    # 3. Load the harmonizer artifact (integrity-verified) and harmonize.
    try:
        bundle = load_harmonizer_bundle(args.harmonizer_dir)
    except ArtifactIntegrityError as exc:
        print(f"Harmonizer artifact problem: {exc}", file=sys.stderr)
        return 3
    harmonizer = bundle["model"]
    harmonized_daily = build_harmonized_daily(
        daily,
        frames[PRODUCT_MODIS],
        frames[PRODUCT_VIIRS],
        harmonizer,
        config.matching,
    )
    harmonized_daily.to_csv(output_dir / "harmonized_daily.csv", index=False)

    # 4. Seasonal baseline + anomalies.
    climatology = SeasonalClimatology(calendar_config.seasonal).fit(
        harmonized_daily["date"], harmonized_daily["value"]
    )
    anomalies = score_anomalies(harmonized_daily, climatology, calendar_config.anomaly)
    anomalies.to_csv(output_dir / "anomaly_records.csv", index=False)

    # 5. Optional forecast from the trained forecaster artifact.
    forecast = None
    forecaster_name = None
    training_cutoff = None
    issue_date = args.as_of or pd.to_datetime(harmonized_daily["date"]).max().date().isoformat()
    if args.forecaster_dir:
        try:
            state = load_forecaster_bundle(args.forecaster_dir)
        except ArtifactIntegrityError as exc:
            print(f"Forecaster artifact problem (forecast unavailable): {exc}", file=sys.stderr)
            state = None
        if state is not None:
            forecaster_name = state.get("model_name", "unknown")
            if forecaster_name == "patchtst":
                if not torch_available():
                    print(
                        "Forecaster artifact requires PyTorch, which is unavailable; "
                        "forecast section will be reported as unavailable.",
                        file=sys.stderr,
                    )
                    state = None
                else:
                    from src.models.patchtst import PatchTSTForecaster

                    forecaster = PatchTSTForecaster().load_state_dict(state)
                    as_of_ts = pd.Timestamp(issue_date)
                    history = harmonized_daily[pd.to_datetime(harmonized_daily["date"]) <= as_of_ts]
                    forecast = forecaster.predict(history)
                    training_cutoff = pd.to_datetime(history["date"]).max().date().isoformat()
                    issue_date = (as_of_ts + pd.Timedelta(days=1)).date().isoformat()
            else:
                # Baseline forecasters stay available without torch.
                from src.calendar.forecast import PersistenceForecaster, SeasonalNaiveForecaster

                factory = (
                    SeasonalNaiveForecaster
                    if forecaster_name == "seasonal_naive"
                    else PersistenceForecaster
                )
                forecaster = factory(calendar_config.forecast)
                as_of_ts = pd.Timestamp(issue_date)
                history = harmonized_daily[pd.to_datetime(harmonized_daily["date"]) <= as_of_ts]
                forecaster.fit(history)
                forecast = forecaster.predict(history)
                training_cutoff = pd.to_datetime(history["date"]).max().date().isoformat()
                issue_date = (as_of_ts + pd.Timedelta(days=1)).date().isoformat()
    if forecast is not None:
        forecast["calibration_status"] = "experimental_uncalibrated"

    # 6. Calendar/forecast records + provenance manifest.
    from src.artifacts.serialization import load_manifest

    harmonizer_manifest = load_manifest(args.harmonizer_dir)
    calendar_records = build_calendar_forecast_records(
        harmonized_daily,
        climatology,
        calendar_config.anomaly,
        forecast=forecast,
        forecast_training_cutoff=training_cutoff,
        forecast_issue_date=issue_date,
    )
    try:
        validate_calendar_forecast_frame(calendar_records)
    except Exception as exc:  # noqa: BLE001
        print(f"Calendar/forecast contract validation failed: {exc}", file=sys.stderr)
        return 4
    calendar_records.to_csv(output_dir / "calendar_forecast_records.csv", index=False)

    provenance = build_manifest(
        pipeline="inference_export",
        config={"harmonization": config.to_dict(), "calendar": calendar_config.to_dict()},
        source_files=[{"path": p.name} for p in sorted(Path(args.source_dir).glob("*.csv"))],
        model={
            "name": harmonizer_manifest["model"]["name"],
            "version": harmonizer_manifest["model"]["version"],
            "sha256": harmonizer_manifest["model"]["sha256"],
        },
        metrics={"daily_summary": summarize_daily(daily)},
        code_commit=resolve_code_commit(REPO_ROOT),
        limitations=[
            "Harmonized values are estimates, not ground truth.",
            "Forecast intervals are experimental/uncalibrated.",
            "No exact-ignition-location claims (regional activity only).",
        ],
        extra={
            "aoi_id": aoi.aoi_id,
            "forecast_available": forecast is not None,
            "forecaster": forecaster_name,
            "forecast_issue_date": issue_date,
            "training_cutoff": training_cutoff,
        },
    )
    artifact_dir = new_artifact_dir(output_dir, "inference_export")
    save_json(artifact_dir / "manifest.json", provenance)

    print(f"Inference complete; records: {output_dir / 'calendar_forecast_records.csv'}")
    print(f"Daily summary: {summarize_daily(daily)}")
    print(f"Forecast available: {forecast is not None} (issue date {issue_date})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
