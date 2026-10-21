#!/usr/bin/env python3
"""Train the burning-activity calendar forecaster (Pillar 2; issues #5 and #10).

Pipeline: harmonized daily CSV -> seasonal climatology + anomaly baseline ->
persistence + seasonal-naive baselines and the PatchTST candidate (skipped with
a clear warning when PyTorch is unavailable) -> rolling-origin backtests with
pinball loss / quantile coverage -> versioned forecaster artifact + metrics.
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
    new_artifact_dir,
    save_forecaster_bundle,
    save_json,
)
from src.calendar.evaluation import compare_forecasters, summarize_backtest
from src.calendar.forecast import (
    PersistenceForecaster,
    SeasonalNaiveForecaster,
    backtest_records_to_frame,
    rolling_origin_backtest,
    torch_available,
)
from src.calendar.records import build_calendar_forecast_records, validate_calendar_forecast_frame
from src.calendar.seasonal import SeasonalClimatology
from src.common.config import CalendarConfig, resolve_code_commit, seed_everything
from src.common.logging import configure_logging, get_logger

LOGGER = get_logger(__name__)

FORECASTER_LIMITATIONS = [
    "Regional activity forecast only; never an exact ignition location/time claim (docs).",
    "Forecast intervals are uncalibrated/experimental until interval calibration is measured.",
    "Gap days are filled with 0.0 and flagged by a coverage channel; the model must not "
    "interpret gaps as confirmed zero activity.",
    "Baseline quantiles are empirical residual quantiles around the point forecast.",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--daily-csv", required=True, help="Harmonized daily series CSV with a 'value' column")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default=None, help="Calendar YAML (defaults used when omitted)")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--skip-patchtst",
        action="store_true",
        help="Skip the PatchTST candidate even when torch is available (baselines only)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    args = build_parser().parse_args(argv)
    config = CalendarConfig.from_yaml(args.config) if args.config else CalendarConfig()
    if args.seed is not None:
        config = CalendarConfig.from_dict({**config.to_dict(), "seed": int(args.seed)})
    seed_everything(config.seed)

    daily = pd.read_csv(args.daily_csv, parse_dates=["date"])
    if "value" not in daily.columns:
        # Raw daily-AOI records (scripts/train_harmonizer.py) carry the per-sensor
        # FRP columns instead of 'value'; derive the best observed series
        # (VIIRS preferred, MODIS-observed days kept, gaps NaN) so the documented
        # "raw daily AOI CSV" input branch is actually reachable (issue #8).
        daily_value_columns = {
            "viirs_coverage",
            "modis_coverage",
            "viirs_frp_sum_mw",
            "modis_frp_sum_mw",
        }
        if not daily_value_columns.issubset(daily.columns):
            print(
                f"Input {args.daily_csv} must contain a 'value' column or the "
                "daily AOI columns (viirs_frp_sum_mw/modis_frp_sum_mw + coverage "
                "flags); produce it with scripts/train_harmonizer.py or "
                "run_inference.py.",
                file=sys.stderr,
            )
            return 1
        from src.data.aggregation import best_observed_value

        daily["value"] = best_observed_value(daily)
        print(
            "No 'value' column found; derived the best observed series "
            "(VIIRS preferred, MODIS-observed days labeled downstream).",
            file=sys.stderr,
        )
    daily = daily.sort_values("date", kind="mergesort").reset_index(drop=True)

    # Accept either a harmonized daily CSV (full column contract) or a raw
    # daily AOI CSV; derive the label columns when the input lacks them.
    import numpy as np

    if "value_type" not in daily.columns:
        viirs_cov = (
            daily["viirs_coverage"].astype(bool)
            if "viirs_coverage" in daily.columns
            else daily["value"].notna()
        )
        modis_cov = (
            daily["modis_coverage"].astype(bool)
            if "modis_coverage" in daily.columns
            else daily["value"].notna()
        )
        daily["value_type"] = np.where(
            viirs_cov,
            "observed_viirs",
            np.where(modis_cov, "harmonized_estimate", "missing_coverage"),
        )
    if "sensor_era" not in daily.columns:
        daily["sensor_era"] = "derived_from_daily"
    if "coverage_status" not in daily.columns:
        daily["coverage_status"] = np.where(
            daily["value_type"].eq("missing_coverage"), "missing", "observed"
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Seasonal baseline + anomalies first (docs: calendar before model complexity).
    climatology = SeasonalClimatology(config.seasonal).fit(daily["date"], daily["value"])
    from src.calendar.anomalies import score_anomalies

    anomalies = score_anomalies(daily, climatology, config.anomaly)
    anomalies.to_csv(output_dir / "anomaly_records.csv", index=False)

    # Rolling-origin backtests for baselines (+ PatchTST when available).
    models: dict[str, callable] = {
        "persistence": lambda: PersistenceForecaster(config.forecast),
        "seasonal_naive": lambda: SeasonalNaiveForecaster(
            config.forecast, config.seasonal
        ),
    }
    use_patchtst = torch_available() and not args.skip_patchtst
    if use_patchtst:
        from src.models.patchtst import PatchTSTForecaster

        models["patchtst"] = lambda: PatchTSTForecaster(config.forecast, seed=config.seed)
    elif not args.skip_patchtst:
        print(
            "PyTorch not installed: PatchTST candidate skipped; "
            "seasonal-naive/persistence baselines still evaluated.",
            file=sys.stderr,
        )

    reports: dict[str, dict] = {}
    backtest_frames: dict[str, pd.DataFrame] = {}
    for name, factory in models.items():
        records = rolling_origin_backtest(
            daily, factory, config.backtest, config.forecast
        )
        reports[name] = summarize_backtest(records, label=name)
        backtest_frames[name] = backtest_records_to_frame(records)

    comparison = compare_forecasters(reports)
    save_json(output_dir / "forecast_backtest_report.json", reports)
    save_json(output_dir / "forecast_baseline_comparison.json", comparison)
    for name, frame in backtest_frames.items():
        frame.to_csv(output_dir / f"backtest_{name}.csv", index=False)

    # Final forecast artifact: train on the full history, forecast the horizon.
    forecaster_name = "patchtst" if use_patchtst else "seasonal_naive"
    forecaster = models[forecaster_name]()
    try:
        forecaster.fit(daily)
        forecast = forecaster.predict(daily)
    except Exception as exc:  # noqa: BLE001
        print(f"Final forecaster training/prediction failed: {exc}", file=sys.stderr)
        return 1

    training_cutoff = pd.to_datetime(daily["date"]).max().date().isoformat()
    issue_date = pd.Timestamp.now().date().isoformat()
    forecast["calibration_status"] = "experimental_uncalibrated"
    calendar_records = build_calendar_forecast_records(
        daily,
        climatology,
        config.anomaly,
        forecast=forecast,
        forecast_training_cutoff=training_cutoff,
        forecast_issue_date=issue_date,
    )
    try:
        validate_calendar_forecast_frame(calendar_records)
    except Exception as exc:  # noqa: BLE001
        print(f"Calendar/forecast contract validation failed: {exc}", file=sys.stderr)
        return 1
    calendar_records.to_csv(output_dir / "calendar_forecast_records.csv", index=False)

    artifact_dir = new_artifact_dir(output_dir, "forecaster")
    if forecaster_name == "patchtst":
        state = forecaster.state_dict()
    else:
        state = {
            "model_name": forecaster.name,
            "model_version": forecaster.version,
            "config": config.forecast.to_dict(),
            "note": "baseline forecaster; no learned weights",
        }
    manifest = build_manifest(
        pipeline="calendar_forecast",
        config=config.to_dict(),
        source_files=[{"path": Path(args.daily_csv).name}],
        model={
            "name": forecaster_name,
            "version": forecaster.version,
            "horizon_days": config.forecast.horizon_days,
            "quantiles": list(config.forecast.quantiles),
        },
        metrics={"backtest": reports},
        code_commit=resolve_code_commit(REPO_ROOT),
        limitations=FORECASTER_LIMITATIONS,
        extra={
            "seed": config.seed,
            "training_cutoff": training_cutoff,
            "forecast_issue_date": issue_date,
            "seasonal_baseline_version": climatology.version,
        },
    )
    model_path = save_forecaster_bundle(artifact_dir, state, manifest)
    print(f"Forecaster ({forecaster_name}) trained and saved: {model_path}")
    print(f"Backtest metrics: {Path(output_dir) / 'forecast_backtest_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
