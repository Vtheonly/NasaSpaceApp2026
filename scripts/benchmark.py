#!/usr/bin/env python3
"""Stage-wise performance benchmark for the offline pipeline (issue #13).

Measures each pipeline stage separately on deterministic synthetic input of a
configurable size and writes ``benchmark_report.json``:

- five-pass gate (read + validate + inventory)
- AOI filtering + daily aggregation
- overlap-period matching
- harmonizer training (LightGBM/Tweedie, chronological early stopping)
- harmonizer inference (per-detection estimates + harmonized daily series)
- seasonal climatology + anomaly scoring
- calendar record assembly + artifact save/load round-trip

The report records the actual runtime environment (Python version, platform,
CPU count, optional torch availability), input sizes, repetitions, and
cold/warm state, so figures are reproducible measurements rather than copied
planning estimates. Graceful-failure behavior is *not* benchmarked here; it is
pinned as passing tests in tests/offline and tests/integration (documented in
docs/MODEL_CONTRACTS.md section 8) so the two evidence types stay separate.
"""

from __future__ import annotations

import argparse
import platform
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.artifacts.serialization import save_json  # noqa: E402
from src.common.config import (  # noqa: E402
    CalendarConfig,
    HarmonizationConfig,
    resolve_code_commit,
)
from src.common.logging import configure_logging, get_logger  # noqa: E402

LOGGER = get_logger(__name__)

AOI_SPEC = "bbox:-123.5,39.5,-120.5,42.0"
START_DATE = "2018-01-01"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--rows-per-product",
        type=int,
        default=20000,
        help="Synthetic rows per product (split across the configured span)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=2555,
        help="Number of calendar days the synthetic records span (default 2018-2024)",
    )
    parser.add_argument("--repeats", type=int, default=1, help="Repetitions per stage (median reported)")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--keep-synthetic-dir",
        default=None,
        help="Optional directory to keep the generated synthetic CSVs for inspection",
    )
    return parser


def generate_synthetic_records(
    rows: int, days: int, seed: int
) -> dict[str, pd.DataFrame]:
    """Deterministic MODIS/VIIRS frames with the required gate columns.

    Detection locations cluster around shared hotspot centers and detection
    times around shared overpass hours (~01/13 UTC slots), mirroring how
    fires and polar orbits actually cluster, so overlap-period matching finds
    positive VIIRS targets even at small benchmark sizes; FRP magnitudes are
    drawn from an exponential similar in scale to real active-fire values.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(START_DATE, periods=days, freq="D")
    # Shared hotspot centers: fires cluster in space, and both sensors must
    # see the SAME clusters or overlap matching finds nothing at small sizes.
    n_hotspots = 6
    hotspot_lat = rng.uniform(39.6, 41.9, size=n_hotspots)
    hotspot_lon = rng.uniform(-123.4, -120.6, size=n_hotspots)
    out: dict[str, pd.DataFrame] = {}
    for product, scale in (("modis", 1.0), ("viirs", 0.4)):
        n = max(rows, 100)
        day_idx = rng.integers(0, days, size=n)
        hour = rng.choice([1, 13], size=n)
        minute = rng.integers(0, 60, size=n)
        center = rng.integers(0, n_hotspots, size=n)
        frame = pd.DataFrame(
            {
                "latitude": hotspot_lat[center] + rng.normal(0.0, 0.006, size=n),
                "longitude": hotspot_lon[center] + rng.normal(0.0, 0.006, size=n),
                "acq_date": dates[day_idx].strftime("%Y-%m-%d"),
                "acq_time": [
                    f"{h:02d}{m:02d}" for h, m in zip(hour, minute, strict=True)
                ],
                "frp": rng.exponential(scale=scale * 30.0, size=n) + 0.1,
                "confidence": rng.choice(["l", "n", "h"], size=n, p=[0.1, 0.3, 0.6]),
                "satellite": "A" if product == "modis" else "NPP",
            }
        )
        # Deterministic sort keeps row order stable for both products.
        frame = frame.sort_values(["acq_date", "acq_time"], kind="mergesort").reset_index(
            drop=True
        )
        out[product] = frame
    return out


def _write_source_dir(
    frames: dict[str, pd.DataFrame], target: Path
) -> None:
    target.mkdir(parents=True, exist_ok=True)
    for product, frame in frames.items():
        frame.to_csv(target / f"{product}_active_fires.csv", index=False)


def _median_of(func, repeats: int) -> tuple[float, object]:
    """Run func() ``repeats`` times; return (median seconds, last result)."""
    timings: list[float] = []
    result = None
    for _ in range(max(1, repeats)):
        start = time.perf_counter()
        result = func()
        timings.append(time.perf_counter() - start)
    return float(np.median(timings)), result


def run_benchmark(args: argparse.Namespace) -> tuple[dict, int]:
    config = HarmonizationConfig()
    calendar_config = CalendarConfig()
    stages: dict[str, dict] = {}

    workdir = Path(args.output_dir)
    workdir.mkdir(parents=True, exist_ok=True)
    source_dir = Path(args.keep_synthetic_dir) if args.keep_synthetic_dir else (
        workdir / "synthetic_source"
    )
    _write_source_dir(
        generate_synthetic_records(args.rows_per_product, args.days, args.seed), source_dir
    )

    # --- Stage: gate ---------------------------------------------------------
    from src.data.gate import FivePassDataGate, assert_gate_passed

    def _gate():
        return FivePassDataGate(config).run(str(source_dir))

    seconds, outcome = _median_of(_gate, args.repeats)
    assert_gate_passed(outcome)  # fail-closed even in benchmark mode
    stages["five_pass_gate"] = {
        "seconds": round(seconds, 3),
        "rows_modis": int(len(outcome.frames["modis"])),
        "rows_viirs": int(len(outcome.frames["viirs"])),
    }

    # --- Stage: AOI filter + daily aggregation -------------------------------
    from src.data.aggregation import build_daily_frame, summarize_daily
    from src.geo.aoi import filter_records, parse_aoi

    aoi = parse_aoi(AOI_SPEC)

    def _aggregate():
        frames = {p: filter_records(f, aoi) for p, f in outcome.frames.items()}
        return build_daily_frame(frames, aoi_id=aoi.aoi_id)

    seconds, daily = _median_of(_aggregate, args.repeats)
    stages["aoi_filter_and_daily_aggregation"] = {
        "seconds": round(seconds, 3),
        "days": int(len(daily)),
    }

    # --- Stage: overlap matching ---------------------------------------------
    from src.data.schema import PRODUCT_MODIS, PRODUCT_VIIRS
    from src.geo.matching import build_matched_records

    modis = filter_records(outcome.frames[PRODUCT_MODIS], aoi)
    viirs = filter_records(outcome.frames[PRODUCT_VIIRS], aoi)

    def _match():
        return build_matched_records(modis, viirs, config.matching)

    seconds, matched = _median_of(_match, args.repeats)
    stages["overlap_matching"] = {
        "seconds": round(seconds, 3),
        "matched_rows": int(len(matched)),
    }

    # Fail fast with an actionable message when the synthetic input cannot
    # train anything (LightGBM would abort with a cryptic zero-label error).
    from src.geo.matching import TARGET_COLUMN, eligible_matches

    eligible = eligible_matches(matched)
    if eligible.empty or float(eligible[TARGET_COLUMN].sum()) <= 0:
        raise RuntimeError(
            "Synthetic input produced no positive calibration targets; "
            "increase --rows-per-product or --days so overlap-period matches "
            "find VIIRS detections with positive FRP."
        )

    # --- Stage: harmonizer training -------------------------------------------
    from src.harmonization.train import run_harmonization_training

    def _train():
        return run_harmonization_training(
            str(source_dir), config=config, aoi=aoi, gate_outcome=outcome
        )

    seconds, result = _median_of(_train, 1)  # training is not repeated
    stages["harmonizer_training"] = {
        "seconds": round(seconds, 3),
        "matched_rows": int(len(result.matched)),
        "boosting_iterations": int(result.model.params.get("num_leaves", 0) or 0) or None,
        "note": "single run (early stopping varies with the sample)",
    }

    # --- Stage: harmonizer inference -------------------------------------------
    from src.harmonization.inference import build_harmonized_daily

    def _infer():
        return build_harmonized_daily(
            daily, modis, viirs, result.model, config.matching
        )

    seconds, harmonized = _median_of(_infer, args.repeats)
    stages["harmonizer_inference"] = {
        "seconds": round(seconds, 3),
        "days": int(len(harmonized)),
    }

    # --- Stage: climatology + anomalies ---------------------------------------
    from src.calendar.anomalies import score_anomalies
    from src.calendar.seasonal import SeasonalClimatology

    def _anomalies():
        clim = SeasonalClimatology(calendar_config.seasonal).fit(
            harmonized["date"], harmonized["value"]
        )
        return clim, score_anomalies(harmonized, clim, calendar_config.anomaly)

    seconds, (clim, anomalies) = _median_of(_anomalies, args.repeats)
    stages["climatology_and_anomalies"] = {
        "seconds": round(seconds, 3),
        "days_scored": int(len(anomalies)),
    }

    # --- Stage: calendar records + artifact save/load -------------------------
    from src.artifacts.manifest import build_manifest
    from src.artifacts.serialization import (
        load_harmonizer_bundle,
        save_harmonizer_bundle,
    )
    from src.calendar.records import build_calendar_forecast_records

    artifact_dir = workdir / "benchmark_harmonizer_artifact"

    def _records_and_roundtrip():
        records = build_calendar_forecast_records(
            harmonized, clim, calendar_config.anomaly
        )
        bundle = {
            "model": result.model,
            "feature_names": result.feature_names,
            "config": result.config_snapshot,
            "aoi_id": result.aoi_id,
        }
        manifest = build_manifest(
            pipeline="benchmark",
            config={},
            source_files=[],
            model={"name": result.model.name, "version": result.model.version},
            metrics={},
            code_commit=resolve_code_commit(REPO_ROOT),
            limitations=["Benchmark artifact: synthetic input, timing evidence only."],
        )
        save_harmonizer_bundle(artifact_dir, bundle, manifest)
        load_harmonizer_bundle(artifact_dir)  # integrity-verified load
        return records

    seconds, records = _median_of(_records_and_roundtrip, args.repeats)
    stages["calendar_records_and_artifact_roundtrip"] = {
        "seconds": round(seconds, 3),
        "rows": int(len(records)),
    }

    report = {
        "schema_version": "1.0.0",
        "pipeline": "stage_benchmark",
        "created_at": datetime.now(UTC).isoformat(),
        "code_commit": resolve_code_commit(REPO_ROOT),
        "methodology": {
            "type": "median of N repetitions per stage (training: single run)",
            "repeats": int(args.repeats),
            "state": "cold (no warm caches between stages beyond pandas internals)",
            "clock": "time.perf_counter",
            "input": (
                "deterministic synthetic records (np.random.default_rng, "
                f"seed {args.seed}); NOT real FIRMS data"
            ),
            "rows_per_product": int(args.rows_per_product),
            "span_days": int(args.days),
            "failure_paths": (
                "graceful-failure behavior is covered by tests/offline and "
                "tests/integration (docs/MODEL_CONTRACTS.md section 8), not "
                "benchmarked here"
            ),
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "cpu_count": __import__("os").cpu_count(),
            "torch_available": None,
        },
        "aoi": AOI_SPEC,
        "stages": stages,
        "daily_summary": summarize_daily(daily),
    }
    from src.calendar.forecast import torch_available

    report["environment"]["torch_available"] = torch_available()
    return report, 0


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    args = build_parser().parse_args(argv)
    try:
        report, code = run_benchmark(args)
    except Exception as exc:  # noqa: BLE001
        LOGGER.error("Benchmark failed: %s", exc)
        print(f"Benchmark failed: {exc}", file=sys.stderr)
        return 1
    out = Path(args.output_dir) / "benchmark_report.json"
    save_json(out, report)
    total = sum(stage["seconds"] for stage in report["stages"].values())
    print(f"Benchmark report: {out}")
    for name, stage in report["stages"].items():
        print(f"  {name}: {stage['seconds']}s")
    print(f"  total (sum of stage medians): {round(total, 3)}s")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
