#!/usr/bin/env python3
"""One-command end-to-end workflow: AOI -> harmonized history -> calendar.

Orchestrates the four existing CLI stages (issue #8) with explicit fail-closed
semantics and a machine-readable run summary:

1. ``validate_data``      - five-pass gate; audit reports written even on failure
2. ``train_harmonizer``   - matched records -> baselines + Tweedie GBDT -> artifact
3. ``train_forecaster``   - seasonal/anomaly baselines + PatchTST (or seasonal-naive
                            without PyTorch) -> rolling-origin backtests -> artifact
4. ``run_inference``      - harmonized daily series, calendar, anomalies, forecast export

Every stage's exit code is propagated; a failing stage aborts the run and later
stages are recorded as ``skipped`` in ``pipeline_run.json``. The forecast stage is
optional (``--forecaster none``) and the inference export then reports the
explicit "no forecast available" state instead of failing (docs/MODEL_CONTRACTS
section 8).
"""

from __future__ import annotations

import argparse
import importlib
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from src.artifacts.serialization import load_harmonizer_bundle, save_json  # noqa: E402
from src.common.config import (  # noqa: E402
    HarmonizationConfig,
    resolve_code_commit,
)
from src.common.logging import configure_logging, get_logger  # noqa: E402

LOGGER = get_logger(__name__)

STAGES = ("validate_data", "train_harmonizer", "train_forecaster", "run_inference")


def _load_script_module(name: str):
    """Import a sibling CLI script as a module (logic stays in scripts/)."""
    return importlib.import_module(name)


def _latest_artifact_dir(base: Path, pipeline: str) -> Path | None:
    """Find the most recently modified ``<pipeline>_*`` artifact dir under base."""
    candidates = sorted(
        (p for p in base.glob(f"{pipeline}_*") if p.is_dir()),
        key=lambda p: p.stat().st_mtime,
    )
    return candidates[-1] if candidates else None


def _run_stage(
    name: str, argv: list[str]
) -> tuple[int, float]:
    """Run one CLI stage, returning (exit_code, seconds). Never raises."""
    module = _load_script_module(name)
    start = time.perf_counter()
    try:
        code = int(module.main(argv))
    except SystemExit as exc:  # argparse errors and explicit exits
        code = int(exc.code or 0)
    except Exception as exc:  # noqa: BLE001
        LOGGER.error("Stage %s crashed: %s", name, exc)
        code = 1
    return code, time.perf_counter() - start


def _build_harmonized_series(
    source_dir: str,
    aoi_spec: str,
    config: HarmonizationConfig,
    harmonizer_dir: Path,
    calendar_dir: Path,
) -> Path:
    """Build the harmonized daily series the forecaster consumes.

    Mirrors run_inference steps 1-3 (gate -> AOI -> daily -> harmonize) so the
    forecaster trains on exactly the series the exported forecast describes
    (docs/ARCHITECTURE.md: "Harmonized daily series" feeds the calendar).
    """
    from src.data.aggregation import build_daily_frame
    from src.data.gate import FivePassDataGate, assert_gate_passed
    from src.data.schema import PRODUCT_MODIS, PRODUCT_VIIRS
    from src.geo.aoi import filter_records, parse_aoi
    from src.harmonization.inference import build_harmonized_daily

    outcome = FivePassDataGate(config).run(source_dir)
    assert_gate_passed(outcome)
    aoi = parse_aoi(aoi_spec)
    frames = {p: filter_records(f, aoi) for p, f in outcome.frames.items()}
    daily = build_daily_frame(frames, aoi_id=aoi.aoi_id)
    bundle = load_harmonizer_bundle(harmonizer_dir)
    harmonized = build_harmonized_daily(
        daily, frames[PRODUCT_MODIS], frames[PRODUCT_VIIRS], bundle["model"], config.matching
    )
    path = calendar_dir / "harmonized_daily.csv"
    harmonized.to_csv(path, index=False)
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, help="Directory with FIRMS MODIS/VIIRS CSV files")
    parser.add_argument("--output-dir", required=True, help="Base directory for all stage outputs")
    parser.add_argument("--aoi", required=True, help="AOI spec, e.g. bbox:-123.5,39.5,-120.5,42.0")
    parser.add_argument("--config", default=None, help="Harmonization YAML")
    parser.add_argument("--calendar-config", default=None, help="Calendar YAML")
    parser.add_argument(
        "--forecaster",
        choices=("auto", "patchtst", "baseline", "none"),
        default="auto",
        help=(
            "auto: PatchTST when PyTorch is installed, seasonal-naive otherwise; "
            "baseline: force seasonal-naive/persistence backtests; "
            "none: skip the forecast stage (inference reports no-forecast state)"
        ),
    )
    parser.add_argument("--as-of", default=None, help="Forecast issue date (YYYY-MM-DD; default last observed)")
    parser.add_argument("--seed", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    args = build_parser().parse_args(argv)
    out = Path(args.output_dir)
    gate_dir = out / "gate"
    train_dir = out / "harmonization"
    calendar_dir = out / "calendar"
    inference_dir = out / "inference"
    for directory in (out, gate_dir, train_dir, calendar_dir, inference_dir):
        directory.mkdir(parents=True, exist_ok=True)

    from src.calendar.forecast import torch_available

    skip_forecast_stage = args.forecaster == "none"
    skip_patchtst = args.forecaster in ("auto", "baseline") or not torch_available()
    if args.forecaster == "patchtst" and not torch_available():
        print(
            "PyTorch is unavailable; --forecaster patchtst falls back to the "
            "seasonal-naive baseline (train_forecaster prints the same warning).",
            file=sys.stderr,
        )
        skip_patchtst = True

    run: dict = {
        "schema_version": "1.0.0",
        "pipeline": "end_to_end_aoi_to_calendar",
        "started_at": datetime.now(UTC).isoformat(),
        "code_commit": resolve_code_commit(REPO_ROOT),
        "source_dir": str(Path(args.source_dir).resolve()),
        "aoi": args.aoi,
        "forecaster_choice": args.forecaster,
        "stage_order": list(STAGES),
        "stages": {},
    }

    # Stage 1 - five-pass gate (fail closed; reports always written).
    code, seconds = _run_stage(
        "validate_data",
        ["--source-dir", args.source_dir, "--output-dir", str(gate_dir)]
        + (["--config", args.config] if args.config else []),
    )
    run["stages"]["validate_data"] = {
        "status": "passed" if code == 0 else "failed",
        "exit_code": code,
        "seconds": round(seconds, 3),
        "reports": [str(gate_dir / "inventory_manifest.json"), str(gate_dir / "dataset_audit_report.json")],
    }
    if code != 0:
        return _finish(run, out, code)

    # Stage 2 - harmonizer training.
    code, seconds = _run_stage(
        "train_harmonizer",
        ["--source-dir", args.source_dir, "--output-dir", str(train_dir), "--aoi", args.aoi]
        + (["--config", args.config] if args.config else [])
        + (["--seed", str(args.seed)] if args.seed is not None else []),
    )
    harmonizer_dir = _latest_artifact_dir(train_dir, "harmonizer")
    run["stages"]["train_harmonizer"] = {
        "status": "passed" if code == 0 else "failed",
        "exit_code": code,
        "seconds": round(seconds, 3),
        "artifact_dir": str(harmonizer_dir) if harmonizer_dir else None,
        "daily_records": str(train_dir / "daily_aoi_records.csv"),
        "matched_records": str(train_dir / "matched_records.csv"),
    }
    if code != 0 or harmonizer_dir is None:
        return _finish(run, out, code or 1)

    # Stage 3 - forecaster training (optional).
    if skip_forecast_stage:
        run["stages"]["train_forecaster"] = {
            "status": "skipped",
            "exit_code": None,
            "seconds": None,
            "reason": "--forecaster none",
        }
        forecaster_dir = None
    else:
        # Build the harmonized daily series first: the forecaster must train on
        # the same series run_inference forecasts from (not the raw daily AOI
        # frame), keeping raw/harmonized histories aligned (issue #8).
        stage_started = time.perf_counter()
        try:
            config = (
                HarmonizationConfig.from_yaml(args.config)
                if args.config
                else HarmonizationConfig()
            )
            harmonized_csv = _build_harmonized_series(
                args.source_dir, args.aoi, config, harmonizer_dir, calendar_dir
            )
            code = 0
        except Exception as exc:  # noqa: BLE001
            LOGGER.error("Harmonized series build failed: %s", exc)
            print(f"Harmonized series build failed: {exc}", file=sys.stderr)
            harmonized_csv = None
            code = 1
        if code != 0:
            run["stages"]["train_forecaster"] = {
                "status": "failed",
                "exit_code": code,
                "seconds": round(time.perf_counter() - stage_started, 3),
                "artifact_dir": None,
                "failed_at": "harmonized_series_build",
            }
            return _finish(run, out, code)
        extra = ["--skip-patchtst"] if skip_patchtst else []
        code, seconds = _run_stage(
            "train_forecaster",
            [
                "--daily-csv",
                str(harmonized_csv),
                "--output-dir",
                str(calendar_dir),
            ]
            + (["--config", args.calendar_config] if args.calendar_config else [])
            + (["--seed", str(args.seed)] if args.seed is not None else [])
            + extra,
        )
        forecaster_dir = _latest_artifact_dir(calendar_dir, "forecaster")
        run["stages"]["train_forecaster"] = {
            "status": "passed" if code == 0 else "failed",
            "exit_code": code,
            "seconds": round(seconds, 3),
            "artifact_dir": str(forecaster_dir) if forecaster_dir else None,
            "patchtst_used": not skip_patchtst,
            "harmonized_series": str(harmonized_csv),
        }
        if code != 0 or forecaster_dir is None:
            return _finish(run, out, code or 1)

    # Stage 4 - inference export (calendar + anomalies + optional forecast).
    code, seconds = _run_stage(
        "run_inference",
        [
            "--source-dir",
            args.source_dir,
            "--harmonizer-dir",
            str(harmonizer_dir),
            "--output-dir",
            str(inference_dir),
            "--aoi",
            args.aoi,
        ]
        + (["--forecaster-dir", str(forecaster_dir)] if forecaster_dir else [])
        + (["--config", args.config] if args.config else [])
        + (["--calendar-config", args.calendar_config] if args.calendar_config else [])
        + (["--as-of", args.as_of] if args.as_of else [])
        + (["--seed", str(args.seed)] if args.seed is not None else []),
    )
    run["stages"]["run_inference"] = {
        "status": "passed" if code == 0 else "failed",
        "exit_code": code,
        "seconds": round(seconds, 3),
        "calendar_records": str(inference_dir / "calendar_forecast_records.csv"),
        "harmonized_daily": str(inference_dir / "harmonized_daily.csv"),
        "anomaly_records": str(inference_dir / "anomaly_records.csv"),
        "forecast_available": forecaster_dir is not None,
    }
    return _finish(run, out, code)


def _finish(run: dict, out: Path, code: int) -> int:
    run["finished_at"] = datetime.now(UTC).isoformat()
    run["overall_status"] = "passed" if code == 0 else "failed"
    save_json(out / "pipeline_run.json", run)
    print(
        f"Pipeline {'completed' if code == 0 else 'failed'} (exit {code}); "
        f"run summary: {out / 'pipeline_run.json'}"
    )
    return code


if __name__ == "__main__":
    raise SystemExit(main())
