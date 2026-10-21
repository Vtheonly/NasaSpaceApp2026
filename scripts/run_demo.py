#!/usr/bin/env python3
"""Reproducible scientific demo: AOI -> harmonized history -> calendar (issue #16).

Runs the full end-to-end pipeline (scripts/run_pipeline.py) on a source
directory, then renders a self-contained ``DEMO_REPORT.md`` plus a
machine-readable ``demo_summary.json`` that:

- compares raw vs harmonized daily activity and marks the sensor-era
  transition;
- reports anomaly status counts and forecast availability/horizon/quantiles;
- links every scientific claim to the evaluation artifact that produced it
  (issue #16 acceptance criterion), and prints the fixed limitations block so
  estimates are never presented as ground truth.

When run on the committed fixtures (default source), the report is explicitly
labeled SYNTHETIC sample evidence; real-data runs must point --source-dir at a
verified FIRMS archive and re-run the five-pass gate first.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import pandas as pd  # noqa: E402

from src.artifacts.serialization import load_json, save_json  # noqa: E402
from src.common.logging import configure_logging, get_logger  # noqa: E402

LOGGER = get_logger(__name__)

SYNTHETIC_NOTE = (
    "All numbers below come from the committed SYNTHETIC sample fixtures "
    "(seed 2026) and demonstrate the workflow only; they are not measurements "
    "of real fire activity and must not be presented as such (docs/"
    "MODEL_CONTRACTS.md section 9, assumption 5)."
)

LIMITATIONS = [
    "Harmonized values are model estimates, not ground truth; they inherit "
    "MODIS detection-selection bias.",
    "Matched-calibration targets describe only fires both sensors could see.",
    "Forecast intervals are experimental/uncalibrated until measured on real data.",
    "Regional activity forecast only; never an exact ignition location/time claim.",
    "Missing observations are reported as missing coverage, never as zero activity.",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        default=str(REPO_ROOT / "tests" / "fixtures" / "sample_aoi"),
        help="FIRMS CSV directory (default: committed SYNTHETIC sample fixtures)",
    )
    parser.add_argument("--output-dir", required=True, help="Demo output directory")
    parser.add_argument("--aoi", default="bbox:-123.5,39.5,-120.5,42.0", help="AOI spec")
    parser.add_argument(
        "--forecaster",
        choices=("auto", "patchtst", "baseline", "none"),
        default="auto",
        help="Forecast choice passed through to run_pipeline",
    )
    parser.add_argument("--config", default=None, help="Harmonization YAML")
    parser.add_argument("--calendar-config", default=None, help="Calendar YAML")
    parser.add_argument("--seed", type=int, default=2026)
    return parser


def _read_csv(path: Path, parse_dates: bool = False) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path, parse_dates=["date"] if parse_dates else None)


def _fmt(value, digits: int = 3) -> str:
    try:
        if pd.isna(value):
            return "n/a"
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _series_comparison(harmonized: pd.DataFrame) -> dict:
    """Raw vs harmonized daily activity summary with the era transition."""
    counts = harmonized["value_type"].value_counts().to_dict()
    observed = harmonized[harmonized["value_type"] == "observed_viirs"]
    harmonized_days = harmonized[harmonized["value_type"] == "harmonized_estimate"]
    transition = None
    if not harmonized_days.empty:
        transition = str(pd.to_datetime(harmonized_days["date"]).min().date())
    return {
        "days_total": int(len(harmonized)),
        "value_type_counts": {k: int(v) for k, v in counts.items()},
        "observed_viirs_days": int(len(observed)),
        "observed_viirs_mean_frp_mw": (
            float(observed["value"].mean()) if len(observed) else None
        ),
        "harmonized_estimate_days": int(len(harmonized_days)),
        "harmonized_mean_frp_mw": (
            float(harmonized_days["value"].mean()) if len(harmonized_days) else None
        ),
        "missing_coverage_days": int(counts.get("missing_coverage", 0)),
        "sensor_era_transition_date": transition,
    }


def _harmonizer_evidence(train_dir: Path) -> dict:
    path = train_dir / "harmonization_evaluation.json"
    evaluation = load_json(path)
    summary = {}
    for split, reports in evaluation.items():
        if reports.get("skipped"):
            continue
        summary[split] = {
            name: report.get("tweedie_deviance") for name, report in reports.items()
        }
    return summary


def _anomaly_summary(calendar: pd.DataFrame) -> dict:
    counts = calendar["anomaly_status"].value_counts(dropna=False).to_dict()
    summary = {str(k): int(v) for k, v in counts.items() if not pd.isna(k)}
    # Forecast rows carry no anomaly score by contract (future days are not
    # "insufficient baseline" claims); count them separately for clarity.
    forecast_days = int(calendar["anomaly_status"].isna().sum())
    if forecast_days:
        summary["not_scored_forecast_days"] = forecast_days
    return summary


def _forecast_summary(calendar: pd.DataFrame) -> dict:
    forecast_rows = calendar[calendar["quality_state"] == "forecast"]
    if forecast_rows.empty:
        return {
            "available": False,
            "state": "no_forecast_available",
            "note": "Forecast columns stay empty; the calendar export still "
            "satisfies the contract (explicit no-forecast state).",
        }
    first = forecast_rows.iloc[0]
    return {
        "available": True,
        "horizon_days": int(len(forecast_rows)),
        "issue_date": str(first["forecast_issue_date"]),
        "training_cutoff": str(first["forecast_training_cutoff"]),
        "model": str(first["forecast_model"]),
        "calibration_status": str(first["forecast_calibration_status"]),
        "quantiles": ["q10", "median", "q90"],
    }


def render_report(
    pipeline_run: dict,
    series: dict,
    harmonizer: dict,
    anomalies: dict,
    forecast: dict,
    daily_summary: dict,
    source_label: str,
) -> str:
    lines: list[str] = []
    lines.append("# Harmonized Burning-Activity Demo Report")
    lines.append("")
    lines.append(f"- Pipeline status: **{pipeline_run['overall_status']}**")
    lines.append(f"- AOI: `{pipeline_run['aoi']}`")
    lines.append(f"- Data source: **{source_label}**")
    lines.append(f"- Code commit: `{pipeline_run.get('code_commit', 'unknown')}`")
    lines.append("")
    lines.append(f"> {SYNTHETIC_NOTE if source_label.startswith('SYNTHETIC') else ''}")
    lines.append("")

    lines.append("## 1. Daily activity (raw vs harmonized)")
    lines.append("")
    lines.append("| Metric | Value | Evidence |")
    lines.append("|---|---|---|")
    lines.append(
        f"| Days in series | {series['days_total']} | `inference/harmonized_daily.csv` |"
    )
    lines.append(
        f"| Observed VIIRS days | {series['observed_viirs_days']} "
        f"(mean FRP {_fmt(series['observed_viirs_mean_frp_mw'])} MW) | "
        "`inference/harmonized_daily.csv` |"
    )
    lines.append(
        f"| Harmonized MODIS-only days | {series['harmonized_estimate_days']} "
        f"(mean FRP {_fmt(series['harmonized_mean_frp_mw'])} MW) | "
        "`inference/harmonized_daily.csv` |"
    )
    lines.append(
        f"| Missing-coverage days | {series['missing_coverage_days']} "
        "(never rendered as zero) | `inference/harmonized_daily.csv` |"
    )
    lines.append(
        f"| Sensor-era transition | {series['sensor_era_transition_date'] or 'n/a'} "
        "(first harmonized-only day) | `inference/harmonized_daily.csv` |"
    )
    lines.append("")
    lines.append(
        f"Series span: {daily_summary.get('date_min')} to {daily_summary.get('date_max')} "
        f"({daily_summary.get('days_observed')} observed of "
        f"{daily_summary.get('days_total')} days; MODIS-observed "
        f"{daily_summary.get('modis_observed_days')}, VIIRS-observed "
        f"{daily_summary.get('viirs_observed_days')})."
    )
    lines.append("")

    lines.append("## 2. Harmonizer evaluation (held-out, chronological)")
    lines.append("")
    lines.append("| Split | identity | calibration | LightGBM/Tweedie |")
    lines.append("|---|---|---|---|")
    for split, reports in harmonizer.items():
        lines.append(
            f"| {split} | {_fmt(reports.get('identity'))} | "
            f"{_fmt(reports.get('calibration'))} | {_fmt(reports.get('lightgbm_tweedie'))} |"
        )
    lines.append("")
    lines.append(
        "Tweedie deviance on held-out chronological splits, model vs mandatory "
        "baselines; evidence: `harmonization/harmonization_evaluation.json`. "
        "On the sample fixtures the learned model must beat both baselines to "
        "count as evidence at all (tests/scientific/test_beats_baselines.py)."
    )
    lines.append("")

    lines.append("## 3. Anomaly status counts")
    lines.append("")
    lines.append("| Status | Days |")
    lines.append("|---|---|")
    for status, count in sorted(anomalies.items()):
        lines.append(f"| {status} | {count} |")
    lines.append("")
    lines.append(
        "Anomaly statuses are relative-to-season descriptions; "
        "`insufficient_baseline`/`missing_coverage` never make a numeric claim. "
        "Evidence: `inference/anomaly_records.csv`."
    )
    lines.append("")

    lines.append("## 4. Forecast")
    lines.append("")
    if forecast["available"]:
        lines.append(f"- Available: **yes** (model `{forecast['model']}`)")
        lines.append(f"- Horizon: {forecast['horizon_days']} days; quantiles {forecast['quantiles']}")
        lines.append(f"- Issue date: {forecast['issue_date']}; training cutoff: {forecast['training_cutoff']}")
        lines.append(f"- Calibration: **{forecast['calibration_status']}**")
    else:
        lines.append(f"- Available: **no** — explicit `{forecast['state']}` state")
        lines.append(f"- {forecast['note']}")
    lines.append("")
    lines.append("Evidence: `inference/calendar_forecast_records.csv`.")
    lines.append("")

    lines.append("## 5. Limitations (must accompany any retelling)")
    lines.append("")
    for item in LIMITATIONS:
        lines.append(f"- {item}")
    lines.append("")
    lines.append(
        "Full contracts, tolerances, and error behavior: "
        "`docs/MODEL_CONTRACTS.md`."
    )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    args = build_parser().parse_args(argv)
    import run_pipeline

    out = Path(args.output_dir)
    pipeline_dir = out / "pipeline"
    code = run_pipeline.main(
        [
            "--source-dir",
            args.source_dir,
            "--output-dir",
            str(pipeline_dir),
            "--aoi",
            args.aoi,
            "--forecaster",
            args.forecaster,
            "--seed",
            str(args.seed),
        ]
        + (["--config", args.config] if args.config else [])
        + (["--calendar-config", args.calendar_config] if args.calendar_config else [])
    )
    pipeline_run = None
    summary_path = pipeline_dir / "pipeline_run.json"
    if summary_path.exists():
        pipeline_run = load_json(summary_path)
    if (
        code != 0
        or pipeline_run is None
        or pipeline_run.get("overall_status") != "passed"
    ):
        print(
            f"Pipeline did not pass (exit {code}); demo report not rendered. "
            f"See {summary_path}",
            file=sys.stderr,
        )
        return code or 1

    harmonized = _read_csv(pipeline_dir / "inference" / "harmonized_daily.csv")
    calendar = _read_csv(pipeline_dir / "inference" / "calendar_forecast_records.csv")
    daily = _read_csv(
        pipeline_dir / "harmonization" / "daily_aoi_records.csv", parse_dates=True
    )
    if harmonized is None or calendar is None:
        print("Pipeline passed but expected inference outputs are missing.", file=sys.stderr)
        return 1

    from src.data.aggregation import summarize_daily

    series = _series_comparison(harmonized)
    harmonizer = _harmonizer_evidence(pipeline_dir / "harmonization")
    anomalies = _anomaly_summary(calendar)
    forecast = _forecast_summary(calendar)
    daily_summary = (
        summarize_daily(daily) if daily is not None else {}
    )

    is_fixture = str(Path(args.source_dir).resolve()).startswith(
        str(REPO_ROOT / "tests" / "fixtures")
    )
    source_label = (
        "SYNTHETIC committed fixtures (tests/fixtures/sample_aoi, seed 2026)"
        if is_fixture
        else f"user source directory {args.source_dir} (five-pass gate applied)"
    )

    report = render_report(
        pipeline_run, series, harmonizer, anomalies, forecast, daily_summary, source_label
    )
    report_path = out / "DEMO_REPORT.md"
    report_path.write_text(report, encoding="utf-8")

    summary = {
        "schema_version": "1.0.0",
        "pipeline": "scientific_demo",
        "source_label": source_label,
        "synthetic_evidence": is_fixture,
        "aoi": pipeline_run["aoi"],
        "seed": args.seed,
        "pipeline_run": pipeline_run,
        "series": series,
        "harmonizer_evaluation": harmonizer,
        "anomaly_status_counts": anomalies,
        "forecast": forecast,
        "limitations": LIMITATIONS,
    }
    save_json(out / "demo_summary.json", summary)

    print(f"Demo report: {report_path}")
    print(f"Machine-readable summary: {out / 'demo_summary.json'}")
    print(f"Forecast available: {forecast['available']}; anomaly statuses: {anomalies}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
