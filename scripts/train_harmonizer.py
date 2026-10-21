#!/usr/bin/env python3
"""Train the MODIS->VIIRS harmonizer (Pillar 1; issues #5 and #10).

Pipeline: five-pass gate (fail closed) -> AOI filter -> daily aggregation ->
overlap-period matching -> chronological split -> identity/calibration
baselines + LightGBM/Tweedie harmonizer -> held-out evaluation with FRP-range
and season slices -> versioned artifact + manifest.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.artifacts.manifest import build_manifest, source_file_entries
from src.artifacts.serialization import (
    load_json,
    new_artifact_dir,
    save_harmonizer_bundle,
    save_json,
)
from src.common.config import HarmonizationConfig, resolve_code_commit, seed_everything
from src.common.logging import configure_logging, get_logger
from src.data.adapters import list_source_files
from src.data.gate import DataGateError, FivePassDataGate, assert_gate_passed
from src.geo.aoi import parse_aoi
from src.harmonization.train import run_harmonization_training

LOGGER = get_logger(__name__)

HARMONIZER_LIMITATIONS = [
    "Harmonized values are model estimates, not ground truth (docs/ARCHITECTURE.md).",
    "Matched-calibration targets inherit MODIS detection-selection bias: fires never "
    "detected by MODIS cannot be recovered by the regression.",
    "Region and land-cover/geometry error slices require auxiliary joins not yet implemented.",
    "Extreme-FRP records are kept as review flags, not removed (docs/TESTING.md).",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, help="Directory with FIRMS MODIS/VIIRS CSV files")
    parser.add_argument("--output-dir", required=True, help="Base directory for reports and artifacts")
    parser.add_argument("--config", default=None, help="Harmonization YAML (defaults used when omitted)")
    parser.add_argument("--aoi", default=None, help="AOI spec, e.g. bbox:-123.5,39.5,-120.5,42.0")
    parser.add_argument("--seed", type=int, default=None, help="Override the configured seed")
    return parser


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    args = build_parser().parse_args(argv)
    config = HarmonizationConfig.from_yaml(args.config) if args.config else HarmonizationConfig()
    if args.seed is not None:
        config = HarmonizationConfig.from_dict({**config.to_dict(), "seed": int(args.seed)})
    seed_everything(config.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    aoi = parse_aoi(args.aoi) if args.aoi else None

    # Gate first so reports exist even when training aborts (fail-closed).
    gate = FivePassDataGate(config)
    outcome = gate.run(args.source_dir)
    save_json(output_dir / "inventory_manifest.json", outcome.inventory_manifest)
    save_json(output_dir / "dataset_audit_report.json", outcome.audit_report)
    try:
        assert_gate_passed(outcome)
    except DataGateError as exc:
        print(f"Five-pass data gate failed; training aborted: {exc}", file=sys.stderr)
        return 1

    try:
        result = run_harmonization_training(
            args.source_dir, config=config, aoi=aoi, gate_outcome=outcome
        )
    except Exception as exc:  # noqa: BLE001
        print(f"Harmonization training failed: {exc}", file=sys.stderr)
        return 1

    # Persist evaluation evidence.
    save_json(output_dir / "harmonization_evaluation.json", result.evaluation)
    save_json(output_dir / "baseline_comparison.json", result.baseline_comparison)
    result.daily.to_csv(output_dir / "daily_aoi_records.csv", index=False)
    result.matched.to_csv(output_dir / "matched_records.csv", index=False)

    # Versioned model artifact + manifest (hash-verified on load).
    artifact_dir = new_artifact_dir(output_dir, "harmonizer")
    bundle = {
        "model": result.model,
        "feature_names": result.feature_names,
        "config": result.config_snapshot,
        "aoi_id": result.aoi_id,
        "split_boundaries": {
            "train_end": config.split.train_end,
            "validation_end": config.split.validation_end,
        },
        "metrics": {
            split: {name: report.get("tweedie_deviance") for name, report in reports.items()}
            for split, reports in result.evaluation.items()
            if not reports.get("skipped")
        },
        "notes": result.notes,
    }
    source_paths = [
        p
        for product in ("modis", "viirs")
        for p in list_source_files(args.source_dir, product)
    ]
    manifest = build_manifest(
        pipeline="harmonization",
        config=result.config_snapshot,
        source_files=source_file_entries(source_paths),
        model={
            "name": result.model.name,
            "version": result.model.version,
            "params": result.model.params,
            "feature_names": result.feature_names,
        },
        metrics=result.evaluation,
        code_commit=resolve_code_commit(REPO_ROOT),
        limitations=HARMONIZER_LIMITATIONS,
        extra={
            "seed": config.seed,
            "aoi_id": result.aoi_id,
            "split_boundaries": {
                "train_end": config.split.train_end,
                "validation_end": config.split.validation_end,
            },
        },
    )
    model_path = save_harmonizer_bundle(artifact_dir, bundle, manifest)
    print(f"Harmonizer trained and saved: {model_path}")
    print(f"Artifact manifest: {Path(artifact_dir) / 'manifest.json'}")
    for split_name, reports in result.evaluation.items():
        if reports.get("skipped"):
            continue
        summary = ", ".join(
            f"{name}: tweedie={report.get('tweedie_deviance'):.3f}"
            for name, report in reports.items()
        )
        print(f"[{split_name}] {summary}")
    _ = load_json  # keep import used for parity with other scripts
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
