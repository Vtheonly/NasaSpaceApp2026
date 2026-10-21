#!/usr/bin/env python3
"""Run the five-pass data gate over a source directory (issue #11).

Writes ``inventory_manifest.json`` and ``dataset_audit_report.json`` into the
output directory in every case, and exits non-zero when the gate fails closed
(so training can never start on unvalidated data).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.artifacts.serialization import save_json
from src.common.config import HarmonizationConfig
from src.common.logging import configure_logging
from src.data.gate import FivePassDataGate, assert_gate_passed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, help="Directory with FIRMS MODIS/VIIRS CSV files")
    parser.add_argument("--output-dir", required=True, help="Where gate reports are written")
    parser.add_argument("--config", default=None, help="Optional harmonization YAML (split boundaries etc.)")
    return parser


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    args = build_parser().parse_args(argv)
    config = HarmonizationConfig.from_yaml(args.config) if args.config else HarmonizationConfig()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    gate = FivePassDataGate(config)
    outcome = gate.run(args.source_dir)
    save_json(output_dir / "inventory_manifest.json", outcome.inventory_manifest)
    save_json(output_dir / "dataset_audit_report.json", outcome.audit_report)

    if outcome.passed:
        print(f"Data gate PASSED; reports written to {output_dir}")
        return 0
    try:
        assert_gate_passed(outcome)
    except Exception as exc:  # noqa: BLE001
        print(f"Data gate FAILED: {exc}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
