"""Performance smoke tests: scripts/benchmark.py (issue #13).

The benchmark report must contain *measured* stage timings with an explicit
methodology block (environment, input size, repetitions, clock, synthetic
label) - reproducible measurements, not copied planning estimates. The
generous upper bounds here only catch pathological regressions so CI stays
stable; real benchmarking runs scripts/benchmark.py at full size and records
the report offline.
"""

from __future__ import annotations

import sys

import pytest

from tests.conftest import REPO_ROOT

SCRIPTS_DIR = REPO_ROOT / "scripts"
for _path in (REPO_ROOT, SCRIPTS_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import benchmark  # noqa: E402

from src.artifacts.serialization import load_json  # noqa: E402

EXPECTED_STAGES = (
    "five_pass_gate",
    "aoi_filter_and_daily_aggregation",
    "overlap_matching",
    "harmonizer_training",
    "harmonizer_inference",
    "climatology_and_anomalies",
    "calendar_records_and_artifact_roundtrip",
)

# Generous regression guards (seconds) - CI machines vary widely.
PER_STAGE_CEILING_SECONDS = 60.0


@pytest.mark.performance
class TestBenchmarkReport:
    def test_report_written_with_all_stages(self, tmp_path):
        code = benchmark.main(
            [
                "--output-dir", str(tmp_path),
                "--rows-per-product", "1500",
                "--days", "365",
                "--repeats", "1",
                "--seed", "2026",
            ]
        )
        assert code == 0
        report = load_json(tmp_path / "benchmark_report.json")
        for stage in EXPECTED_STAGES:
            assert stage in report["stages"], stage
            assert report["stages"][stage]["seconds"] > 0

    def test_methodology_is_recorded(self, tmp_path):
        code = benchmark.main(
            ["--output-dir", str(tmp_path), "--rows-per-product", "1200", "--days", "365"]
        )
        assert code == 0
        report = load_json(tmp_path / "benchmark_report.json")
        method = report["methodology"]
        assert method["repeats"] >= 1
        assert method["clock"] == "time.perf_counter"
        assert "synthetic" in method["input"].lower()
        assert method["rows_per_product"] == 1200
        assert "tests/offline" in method["failure_paths"]
        # Environment block: actual measured runtime, not planning estimates.
        env = report["environment"]
        assert env["python"]
        assert isinstance(env["cpu_count"], int)
        assert "torch_available" in env

    def test_stage_times_bounded(self, tmp_path):
        code = benchmark.main(
            ["--output-dir", str(tmp_path), "--rows-per-product", "1500", "--days", "365"]
        )
        assert code == 0
        report = load_json(tmp_path / "benchmark_report.json")
        for stage, entry in report["stages"].items():
            assert entry["seconds"] < PER_STAGE_CEILING_SECONDS, f"{stage} regressed"

    def test_report_records_row_counts_and_daily_summary(self, tmp_path):
        code = benchmark.main(
            ["--output-dir", str(tmp_path), "--rows-per-product", "1200", "--days", "365"]
        )
        assert code == 0
        report = load_json(tmp_path / "benchmark_report.json")
        stages = report["stages"]
        assert stages["five_pass_gate"]["rows_modis"] > 0
        assert stages["five_pass_gate"]["rows_viirs"] > 0
        assert stages["overlap_matching"]["matched_rows"] >= 0
        assert report["daily_summary"]["days_total"] > 0
