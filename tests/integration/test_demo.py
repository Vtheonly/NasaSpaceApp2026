"""Scientific demo tests: scripts/run_demo.py (issue #16).

The demo must be reproducible end-to-end on the committed synthetic fixtures,
render a report whose every claim links to evaluation evidence, and carry the
fixed limitations + synthetic-data labels (never present sample numbers as
real measurements).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

from tests.conftest import SAMPLE_DIR

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
for _path in (REPO_ROOT, SCRIPTS_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import run_demo  # noqa: E402

from src.artifacts.serialization import load_json  # noqa: E402


@pytest.fixture(scope="module")
def demo_run(tmp_path_factory):
    out = tmp_path_factory.mktemp("demo")
    code = run_demo.main(["--output-dir", str(out), "--seed", "2026"])
    assert code == 0
    return out


class TestDemoReport:
    def test_report_and_summary_written(self, demo_run):
        report = demo_run / "DEMO_REPORT.md"
        summary = demo_run / "demo_summary.json"
        assert report.exists() and summary.exists()
        assert "Harmonized Burning-Activity Demo Report" in report.read_text()

    def test_every_claim_has_evidence_links(self, demo_run):
        text = (demo_run / "DEMO_REPORT.md").read_text()
        assert "Evidence" in text
        # Each evaluation section points at its artifact file.
        for evidence in (
            "harmonization_evaluation.json",
            "anomaly_records.csv",
            "calendar_forecast_records.csv",
        ):
            assert evidence in text, evidence

    def test_synthetic_label_is_impossible_to_miss(self, demo_run):
        text = (demo_run / "DEMO_REPORT.md").read_text()
        summary = load_json(demo_run / "demo_summary.json")
        assert "SYNTHETIC" in text
        assert summary["synthetic_evidence"] is True
        assert "SYNTHETIC" in summary["source_label"]

    def test_raw_vs_harmonized_section_is_quantified(self, demo_run):
        summary = load_json(demo_run / "demo_summary.json")
        series = summary["series"]
        counts = series["value_type_counts"]
        assert counts["observed_viirs"] > 0
        assert counts["harmonized_estimate"] > 0
        assert counts["missing_coverage"] > 0
        # Counts must add up to the full series (no dropped days).
        harmonized = pd.read_csv(demo_run / "pipeline" / "inference" / "harmonized_daily.csv")
        assert sum(counts.values()) == len(harmonized)
        assert series["sensor_era_transition_date"] is not None

    def test_anomaly_counts_match_records(self, demo_run):
        summary = load_json(demo_run / "demo_summary.json")
        anomalies = summary["anomaly_status_counts"]
        calendar = pd.read_csv(demo_run / "pipeline" / "inference" / "calendar_forecast_records.csv")
        scored = calendar["anomaly_status"].dropna()
        assert sum(anomalies.values()) == len(calendar)
        for status in scored.unique():
            assert anomalies.get(status) == int((scored == status).sum())

    def test_harmonizer_table_reports_both_baselines(self, demo_run):
        summary = load_json(demo_run / "demo_summary.json")
        evaluation = summary["harmonizer_evaluation"]
        assert {"validation", "test"}.issubset(evaluation)
        for reports in evaluation.values():
            assert {"identity", "calibration", "lightgbm_tweedie"}.issubset(reports)
            assert all(v is not None for v in reports.values())

    def test_forecast_state_is_explicit(self, demo_run):
        summary = load_json(demo_run / "demo_summary.json")
        forecast = summary["forecast"]
        assert forecast["available"] is True
        assert forecast["horizon_days"] == 14
        assert forecast["calibration_status"] == "experimental_uncalibrated"

    def test_limitations_block_present(self, demo_run):
        text = (demo_run / "DEMO_REPORT.md").read_text()
        summary = load_json(demo_run / "demo_summary.json")
        assert "Limitations" in text
        assert any("not ground truth" in line for line in summary["limitations"])
        assert any("ignition" in line.lower() for line in summary["limitations"])


class TestDemoFailureAndFallback:
    def test_demo_aborts_when_pipeline_fails(self, tmp_path, monkeypatch):
        """A failing pipeline must never render a half-truthful report."""
        import types

        def _fail(argv=None):
            return 1

        fake_module = types.ModuleType("run_pipeline")
        fake_module.main = staticmethod(_fail)
        # run_demo imports run_pipeline inside main(); swap the sys.modules
        # entry so the in-function import resolves to the fake.
        monkeypatch.setitem(sys.modules, "run_pipeline", fake_module)
        code = run_demo.main(["--output-dir", str(tmp_path / "demo_bad")])
        assert code != 0
        assert not (tmp_path / "demo_bad" / "DEMO_REPORT.md").exists()

    def test_forecaster_none_demo_reports_no_forecast(self, tmp_path):
        out = tmp_path / "demo_nofc"
        code = run_demo.main(["--output-dir", str(out), "--forecaster", "none", "--seed", "2026"])
        assert code == 0
        summary = load_json(out / "demo_summary.json")
        assert summary["forecast"]["available"] is False
        assert summary["forecast"]["state"] == "no_forecast_available"
        text = (out / "DEMO_REPORT.md").read_text()
        assert "no_forecast_available" in text

    def test_user_source_dir_is_not_labeled_synthetic(self, tmp_path):
        """Pointing at a copied (non-fixture) source must drop the SYNTHETIC label
        and record the gate-applied provenance instead."""
        out = tmp_path / "demo_usersrc"
        user_dir = tmp_path / "user_source"
        user_dir.mkdir()
        for name in ("modis_active_fires.csv", "viirs_active_fires.csv"):
            (user_dir / name).write_bytes((SAMPLE_DIR / name).read_bytes())
        code = run_demo.main(["--output-dir", str(out), "--source-dir", str(user_dir), "--seed", "2026"])
        assert code == 0
        summary = load_json(out / "demo_summary.json")
        assert summary["synthetic_evidence"] is False
        assert "five-pass gate applied" in summary["source_label"]
