"""End-to-end orchestrator tests: scripts/run_pipeline.py (issue #8).

A reviewer must be able to complete the core user journey - AOI -> harmonized
history -> calendar - with one command, and every failure mode must surface as
an explicit recorded stage status in ``pipeline_run.json`` (graceful failure,
issue #13): gate failures abort before training, a skipped forecaster still
produces a no-forecast calendar export, and empty AOIs never fabricate zero
activity.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

from tests.conftest import AOI_SPEC, SAMPLE_DIR

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
for _path in (REPO_ROOT, SCRIPTS_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import run_pipeline  # noqa: E402

from src.artifacts.serialization import load_json  # noqa: E402
from src.calendar.records import CALENDAR_FORECAST_COLUMNS  # noqa: E402


def _run_pipeline(workdir: Path, *extra: str, aoi: str = AOI_SPEC) -> tuple[int, dict]:
    out = workdir / ("pipeline_" + "_".join(x.replace("/", "_") for x in extra) or "pipeline")
    code = run_pipeline.main(
        ["--source-dir", str(SAMPLE_DIR), "--output-dir", str(out), "--aoi", aoi, "--seed", "2026", *extra]
    )
    run_summary = load_json(out / "pipeline_run.json")
    return code, run_summary


class TestFullPipeline:
    def test_all_stages_pass_and_outputs_exist(self, tmp_path):
        code, run = _run_pipeline(tmp_path)
        assert code == 0
        assert run["overall_status"] == "passed"
        for stage, entry in run["stages"].items():
            assert entry["status"] == "passed", stage
        # Machine-readable evidence per stage (issue #13 style timing records).
        for stage in ("validate_data", "train_harmonizer", "train_forecaster", "run_inference"):
            assert run["stages"][stage]["seconds"] >= 0
        inference = Path(run["stages"]["run_inference"]["calendar_records"])
        assert inference.exists()
        calendar = pd.read_csv(inference)
        assert set(CALENDAR_FORECAST_COLUMNS).issubset(calendar.columns)
        # Contract: forecast rows exist only because a forecaster was trained.
        assert (calendar["quality_state"] == "forecast").any()
        assert not calendar["forecast_model"].dropna().empty

    def test_gate_reports_written(self, tmp_path):
        _, run = _run_pipeline(tmp_path)
        gate = run["stages"]["validate_data"]
        for report in gate["reports"]:
            assert Path(report).exists(), report

    def test_harmonizer_artifact_is_versioned_and_hash_verified(self, tmp_path):
        _, run = _run_pipeline(tmp_path)
        artifact = Path(run["stages"]["train_harmonizer"]["artifact_dir"])
        assert artifact.name.startswith("harmonizer_")
        assert (artifact / "model.joblib").exists()
        assert (artifact / "manifest.json").exists()
        # Loading verifies the recorded SHA-256 (ArtifactIntegrityError otherwise).
        from src.artifacts.serialization import load_harmonizer_bundle

        bundle = load_harmonizer_bundle(artifact)
        assert bundle["model"] is not None

    def test_forecaster_trained_on_harmonized_series(self, tmp_path):
        _, run = _run_pipeline(tmp_path)
        series = Path(run["stages"]["train_forecaster"]["harmonized_series"])
        harmonized = pd.read_csv(series)
        assert {"value", "value_type", "sensor_era", "coverage_status"}.issubset(
            harmonized.columns
        )
        assert run["stages"]["train_forecaster"]["artifact_dir"]

    def test_pipeline_run_json_is_valid_json_with_stage_order(self, tmp_path):
        code, run = _run_pipeline(tmp_path)
        assert code == 0
        expected = ["validate_data", "train_harmonizer", "train_forecaster", "run_inference"]
        # The stages mapping is key-sorted on disk; execution order is recorded
        # explicitly in stage_order so consumers never rely on dict order.
        assert run["stage_order"] == expected
        assert set(run["stages"].keys()) == set(expected)
        raw = json.loads(json.dumps(run))  # round-trip serializable
        assert raw["schema_version"] == "1.0.0"


class TestForecasterSkipped:
    def test_forecaster_none_yields_explicit_no_forecast_state(self, tmp_path):
        code, run = _run_pipeline(tmp_path, "--forecaster", "none")
        assert code == 0
        assert run["stages"]["train_forecaster"]["status"] == "skipped"
        calendar = pd.read_csv(run["stages"]["run_inference"]["calendar_records"])
        # No forecast rows; forecast columns stay NaN (explicit state).
        assert not (calendar["quality_state"] == "forecast").any()
        assert calendar["forecast_model"].isna().all()
        # The calendar itself is still complete and contract-valid.
        from src.calendar.records import validate_calendar_forecast_frame

        validate_calendar_forecast_frame(calendar)


class TestGracefulFailures:
    def test_gate_failure_aborts_before_training(self, tmp_path):
        """A dirty source directory fails the gate with reports written."""
        bad_source = tmp_path / "bad_source"
        bad_source.mkdir()
        # A structural missing-column error is recorded in the gate reports.
        import shutil

        shutil.copy(
            Path(REPO_ROOT) / "tests" / "fixtures" / "raw_dirty" / "modis_missing_column.csv",
            bad_source / "modis_missing_column.csv",
        )
        out = tmp_path / "pipeline_badgate"
        code = run_pipeline.main(
            [
                "--source-dir", str(bad_source),
                "--output-dir", str(out),
                "--aoi", AOI_SPEC,
            ]
        )
        assert code != 0
        run = load_json(out / "pipeline_run.json")
        assert run["stages"]["validate_data"]["status"] == "failed"
        assert "train_harmonizer" not in run["stages"]
        # Reports are still written even though the gate failed (fail-closed).
        assert (out / "gate" / "dataset_audit_report.json").exists()

    def test_missing_source_dir_is_recorded_not_raised(self, tmp_path):
        """A nonexistent source dir raises inside the gate; the orchestrator
        records the failure and never escapes with a traceback."""
        out = tmp_path / "pipeline_nodir"
        code = run_pipeline.main(
            [
                "--source-dir", str(tmp_path / "no_such_dir"),
                "--output-dir", str(out),
                "--aoi", AOI_SPEC,
            ]
        )
        assert code != 0
        run = load_json(out / "pipeline_run.json")
        assert run["stages"]["validate_data"]["status"] == "failed"
        assert "train_harmonizer" not in run["stages"]

    def test_empty_aoi_never_fabricates_activity(self, tmp_path):
        code, run = _run_pipeline(
            tmp_path, aoi="bbox:0.0,0.0,1.0,1.0"  # far from the fixture data
        )
        assert code != 0
        assert run["stages"]["train_harmonizer"]["status"] == "failed"
        # No inference stage ran: no silent zero-activity series was produced.
        assert "run_inference" not in run["stages"]

    def test_crashing_stage_is_recorded_not_raised(self, tmp_path, monkeypatch):
        """A stage that raises is converted into a recorded failure (never a
        traceback escaping the orchestrator)."""
        import train_harmonizer as script

        def _boom(argv=None):
            raise RuntimeError("simulated stage crash")

        monkeypatch.setattr(script, "main", _boom)
        out = tmp_path / "pipeline_crash"
        code = run_pipeline.main(
            [
                "--source-dir", str(SAMPLE_DIR),
                "--output-dir", str(out),
                "--aoi", AOI_SPEC,
            ]
        )
        assert code != 0
        run = load_json(out / "pipeline_run.json")
        assert run["stages"]["train_harmonizer"]["status"] == "failed"


@pytest.mark.parametrize("choice", ["auto", "baseline"])
class TestForecasterChoices:
    def test_baseline_paths_produce_a_forecaster_artifact(self, tmp_path, choice):
        code, run = _run_pipeline(tmp_path, "--forecaster", choice)
        assert code == 0
        forecaster = run["stages"]["train_forecaster"]
        assert forecaster["status"] == "passed"
        assert Path(forecaster["artifact_dir"]).exists()
