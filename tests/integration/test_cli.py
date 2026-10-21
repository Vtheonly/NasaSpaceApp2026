"""CLI integration tests: run the committed scripts end-to-end as subprocesses.

Covers issue #13 graceful failure paths (dirty data exits non-zero with
reports written; empty AOI produces an explicit empty state) and the full
offline workflow: validate -> train harmonizer -> train forecaster -> infer.
"""

import json

import pandas as pd
import pytest

from tests.conftest import (
    AOI_SPEC,
    DIRTY_DIR,
    SAMPLE_DIR,
    run_script,
    tiny_calendar_config,
)


@pytest.fixture(scope="module")
def workdir(tmp_path_factory):
    return tmp_path_factory.mktemp("cli")


@pytest.fixture(scope="module")
def trained_harmonizer_dir(workdir):
    """Train the harmonizer once for the CLI module; return artifact dir."""
    output = workdir / "train_harmonizer"
    run_script(
        "train_harmonizer.py",
        "--source-dir", str(SAMPLE_DIR),
        "--output-dir", str(output),
        "--aoi", AOI_SPEC,
    )
    artifact_dirs = sorted(
        d for d in output.iterdir() if d.is_dir() and d.name.startswith("harmonizer_")
    )
    assert artifact_dirs, "expected a versioned harmonizer artifact directory"
    return artifact_dirs[-1]


class TestValidateData:
    def test_clean_source_passes_and_writes_reports(self, workdir):
        output = workdir / "validate_clean"
        run_script(
            "validate_data.py",
            "--source-dir", str(SAMPLE_DIR),
            "--output-dir", str(output),
        )
        assert (output / "inventory_manifest.json").is_file()
        assert (output / "dataset_audit_report.json").is_file()
        report = json.loads((output / "dataset_audit_report.json").read_text())
        assert report["gate_passed"] is True

    def test_dirty_source_fails_closed_with_reports(self, workdir):
        output = workdir / "validate_dirty"
        proc = run_script(
            "validate_data.py",
            "--source-dir", str(DIRTY_DIR),
            "--output-dir", str(output),
            expect=1,
        )
        assert "gate FAILED" in proc.stderr or "failed" in proc.stderr.lower()
        assert (output / "inventory_manifest.json").is_file()
        report = json.loads((output / "dataset_audit_report.json").read_text())
        assert report["gate_passed"] is False


class TestTrainHarmonizer:
    def test_exit_zero_and_outputs(self, workdir, trained_harmonizer_dir):
        output = workdir / "train_harmonizer"
        # (training ran in the module fixture; verify its outputs here)
        assert (output / "harmonization_evaluation.json").is_file()
        assert (output / "baseline_comparison.json").is_file()
        assert (output / "daily_aoi_records.csv").is_file()
        assert (output / "matched_records.csv").is_file()
        evaluation = json.loads((output / "harmonization_evaluation.json").read_text())
        assert "validation" in evaluation and "test" in evaluation

    def test_manifest_contract(self, trained_harmonizer_dir):
        manifest = json.loads((trained_harmonizer_dir / "manifest.json").read_text())
        for key in ("schema_version", "created_at", "pipeline", "code_commit", "config",
                    "source_files", "model", "metrics", "limitations"):
            assert key in manifest
        assert len(manifest["model"]["sha256"]) == 64
        assert manifest["limitations"]  # documented limitations are mandatory

    def test_model_hash_matches_file(self, trained_harmonizer_dir):
        import hashlib

        model_file = trained_harmonizer_dir / "model.joblib"
        manifest = json.loads((trained_harmonizer_dir / "manifest.json").read_text())
        actual = hashlib.sha256(model_file.read_bytes()).hexdigest()
        assert actual == manifest["model"]["sha256"]


class TestTrainForecaster:
    @pytest.fixture(scope="class")
    def forecaster_output(self, workdir, tmp_path_factory, trained_harmonizer_dir):
        # Depends on trained_harmonizer_dir so the daily CSV is guaranteed to exist.
        # Tiny calendar config so the subprocess stays fast.
        tiny_yaml = tmp_path_factory.mktemp("cfg") / "calendar_tiny.yaml"
        tiny_calendar_config(min_train_days=120).to_dict()  # sanity: serializable
        import yaml

        with open(tiny_yaml, "w", encoding="utf-8") as fh:
            yaml.safe_dump(tiny_calendar_config().to_dict(), fh)
        daily = pd.read_csv(workdir / "train_harmonizer" / "daily_aoi_records.csv")
        # Build a value column: prefer observed VIIRS, else MODIS observed sum.
        daily["value"] = daily["viirs_frp_sum_mw"].where(daily["viirs_coverage"].astype(bool))
        daily.loc[daily["viirs_coverage"].astype(bool) == False, "value"] = daily[  # noqa: E712
            "modis_frp_sum_mw"
        ]
        daily_csv = workdir / "daily_with_value.csv"
        daily.to_csv(daily_csv, index=False)
        output = workdir / "train_forecaster"
        run_script(
            "train_forecaster.py",
            "--daily-csv", str(daily_csv),
            "--output-dir", str(output),
            "--config", str(tiny_yaml),
        )
        return output

    def test_backtest_reports_written(self, forecaster_output):
        report = json.loads(
            (forecaster_output / "forecast_backtest_report.json").read_text()
        )
        for name in ("persistence", "seasonal_naive"):
            assert name in report
            assert report[name]["n_evaluated_origins"] >= 1
        assert (forecaster_output / "forecast_baseline_comparison.json").is_file()
        assert (forecaster_output / "calendar_forecast_records.csv").is_file()

    def test_calendar_records_contract_from_cli(self, forecaster_output):
        from src.calendar.records import CALENDAR_FORECAST_COLUMNS, validate_calendar_forecast_frame

        records = pd.read_csv(forecaster_output / "calendar_forecast_records.csv")
        validate_calendar_forecast_frame(records)
        missing = [c for c in CALENDAR_FORECAST_COLUMNS if c not in records.columns]
        assert not missing
        assert (records["quality_state"] == "forecast").sum() == 14

    def test_forecaster_artifact_created(self, forecaster_output):
        artifact_dirs = [
            d for d in forecaster_output.iterdir() if d.is_dir() and d.name.startswith("forecaster_")
        ]
        assert artifact_dirs
        manifest = json.loads(artifact_dirs[0].joinpath("manifest.json").read_text())
        assert manifest["pipeline"] == "calendar_forecast"
        assert manifest["model"]["name"] == "patchtst"


class TestRunInference:
    def test_full_inference_flow(self, workdir, trained_harmonizer_dir):
        output = workdir / "run_inference"
        proc = run_script(
            "run_inference.py",
            "--source-dir", str(SAMPLE_DIR),
            "--harmonizer-dir", str(trained_harmonizer_dir),
            "--output-dir", str(output),
            "--aoi", AOI_SPEC,
        )
        records_path = output / "calendar_forecast_records.csv"
        assert records_path.is_file()
        assert (output / "harmonized_daily.csv").is_file()
        assert (output / "anomaly_records.csv").is_file()
        records = pd.read_csv(records_path)
        from src.calendar.records import CALENDAR_FORECAST_COLUMNS

        missing = [c for c in CALENDAR_FORECAST_COLUMNS if c not in records.columns]
        assert not missing
        # No forecaster passed -> no forecast rows, and the flow still succeeds.
        assert (records["quality_state"] == "forecast").sum() == 0
        assert "Inference complete" in proc.stdout

    def test_empty_aoi_explicit_state(self, workdir, trained_harmonizer_dir):
        output = workdir / "run_inference_empty"
        proc = run_script(
            "run_inference.py",
            "--source-dir", str(SAMPLE_DIR),
            "--harmonizer-dir", str(trained_harmonizer_dir),
            "--output-dir", str(output),
            "--aoi", "bbox:10.0,10.0,11.0,11.0",  # nowhere near the fixture AOI
            expect=2,
        )
        empty_state = json.loads((output / "calendar_forecast_records.json").read_text())
        assert empty_state["status"] == "empty_aoi"
        assert empty_state["records"] == []
        assert "empty_aoi" in proc.stderr or "no validated observations" in proc.stderr.lower()

    def test_missing_harmonizer_dir_exits_nonzero(self, workdir):
        proc = run_script(
            "run_inference.py",
            "--source-dir", str(SAMPLE_DIR),
            "--harmonizer-dir", str(workdir / "does_not_exist"),
            "--output-dir", str(workdir / "run_inference_bad"),
            "--aoi", AOI_SPEC,
            expect=None,
        )
        assert proc.returncode != 0
        assert "manifest" in proc.stderr.lower() or "artifact" in proc.stderr.lower()

    def test_gate_failure_blocks_inference(self, workdir, trained_harmonizer_dir):
        output = workdir / "run_inference_dirty"
        proc = run_script(
            "run_inference.py",
            "--source-dir", str(DIRTY_DIR),
            "--harmonizer-dir", str(trained_harmonizer_dir),
            "--output-dir", str(output),
            "--aoi", AOI_SPEC,
            expect=1,
        )
        assert "gate failed" in proc.stderr.lower()
        assert (output / "dataset_audit_report.json").is_file()  # evidence still written
