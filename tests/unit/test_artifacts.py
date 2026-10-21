"""Unit tests for artifact contracts, manifests, and integrity checks (R-10, R-14)."""


import numpy as np
import pandas as pd
import pytest

from src.artifacts.contracts import (
    SCHEMA_VERSION,
    ContractError,
    validate_daily_frame,
    validate_harmonized_daily,
    validate_manifest,
)
from src.artifacts.manifest import (
    MANIFEST_FILENAME,
    build_manifest,
    sha256_bytes,
    sha256_file,
    verify_model_integrity,
)
from src.artifacts.serialization import (
    FORECASTER_MODEL_FILENAME,
    HARMONIZER_MODEL_FILENAME,
    ArtifactIntegrityError,
    load_forecaster_bundle,
    load_harmonizer_bundle,
    load_json,
    save_forecaster_bundle,
    save_harmonizer_bundle,
    save_json,
)


class TestJson:
    def test_save_load_roundtrip(self, tmp_path):
        payload = {"b": 2, "a": 1, "nested": {"x": [1, 2]}}
        save_json(tmp_path / "x.json", payload)
        assert load_json(tmp_path / "x.json") == payload

    def test_save_json_is_atomic(self, tmp_path):
        target = tmp_path / "x.json"
        save_json(target, {"ok": True})
        # No stray temp files left behind.
        assert [p.name for p in tmp_path.iterdir()] == ["x.json"]

    def test_load_missing_raises_clear_error(self, tmp_path):
        with pytest.raises(ArtifactIntegrityError, match="missing"):
            load_json(tmp_path / "nope.json")

    def test_load_corrupt_raises_clear_error(self, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text("{not json", encoding="utf-8")
        with pytest.raises(ArtifactIntegrityError, match="Corrupt"):
            load_json(bad)


class TestManifest:
    def test_build_manifest_contains_required_keys(self, tmp_path):
        source = tmp_path / "data.csv"
        source.write_text("a\n1\n", encoding="utf-8")
        manifest = build_manifest(
            pipeline="harmonization",
            config={"seed": 1},
            source_files=[{"path": source.name, "sha256": sha256_file(source)}],
            model={"name": "lightgbm_tweedie", "version": "1.0.0", "sha256": "x" * 64},
            metrics={"tweedie": 1.0},
            code_commit="abc123",
            limitations=["l1"],
        )
        validate_manifest(manifest)
        assert manifest["schema_version"] == SCHEMA_VERSION
        assert manifest["model"]["name"] == "lightgbm_tweedie"

    def test_validate_manifest_rejects_missing_keys(self):
        with pytest.raises(ContractError, match="missing required keys"):
            validate_manifest({"schema_version": SCHEMA_VERSION})

    def test_validate_manifest_rejects_wrong_version(self, tmp_path):
        manifest = build_manifest(
            pipeline="p",
            config={},
            source_files=[],
            model={"name": "m", "version": "1", "sha256": "x"},
            metrics={},
            code_commit="c",
            limitations=[],
        )
        manifest["schema_version"] = "0.0.1"
        with pytest.raises(ContractError, match="schema_version"):
            validate_manifest(manifest)

    def test_validate_manifest_rejects_bad_model_section(self):
        with pytest.raises(ContractError, match="manifest.model"):
            validate_manifest(
                {
                    "schema_version": SCHEMA_VERSION,
                    "created_at": "2026-01-01T00:00:00+00:00",
                    "pipeline": "p",
                    "config": {},
                    "source_files": [],
                    "model": {"name": "m"},
                    "metrics": {},
                    "limitations": [],
                    "code_commit": "c",
                }
            )


class TestIntegrity:
    def test_sha256_bytes(self):
        assert sha256_bytes(b"abc") == sha256_bytes(b"abc")
        assert len(sha256_bytes(b"abc")) == 64

    def test_verify_model_integrity_detects_tampering(self, tmp_path):
        model_path = tmp_path / HARMONIZER_MODEL_FILENAME
        model_path.write_bytes(b"original weights")
        manifest = {"model": {"sha256": sha256_file(model_path)}}
        assert verify_model_integrity(tmp_path, manifest, HARMONIZER_MODEL_FILENAME)
        model_path.write_bytes(b"tampered weights")
        with pytest.raises(ArtifactIntegrityError, match="integrity"):
            verify_model_integrity(tmp_path, manifest, HARMONIZER_MODEL_FILENAME)

    def test_verify_model_integrity_missing_file(self, tmp_path):
        with pytest.raises(ArtifactIntegrityError, match="missing"):
            verify_model_integrity(tmp_path, {"model": {"sha256": "x"}}, "model.joblib")

    def test_manifest_without_hash_rejected(self, tmp_path):
        (tmp_path / HARMONIZER_MODEL_FILENAME).write_bytes(b"w")
        with pytest.raises(ArtifactIntegrityError, match="does not record"):
            verify_model_integrity(tmp_path, {"model": {}}, HARMONIZER_MODEL_FILENAME)


class TestBundles:
    def test_harmonizer_bundle_roundtrip(self, tmp_path, training_result):
        manifest = build_manifest(
            pipeline="harmonization",
            config=training_result.config_snapshot,
            source_files=[],
            model={
                "name": training_result.model.name,
                "version": training_result.model.version,
                "params": training_result.model.params,
            },
            metrics=training_result.evaluation,
            code_commit="test",
            limitations=[],
        )
        bundle = {
            "model": training_result.model,
            "feature_names": training_result.feature_names,
        }
        save_harmonizer_bundle(tmp_path, bundle, manifest)
        loaded = load_harmonizer_bundle(tmp_path)
        assert loaded["feature_names"] == training_result.feature_names
        # Predictions must be identical after the save/load roundtrip.
        rng = np.random.default_rng(0)
        probe = pd.DataFrame(
            {name: rng.uniform(0, 50, 20) for name in training_result.feature_names}
        )
        assert np.allclose(
            loaded["model"].predict(probe),
            training_result.model.predict(probe),
            rtol=1e-12,
        )

    def test_forecaster_bundle_roundtrip(self, tmp_path):
        state = {
            "model_name": "seasonal_naive",
            "model_version": "1.0.0",
            "config": {"horizon_days": 14},
            "note": "baseline forecaster; no learned weights",
        }
        manifest = build_manifest(
            pipeline="calendar_forecast",
            config={},
            source_files=[],
            model={"name": "seasonal_naive", "version": "1.0.0", "horizon_days": 14},
            metrics={},
            code_commit="test",
            limitations=[],
        )
        path = save_forecaster_bundle(tmp_path, state, manifest)
        assert path.endswith(FORECASTER_MODEL_FILENAME)
        assert (tmp_path / MANIFEST_FILENAME).is_file()
        loaded = load_forecaster_bundle(tmp_path)
        assert loaded["model_name"] == "seasonal_naive"

    def test_bundle_integrity_error_on_corrupt_manifest(self, tmp_path, training_result):
        manifest = build_manifest(
            pipeline="harmonization",
            config={},
            source_files=[],
            model={"name": "m", "version": "1"},
            metrics={},
            code_commit="c",
            limitations=[],
        )
        save_harmonizer_bundle(
            tmp_path, {"model": training_result.model, "feature_names": []}, manifest
        )
        # Corrupt the model file after saving -> hash mismatch on load.
        model_file = tmp_path / HARMONIZER_MODEL_FILENAME
        model_file.write_bytes(b"corrupted")
        with pytest.raises(ArtifactIntegrityError):
            load_harmonizer_bundle(tmp_path)


class TestFrameContracts:
    def test_daily_frame_contract(self, daily_frame):
        validate_daily_frame(daily_frame)

    def test_harmonized_daily_contract(self, gate_outcome, training_result, aoi):
        from src.data.schema import PRODUCT_MODIS, PRODUCT_VIIRS
        from src.geo.aoi import filter_records
        from src.harmonization.inference import build_harmonized_daily

        frames = {p: filter_records(f, aoi) for p, f in gate_outcome.frames.items()}
        harmonized = build_harmonized_daily(
            training_result.daily,
            frames[PRODUCT_MODIS],
            frames[PRODUCT_VIIRS],
            training_result.model,
        )
        validate_harmonized_daily(harmonized)

    def test_empty_frame_rejected(self):
        with pytest.raises(ContractError, match="empty"):
            validate_daily_frame(pd.DataFrame())
