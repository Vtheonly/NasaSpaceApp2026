"""Artifact serialization: atomic JSON writes, model bundles, integrity checks.

Model bundles are stored as a directory holding ``manifest.json`` plus the
model file (``model.joblib`` for the harmonizer, ``model.json`` for the
PatchTST forecaster - pure-JSON state keeps torch loading policy-safe).
Loading always verifies the recorded SHA-256 before use.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import UTC
from pathlib import Path
from typing import Any

import joblib

from src.artifacts.contracts import validate_manifest
from src.artifacts.manifest import MANIFEST_FILENAME, sha256_file, verify_model_integrity

HARMONIZER_MODEL_FILENAME = "model.joblib"
FORECASTER_MODEL_FILENAME = "model.json"


class ArtifactIntegrityError(RuntimeError):
    """Raised when an artifact is missing, corrupt, or fails hash verification."""


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    """Atomically write JSON (tmp file + rename) so artifacts never land half-written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True, default=_json_default)
        os.replace(tmp_name, path)
    except Exception:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
        raise


def _json_default(obj: Any):
    import numpy as np

    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def load_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise ArtifactIntegrityError(f"Required JSON artifact missing: {path}")
    try:
        with open(path, encoding="utf-8") as fh:
            payload = json.load(fh)
    except json.JSONDecodeError as exc:
        raise ArtifactIntegrityError(f"Corrupt JSON artifact {path}: {exc}") from exc
    return payload


def load_manifest(artifact_dir: str | Path) -> dict[str, Any]:
    """Load and contract-validate ``manifest.json`` inside an artifact directory."""
    manifest = load_json(Path(artifact_dir) / MANIFEST_FILENAME)
    validate_manifest(manifest)
    return manifest


# ---------------------------------------------------------------------------
# Harmonizer bundle
# ---------------------------------------------------------------------------


def save_harmonizer_bundle(
    artifact_dir: str | Path,
    bundle: dict[str, Any],
    manifest: dict[str, Any],
) -> str:
    """Persist the harmonizer model bundle plus its manifest.

    The manifest's ``model.sha256`` is computed after writing the model file so
    loading can always verify integrity end-to-end.
    """
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    model_path = artifact_dir / HARMONIZER_MODEL_FILENAME
    joblib.dump(bundle, model_path)
    manifest["model"]["sha256"] = sha256_file(model_path)
    save_json(artifact_dir / MANIFEST_FILENAME, manifest)
    return str(model_path)


def load_harmonizer_bundle(artifact_dir: str | Path) -> dict[str, Any]:
    """Load the harmonizer bundle with manifest + SHA-256 verification."""
    manifest = load_manifest(artifact_dir)
    model_path = verify_model_integrity(
        artifact_dir, manifest, HARMONIZER_MODEL_FILENAME
    )
    try:
        bundle = joblib.load(model_path)
    except Exception as exc:  # noqa: BLE001
        raise ArtifactIntegrityError(f"Could not load harmonizer model {model_path}: {exc}") from exc
    if not isinstance(bundle, dict) or "model" not in bundle:
        raise ArtifactIntegrityError(
            f"Harmonizer bundle in {artifact_dir} has an unexpected structure"
        )
    return bundle


# ---------------------------------------------------------------------------
# Forecaster bundle (PatchTST; pure JSON state)
# ---------------------------------------------------------------------------


def save_forecaster_bundle(
    artifact_dir: str | Path,
    state: dict[str, Any],
    manifest: dict[str, Any],
) -> str:
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    model_path = artifact_dir / FORECASTER_MODEL_FILENAME
    save_json(model_path, state)
    manifest["model"]["sha256"] = sha256_file(model_path)
    save_json(artifact_dir / MANIFEST_FILENAME, manifest)
    return str(model_path)


def load_forecaster_bundle(artifact_dir: str | Path) -> dict[str, Any]:
    manifest = load_manifest(artifact_dir)
    model_path = verify_model_integrity(
        artifact_dir, manifest, FORECASTER_MODEL_FILENAME
    )
    return load_json(model_path)


def new_artifact_dir(
    base_dir: str | Path, pipeline: str, timestamp: str | None = None
) -> Path:
    """Create a fresh, timestamped artifact directory (versioned outputs)."""
    stamp = timestamp or _utc_stamp()
    artifact_dir = Path(base_dir) / f"{pipeline}_{stamp}"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    return artifact_dir


def _utc_stamp() -> str:
    from datetime import datetime

    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


__all__ = [
    "ArtifactIntegrityError",
    "save_json",
    "load_json",
    "load_manifest",
    "save_harmonizer_bundle",
    "load_harmonizer_bundle",
    "save_forecaster_bundle",
    "load_forecaster_bundle",
    "new_artifact_dir",
    "HARMONIZER_MODEL_FILENAME",
    "FORECASTER_MODEL_FILENAME",
]
