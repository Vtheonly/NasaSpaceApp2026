"""Manifest construction and integrity verification.

Every artifact directory contains ``manifest.json`` with source hashes,
configuration, code commit, model hash, metrics, and limitations. Integrity is
verified on load: a hash mismatch raises :class:`ArtifactIntegrityError` instead
of silently using a corrupted model (issue #13 graceful failure paths).
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.artifacts.contracts import SCHEMA_VERSION

MANIFEST_FILENAME = "manifest.json"


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def build_manifest(
    pipeline: str,
    config: dict[str, Any],
    source_files: list[dict[str, Any]],
    model: dict[str, Any],
    metrics: dict[str, Any],
    code_commit: str,
    limitations: list[str],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the artifact manifest (docs/ARCHITECTURE.md artifact contracts)."""
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "created_at": _now_iso(),
        "pipeline": pipeline,
        "code_commit": code_commit,
        "config": config,
        "source_files": source_files,
        "model": model,
        "metrics": metrics,
        "limitations": list(limitations),
    }
    if extra:
        manifest.update(extra)
    return manifest


def verify_model_integrity(artifact_dir: str | Path, manifest: dict[str, Any], model_filename: str) -> str:
    """Verify the model file hash against the manifest; return the verified path.

    Raises :class:`ArtifactIntegrityError` on missing files or hash mismatch.
    """
    from src.artifacts.serialization import ArtifactIntegrityError

    artifact_dir = Path(artifact_dir)
    model_path = artifact_dir / model_filename
    if not model_path.is_file():
        raise ArtifactIntegrityError(
            f"Model file {model_filename} is missing from artifact directory {artifact_dir}"
        )
    actual = sha256_file(model_path)
    expected = manifest.get("model", {}).get("sha256")
    if expected is None:
        raise ArtifactIntegrityError("Manifest does not record a model sha256; cannot verify")
    if actual != expected:
        raise ArtifactIntegrityError(
            f"Model file {model_filename} failed integrity check: manifest sha256 "
            f"{expected[:12]}... but file has {actual[:12]}..."
        )
    return str(model_path)


def source_file_entries(paths: list[str | Path], rows: int | None = None) -> list[dict[str, Any]]:
    """Build the source_files manifest section with SHA-256 hashes."""
    entries = []
    for path in paths:
        path = Path(path)
        entry: dict[str, Any] = {
            "path": path.name,
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
        entries.append(entry)
    if rows is not None:
        entries[0]["rows"] = rows
    return entries


__all__ = [
    "MANIFEST_FILENAME",
    "build_manifest",
    "sha256_file",
    "sha256_bytes",
    "verify_model_integrity",
    "source_file_entries",
]
