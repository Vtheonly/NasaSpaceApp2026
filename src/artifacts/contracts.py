"""Artifact contracts: required columns, label vocabularies, manifest schema.

Every artifact frame must satisfy its contract before serialization and after
loading (round-trip integrity). Column checks are explicit and list the
offending columns in the error message (graceful failure, issue #13).
"""

from __future__ import annotations

from typing import Any

import pandas as pd

SCHEMA_VERSION = "1.0.0"

#: docs/ARCHITECTURE.md - artifact contracts.
MANIFEST_REQUIRED_KEYS: tuple[str, ...] = (
    "schema_version",
    "created_at",
    "pipeline",
    "code_commit",
    "config",
    "source_files",
    "model",
    "metrics",
    "limitations",
)

VALUE_TYPES: frozenset[str] = frozenset(
    {"observed_viirs", "harmonized_estimate", "missing_coverage"}
)
COVERAGE_STATUSES: frozenset[str] = frozenset({"observed", "missing"})
ANOMALY_STATUSES: frozenset[str] = frozenset(
    {
        "normal",
        "anomaly_high",
        "anomaly_low",
        "insufficient_baseline",
        "missing_coverage",
    }
)


class ContractError(ValueError):
    """Raised when an artifact violates its documented contract."""


def validate_columns(frame: pd.DataFrame, required: tuple[str, ...], kind: str) -> None:
    if frame is None or len(frame) == 0:
        raise ContractError(f"{kind} artifact is empty; expected a non-empty frame")
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise ContractError(f"{kind} artifact is missing required columns: {missing}")


def validate_matched_frame(frame: pd.DataFrame) -> None:
    from src.geo.matching import MATCHED_COLUMNS

    validate_columns(frame, MATCHED_COLUMNS, "matched calibration records")


def validate_daily_frame(frame: pd.DataFrame) -> None:
    from src.data.aggregation import DAILY_AOI_COLUMNS

    validate_columns(frame, DAILY_AOI_COLUMNS, "daily AOI records")


def validate_harmonized_daily(frame: pd.DataFrame) -> None:
    from src.harmonization.inference import HARMONIZED_DAILY_COLUMNS

    validate_columns(frame, HARMONIZED_DAILY_COLUMNS, "harmonized daily series")
    bad = set(frame["value_type"].astype(str)) - VALUE_TYPES
    if bad:
        raise ContractError(f"harmonized daily series has invalid value_type labels: {bad}")


def validate_anomaly_frame(frame: pd.DataFrame) -> None:
    from src.calendar.anomalies import ANOMALY_COLUMNS

    validate_columns(frame, ANOMALY_COLUMNS, "anomaly records")
    bad = set(frame["status"].astype(str)) - ANOMALY_STATUSES
    if bad:
        raise ContractError(f"anomaly records have invalid statuses: {bad}")
    missing = frame["status"].eq("missing_coverage") & frame["ratio"].notna()
    if missing.any():
        raise ContractError(
            "missing-coverage days must not carry a numeric anomaly ratio "
            "(missing coverage is not zero activity)"
        )


def validate_manifest(manifest: dict[str, Any]) -> None:
    if not isinstance(manifest, dict):
        raise ContractError("manifest must be a mapping")
    missing = [k for k in MANIFEST_REQUIRED_KEYS if k not in manifest]
    if missing:
        raise ContractError(f"artifact manifest is missing required keys: {missing}")
    model = manifest.get("model")
    if not isinstance(model, dict) or not {"name", "version", "sha256"}.issubset(model):
        raise ContractError("manifest.model must contain name, version, and sha256")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ContractError(
            f"manifest schema_version {manifest.get('schema_version')!r} does not "
            f"match the supported version {SCHEMA_VERSION!r}"
        )


__all__ = [
    "SCHEMA_VERSION",
    "MANIFEST_REQUIRED_KEYS",
    "ContractError",
    "VALUE_TYPES",
    "COVERAGE_STATUSES",
    "ANOMALY_STATUSES",
    "validate_columns",
    "validate_matched_frame",
    "validate_daily_frame",
    "validate_harmonized_daily",
    "validate_anomaly_frame",
    "validate_manifest",
]
