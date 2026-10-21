"""Versioned artifacts: contracts, manifests, hashing, serialization.

GitHub holds only source/tests/fixtures; trained models and processed outputs
are versioned artifacts with manifests recording source hashes, configuration,
code commit, model hash, metrics, and limitations (docs/ARCHITECTURE.md
"Artifact contracts").
"""

from src.artifacts.contracts import (
    ANOMALY_STATUSES,
    COVERAGE_STATUSES,
    MANIFEST_REQUIRED_KEYS,
    SCHEMA_VERSION,
    VALUE_TYPES,
    ContractError,
    validate_anomaly_frame,
    validate_daily_frame,
    validate_harmonized_daily,
    validate_manifest,
    validate_matched_frame,
)
from src.artifacts.serialization import (
    ArtifactIntegrityError,
    load_forecaster_bundle,
    load_harmonizer_bundle,
    load_json,
    load_manifest,
    save_forecaster_bundle,
    save_harmonizer_bundle,
    save_json,
)

__all__ = [
    "ContractError",
    "ArtifactIntegrityError",
    "SCHEMA_VERSION",
    "MANIFEST_REQUIRED_KEYS",
    "VALUE_TYPES",
    "COVERAGE_STATUSES",
    "ANOMALY_STATUSES",
    "validate_matched_frame",
    "validate_daily_frame",
    "validate_harmonized_daily",
    "validate_anomaly_frame",
    "validate_manifest",
    "save_json",
    "load_json",
    "load_manifest",
    "save_harmonizer_bundle",
    "load_harmonizer_bundle",
    "save_forecaster_bundle",
    "load_forecaster_bundle",
]
