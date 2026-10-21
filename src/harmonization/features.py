"""Feature construction for the MODIS -> VIIRS-equivalent harmonizer.

Feature contract (docs/ARCHITECTURE.md Pillar 1 "Candidate covariates"):

- Only MODIS-side and temporal-context features are allowed. Anything derived
  from VIIRS is a target and would leak; :func:`assert_no_target_leakage`
  enforces this invariant and is exercised by the scientific test suite.
- Optional covariates (brightness, scan, track, confidence, daynight) are used
  only when the columns actually exist in the matched records
  (docs/RESEARCH_AND_DATA.md: do not assume candidate fields exist).
- Temporal context uses circular day-of-year encodings; no future information
  is ever used (features describe the current detection only).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.schema import pseudo_day_of_year

FEATURE_VERSION = "modis_side_v1"

#: Always-present numeric features.
BASE_FEATURES: tuple[str, ...] = (
    "modis_frp_mw",
    "log_modis_frp",
    "footprint_radius_deg",
    "doy_sin",
    "doy_cos",
    "modis_daily_active_pixels",
)

#: Optional features included only when their source column exists.
OPTIONAL_FEATURES: tuple[str, ...] = (
    "modis_brightness",
    "modis_scan_deg",
    "modis_track_deg",
    "modis_confidence_ordinal",
    "is_night",
)

_FORBIDDEN_SUBSTRINGS = ("viirs", "target")


def assert_no_target_leakage(feature_names: list[str]) -> None:
    """Raise if any feature name hints at target/VIIRS-derived information."""
    for name in feature_names:
        lowered = name.lower()
        if any(bad in lowered for bad in _FORBIDDEN_SUBSTRINGS):
            raise ValueError(f"Feature {name!r} derives from the target (VIIRS side); forbidden")


def build_feature_matrix(
    records: pd.DataFrame, extra_context: pd.DataFrame | None = None
) -> tuple[pd.DataFrame, list[str]]:
    """Build the harmonizer feature matrix from MODIS-side records.

    ``records`` must contain ``modis_frp_mw``, ``modis_acq_datetime`` and
    ``footprint_radius_deg`` (both the matched frame and the inference pseudo
    frame satisfy this). Returns ``(X, feature_names)`` with a stable column
    order. ``extra_context`` is ignored for now and reserved for documented
    auxiliary joins (land cover, elevation).
    """
    if records.empty:
        raise ValueError("Cannot build features from an empty record set")

    frp = pd.to_numeric(records["modis_frp_mw"], errors="coerce")
    timestamps = pd.to_datetime(records["modis_acq_datetime"])
    doy = timestamps.map(pseudo_day_of_year).astype(float)
    angle = 2.0 * np.pi * (doy - 1.0) / 365.0

    features = pd.DataFrame(index=records.index)
    features["modis_frp_mw"] = frp
    features["log_modis_frp"] = np.log1p(frp.clip(lower=0.0))
    features["footprint_radius_deg"] = pd.to_numeric(
        records["footprint_radius_deg"], errors="coerce"
    )
    features["doy_sin"] = np.sin(angle)
    features["doy_cos"] = np.cos(angle)

    # Detection density: number of MODIS detections in the AOI on that date.
    dates = timestamps.dt.normalize()
    density = records.groupby(dates)["modis_frp_mw"].transform("size")
    features["modis_daily_active_pixels"] = density.astype(float)

    if "modis_brightness" in records.columns:
        features["modis_brightness"] = pd.to_numeric(
            records["modis_brightness"], errors="coerce"
        )
    if "modis_scan_deg" in records.columns:
        features["modis_scan_deg"] = pd.to_numeric(records["modis_scan_deg"], errors="coerce")
    if "modis_track_deg" in records.columns:
        features["modis_track_deg"] = pd.to_numeric(records["modis_track_deg"], errors="coerce")
    if "modis_confidence_ordinal" in records.columns:
        features["modis_confidence_ordinal"] = pd.to_numeric(
            records["modis_confidence_ordinal"], errors="coerce"
        )
    if "modis_daynight" in records.columns:
        features["is_night"] = (
            records["modis_daynight"].astype(str).str.upper().eq("N").astype(float)
        )

    feature_names = list(features.columns)
    assert_no_target_leakage(feature_names)
    features = features.replace([np.inf, -np.inf], np.nan)
    return features, feature_names


__all__ = [
    "FEATURE_VERSION",
    "BASE_FEATURES",
    "OPTIONAL_FEATURES",
    "build_feature_matrix",
    "assert_no_target_leakage",
]
