"""Harmonization inference: apply a trained model to MODIS observations.

Produces per-detection VIIRS-equivalent FRP estimates (labeled ``estimated``,
never ground truth) and the harmonized daily AOI series in which:

- days with VIIRS observations use the observed VIIRS sum (``observed_viirs``),
- MODIS-only days use the summed harmonized estimate (``harmonized_estimate``),
- days without sensor coverage are ``missing_coverage`` and carry NaN values -

following the docs rule "label observed, harmonized, estimated, forecast,
cached, and sample data distinctly" and "missing observations are not the same
as zero fire activity".
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.common.config import MatchingConfig
from src.common.logging import get_logger
from src.data.aggregation import EmptyAOIError
from src.data.schema import COL_FRP, QUARANTINE_COLUMN
from src.geo.matching import (
    _apply_confidence_filter,
    _footprint_radius,
)
from src.harmonization.features import build_feature_matrix

LOGGER = get_logger(__name__)

VALUE_TYPE_OBSERVED = "observed_viirs"
VALUE_TYPE_HARMONIZED = "harmonized_estimate"
VALUE_TYPE_MISSING = "missing_coverage"

HARMONIZED_DAILY_COLUMNS: tuple[str, ...] = (
    "date",
    "aoi_id",
    "value",
    "value_type",
    "sensor_era",
    "coverage_status",
    "observed_viirs_frp_sum_mw",
    "harmonized_modis_frp_sum_mw",
    "observed_modis_frp_sum_mw",
    "modis_coverage",
    "viirs_coverage",
)


def build_pseudo_records(
    modis: pd.DataFrame, matching_config: MatchingConfig | None = None
) -> pd.DataFrame:
    """Mirror the MODIS-side columns of the matched frame from raw observations.

    Applies the same confidence filter and duplicate policy as training-time
    matching so feature distributions stay consistent between train/inference.
    """
    matching_config = matching_config or MatchingConfig()
    from src.data.schema import PRODUCT_MODIS

    work = _apply_confidence_filter(modis, matching_config.min_confidence, PRODUCT_MODIS)
    if QUARANTINE_COLUMN in work.columns:
        work = work[work[QUARANTINE_COLUMN].isna()]
    if matching_config.deduplicate_modis and not work.empty:
        work = work.drop_duplicates(
            subset=[
                c
                for c in ("latitude", "longitude", "acq_datetime", "frp", "source_file")
                if c in work.columns
            ]
        )
    if work.empty:
        return pd.DataFrame(
            columns=[
                "modis_frp_mw",
                "modis_acq_datetime",
                "footprint_radius_deg",
                "match_date",
            ]
        )
    records = pd.DataFrame(
        {
            "modis_frp_mw": pd.to_numeric(work[COL_FRP], errors="coerce"),
            "modis_acq_datetime": pd.to_datetime(work["acq_datetime"]),
        }
    )
    records["footprint_radius_deg"] = [
        _footprint_radius(row, matching_config) for _, row in work.iterrows()
    ]
    for passthrough in ("brightness", "scan", "track", "daynight", "confidence_ordinal"):
        if passthrough in work.columns:
            records[f"modis_{passthrough}"] = work[passthrough].to_numpy()
    records["match_date"] = records["modis_acq_datetime"].dt.normalize().dt.date.astype(str)
    return records.reset_index(drop=True)


def harmonize_records(
    modis: pd.DataFrame, harmonizer, matching_config: MatchingConfig | None = None
) -> pd.DataFrame:
    """Per-detection VIIRS-equivalent FRP estimates (MW), labeled as estimates."""
    pseudo = build_pseudo_records(modis, matching_config)
    if pseudo.empty:
        return pseudo.assign(estimated_frp_mw=np.nan)
    X, _ = build_feature_matrix(pseudo)
    estimates = harmonizer.predict(X)
    pseudo["estimated_frp_mw"] = estimates
    return pseudo


def build_harmonized_daily(
    daily: pd.DataFrame,
    modis: pd.DataFrame,
    viirs: pd.DataFrame,
    harmonizer,
    matching_config: MatchingConfig | None = None,
) -> pd.DataFrame:
    """Combine observed and harmonized evidence into the daily AOI series.

    ``daily`` is the daily AOI frame from :mod:`src.data.aggregation`; the
    returned frame keeps every calendar day, so missing-coverage days remain
    explicit rather than being dropped or zero-filled.
    """
    if daily.empty:
        raise EmptyAOIError("Empty daily frame; nothing to harmonize")

    per_detection = harmonize_records(modis, harmonizer, matching_config)
    if not per_detection.empty:
        est_by_date = (
            per_detection.groupby("match_date")["estimated_frp_mw"].sum(min_count=1)
        )
    else:
        est_by_date = pd.Series(dtype=float)

    out = daily.copy()
    dates = pd.to_datetime(out["date"])
    date_keys = dates.dt.date.astype(str)

    out["observed_viirs_frp_sum_mw"] = np.where(
        out["viirs_coverage"].astype(bool), out["viirs_frp_sum_mw"], np.nan
    )
    est_values = date_keys.map(est_by_date).astype(float).to_numpy()
    out["harmonized_modis_frp_sum_mw"] = np.where(
        out["modis_coverage"].astype(bool),
        np.nan_to_num(est_values, nan=0.0),
        np.nan,
    )
    out["observed_modis_frp_sum_mw"] = np.where(
        out["modis_coverage"].astype(bool), out["modis_frp_sum_mw"], np.nan
    )

    viirs_cov = out["viirs_coverage"].astype(bool)
    modis_cov = out["modis_coverage"].astype(bool)

    value = pd.Series(np.nan, index=out.index, dtype=float)
    value_type = pd.Series(VALUE_TYPE_MISSING, index=out.index, dtype=object)
    value[viirs_cov] = out.loc[viirs_cov, "viirs_frp_sum_mw"]
    value_type[viirs_cov] = VALUE_TYPE_OBSERVED
    harmonized_mask = modis_cov & ~viirs_cov
    value[harmonized_mask] = out.loc[harmonized_mask, "harmonized_modis_frp_sum_mw"]
    value_type[harmonized_mask] = VALUE_TYPE_HARMONIZED
    out["value"] = value
    out["value_type"] = value_type

    era = []
    for m, v in zip(modis_cov, viirs_cov, strict=False):
        if m and v:
            era.append("modis+viirs")
        elif v:
            era.append("viirs_only")
        elif m:
            era.append("modis_only")
        else:
            era.append("no_coverage")
    out["sensor_era"] = era

    columns = [
        c for c in HARMONIZED_DAILY_COLUMNS if c in out.columns
    ]
    return out[columns].reset_index(drop=True)


__all__ = [
    "VALUE_TYPE_OBSERVED",
    "VALUE_TYPE_HARMONIZED",
    "VALUE_TYPE_MISSING",
    "HARMONIZED_DAILY_COLUMNS",
    "build_pseudo_records",
    "harmonize_records",
    "build_harmonized_daily",
]
