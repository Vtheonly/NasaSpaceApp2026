"""Daily AOI aggregation (docs/ARCHITECTURE.md: "AOI filtering and daily
aggregation"; artifact contract "Daily AOI records").

Deterministic by construction (SC-02): identical inputs produce identical
outputs. Missing days are never fabricated as zero-activity observations:
``coverage_status`` and per-sensor ``*_coverage`` flags carry the distinction
between "no observation coverage" and "zero detections" (docs: missing
observations are not the same as zero fire activity).
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.common.logging import get_logger
from src.data.schema import (
    COL_FRP,
    PRODUCT_MODIS,
    QUARANTINE_COLUMN,
    SUPPORTED_PRODUCTS,
)

LOGGER = get_logger(__name__)

AOI_ID_COLUMN = "aoi_id"
DATE_COLUMN = "date"
COVERAGE_STATUS_COLUMN = "coverage_status"

STATUS_OBSERVED = "observed"
STATUS_MISSING = "missing"

#: Column contract of the daily AOI frame (docs/ARCHITECTURE.md artifact contracts).
DAILY_AOI_COLUMNS: tuple[str, ...] = (
    DATE_COLUMN,
    AOI_ID_COLUMN,
    COVERAGE_STATUS_COLUMN,
    "modis_coverage",
    "viirs_coverage",
    "modis_active_pixels",
    "viirs_active_pixels",
    "modis_frp_sum_mw",
    "viirs_frp_sum_mw",
    "modis_frp_mean_mw",
    "viirs_frp_mean_mw",
    "modis_confidence_mean",
    "viirs_confidence_mean",
)


class EmptyAOIError(ValueError):
    """Raised when no validated observations remain inside the AOI."""


def _product_daily(frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate one product frame into per-day summaries."""
    if frame is None or frame.empty:
        return pd.DataFrame(
            columns=["date", "active_pixels", "frp_sum_mw", "frp_mean_mw", "confidence_mean"]
        )
    work = frame[frame[QUARANTINE_COLUMN].isna()] if QUARANTINE_COLUMN in frame.columns else frame
    work = work.copy()
    work["date"] = pd.to_datetime(work["acq_datetime"]).dt.normalize()
    frp = pd.to_numeric(work[COL_FRP], errors="coerce")
    grouped = work.groupby("date", sort=True)
    out = pd.DataFrame(
        {
            "active_pixels": grouped.size(),
            "frp_sum_mw": frp.groupby(work["date"]).sum(min_count=1),
            "frp_mean_mw": frp.groupby(work["date"]).mean(),
        }
    )
    if "confidence_ordinal" in work.columns and work["confidence_ordinal"].notna().any():
        out["confidence_mean"] = (
            pd.to_numeric(work["confidence_ordinal"], errors="coerce")
            .groupby(work["date"])
            .mean()
        )
    else:
        out["confidence_mean"] = float("nan")
    out = out.reset_index()
    return out


def build_daily_frame(
    frames: dict[str, pd.DataFrame],
    aoi_id: str = "global",
    aoi_filter=None,
    include_missing_days: bool = True,
) -> pd.DataFrame:
    """Build the wide daily AOI frame from validated per-product frames.

    Parameters
    ----------
    frames:
        Mapping product -> observation frame (as produced by the five-pass gate).
    aoi_id:
        Identifier recorded on every row for provenance.
    aoi_filter:
        Optional callable ``(df) -> df`` applied before aggregation (e.g. AOI
        filtering); must never mutate the input frame.
    include_missing_days:
        When true, the full calendar span between the first and last observed
        days is reindexed and gap days are marked ``coverage_status='missing'``.
    """
    if aoi_filter is not None:
        frames = {product: aoi_filter(frame) for product, frame in frames.items()}

    daily = {
        product: _product_daily(frame)
        for product, frame in frames.items()
        if product in SUPPORTED_PRODUCTS
    }
    non_empty = {p: d for p, d in daily.items() if not d.empty}
    if not non_empty:
        raise EmptyAOIError(
            "No validated observations remain after AOI filtering; "
            "cannot build a daily series (expected an explicit empty state in the UI)."
        )

    all_dates = pd.DatetimeIndex(
        sorted(set().union(*[set(d["date"]) for d in non_empty.values()]))
    )
    if include_missing_days:
        all_dates = pd.date_range(all_dates.min(), all_dates.max(), freq="D")

    out = pd.DataFrame({DATE_COLUMN: all_dates})
    out[AOI_ID_COLUMN] = aoi_id

    for product, daily_df in daily.items():
        prefix = "modis" if product == PRODUCT_MODIS else "viirs"
        if daily_df.empty:
            out[f"{prefix}_coverage"] = False
            out[f"{prefix}_active_pixels"] = 0
            out[f"{prefix}_frp_sum_mw"] = 0.0
            out[f"{prefix}_frp_mean_mw"] = float("nan")
            out[f"{prefix}_confidence_mean"] = float("nan")
            continue
        merged = out.merge(daily_df, on="date", how="left")
        covered = merged["active_pixels"].notna() & (merged["active_pixels"] > 0)
        out[f"{prefix}_coverage"] = covered.to_numpy()
        out[f"{prefix}_active_pixels"] = merged["active_pixels"].fillna(0).astype(int).to_numpy()
        out[f"{prefix}_frp_sum_mw"] = (
            merged["frp_sum_mw"].fillna(0.0).where(covered, 0.0).to_numpy()
        )
        out[f"{prefix}_frp_mean_mw"] = merged["frp_mean_mw"].to_numpy()
        out[f"{prefix}_confidence_mean"] = merged["confidence_mean"].to_numpy()

    modis_cov = out.get("modis_coverage", pd.Series(False, index=out.index)).astype(bool)
    viirs_cov = out.get("viirs_coverage", pd.Series(False, index=out.index)).astype(bool)
    out[COVERAGE_STATUS_COLUMN] = [
        STATUS_OBSERVED if (m or v) else STATUS_MISSING for m, v in zip(modis_cov, viirs_cov, strict=False)
    ]

    ordered = [c for c in DAILY_AOI_COLUMNS]
    for c in ordered:
        if c not in out.columns:
            out[c] = pd.Series([pd.NA] * len(out), index=out.index)
    out = out[list(DAILY_AOI_COLUMNS)].sort_values(DATE_COLUMN, kind="mergesort")
    return out.reset_index(drop=True)


def best_observed_value(daily: pd.DataFrame) -> pd.Series:
    """Return the best available *observed* regional FRP value per day.

    VIIRS observations are preferred (native scale); MODIS-observed days map to
    the MODIS value but are labeled separately downstream. Missing days
    produce NaN - never a silent zero.
    """
    value = pd.Series(float("nan"), index=daily.index)
    viirs_mask = daily["viirs_coverage"].astype(bool)
    modis_mask = daily["modis_coverage"].astype(bool) & ~viirs_mask
    value[viirs_mask] = daily.loc[viirs_mask, "viirs_frp_sum_mw"]
    value[modis_mask] = daily.loc[modis_mask, "modis_frp_sum_mw"]
    return value


def summarize_daily(daily: pd.DataFrame) -> dict[str, Any]:
    """Small deterministic summary used in reports and manifests."""
    observed = daily[daily[COVERAGE_STATUS_COLUMN] == STATUS_OBSERVED]
    return {
        "days_total": int(len(daily)),
        "days_observed": int(len(observed)),
        "days_missing_coverage": int((daily[COVERAGE_STATUS_COLUMN] == STATUS_MISSING).sum()),
        "modis_observed_days": int(daily["modis_coverage"].astype(bool).sum()),
        "viirs_observed_days": int(daily["viirs_coverage"].astype(bool).sum()),
        "date_min": str(daily[DATE_COLUMN].min().date()),
        "date_max": str(daily[DATE_COLUMN].max().date()),
    }
