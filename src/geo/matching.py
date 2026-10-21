"""Overlap-period MODIS/VIIRS sensor matching (docs/ARCHITECTURE.md Pillar 1).

Matching rules (all recorded in the artifact manifest):

- **Temporal tolerance:** VIIRS detections within ``temporal_tolerance_minutes``
  of the MODIS overpass timestamp are eligible (default 90 minutes, configurable).
- **Spatial rule:** a VIIRS detection is inside a MODIS footprint when its center
  lies within the axis-aligned box of half-width ``radius`` around the MODIS
  pixel, where ``radius`` is derived from the MODIS ``scan``/``track`` pixel size
  (degrees) and clipped to configured bounds.
- **Multiple VIIRS detections inside one footprint** are *summed* into a single
  matched target (summed VIIRS FRP in MW within the MODIS footprint).
- **Duplicate policy:** exact duplicate MODIS rows are dropped before matching
  (the five-pass gate already quarantines them; this is a defensive no-op).
- **Confidence filter:** records below ``min_confidence`` (ordinal of l/n/h) are
  excluded from matching entirely.
- **Coverage rule:** when the AOI had *no* VIIRS observation coverage on a
  match day, the match is marked ineligible with reason ``no_viirs_coverage``
  and the target is NaN - missing coverage is never treated as zero activity.
"""

from __future__ import annotations

import pandas as pd

from src.common.config import MatchingConfig
from src.common.logging import get_logger
from src.data.schema import (
    COL_FRP,
    CONFIDENCE_ORDINAL,
    PRODUCT_MODIS,
    PRODUCT_VIIRS,
    QUARANTINE_COLUMN,
)

LOGGER = get_logger(__name__)

MATCH_ID_COLUMN = "match_id"
TARGET_COLUMN = "viirs_frp_sum_mw"
MATCHED_COUNT_COLUMN = "viirs_matched_count"
VIIRS_COVERAGE_COLUMN = "viirs_coverage"
ELIGIBLE_COLUMN = "eligible"
INELIGIBILITY_REASON_COLUMN = "ineligibility_reason"
FOOTPRINT_RADIUS_COLUMN = "footprint_radius_deg"
DATE_COLUMN = "match_date"

#: Column contract of the matched calibration records (docs/ARCHITECTURE.md).
MATCHED_COLUMNS: tuple[str, ...] = (
    MATCH_ID_COLUMN,
    "aoi_id",
    "modis_acq_datetime",
    "modis_latitude",
    "modis_longitude",
    "modis_frp_mw",
    FOOTPRINT_RADIUS_COLUMN,
    "temporal_tolerance_minutes",
    MATCHED_COUNT_COLUMN,
    TARGET_COLUMN,
    VIIRS_COVERAGE_COLUMN,
    ELIGIBLE_COLUMN,
    INELIGIBILITY_REASON_COLUMN,
    DATE_COLUMN,
)

REASON_NO_VIIRS_COVERAGE = "no_viirs_coverage"
TARGET_DEFINITION = (
    "summed VIIRS FRP (MW) of detections inside the MODIS footprint box within the "
    "temporal tolerance; NaN when the AOI lacks VIIRS coverage that day"
)


def _apply_confidence_filter(
    frame: pd.DataFrame, min_confidence: str, product: str
) -> pd.DataFrame:
    if frame.empty or "confidence_ordinal" not in frame.columns:
        return frame
    threshold = CONFIDENCE_ORDINAL.get(str(min_confidence).lower())
    if threshold is None:
        raise ValueError(
            f"min_confidence must be one of {sorted(CONFIDENCE_ORDINAL)}; got {min_confidence!r}"
        )
    if threshold <= 0:
        return frame
    ordinal = pd.to_numeric(frame["confidence_ordinal"], errors="coerce")
    keep = ordinal.isna() | (ordinal >= threshold)
    dropped = int((~keep).sum())
    if dropped:
        LOGGER.info("[%s] confidence filter excluded %d records", product, dropped)
    return frame[keep].reset_index(drop=True)


def _footprint_radius(row: pd.Series, config: MatchingConfig) -> float:
    scan = pd.to_numeric(row.get("scan"), errors="coerce")
    track = pd.to_numeric(row.get("track"), errors="coerce")
    sizes = [s for s in (scan, track) if s == s and s > 0]
    if sizes:
        radius = max(sizes) / 2.0
    else:
        radius = config.default_footprint_radius_deg
    return float(
        min(max(radius, config.min_footprint_radius_deg), config.max_footprint_radius_deg)
    )


def build_matched_records(
    modis: pd.DataFrame,
    viirs: pd.DataFrame,
    config: MatchingConfig | None = None,
    aoi_id: str = "global",
) -> pd.DataFrame:
    """Construct matched MODIS -> summed VIIRS-FRP calibration records.

    Both input frames must be gate-clean observation frames for a single AOI
    (filter with :func:`src.geo.aoi.filter_records` first). The output is
    deterministic: identical inputs and config produce identical match IDs.
    """
    config = config or MatchingConfig()
    tolerance = int(config.temporal_tolerance_minutes)

    modis = _apply_confidence_filter(modis, config.min_confidence, PRODUCT_MODIS)
    viirs = _apply_confidence_filter(viirs, config.min_confidence, PRODUCT_VIIRS)

    if config.deduplicate_modis and not modis.empty:
        before = len(modis)
        modis = modis.drop_duplicates(
            subset=[
                c
                for c in ("latitude", "longitude", "acq_datetime", "frp", "source_file")
                if c in modis.columns
            ]
        )
        dropped = before - len(modis)
        if dropped:
            LOGGER.info("duplicate policy dropped %d MODIS rows before matching", dropped)

    viirs_by_date: dict[pd.Timestamp, pd.DataFrame] = {}
    if not viirs.empty:
        viirs_clean = viirs[viirs[QUARANTINE_COLUMN].isna()] if QUARANTINE_COLUMN in viirs else viirs
        viirs_dates = pd.to_datetime(viirs_clean["acq_datetime"]).dt.normalize()
        for date, group in viirs_clean.groupby(viirs_dates, sort=True):
            viirs_by_date[date] = group.sort_values(
                ["acq_datetime", "latitude", "longitude"], kind="mergesort"
            )

    modis = modis.sort_values(
        ["acq_datetime", "latitude", "longitude"], kind="mergesort"
    ).reset_index(drop=True)
    if QUARANTINE_COLUMN in modis.columns:
        modis = modis[modis[QUARANTINE_COLUMN].isna()].reset_index(drop=True)

    rows: list[dict] = []
    seq = 0
    for _, row in modis.iterrows():
        ts = row["acq_datetime"]
        date = pd.Timestamp(ts).normalize()
        # Always pool the day-of and the adjacent day buckets so near-midnight
        # overpasses can still match (tolerance window may cross a date boundary).
        buckets = [
            viirs_by_date.get(date - pd.Timedelta(days=1)),
            viirs_by_date.get(date),
            viirs_by_date.get(date + pd.Timedelta(days=1)),
        ]
        nearby = pd.concat([b for b in buckets if b is not None]) if any(
            b is not None for b in buckets
        ) else None
        radius = _footprint_radius(row, config)
        count = 0
        target = 0.0
        coverage = False
        if nearby is not None and len(nearby) > 0:
            v_ts = pd.to_datetime(nearby["acq_datetime"])
            delta_minutes = (v_ts - ts).dt.total_seconds().abs() / 60.0
            within_time = delta_minutes <= tolerance
            coverage = bool(within_time.any())
            in_window = nearby[within_time]
            if len(in_window) > 0:
                v_lat = pd.to_numeric(in_window["latitude"], errors="coerce")
                v_lon = pd.to_numeric(in_window["longitude"], errors="coerce")
                d_lat = (v_lat - float(row["latitude"])).abs()
                d_lon = (v_lon - float(row["longitude"])).abs()
                inside = (d_lat <= radius) & (d_lon <= radius)
                matched = in_window[inside]
                count = int(len(matched))
                if count:
                    target = float(
                        pd.to_numeric(matched[COL_FRP], errors="coerce").fillna(0.0).sum()
                    )

        eligible = coverage
        reason = None if coverage else REASON_NO_VIIRS_COVERAGE
        seq += 1
        record = {
            MATCH_ID_COLUMN: f"m{seq:06d}",
            "aoi_id": aoi_id,
            "modis_acq_datetime": ts,
            "modis_latitude": float(row["latitude"]),
            "modis_longitude": float(row["longitude"]),
            "modis_frp_mw": float(pd.to_numeric(row[COL_FRP], errors="coerce")),
            FOOTPRINT_RADIUS_COLUMN: radius,
            "temporal_tolerance_minutes": tolerance,
            MATCHED_COUNT_COLUMN: count,
            TARGET_COLUMN: target if coverage else float("nan"),
            VIIRS_COVERAGE_COLUMN: coverage,
            ELIGIBLE_COLUMN: eligible,
            INELIGIBILITY_REASON_COLUMN: reason,
            DATE_COLUMN: pd.Timestamp(ts).date().isoformat(),
        }
        for passthrough in ("brightness", "scan", "track", "daynight", "confidence_ordinal"):
            if passthrough in modis.columns:
                record[f"modis_{passthrough}"] = row.get(passthrough)
        rows.append(record)

    matched = pd.DataFrame(rows)
    if matched.empty:
        matched = pd.DataFrame(columns=list(MATCHED_COLUMNS))
    return matched


def eligible_matches(matched: pd.DataFrame) -> pd.DataFrame:
    """Return only eligible matches (VIIRS coverage existed that day).

    Indexes of the source frame are preserved so chronological splits and
    feature matrices stay row-aligned.
    """
    if matched.empty:
        return matched
    return matched[matched[ELIGIBLE_COLUMN].astype(bool)]


__all__ = [
    "MATCHED_COLUMNS",
    "TARGET_COLUMN",
    "build_matched_records",
    "eligible_matches",
    "REASON_NO_VIIRS_COVERAGE",
    "TARGET_DEFINITION",
]
