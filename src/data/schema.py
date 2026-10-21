"""FIRMS product schemas and normalized column contracts.

The harmonization pipeline normalizes processed NASA FIRMS MODIS and VIIRS
active-fire CSV exports into a single observation frame. Per
docs/RESEARCH_AND_DATA.md, candidate features (brightness temperature, scan
angle, view geometry, ...) are OPTIONAL: the code must only use fields that are
actually present in the selected products.

Units: FRP is expressed in megawatts (MW) for both products and is never
silently converted (docs/ARCHITECTURE.md artifact contracts).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

PRODUCT_MODIS = "modis"
PRODUCT_VIIRS = "viirs"
SUPPORTED_PRODUCTS = (PRODUCT_MODIS, PRODUCT_VIIRS)

#: FRP units used throughout the pipeline and recorded in every manifest.
FRP_UNITS = "MW"

#: Normalized column names shared by both products.
COL_LATITUDE = "latitude"
COL_LONGITUDE = "longitude"
COL_ACQ_DATE = "acq_date"
COL_ACQ_TIME = "acq_time"
COL_FRP = "frp"

#: Columns required by the five-pass gate (docs/TESTING.md pass 1).
REQUIRED_COLUMNS: tuple[str, ...] = (
    COL_LATITUDE,
    COL_LONGITUDE,
    COL_ACQ_DATE,
    COL_ACQ_TIME,
    COL_FRP,
)

#: Optional product fields; used for features only when present.
OPTIONAL_COLUMNS: dict[str, str] = {
    "brightness": "brightness temperature (K); MODIS: `brightness`, VIIRS: `bright_ti4`",
    "scan": "along-scan pixel size (degrees)",
    "track": "along-track pixel size (degrees)",
    "confidence": "detection confidence label (l/n/h) or numeric value",
    "daynight": "daytime/nighttime flag (D/N)",
    "satellite": "platform name (e.g. Terra/Aqua/S-NPP)",
    "instrument": "instrument name",
    "version": "collection/product version string",
}

#: Raw-to-normalized column aliases accepted by the adapter.
COLUMN_ALIASES: dict[str, dict[str, str]] = {
    PRODUCT_MODIS: {
        "bright_ti4": "brightness",  # tolerated for symmetric ingestion
        "bright_t31": "brightness_t31",
    },
    PRODUCT_VIIRS: {
        "bright_ti4": "brightness",
        "bright_ti5": "brightness_t31",
    },
}

#: Confidence ordinal mapping. FIRMS low/normal/high labels become 0/1/2 so a
#: configurable minimum-confidence filter can be applied consistently.
CONFIDENCE_ORDINAL: dict[str, int] = {"l": 0, "n": 1, "h": 2}

#: Quarantine reason codes (docs/TESTING.md pass 2: quarantine with reason codes
#: rather than blindly deleting outliers).
REASON_LAT_OUT_OF_BOUNDS = "lat_out_of_bounds"
REASON_LON_OUT_OF_BOUNDS = "lon_out_of_bounds"
REASON_COORDS_MISSING = "coords_missing"
REASON_DATE_INVALID = "acq_date_invalid"
REASON_TIME_INVALID = "acq_time_invalid"
REASON_FRP_MISSING = "frp_missing"
REASON_FRP_NONFINITE = "frp_nonfinite"
REASON_FRP_NEGATIVE = "frp_negative"
REASON_DUPLICATE_ROW = "duplicate_row"

QUARANTINE_COLUMN = "quarantine_reason"

#: Coordinate bounds.
LAT_MIN, LAT_MAX = -90.0, 90.0
LON_MIN, LON_MAX = -180.0, 180.0


def confidence_ordinal(value: Any) -> int | None:
    """Map a confidence value to an ordinal (l/n/h -> 0/1/2).

    Numeric confidence values between 0 and 100 are scaled to 0..2 by thirds so
    that VIIRS-style numeric confidence can be filtered with the same scale.
    Returns ``None`` for missing/unmappable values.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, (int, np.integer, float, np.floating)):
        num = float(value)
        if 0 <= num <= 100:
            return int(min(num / 100 * 3, 2.999))  # 0..33 -> 0, 34..66 -> 1, 67..100 -> 2
        return None
    text = str(value).strip().lower()
    return CONFIDENCE_ORDINAL.get(text)


def parse_acq_datetime(acq_date: Any, acq_time: Any) -> pd.Timestamp | None:
    """Parse FIRMS ``acq_date`` (YYYY-MM-DD) + ``acq_time`` (HHMM UTC) into a
    timezone-naive UTC timestamp. Returns ``None`` for invalid values."""
    if acq_time is None or (isinstance(acq_time, float) and np.isnan(acq_time)):
        return None
    try:
        date_ts = pd.to_datetime(str(acq_date).strip(), format="%Y-%m-%d", errors="raise")
        time_int = int(float(acq_time))
        if time_int < 0 or time_int > 2359:
            return None
        hour, minute = divmod(time_int, 100)
        if hour > 23 or minute > 59:
            return None
        return date_ts + pd.Timedelta(hours=hour, minutes=minute)
    except (ValueError, TypeError, OverflowError):
        return None


def pseudo_day_of_year(timestamp: pd.Timestamp) -> int:
    """Return the pseudo day-of-year in 1..365 with Feb 29 pooled into Feb 28.

    This implements the documented leap-year policy
    (``seasonal.leap_policy = pool_feb29_with_feb28``) used by the seasonal
    baseline so that leap-day observations never form a one-sample bucket.
    """
    doy = timestamp.dayofyear
    is_leap = timestamp.is_leap_year
    if is_leap and doy > 59:  # Mar 1 (doy 60 in leap years) and later shift down by one
        doy -= 1
    return doy
