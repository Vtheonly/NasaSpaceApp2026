"""Anomaly scoring against the seasonal baseline (challenge R-07, SC-05).

Score definition (documented; thresholds must be validated before alerting):

``ratio = (observed + eps) / (baseline + eps)``

with statuses:

- ``normal``                - ratio between low and high thresholds,
- ``anomaly_high``          - ratio >= high threshold,
- ``anomaly_low``           - ratio <= low threshold,
- ``insufficient_baseline`` - fewer pooled samples than configured,
- ``missing_coverage``       - no sensor coverage that day (score NaN; never
  treated as zero activity).

Anomaly scores are relative-to-season statements, not ignition claims.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.calendar.seasonal import SeasonalClimatology
from src.common.config import AnomalyConfig
from src.data.aggregation import COVERAGE_STATUS_COLUMN, DATE_COLUMN, STATUS_OBSERVED

SCORE_VERSION = "ratio_v1"

STATUS_NORMAL = "normal"
STATUS_ANOMALY_HIGH = "anomaly_high"
STATUS_ANOMALY_LOW = "anomaly_low"
STATUS_INSUFFICIENT_BASELINE = "insufficient_baseline"
STATUS_MISSING_COVERAGE = "missing_coverage"

ANOMALY_COLUMNS: tuple[str, ...] = (
    DATE_COLUMN,
    "observed",
    "baseline",
    "ratio",
    "status",
    "baseline_sample_count",
    "baseline_version",
    "score_version",
)


def score_anomalies(
    daily: pd.DataFrame,
    climatology: SeasonalClimatology,
    config: AnomalyConfig | None = None,
    value_column: str = "value",
    baseline_dates: pd.Series | None = None,
) -> pd.DataFrame:
    """Score daily values against the fitted climatology.

    ``baseline_dates`` optionally restricts which rows of ``daily`` were used
    to fit the climatology (leakage tests verify no future dates are used).
    """
    config = config or AnomalyConfig()
    if not climatology.fitted:
        raise RuntimeError("score_anomalies requires a fitted SeasonalClimatology")

    out = daily.copy().reset_index(drop=True)
    dates = pd.to_datetime(out[DATE_COLUMN])
    baselines, counts = climatology.baseline(dates)
    out["baseline"] = baselines
    out["baseline_sample_count"] = counts

    observed = pd.to_numeric(out[value_column], errors="coerce").to_numpy(dtype=float)
    out["observed"] = observed
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = (observed + config.ratio_epsilon) / (np.asarray(baselines) + config.ratio_epsilon)
    out["ratio"] = ratio

    coverage_observed = (
        out[COVERAGE_STATUS_COLUMN].eq(STATUS_OBSERVED).to_numpy()
        if COVERAGE_STATUS_COLUMN in out.columns
        else ~np.isnan(observed)
    )
    status = np.empty(len(out), dtype=object)
    for i in range(len(out)):
        if not coverage_observed[i]:
            status[i] = STATUS_MISSING_COVERAGE
        elif np.isnan(baselines[i]):
            status[i] = STATUS_INSUFFICIENT_BASELINE
        elif ratio[i] >= config.high_threshold:
            status[i] = STATUS_ANOMALY_HIGH
        elif ratio[i] <= config.low_threshold:
            status[i] = STATUS_ANOMALY_LOW
        else:
            status[i] = STATUS_NORMAL

    # Missing-coverage days must not carry a numeric score.
    out.loc[~coverage_observed, "ratio"] = np.nan
    out["status"] = status
    out["baseline_version"] = climatology.version
    out["score_version"] = SCORE_VERSION

    columns = [c for c in ANOMALY_COLUMNS if c in out.columns]
    return out[columns]


__all__ = [
    "score_anomalies",
    "SCORE_VERSION",
    "STATUS_NORMAL",
    "STATUS_ANOMALY_HIGH",
    "STATUS_ANOMALY_LOW",
    "STATUS_INSUFFICIENT_BASELINE",
    "STATUS_MISSING_COVERAGE",
    "ANOMALY_COLUMNS",
]
