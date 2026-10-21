"""Seasonal day-of-year climatology baseline (docs: "Build seasonal baselines
and anomaly scoring first").

Policy (documented in docs/MODEL_CONTRACTS.md):

- pseudo day-of-year 1..365 with Feb 29 pooled into Feb 28 (leap policy),
- circular +/- ``window_days`` smoothing window across year boundaries,
- a calendar day with fewer than ``min_samples_per_day`` pooled samples gets a
  NaN baseline and the anomaly pipeline reports ``insufficient_baseline``
  instead of guessing,
- only days with actual observation coverage contribute samples: missing
  coverage is never treated as zero fire activity.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.common.config import SeasonalConfig
from src.data.schema import pseudo_day_of_year

BASELINE_VERSION = "doy_climatology_v1"


class SeasonalClimatology:
    """Day-of-year climatology over observed daily values."""

    version = BASELINE_VERSION

    def __init__(self, config: SeasonalConfig | None = None):
        self.config = config or SeasonalConfig()
        self._samples: pd.DataFrame | None = None

    @staticmethod
    def _circular_distance(a: np.ndarray, b: int) -> np.ndarray:
        d = np.abs(a - b)
        return np.minimum(d, 365 - d)

    def fit(self, dates: pd.Series, values: pd.Series) -> SeasonalClimatology:
        dates = pd.to_datetime(dates)
        values = pd.Series(values, index=dates.index, dtype=float)
        doy = dates.map(pseudo_day_of_year).astype(int).to_numpy()
        self._samples = pd.DataFrame({"doy": doy, "value": values.to_numpy()})
        self.n_samples_ = int(len(self._samples))
        return self

    @property
    def fitted(self) -> bool:
        return self._samples is not None and self.n_samples_ > 0

    def _baseline_for_doy(self, doy: int) -> tuple[float, int]:
        samples = self._samples
        dist = self._circular_distance(samples["doy"].to_numpy(), doy)
        window = dist <= self.config.window_days
        selected = samples[window]
        if len(selected) < self.config.min_samples_per_day:
            return float("nan"), int(len(selected))
        return float(selected["value"].mean()), int(len(selected))

    def baseline(self, dates: pd.Series) -> tuple[np.ndarray, np.ndarray]:
        """Return (baseline, sample_count) arrays for the requested dates."""
        if not self.fitted:
            raise RuntimeError("SeasonalClimatology.baseline called before fit")
        dates = pd.to_datetime(dates)
        doys = dates.map(pseudo_day_of_year).astype(int).to_numpy()
        cache: dict[int, tuple[float, int]] = {}
        baselines = np.empty(len(doys), dtype=float)
        counts = np.empty(len(doys), dtype=int)
        for i, doy in enumerate(doys):
            if doy not in cache:
                cache[doy] = self._baseline_for_doy(int(doy))
            baselines[i], counts[i] = cache[doy]
        return baselines, counts

    def seasonal_naive(self, dates: pd.Series) -> np.ndarray:
        """Seasonal-naive values: same pseudo day-of-year mean (climatology).

        Used as the seasonal-naive forecast baseline; NaN where the baseline
        is unsupported by samples.
        """
        baselines, _ = self.baseline(dates)
        return baselines


__all__ = ["SeasonalClimatology", "BASELINE_VERSION"]
