"""Forecast baselines and the rolling-origin backtest protocol.

docs/TESTING.md: "Use rolling-origin backtests against seasonal-naive and
persistence baselines; report suitable point metrics and quantile
coverage/pinball loss. Record forecast issue time and training cutoff. Never
randomly shuffle temporal observations into train/test."

Baseline quantiles are empirical residual quantiles around the point forecast
(uncalibrated; labeled experimental in reports). Missing-coverage days are
excluded from baseline fitting and from metric computation, never treated as
zeros.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.calendar.seasonal import SeasonalClimatology
from src.common.config import BacktestConfig, ForecastConfig, SeasonalConfig
from src.common.logging import get_logger
from src.data.aggregation import DATE_COLUMN

LOGGER = get_logger(__name__)

FORECAST_POINT_COLUMN = "point"
FORECAST_QUANTILES_COLUMN = "quantiles"
FORECAST_QUANTILE_LEVELS_COLUMN = "quantile_levels"


def torch_available() -> bool:
    """True when the optional PyTorch dependency is importable."""
    try:
        return importlib.util.find_spec("torch") is not None
    except Exception:  # pragma: no cover
        return False


class ForecasterFitError(RuntimeError):
    """Raised when a forecaster cannot be fit with the provided history."""


class _BaseForecaster:
    """Common fit/predict plumbing for the baseline forecasters."""

    name = "base"
    version = "1.0.0"

    def __init__(self, forecast_config: ForecastConfig | None = None):
        self.config = forecast_config or ForecastConfig()
        self._residuals: np.ndarray | None = None
        self._last_history: pd.DataFrame | None = None

    def _observed(self, daily: pd.DataFrame) -> pd.DataFrame:
        """Rows with a usable numeric value (gap days excluded, never zeroed)."""
        values = pd.to_numeric(daily["value"], errors="coerce")
        return daily[values.notna()].reset_index(drop=True)

    def _empirical_quantiles(self, point: np.ndarray) -> np.ndarray:
        """Point + residual empirical quantiles, sorted and clipped at zero."""
        if self._residuals is not None and len(self._residuals) > 0:
            qs = np.quantile(self._residuals, self.config.quantiles)
        else:
            qs = np.zeros(len(self.config.quantiles))
        spread = np.sort(qs)[None, :]  # (1, Q)
        out = point[:, None] + spread
        return np.clip(np.sort(out, axis=1), 0.0, None)

    def _horizon_dates(self, history: pd.DataFrame) -> pd.DatetimeIndex:
        last = pd.to_datetime(history[DATE_COLUMN]).max()
        return pd.date_range(last + pd.Timedelta(days=1), periods=self.config.horizon_days, freq="D")

    def _pack(self, point: np.ndarray, quantiles: np.ndarray, horizon_dates: pd.DatetimeIndex) -> dict[str, Any]:
        return {
            "model_name": self.name,
            "model_version": self.version,
            "point": np.asarray(point, dtype=float),
            "quantiles": np.asarray(quantiles, dtype=float),
            "quantile_levels": list(self.config.quantiles),
            "horizon_dates": [d.date().isoformat() for d in horizon_dates],
            "training_history": None,
        }


class PersistenceForecaster(_BaseForecaster):
    """Persistence baseline: the last observed value continues for H steps."""

    name = "persistence"

    def fit(self, daily: pd.DataFrame) -> PersistenceForecaster:
        observed = self._observed(daily)
        if observed.empty:
            raise ForecasterFitError("persistence fit needs at least one observed day")
        values = pd.to_numeric(observed["value"], errors="coerce").to_numpy(dtype=float)
        self._last_history = daily
        self._last_value = float(values[-1])
        if len(values) >= 3:
            self._residuals = np.diff(values)
        else:
            self._residuals = None
        return self

    def predict(self, daily: pd.DataFrame | None = None) -> dict[str, Any]:
        history = daily if daily is not None else self._last_history
        if history is None:
            raise RuntimeError("persistence predict requires fit(history) or predict(daily)")
        point = np.full(self.config.horizon_days, self._last_value)
        quantiles = self._empirical_quantiles(point)
        return self._pack(point, quantiles, self._horizon_dates(history))


class SeasonalNaiveForecaster(_BaseForecaster):
    """Seasonal-naive baseline: the day-of-year climatology continues.

    Where the climatology is unsupported (insufficient samples) the baseline
    falls back to the last observed value and flags that in the result.
    """

    name = "seasonal_naive"

    def __init__(self, forecast_config: ForecastConfig | None = None, seasonal_config: SeasonalConfig | None = None):
        super().__init__(forecast_config)
        self.seasonal_config = seasonal_config or SeasonalConfig()
        self._climatology: SeasonalClimatology | None = None

    def fit(self, daily: pd.DataFrame) -> SeasonalNaiveForecaster:
        observed = self._observed(daily)
        if observed.empty:
            raise ForecasterFitError("seasonal naive fit needs at least one observed day")
        self._last_history = daily
        self._climatology = SeasonalClimatology(self.seasonal_config).fit(
            pd.to_datetime(observed[DATE_COLUMN]), pd.to_numeric(observed["value"], errors="coerce")
        )
        # Residuals of the climatology on the fitted history.
        baselines, _ = self._climatology.baseline(pd.to_datetime(observed[DATE_COLUMN]))
        values = pd.to_numeric(observed["value"], errors="coerce").to_numpy(dtype=float)
        valid = ~np.isnan(baselines)
        self._residuals = values[valid] - baselines[valid] if valid.any() else None
        self._last_value = float(values[-1])
        return self

    def predict(self, daily: pd.DataFrame | None = None) -> dict[str, Any]:
        history = daily if daily is not None else self._last_history
        if history is None:
            raise RuntimeError("seasonal naive predict requires fit(history) or predict(daily)")
        if self._climatology is None:
            raise RuntimeError("seasonal naive predict called before fit")
        horizon_dates = self._horizon_dates(history)
        baselines, counts = self._climatology.baseline(horizon_dates)
        unsupported = np.isnan(baselines)
        point = np.where(unsupported, self._last_value, baselines)
        quantiles = self._empirical_quantiles(point)
        packed = self._pack(point, quantiles, horizon_dates)
        packed["n_unsupported_baseline_steps"] = int(unsupported.sum())
        return packed


# ---------------------------------------------------------------------------
# Rolling-origin backtest
# ---------------------------------------------------------------------------


@dataclass
class BacktestRecord:
    origin: str
    training_cutoff: str
    model_name: str
    model_version: str
    horizon_dates: list[str]
    point: np.ndarray
    quantiles: np.ndarray
    quantile_levels: list[float]
    actuals: np.ndarray
    n_train_days_observed: int
    skipped_reason: str | None = None
    training_history: dict[str, Any] | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "origin": self.origin,
            "training_cutoff": self.training_cutoff,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "horizon_dates": self.horizon_dates,
            "point": self.point.tolist(),
            "quantiles": self.quantiles.tolist(),
            "quantile_levels": self.quantile_levels,
            "actuals": self.actuals.tolist(),
            "n_train_days_observed": self.n_train_days_observed,
            "skipped_reason": self.skipped_reason,
            "training_history": self.training_history,
            **self.extra,
        }


def rolling_origin_backtest(
    daily: pd.DataFrame,
    forecaster_factory: Callable[[], Any],
    backtest_config: BacktestConfig | None = None,
    forecast_config: ForecastConfig | None = None,
) -> list[BacktestRecord]:
    """Evaluate a forecaster over multiple rolling origins.

    For each origin ``o`` the forecaster only receives history with dates
    ``<= o - 1`` (the training cutoff); predictions cover ``o .. o+H-1``.
    Records carry issue time, cutoff, predictions, quantiles, and actuals
    (NaN where coverage was missing - excluded from metrics downstream).
    """
    backtest_config = backtest_config or BacktestConfig()
    forecast_config = forecast_config or ForecastConfig()
    horizon = forecast_config.horizon_days

    work = daily.sort_values(DATE_COLUMN, kind="mergesort").reset_index(drop=True)
    dates = pd.to_datetime(work[DATE_COLUMN])
    values = pd.to_numeric(work["value"], errors="coerce")

    if len(work) == 0:
        raise ForecasterFitError("rolling_origin_backtest requires a non-empty daily frame")

    last_date = dates.max()
    # Candidate origins leave a full observed horizon for evaluation.
    candidates: list[pd.Timestamp] = []
    step = backtest_config.origin_spacing_days
    for k in range(backtest_config.n_origins):
        origin = last_date - pd.Timedelta(days=horizon - 1 + k * step)
        candidates.append(origin.normalize())

    records: list[BacktestRecord] = []
    for origin in candidates:
        cutoff = origin - pd.Timedelta(days=1)
        history = work[dates <= cutoff].reset_index(drop=True)
        n_observed = int(
            pd.to_numeric(history["value"], errors="coerce").notna().sum()
        )
        if n_observed < backtest_config.min_train_days:
            records.append(
                BacktestRecord(
                    origin=origin.date().isoformat(),
                    training_cutoff=cutoff.date().isoformat(),
                    model_name="skipped",
                    model_version="-",
                    horizon_dates=[
                        d.date().isoformat()
                        for d in pd.date_range(origin, periods=horizon, freq="D")
                    ],
                    point=np.full(horizon, np.nan),
                    quantiles=np.full((horizon, len(forecast_config.quantiles)), np.nan),
                    quantile_levels=list(forecast_config.quantiles),
                    actuals=np.full(horizon, np.nan),
                    n_train_days_observed=n_observed,
                    skipped_reason=(
                        f"insufficient observed history ({n_observed} < "
                        f"{backtest_config.min_train_days} days)"
                    ),
                )
            )
            continue

        forecaster = forecaster_factory()
        forecaster.fit(history)
        prediction = forecaster.predict(history)

        # Leakage invariant: no training date beyond the cutoff.
        history_dates = pd.to_datetime(history[DATE_COLUMN])
        if not history_dates.empty and history_dates.max() > cutoff:
            raise RuntimeError("leakage detected: training history extends past the cutoff")

        horizon_dates = pd.date_range(origin, periods=horizon, freq="D")
        actual_lookup = pd.Series(values.to_numpy(), index=dates)
        actuals = actual_lookup.reindex(horizon_dates).to_numpy(dtype=float)

        records.append(
            BacktestRecord(
                origin=origin.date().isoformat(),
                training_cutoff=cutoff.date().isoformat(),
                model_name=prediction["model_name"],
                model_version=prediction["model_version"],
                horizon_dates=[d.date().isoformat() for d in horizon_dates],
                point=np.asarray(prediction["point"], dtype=float),
                quantiles=np.asarray(prediction["quantiles"], dtype=float),
                quantile_levels=list(prediction["quantile_levels"]),
                actuals=actuals,
                n_train_days_observed=n_observed,
                training_history=prediction.get("training_history"),
            )
        )
    return records


def backtest_records_to_frame(records: list[BacktestRecord]) -> pd.DataFrame:
    """Flatten backtest records into a tidy frame (one row per origin x step x quantile)."""
    rows = []
    for rec in records:
        for h, date in enumerate(rec.horizon_dates):
            row = {
                "origin": rec.origin,
                "training_cutoff": rec.training_cutoff,
                "model_name": rec.model_name,
                "model_version": rec.model_version,
                "step": h + 1,
                "date": date,
                "point": rec.point[h],
                "actual": rec.actuals[h],
                "n_train_days_observed": rec.n_train_days_observed,
                "skipped_reason": rec.skipped_reason,
            }
            for tau, q in zip(rec.quantile_levels, rec.quantiles[h], strict=True):
                row[f"q{tau:g}"] = q
            rows.append(row)
    return pd.DataFrame(rows)


__all__ = [
    "PersistenceForecaster",
    "SeasonalNaiveForecaster",
    "ForecasterFitError",
    "BacktestRecord",
    "rolling_origin_backtest",
    "backtest_records_to_frame",
    "torch_available",
]
