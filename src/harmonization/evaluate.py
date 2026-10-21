"""Held-out chronological evaluation for the harmonization pillar.

docs/TESTING.md: "Report bias and error by FRP range, region, season, land
cover/geometry where sample size supports it. Test matching-tolerance
sensitivity, zero inflation, extreme observations..." Region/land-cover slices
require auxiliary sources and are recorded as a documented limitation until
those joins exist; FRP-range and season slices are implemented.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

#: FRP-range bins for slice reporting (MW). Boundaries are documented defaults.
FRP_RANGE_BINS: tuple[float, ...] = (0.0, 10.0, 50.0, np.inf)
FRP_RANGE_LABELS: tuple[str, ...] = ("0-10", "10-50", "50+")


def tweedie_deviance(y_true, y_pred, power: float = 1.5) -> float:
    """Mean Tweedie deviance for 1 < power < 2 (primary harmonization metric).

    Closed-form unit deviance (the same definition as scikit-learn's
    ``mean_tweedie_deviance``; implemented locally and cross-verified against
    it to 1e-12 so the offline wheelhouse stays minimal):

        d(y, mu) = 2 * { y^(2-p) / ((1-p)(2-p))
                         - y * mu^(1-p) / (1-p)
                         + mu^(2-p) / (2-p) }

    For power 1.5 this reduces to -8*sqrt(y) + 4*y/sqrt(mu) + 4*sqrt(mu); the
    deviance is minimized at the conditional mean, so it pairs canonically
    with the LightGBM Tweedie objective. Zero FRP observations are handled
    natively. Lower is better.
    """
    if not 1.0 < power < 2.0:
        raise ValueError(f"Tweedie power must be in (1, 2); got {power}")
    y = np.asarray(y_true, dtype=float)
    mu = np.clip(np.asarray(y_pred, dtype=float), 1e-9, None)
    p = power
    unit = (
        np.power(y, 2.0 - p) / ((1.0 - p) * (2.0 - p))
        - y * np.power(mu, 1.0 - p) / (1.0 - p)
        + np.power(mu, 2.0 - p) / (2.0 - p)
    )
    return float(np.mean(2.0 * unit))


def rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred) -> float:
    return float(
        np.mean(np.abs(np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)))
    )


def bias(y_true, y_pred) -> float:
    """Mean signed error (prediction minus observation)."""
    return float(np.mean(np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)))


def ratio_of_totals(y_true, y_pred) -> float:
    """Total predicted FRP divided by total observed FRP (mass conservation view)."""
    total_true = float(np.sum(y_true))
    if total_true <= 0:
        return float("nan")
    return float(np.sum(y_pred) / total_true)


def season_from_month(months: pd.Series) -> pd.Series:
    """Map month numbers to meteorological seasons (DJF/MAM/JJA/SON)."""
    season = np.select(
        [months.isin([12, 1, 2]), months.isin([3, 4, 5]), months.isin([6, 7, 8])],
        ["DJF", "MAM", "JJA"],
        default="SON",
    )
    return pd.Series(season, index=months.index)


def _slice_metrics(y_true: np.ndarray, y_pred: np.ndarray, power: float) -> dict[str, float]:
    if len(y_true) == 0:
        return {"n": 0}
    return {
        "n": int(len(y_true)),
        "tweedie_deviance": tweedie_deviance(y_true, y_pred, power),
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "bias": bias(y_true, y_pred),
        "ratio_of_totals": ratio_of_totals(y_true, y_pred),
    }


def evaluate_predictions(
    name: str,
    y_true,
    y_pred,
    frp_source: pd.Series | None = None,
    months: pd.Series | None = None,
    power: float = 1.5,
) -> dict[str, Any]:
    """Evaluate one predictor with optional FRP-range and season slices."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    report: dict[str, Any] = {
        "name": name,
        "n": int(len(y_true)),
        "tweedie_deviance": tweedie_deviance(y_true, y_pred, power),
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "bias": bias(y_true, y_pred),
        "ratio_of_totals": ratio_of_totals(y_true, y_pred),
    }
    if frp_source is not None:
        frp = np.asarray(pd.to_numeric(frp_source, errors="coerce").fillna(0.0), dtype=float)
        bins = pd.cut(frp, bins=list(FRP_RANGE_BINS), labels=FRP_RANGE_LABELS, right=False)
        report["by_frp_range"] = {
            str(label): _slice_metrics(
                y_true[np.asarray(bins == label)], y_pred[np.asarray(bins == label)], power
            )
            for label in FRP_RANGE_LABELS
            if bool((bins == label).any())
        }
    if months is not None:
        seasons = season_from_month(pd.Series(months).reset_index(drop=True))
        report["by_season"] = {
            season: _slice_metrics(
                y_true[np.asarray(seasons == season)],
                y_pred[np.asarray(seasons == season)],
                power,
            )
            for season in ("DJF", "MAM", "JJA", "SON")
            if bool((seasons == season).any())
        }
    return report


def compare_predictions(
    predictions: dict[str, Callable[[pd.DataFrame], np.ndarray] | np.ndarray],
    X: pd.DataFrame,
    y_true,
    frp_source: pd.Series | None = None,
    months: pd.Series | None = None,
    power: float = 1.5,
) -> dict[str, Any]:
    """Evaluate multiple predictors over the same held-out set."""
    y_true = np.asarray(pd.Series(y_true).reset_index(drop=True), dtype=float)
    out: dict[str, Any] = {}
    for name, pred in predictions.items():
        y_pred = pred(X) if callable(pred) else np.asarray(pred, dtype=float)
        out[name] = evaluate_predictions(name, y_true, y_pred, frp_source, months, power)
    return out


def beats_baseline(
    model_report: dict[str, Any],
    baseline_report: dict[str, Any],
    metric: str = "tweedie_deviance",
) -> tuple[bool, float]:
    """Return (beats, margin). Lower metric is better for deviance/rmse/mae."""
    if model_report.get("n", 0) == 0 or baseline_report.get("n", 0) == 0:
        return False, float("nan")
    margin = float(baseline_report[metric] - model_report[metric])
    return margin > 0, margin


__all__ = [
    "FRP_RANGE_BINS",
    "FRP_RANGE_LABELS",
    "tweedie_deviance",
    "rmse",
    "mae",
    "bias",
    "ratio_of_totals",
    "season_from_month",
    "evaluate_predictions",
    "compare_predictions",
    "beats_baseline",
]
