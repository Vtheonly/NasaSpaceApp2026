"""Probabilistic forecast evaluation: pinball loss, quantile coverage, point metrics.

docs/CHALLENGE_AND_REQUIREMENTS.md R-08 scientific rule: "Report quantile
coverage/calibration; label uncalibrated intervals as experimental."
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.calendar.forecast import BacktestRecord
from src.harmonization.evaluate import bias, mae, rmse

CALIBRATION_STATUS_EXPERIMENTAL = "experimental_uncalibrated"


def pinball_loss(actual: np.ndarray, prediction: np.ndarray, tau: float) -> np.ndarray:
    """Element-wise pinball (quantile) loss.

    pinball(y, yhat, tau) = max(tau*(y-yhat), (tau-1)*(y-yhat)); NaN actuals are
    dropped by the aggregation helpers, never imputed.
    """
    actual = np.asarray(actual, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    diff = actual - prediction
    return np.maximum(tau * diff, (tau - 1.0) * diff)


def mean_pinball(actual: np.ndarray, prediction: np.ndarray, tau: float) -> float:
    losses = pinball_loss(actual, prediction, tau)
    valid = ~np.isnan(losses)
    if not valid.any():
        return float("nan")
    return float(np.mean(losses[valid]))


def quantile_coverage(
    actual: np.ndarray, q_low: np.ndarray, q_high: np.ndarray
) -> float | None:
    """Fraction of observed actuals inside [q_low, q_high] (None when empty)."""
    actual = np.asarray(actual, dtype=float)
    q_low = np.asarray(q_low, dtype=float)
    q_high = np.asarray(q_high, dtype=float)
    valid = ~np.isnan(actual)
    if not valid.any():
        return None
    inside = (actual[valid] >= q_low[valid]) & (actual[valid] <= q_high[valid])
    return float(np.mean(inside))


def summarize_backtest(
    records: list[BacktestRecord], label: str = "model"
) -> dict[str, Any]:
    """Aggregate rolling-origin records into a single metrics summary."""
    all_actuals: list[np.ndarray] = []
    all_points: list[np.ndarray] = []
    per_tau: dict[float, list[np.ndarray]] = {}
    q_low_idx: int | None = None
    q_high_idx: int | None = None

    evaluated = [r for r in records if r.skipped_reason is None]
    if not evaluated:
        return {
            "label": label,
            "n_origins": len(records),
            "n_evaluated_origins": 0,
            "skipped_reasons": sorted({r.skipped_reason for r in records if r.skipped_reason}),
            "note": "no evaluated origins; cannot compute metrics",
        }

    for rec in evaluated:
        all_actuals.append(np.asarray(rec.actuals, dtype=float))
        all_points.append(np.asarray(rec.point, dtype=float))
        for idx, tau in enumerate(rec.quantile_levels):
            per_tau.setdefault(float(tau), []).append(np.asarray(rec.quantiles)[:, idx])
        if q_low_idx is None:
            q_low_idx = 0
            q_high_idx = len(rec.quantile_levels) - 1  # quantiles are sorted ascending

    actuals = np.concatenate(all_actuals)
    points = np.concatenate(all_points)
    report: dict[str, Any] = {
        "label": label,
        "model_name": evaluated[0].model_name,
        "model_version": evaluated[0].model_version,
        "n_origins": len(records),
        "n_evaluated_origins": len(evaluated),
        "n_forecast_points": int(len(actuals)),
        "n_observed_actuals": int((~np.isnan(actuals)).sum()),
        "mean_pinball": {
            f"{tau:g}": mean_pinball(actuals, np.concatenate(preds), tau)
            for tau, preds in per_tau.items()
        },
        "rmse": rmse(actuals[~np.isnan(actuals)], points[~np.isnan(actuals)]),
        "mae": mae(actuals[~np.isnan(actuals)], points[~np.isnan(actuals)]),
        "bias": bias(actuals[~np.isnan(actuals)], points[~np.isnan(actuals)]),
        "calibration_status": CALIBRATION_STATUS_EXPERIMENTAL,
    }
    if q_high_idx is not None:
        q_low = np.concatenate(
            [np.asarray(r.quantiles)[:, q_low_idx] for r in evaluated]
        )
        q_high = np.concatenate(
            [np.asarray(r.quantiles)[:, q_high_idx] for r in evaluated]
        )
        coverage = quantile_coverage(actuals, q_low, q_high)
        report["interval_coverage"] = {
            "q_low": float(evaluated[0].quantile_levels[q_low_idx]),
            "q_high": float(evaluated[0].quantile_levels[q_high_idx]),
            "coverage": coverage,
            "nominal": float(evaluated[0].quantile_levels[q_high_idx])
            - float(evaluated[0].quantile_levels[q_low_idx]),
            "calibration_status": CALIBRATION_STATUS_EXPERIMENTAL,
        }
    return report


def compare_forecasters(
    reports: dict[str, dict[str, Any]], baseline_names: tuple[str, ...] = ("seasonal_naive", "persistence")
) -> dict[str, Any]:
    """Relative comparison of every model against the documented baselines."""
    comparison: dict[str, Any] = {}
    for name, report in reports.items():
        if report.get("n_evaluated_origins", 0) == 0:
            continue
        entry: dict[str, Any] = {}
        for baseline in baseline_names:
            base = reports.get(baseline)
            if not base or base.get("n_evaluated_origins", 0) == 0:
                continue
            median_pinball = report["mean_pinball"].get("0.5")
            base_pinball = base["mean_pinball"].get("0.5")
            if median_pinball is None or base_pinball is None:
                continue
            entry[baseline] = {
                "median_pinball_delta_vs_baseline": float(base_pinball - median_pinball),
                "beats_baseline_on_median_pinball": bool(
                    base_pinball - median_pinball > 0
                ),
                "rmse_delta_vs_baseline": float(base["rmse"] - report["rmse"]),
            }
        comparison[name] = entry
    return comparison


__all__ = [
    "pinball_loss",
    "mean_pinball",
    "quantile_coverage",
    "summarize_backtest",
    "compare_forecasters",
    "CALIBRATION_STATUS_EXPERIMENTAL",
]
