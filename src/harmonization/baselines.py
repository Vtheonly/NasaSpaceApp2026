"""Simple baselines required by the scientific acceptance rules.

docs/CHALLENGE_AND_REQUIREMENTS.md: "Compare harmonization against
identity/no-correction and simple calibration baselines." A learned model must
beat or explain the difference versus these baselines on held-out data (SC-03).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class IdentityBaseline:
    """No-correction baseline: predicted VIIRS-equivalent FRP = MODIS FRP."""

    name = "identity"
    version = "1.0.0"

    def fit(self, X: pd.DataFrame, y: pd.Series) -> IdentityBaseline:
        self.scale_factor_ = 1.0
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(
            pd.to_numeric(X["modis_frp_mw"], errors="coerce").fillna(0.0), dtype=float
        )

    @property
    def params(self) -> dict:
        return {"scale_factor": 1.0}


class CalibrationBaseline:
    """Global scale-factor calibration fitted through the origin.

    ``k = sum(x*y) / sum(x*x)`` minimizes squared error of ``y ~ k*x`` on the
    training set. Falls back to ``k = 1`` (identity) when the source has zero
    energy, so the baseline always predicts.
    """

    name = "calibration"
    version = "1.0.0"

    def __init__(self) -> None:
        self.scale_factor_ = 1.0

    def fit(self, X: pd.DataFrame, y) -> CalibrationBaseline:
        x = np.asarray(
            pd.to_numeric(X["modis_frp_mw"], errors="coerce").fillna(0.0), dtype=float
        )
        yv = np.asarray(pd.Series(y).astype(float).fillna(0.0), dtype=float)
        denom = float(np.dot(x, x))
        if denom > 0:
            self.scale_factor_ = float(np.dot(x, yv) / denom)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.scale_factor_ * np.asarray(
            pd.to_numeric(X["modis_frp_mw"], errors="coerce").fillna(0.0), dtype=float
        )

    @property
    def params(self) -> dict:
        return {"scale_factor": self.scale_factor_}


__all__ = ["IdentityBaseline", "CalibrationBaseline"]
