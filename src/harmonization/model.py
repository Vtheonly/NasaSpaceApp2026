"""LightGBM Tweedie gradient-boosted harmonizer (selected primary candidate).

docs/ARCHITECTURE.md Pillar 1: "The primary candidate is LightGBM regression
with a Tweedie objective; XGBoost may be compared if justified." The Tweedie
objective handles the zero inflation of matched targets (MODIS detections with
no VIIRS detection inside the footprint) without treating zeros as missing.

Overfitting control uses chronological early stopping: the last year of the
training period is held out as an internal stopping set (never the validation
or test periods), and boosting stops when it stops improving. The reported
best iteration is recorded in the artifact manifest.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.common.config import HarmonizerModelConfig
from src.common.logging import get_logger

LOGGER = get_logger(__name__)


class ModelNotFittedError(RuntimeError):
    """Raised when predict/information is requested before fit."""


class TorchLikeDependencyError(RuntimeError):
    """Clear, offline-friendly error when an optional backend is missing."""


def require_lightgbm():
    """Import LightGBM lazily with a clear, offline-friendly error message."""
    try:
        import lightgbm  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise TorchLikeDependencyError(
            "LightGBM is required for the selected Tweedie harmonizer. Install the project "
            "dependencies (pip install lightgbm) or use the identity/calibration baselines."
        ) from exc
    return lightgbm


class TweedieGBDTHarmonizer:
    """Wrapper with a stable fit/predict interface and reproducible seeding."""

    name = "lightgbm_tweedie"
    version = "1.0.0"

    def __init__(
        self,
        model_config: HarmonizerModelConfig | None = None,
        seed: int = 2026,
        feature_names: list[str] | None = None,
    ):
        self.model_config = model_config or HarmonizerModelConfig()
        self.seed = int(seed)
        self.feature_names = feature_names
        self._model = None

    @property
    def params(self) -> dict:
        """Recorded training parameters (native LightGBM train API)."""
        return {
            "objective": "tweedie",
            "tweedie_variance_power": self.model_config.tweedie_variance_power,
            "num_boost_round": self.model_config.n_estimators,
            "learning_rate": self.model_config.learning_rate,
            "num_leaves": self.model_config.num_leaves,
            "min_child_samples": self.model_config.min_child_samples,
            "feature_fraction": self.model_config.feature_fraction,
            "bagging_fraction": self.model_config.bagging_fraction,
            "bagging_freq": self.model_config.bagging_freq,
            "lambda_l2": self.model_config.lambda_l2,
            "early_stopping_rounds": self.model_config.early_stopping_rounds,
            "early_stopping_days": self.model_config.early_stopping_days,
            "seed": self.seed,
        }

    def fit(
        self,
        X: pd.DataFrame,
        y,
        stop_dates: pd.Series | None = None,
        stopping_days: int = 365,
    ) -> TweedieGBDTHarmonizer:
        """Fit with chronological early stopping.

        ``stop_dates`` (matched record timestamps) splits the last
        ``stopping_days`` days of the training period into an internal stopping
        set. When it is None or the tail is too small, all data is used and no
        early stopping occurs. Validation/test periods must never be passed
        here (leakage; the training orchestrator enforces this).
        """
        lgbm = require_lightgbm()
        y_array = np.asarray(pd.Series(y).astype(float), dtype=float)
        if np.isnan(y_array).any():
            raise ValueError("Training targets must not contain NaN (check coverage eligibility)")
        if (y_array < 0).any():
            raise ValueError("Tweedie targets must be non-negative")
        self.feature_names = list(X.columns)

        X_fit, y_fit = X, y_array
        valid_sets = None
        callbacks = []
        if stop_dates is not None:
            dates = pd.to_datetime(pd.Series(stop_dates).reset_index(drop=True))
            cutoff = dates.max() - pd.Timedelta(days=self.model_config.early_stopping_days)
            stop_mask = (dates <= cutoff).to_numpy()
            if stop_mask.sum() >= 30 and (~stop_mask).sum() >= 10:
                X_fit, y_fit = X.loc[stop_mask], y_array[stop_mask]
                stop_set = lgbm.Dataset(X.loc[~stop_mask], label=y_array[~stop_mask])
                valid_sets = [stop_set]
                callbacks.append(
                    lgbm.early_stopping(self.model_config.early_stopping_rounds, verbose=False)
                )

        # Native LightGBM training API: no scikit-learn dependency, which keeps
        # the offline wheelhouse minimal (docs/DEVELOPMENT.md offline staging).
        params: dict = {
            "objective": "tweedie",
            "tweedie_variance_power": self.model_config.tweedie_variance_power,
            "learning_rate": self.model_config.learning_rate,
            "num_leaves": self.model_config.num_leaves,
            "min_child_samples": self.model_config.min_child_samples,
            "feature_fraction": self.model_config.feature_fraction,
            "bagging_fraction": self.model_config.bagging_fraction,
            "bagging_freq": self.model_config.bagging_freq,
            "lambda_l2": self.model_config.lambda_l2,
            "seed": self.seed,
            "deterministic": True,
            "force_row_wise": True,
            "verbosity": -1,
        }
        train_set = lgbm.Dataset(X_fit, label=y_fit)
        self._model = lgbm.train(
            params,
            train_set,
            num_boost_round=self.model_config.n_estimators,
            valid_sets=valid_sets,
            callbacks=callbacks or None,
        )
        best = getattr(self._model, "best_iteration", None)
        # LightGBM reports 0 when early stopping never triggered; normalize to None.
        self.best_iteration_ = int(best) if best else None
        if self.best_iteration_:
            LOGGER.info("early stopping chose %d boosting iterations", self.best_iteration_)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._model is None:
            raise ModelNotFittedError(
                "TweedieGBDTHarmonizer.predict called before fit; train the model first"
            )
        if self.feature_names is not None and list(X.columns) != self.feature_names:
            raise ValueError(
                f"Feature mismatch: trained on {self.feature_names}, got {list(X.columns)}"
            )
        raw = self._model.predict(X, num_iteration=self.best_iteration_)
        # VIIRS-equivalent FRP is a non-negative physical quantity.
        return np.clip(np.asarray(raw, dtype=float), 0.0, None)

    @property
    def feature_importances(self) -> dict[str, float]:
        if self._model is None:
            raise ModelNotFittedError("feature_importances requires a fitted model")
        importance = self._model.feature_importance()
        return {
            name: float(value)
            for name, value in zip(self.feature_names, importance, strict=True)
        }

    @property
    def n_boosted_trees(self) -> int:
        if self._model is None:
            raise ModelNotFittedError("n_boosted_trees requires a fitted model")
        return int(self._model.num_trees())


__all__ = [
    "TweedieGBDTHarmonizer",
    "ModelNotFittedError",
    "TorchLikeDependencyError",
    "require_lightgbm",
]
