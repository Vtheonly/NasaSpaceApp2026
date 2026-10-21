"""Unit tests for the LightGBM Tweedie harmonizer wrapper."""

import numpy as np
import pandas as pd
import pytest

from src.common.config import HarmonizerModelConfig
from src.harmonization.model import (
    ModelNotFittedError,
    TweedieGBDTHarmonizer,
    require_lightgbm,
)


def make_xy(n=400, seed=5):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.1, 100, n)
    frame = pd.DataFrame(
        {
            "modis_frp_mw": x,
            "log_modis_frp": np.log1p(x),
            "doy_sin": rng.uniform(-1, 1, n),
            "doy_cos": rng.uniform(-1, 1, n),
        }
    )
    y = 1.25 * x * rng.lognormal(0, 0.1, n)
    return frame, y


class TestFitPredict:
    def test_predictions_are_non_negative_and_shaped(self):
        X, y = make_xy()
        model = TweedieGBDTHarmonizer(HarmonizerModelConfig(), seed=1).fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (len(y),)
        assert (preds >= 0).all()

    def test_predict_before_fit_raises(self):
        model = TweedieGBDTHarmonizer()
        with pytest.raises(ModelNotFittedError):
            model.predict(make_xy(10)[0])

    def test_feature_mismatch_raises(self):
        X, y = make_xy()
        model = TweedieGBDTHarmonizer(HarmonizerModelConfig(), seed=1).fit(X, y)
        with pytest.raises(ValueError, match="Feature mismatch"):
            model.predict(X.drop(columns=["doy_sin"]))

    def test_nan_targets_rejected(self):
        X, y = make_xy(50)
        y[3] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            TweedieGBDTHarmonizer(HarmonizerModelConfig(), seed=1).fit(X, y)

    def test_negative_targets_rejected(self):
        X, y = make_xy(50)
        y[3] = -1.0
        with pytest.raises(ValueError, match="non-negative"):
            TweedieGBDTHarmonizer(HarmonizerModelConfig(), seed=1).fit(X, y)

    def test_feature_importances_available_after_fit(self):
        X, y = make_xy(200)
        model = TweedieGBDTHarmonizer(HarmonizerModelConfig(), seed=1).fit(X, y)
        importances = model.feature_importances
        assert set(importances) == set(X.columns)
        assert sum(importances.values()) > 0


class TestReproducibility:
    def test_same_seed_same_predictions(self):
        X, y = make_xy(200, seed=11)
        first = TweedieGBDTHarmonizer(HarmonizerModelConfig(), seed=42).fit(X, y)
        second = TweedieGBDTHarmonizer(HarmonizerModelConfig(), seed=42).fit(X, y)
        assert first.predict(X) == pytest.approx(second.predict(X), rel=1e-12)

    def test_learns_calibration_factor(self):
        # With no noise the model should track the 1.25x relationship closely.
        rng = np.random.default_rng(9)
        x = np.sort(rng.uniform(1, 100, 300))
        X = pd.DataFrame({"modis_frp_mw": x, "log_modis_frp": np.log1p(x)})
        y = 1.25 * x
        model = TweedieGBDTHarmonizer(
            HarmonizerModelConfig(n_estimators=150, learning_rate=0.1), seed=7
        ).fit(X, y)
        preds = model.predict(X)
        # Mean ratio should be near 1.25 on the mid-to-high FRP range.
        mid = x > 20
        assert np.mean(preds[mid] / x[mid]) == pytest.approx(1.25, rel=0.2)


class TestEarlyStopping:
    def test_early_stopping_picks_iteration(self):
        X, y = make_xy(730, seed=13)
        stop_dates = pd.Series(pd.date_range("2020-01-01", periods=730, freq="D"))
        model = TweedieGBDTHarmonizer(HarmonizerModelConfig(), seed=3).fit(
            X, y, stop_dates=stop_dates
        )
        assert model.best_iteration_ is not None
        assert 0 < model.best_iteration_ <= HarmonizerModelConfig().n_estimators

    def test_without_stop_dates_uses_all_iterations(self):
        X, y = make_xy(100, seed=14)
        model = TweedieGBDTHarmonizer(
            HarmonizerModelConfig(n_estimators=20, learning_rate=0.1), seed=3
        ).fit(X, y)
        assert model.best_iteration_ is None
        assert model.predict(X).shape == (100,)


def test_require_lightgbm_returns_module():
    import lightgbm

    assert require_lightgbm() is lightgbm


def test_params_record_configuration():
    model = TweedieGBDTHarmonizer(HarmonizerModelConfig(num_leaves=7), seed=99)
    params = model.params
    assert params["objective"] == "tweedie"
    assert params["num_leaves"] == 7
    assert params["random_state"] == 99
    assert params["lambda_l2"] == HarmonizerModelConfig().lambda_l2
