"""Unit tests for identity and calibration baselines (SC-03 foundations)."""

import numpy as np
import pandas as pd
import pytest

from src.harmonization.baselines import CalibrationBaseline, IdentityBaseline


def make_X(frp):
    return pd.DataFrame({"modis_frp_mw": frp, "other": np.arange(len(frp), dtype=float)})


class TestIdentityBaseline:
    def test_prediction_is_source_frp(self):
        X = make_X([10.0, 20.0, 30.0])
        model = IdentityBaseline().fit(X, [12.0, 22.0, 33.0])
        preds = model.predict(X)
        assert preds == pytest.approx([10.0, 20.0, 30.0])

    def test_name_and_params(self):
        model = IdentityBaseline()
        assert model.name == "identity"
        assert model.params == {"scale_factor": 1.0}


class TestCalibrationBaseline:
    def test_recovers_known_scale_factor(self):
        rng = np.random.default_rng(3)
        x = rng.uniform(1, 100, 200)
        y = 2.5 * x
        X = make_X(x)
        model = CalibrationBaseline().fit(X, y)
        assert model.scale_factor_ == pytest.approx(2.5, rel=1e-6)
        assert model.predict(X) == pytest.approx(2.5 * x)

    def test_noise_averages_to_correct_factor(self):
        rng = np.random.default_rng(4)
        x = rng.uniform(1, 100, 500)
        y = 1.3 * x + rng.normal(0, 0.1 * x)
        model = CalibrationBaseline().fit(make_X(x), y)
        assert model.scale_factor_ == pytest.approx(1.3, rel=0.05)

    def test_zero_energy_falls_back_to_identity(self):
        X = make_X([0.0, 0.0, 0.0])
        model = CalibrationBaseline().fit(X, [0.0, 0.0, 0.0])
        assert model.scale_factor_ == 1.0

    def test_predict_before_fit_uses_default_factor(self):
        # Calibration starts at k=1 and fit is required for a meaningful factor.
        X = make_X([4.0])
        model = CalibrationBaseline()
        assert model.predict(X) == pytest.approx([4.0])


def test_baselines_handle_nan_like_missing(tmp_path):
    # Baselines must never silently fabricate values for missing FRP.
    X = make_X([10.0, np.nan])
    model = IdentityBaseline().fit(X, [10.0, np.nan])
    preds = model.predict(X)
    assert preds[0] == pytest.approx(10.0)
    assert preds[1] == pytest.approx(0.0)  # documented NaN->0 fill in baselines
