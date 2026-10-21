"""Unit tests for the PatchTST forecaster architecture (docs Pillar 2 candidate)."""

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")  # optional dependency: skip suite without it

from src.common.config import (  # noqa: E402
    ForecastConfig,
    PatchTSTParams,
    TrainerConfig,
)
from src.models.patchtst import (  # noqa: E402
    PatchTSTForecaster,
    TorchDependencyError,
)
from tests.conftest import make_daily_series  # noqa: E402


def tiny_forecast_config(**overrides) -> ForecastConfig:
    defaults = dict(
        horizon_days=14,
        quantiles=(0.1, 0.5, 0.9),
        patchtst=PatchTSTParams(
            context_len=48,
            patch_len=8,
            stride=4,
            d_model=16,
            n_encoder_blocks=1,
            n_heads=2,
            dropout=0.0,
            dim_feedforward=32,
        ),
        trainer=TrainerConfig(epochs=12, batch_size=16, learning_rate=0.05, patience=4),
    )
    defaults.update(overrides)
    return ForecastConfig(**defaults)


class TestWindowConstruction:
    def test_window_shapes(self):
        forecaster = PatchTSTForecaster(tiny_forecast_config(), seed=1)
        matrix = np.random.default_rng(0).normal(size=(200, 3))
        X, y = forecaster.make_windows(matrix, horizon=14)
        assert X.shape == (200 - 48 - 14 + 1, 3, 48)
        assert y.shape == (200 - 48 - 14 + 1, 14)

    def test_windows_never_leak_future(self):
        forecaster = PatchTSTForecaster(tiny_forecast_config(), seed=1)
        matrix = np.arange(100, dtype=float).reshape(-1, 1)  # unique value per day
        X, y = forecaster.make_windows(matrix, horizon=5)
        for i in range(len(X)):
            assert X[i].max() < y[i].min()  # every target day is strictly after the window
            assert list(y[i]) == list(range(int(X[i].max()) + 1, int(X[i].max()) + 1 + 5))

    def test_short_series_raises(self):
        forecaster = PatchTSTForecaster(tiny_forecast_config(), seed=1)
        with pytest.raises(ValueError, match="shorter than"):
            forecaster.make_windows(np.zeros((20, 3), dtype=float), horizon=14)


class TestArchitectureContract:
    def test_patch_count_formula(self):
        params = PatchTSTParams(context_len=90, patch_len=16, stride=8)
        assert params.n_patches == 1 + (90 - 16) // 8 == 10

    def test_patch_len_cannot_exceed_context(self):
        with pytest.raises(ValueError, match="patch_len"):
            _ = PatchTSTParams(context_len=10, patch_len=16).n_patches

    def test_head_dim_divisible_by_heads(self):
        config = tiny_forecast_config()
        broken = ForecastConfig(
            horizon_days=14,
            quantiles=config.quantiles,
            patchtst=PatchTSTParams(context_len=48, patch_len=8, stride=4, d_model=18, n_heads=4),
        )
        with pytest.raises(ValueError, match="divisible"):
            PatchTSTForecaster(broken)

    def test_quantiles_must_contain_median(self):
        with pytest.raises(ValueError, match="median"):
            ForecastConfig.from_dict({"quantiles": [0.1, 0.9]})

    def test_quantiles_must_be_probabilities(self):
        with pytest.raises(ValueError, match="quantiles"):
            ForecastConfig.from_dict({"quantiles": [0.1, 1.5, 0.5]})


class TestPredictionContract:
    def test_output_shapes_and_quantile_ordering(self, synthetic_daily):
        forecaster = PatchTSTForecaster(tiny_forecast_config(), seed=2026).fit(synthetic_daily)
        result = forecaster.predict()
        assert result["point"].shape == (14,)
        assert result["quantiles"].shape == (14, 3)
        assert result["quantile_levels"] == [0.1, 0.5, 0.9]
        # Monotone quantiles at every horizon step (docs requirement).
        assert (np.diff(result["quantiles"], axis=1) >= 0).all()
        # Non-negative FRP forecasts.
        assert (result["quantiles"] >= 0).all()

    def test_point_forecast_is_median_quantile(self, synthetic_daily):
        forecaster = PatchTSTForecaster(tiny_forecast_config(), seed=2026).fit(synthetic_daily)
        result = forecaster.predict()
        assert np.allclose(result["point"], result["quantiles"][:, 1])

    def test_predict_before_fit_raises(self):
        with pytest.raises(RuntimeError, match="before fit"):
            PatchTSTForecaster(tiny_forecast_config()).predict(make_daily_series(periods=80))

    def test_channel_independence(self, synthetic_daily):
        """Target-channel forecasts must not depend on covariate channels."""
        config = tiny_forecast_config()
        forecaster = PatchTSTForecaster(config, seed=2026)
        # Fit on the plain series.
        forecaster.fit(synthetic_daily)
        base = forecaster.predict(synthetic_daily)

        # Perturb ONLY the covariate channels of recent history.
        perturbed = synthetic_daily.copy()
        perturbed["coverage_status"] = "observed"
        perturbed["modis_active_pixels"] = 999
        perturbed["viirs_active_pixels"] = 777
        changed = forecaster.predict(perturbed)

        # Same value channel -> identical target forecasts (channel independence).
        assert np.allclose(base["point"], changed["point"], rtol=1e-6)

    def test_gap_days_flagged_by_coverage_channel(self):
        daily = make_daily_series(periods=200, gap_fraction=0.1)
        forecaster = PatchTSTForecaster(tiny_forecast_config(), seed=1)
        matrix, channels = forecaster.build_channel_matrix(daily)
        gap_rows = daily["value"].isna().to_numpy()
        assert channels == ["value", "active_pixels", "coverage"]
        # Gap policy: value filled with 0.0, coverage channel 0 (docs: explicit,
        # never silently treat missing coverage as zero activity).
        assert (matrix[gap_rows, 0] == 0.0).all()
        assert (matrix[gap_rows, 2] == 0.0).all()
        assert (matrix[~gap_rows, 2] == 1.0).all()


class TestLearning:
    def test_training_loss_decreases(self):
        daily = make_daily_series(periods=400, seed=42, gap_fraction=0.0)
        forecaster = PatchTSTForecaster(
            tiny_forecast_config(trainer=TrainerConfig(epochs=15, batch_size=32, learning_rate=0.05, patience=50)),
            seed=2026,
        ).fit(daily)
        history = forecaster.training_history
        assert history["epochs_run"] >= 5
        first, last = history["train_losses"][0], history["train_losses"][-1]
        assert last < first  # optimization actually reduces pinball loss

    def test_learns_constant_series(self):
        dates = pd.date_range("2022-01-01", periods=365)
        daily = pd.DataFrame({"date": dates, "value": 25.0, "coverage_status": "observed"})
        forecaster = PatchTSTForecaster(
            tiny_forecast_config(trainer=TrainerConfig(epochs=25, batch_size=32, learning_rate=0.05, patience=50)),
            seed=2026,
        ).fit(daily)
        result = forecaster.predict()
        assert np.allclose(result["point"], 25.0, atol=4.0)

    def test_deterministic_fit_with_fixed_seed(self):
        daily = make_daily_series(periods=200, seed=3, gap_fraction=0.0)
        first = PatchTSTForecaster(tiny_forecast_config(), seed=7).fit(daily).predict()
        second = PatchTSTForecaster(tiny_forecast_config(), seed=7).fit(daily).predict()
        assert np.allclose(first["point"], second["point"], rtol=1e-9)


class TestSerialization:
    def test_state_dict_roundtrip_preserves_predictions(self):
        daily = make_daily_series(periods=200, seed=5, gap_fraction=0.0)
        forecaster = PatchTSTForecaster(tiny_forecast_config(), seed=11).fit(daily)
        state = forecaster.state_dict()
        restored = PatchTSTForecaster().load_state_dict(state)
        assert np.allclose(
            forecaster.predict()["quantiles"], restored.predict(daily)["quantiles"], rtol=1e-9
        )

    def test_state_records_provenance(self):
        daily = make_daily_series(periods=120, seed=5, gap_fraction=0.0)
        forecaster = PatchTSTForecaster(tiny_forecast_config(), seed=11).fit(daily)
        state = forecaster.state_dict()
        assert state["model_name"] == "patchtst"
        assert state["seed"] == 11
        assert state["training_history"]["seed"] == 11
        assert len(state["torch_state"]) > 0


class TestGracefulFailureWithoutTorch:
    def test_clear_error_when_torch_missing(self, monkeypatch, synthetic_daily):
        import sys

        monkeypatch.setitem(sys.modules, "torch", None)
        with pytest.raises(TorchDependencyError, match="PyTorch is required"):
            PatchTSTForecaster(tiny_forecast_config()).fit(synthetic_daily)

    def test_error_mentions_baselines_remain_available(self, monkeypatch, synthetic_daily):
        import sys

        monkeypatch.setitem(sys.modules, "torch", None)
        forecaster = PatchTSTForecaster(tiny_forecast_config())
        with pytest.raises(TorchDependencyError, match="baselines remain available"):
            forecaster.fit(synthetic_daily)
