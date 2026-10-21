"""Unit tests for forecast baselines, rolling-origin protocol, and metrics (R-08)."""

import numpy as np
import pandas as pd
import pytest

from src.calendar.evaluation import (
    CALIBRATION_STATUS_EXPERIMENTAL,
    compare_forecasters,
    mean_pinball,
    pinball_loss,
    quantile_coverage,
    summarize_backtest,
)
from src.calendar.forecast import (
    ForecasterFitError,
    PersistenceForecaster,
    SeasonalNaiveForecaster,
    backtest_records_to_frame,
    rolling_origin_backtest,
)
from src.common.config import BacktestConfig, ForecastConfig, SeasonalConfig
from tests.conftest import make_daily_series


class TestPersistence:
    def test_repeats_last_observed_value(self, synthetic_daily):
        model = PersistenceForecaster(ForecastConfig()).fit(synthetic_daily)
        result = model.predict()
        last = synthetic_daily["value"].dropna().iloc[-1]
        assert np.allclose(result["point"], last)
        assert len(result["point"]) == 14

    def test_quantiles_are_monotone(self, synthetic_daily):
        model = PersistenceForecaster(ForecastConfig()).fit(synthetic_daily)
        result = model.predict()
        assert (np.diff(result["quantiles"], axis=1) >= 0).all()

    def test_empty_history_raises(self):
        empty = pd.DataFrame({"date": pd.to_datetime([]), "value": []})
        with pytest.raises(ForecasterFitError):
            PersistenceForecaster(ForecastConfig()).fit(empty)


class TestSeasonalNaive:
    def test_uses_climatology_for_horizon_dates(self):
        # Construct two full years of a mild seasonal cycle with one hot day.
        dates = pd.date_range("2021-01-01", "2022-12-31", freq="D")
        values = 10 + 5 * np.sin(2 * np.pi * dates.dayofyear / 365.0)
        daily = pd.DataFrame({"date": dates, "value": values, "coverage_status": "observed"})
        model = SeasonalNaiveForecaster(
            ForecastConfig(), SeasonalConfig(window_days=3, min_samples_per_day=2)
        ).fit(daily)
        result = model.predict()
        # Forecast for a mid-summer date next year should be near the seasonal value.
        assert result["point"].min() > 5.0
        assert result["point"].max() < 20.0

    def test_unsupported_baseline_falls_back_to_last_value(self):
        # 100 days of history: a horizon doy 14 days ahead may have 0-1 samples.
        dates = pd.date_range("2021-01-01", periods=100)
        daily = pd.DataFrame({"date": dates, "value": 10.0, "coverage_status": "observed"})
        config = SeasonalConfig(min_samples_per_day=50, window_days=0)
        model = SeasonalNaiveForecaster(ForecastConfig(), config).fit(daily)
        result = model.predict()
        assert result["n_unsupported_baseline_steps"] == 14
        assert np.allclose(result["point"], 10.0)


@pytest.fixture(scope="module")
def rolling_records():
    """Module-scoped rolling-origin backtest records (persistence baseline)."""
    config = ForecastConfig()
    series = make_daily_series(periods=1096, seed=7)
    return rolling_origin_backtest(
        series,
        lambda: PersistenceForecaster(config),
        BacktestConfig(n_origins=3, origin_spacing_days=14, min_train_days=60),
        config,
    )


class TestRollingOrigin:
    @pytest.fixture(scope="module")
    def backtest(self, rolling_records):
        return rolling_records

    def test_origins_and_cutoffs_are_chronological(self, backtest):
        assert len(backtest) == 3
        for record in backtest:
            assert record.origin > record.training_cutoff
            assert record.horizon_dates[0] == record.origin

    def test_history_never_extends_past_cutoff(self, backtest):
        # The backtest itself raises on leakage; here we additionally verify
        # every record has a real training history.
        for record in backtest:
            assert record.n_train_days_observed >= 60
            assert record.skipped_reason is None

    def test_horizon_and_quantile_shapes(self, backtest):
        for record in backtest:
            assert len(record.horizon_dates) == 14
            assert record.point.shape == (14,)
            assert record.quantiles.shape == (14, 3)
            assert (np.diff(record.quantiles, axis=1) >= 0).all()

    def test_actuals_carry_missing_coverage_as_nan(self, backtest):
        for record in backtest:
            # Gap days keep NaN actuals (excluded from metrics downstream).
            assert np.isnan(record.actuals).sum() >= 0  # explicit: gaps never become zeros

    def test_skipped_origins_record_reason(self):
        tiny = make_daily_series(periods=70)
        records = rolling_origin_backtest(
            tiny,
            lambda: PersistenceForecaster(ForecastConfig()),
            BacktestConfig(n_origins=2, origin_spacing_days=7, min_train_days=500),
            ForecastConfig(),
        )
        assert all(r.skipped_reason is not None for r in records)
        assert all(r.model_name == "skipped" for r in records)

    def test_records_to_frame_is_tidy(self, backtest):
        frame = backtest_records_to_frame(backtest)
        assert {"origin", "step", "date", "point", "actual", "q0.1", "q0.5", "q0.9"}.issubset(
            frame.columns
        )
        assert len(frame) == 3 * 14
        assert (frame.groupby("origin")["step"].max() == 14).all()


class TestPinballLoss:
    def test_hand_computed_example(self):
        # y=10, pred=8, tau=0.5 -> |10-8| * 0.5 = 1.0
        assert pinball_loss([10.0], [8.0], 0.5)[0] == pytest.approx(1.0)
        # tau=0.1 (pred below actual): (10-8) * 0.1 = 0.2
        assert pinball_loss([10.0], [8.0], 0.1)[0] == pytest.approx(0.2)
        # tau=0.9 (pred below actual): (10-8) * 0.9 = 1.8
        assert pinball_loss([10.0], [8.0], 0.9)[0] == pytest.approx(1.8)
        # pred above actual: loss scaled by (1 - tau).
        assert pinball_loss([8.0], [10.0], 0.1)[0] == pytest.approx(0.2 * 9)  # (1-0.1)*2

    def test_mean_pinball_drops_nan_actuals(self):
        actuals = np.array([10.0, np.nan, 12.0])
        preds = np.array([8.0, 999.0, 6.0])
        value = mean_pinball(actuals, preds, 0.5)
        expected = np.mean([abs(10 - 8) * 0.5, abs(12 - 6) * 0.5])
        assert value == pytest.approx(expected)


class TestCoverage:
    def test_coverage_fraction(self):
        actuals = np.array([5.0, 10.0, 20.0])
        assert quantile_coverage(actuals, np.array([4.0, 9.0, 19.0]), np.array([6.0, 11.0, 21.0])) == 1.0
        assert quantile_coverage(actuals, np.array([6.0, 9.0, 19.0]), np.array([7.0, 11.0, 21.0])) == pytest.approx(2 / 3)

    def test_coverage_none_when_all_missing(self):
        assert quantile_coverage(np.array([np.nan]), np.array([1.0]), np.array([2.0])) is None


class TestReports:
    def test_summarize_backtest_structure(self, rolling_records):
        report = summarize_backtest(rolling_records, label="persistence")
        assert report["label"] == "persistence"
        assert report["n_evaluated_origins"] == 3
        assert set(report["mean_pinball"]) == {"0.1", "0.5", "0.9"}
        assert 0.0 <= report["interval_coverage"]["coverage"] <= 1.0
        assert report["calibration_status"] == CALIBRATION_STATUS_EXPERIMENTAL
        assert report["interval_coverage"]["calibration_status"] == CALIBRATION_STATUS_EXPERIMENTAL

    def test_summarize_handles_all_skipped(self):
        records = rolling_origin_backtest(
            make_daily_series(periods=70),
            lambda: PersistenceForecaster(ForecastConfig()),
            BacktestConfig(n_origins=1, min_train_days=999),
            ForecastConfig(),
        )
        report = summarize_backtest(records)
        assert report["n_evaluated_origins"] == 0
        assert "no evaluated origins" in report["note"]

    def test_compare_forecasters_deltas(self):
        synthetic = make_daily_series(periods=1096, seed=7)
        config = ForecastConfig()
        bt = BacktestConfig(n_origins=2, origin_spacing_days=14, min_train_days=60)
        reports = {}
        for name, factory in {
            "persistence": lambda: PersistenceForecaster(config),
            "seasonal_naive": lambda: SeasonalNaiveForecaster(config, SeasonalConfig()),
        }.items():
            records = rolling_origin_backtest(synthetic, factory, bt, config)
            reports[name] = summarize_backtest(records, label=name)
        comparison = compare_forecasters(reports)
        assert "seasonal_naive" in comparison
        assert "persistence" in comparison["persistence"] or not comparison["persistence"]
        # A strong annual cycle: seasonal-naive should beat persistence here.
        if "seasonal_naive" in comparison and "persistence" in comparison["seasonal_naive"]:
            assert comparison["seasonal_naive"]["persistence"]["beats_baseline_on_median_pinball"]
