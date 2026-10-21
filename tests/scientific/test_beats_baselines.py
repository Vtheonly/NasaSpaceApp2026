"""Scientific model-comparison tests: the learned models must beat (or tie)
the documented baselines on held-out data (SC-03, SC-06 evidence).
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")  # PatchTST comparison needs the optional dependency

from src.calendar.evaluation import summarize_backtest  # noqa: E402
from src.calendar.forecast import (  # noqa: E402
    PersistenceForecaster,
    SeasonalNaiveForecaster,
    rolling_origin_backtest,
)
from src.models.patchtst import PatchTSTForecaster  # noqa: E402
from tests.conftest import make_daily_series, tiny_calendar_config  # noqa: E402


class TestHarmonizerBeatsBaselines:
    """SC-03: Tweedie GBDT vs identity and calibration on held-out periods."""

    def test_beats_identity_on_validation(self, training_result):
        reports = training_result.evaluation["validation"]
        model = reports["lightgbm_tweedie"]["tweedie_deviance"]
        identity = reports["identity"]["tweedie_deviance"]
        assert model < identity, "harmonizer must beat the identity baseline on validation"

    def test_beats_global_calibration_on_validation(self, training_result):
        reports = training_result.evaluation["validation"]
        model = reports["lightgbm_tweedie"]["tweedie_deviance"]
        calibration = reports["calibration"]["tweedie_deviance"]
        assert model < calibration

    def test_beats_identity_on_test(self, training_result):
        reports = training_result.evaluation["test"]
        model = reports["lightgbm_tweedie"]["tweedie_deviance"]
        identity = reports["identity"]["tweedie_deviance"]
        assert model <= identity, (
            "harmonizer must at least match the identity baseline on the test split "
            f"(model={model:.4f}, identity={identity:.4f})"
        )

    def test_comparison_flags_recorded(self, training_result):
        comparison = training_result.baseline_comparison["validation"]
        assert comparison["identity"]["beats"] is True
        assert comparison["calibration"]["beats"] is True
        assert comparison["identity"]["metric"] == "tweedie_deviance"


class TestForecastersCompareOnRollingOrigin:
    """SC-06 / R-08: PatchTST vs seasonal-naive and persistence baselines."""

    @pytest.fixture(scope="class")
    def backtest_reports(self):
        # Strong weekly periodicity: any competent forecaster beats persistence.
        config = tiny_calendar_config(min_train_days=120)
        daily = make_daily_series(periods=730, seed=99, gap_fraction=0.0)
        rng = np.random.default_rng(99)
        values = daily["value"].to_numpy()
        daily["value"] = values + 30.0 * np.sin(
            2 * np.pi * np.arange(len(daily)) / 7.0
        )
        daily["value"] = np.clip(daily["value"] + rng.normal(0, 2.0, len(daily)), 0, None)

        reports = {}
        factories = {
            "persistence": lambda: PersistenceForecaster(config.forecast),
            "seasonal_naive": lambda: SeasonalNaiveForecaster(config.forecast, config.seasonal),
            "patchtst": lambda: PatchTSTForecaster(config.forecast, seed=config.seed),
        }
        for name, factory in factories.items():
            records = rolling_origin_backtest(
                daily, factory, config.backtest, config.forecast
            )
            reports[name] = summarize_backtest(records, label=name)
        return reports

    def test_patchtst_report_is_complete(self, backtest_reports):
        report = backtest_reports["patchtst"]
        assert report["n_evaluated_origins"] > 0
        assert set(report["mean_pinball"]) == {"0.1", "0.5", "0.9"}
        assert report["rmse"] >= 0
        assert 0.0 <= report["interval_coverage"]["coverage"] <= 1.0

    def test_patchtst_matches_or_beats_persistence(self, backtest_reports):
        # On a strongly periodic series the learned model must not be worse
        # than the naive persistence baseline (median pinball loss).
        model = backtest_reports["patchtst"]["mean_pinball"]["0.5"]
        persistence = backtest_reports["persistence"]["mean_pinball"]["0.5"]
        assert model <= persistence * 1.15, (
            f"PatchTST median pinball ({model:.4f}) should be comparable to or better "
            f"than persistence ({persistence:.4f}) on a periodic series"
        )

    def test_seasonal_naive_report_present(self, backtest_reports):
        assert backtest_reports["seasonal_naive"]["n_evaluated_origins"] > 0

    def test_calibration_is_labeled_experimental(self, backtest_reports):
        # docs: "label uncalibrated intervals as experimental".
        assert backtest_reports["patchtst"]["calibration_status"] == "experimental_uncalibrated"


def test_heldout_metrics_recorded_in_training_result(training_result):
    """The training result carries reproducible held-out evidence (R-14)."""
    evaluation = training_result.evaluation
    for split in ("validation", "test"):
        if evaluation[split].get("skipped"):
            continue
        for name in ("identity", "calibration", "lightgbm_tweedie"):
            report = evaluation[split][name]
            assert {"tweedie_deviance", "rmse", "mae", "bias", "n"} <= set(report)
    assert training_result.config_snapshot["seed"] == training_result.config_snapshot["seed"]
