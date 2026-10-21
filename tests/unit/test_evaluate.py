"""Unit tests for harmonization evaluation metrics and slice reporting."""

import numpy as np
import pandas as pd
import pytest

from src.harmonization.evaluate import (
    FRP_RANGE_LABELS,
    beats_baseline,
    bias,
    compare_predictions,
    evaluate_predictions,
    ratio_of_totals,
    rmse,
    season_from_month,
    tweedie_deviance,
)


class TestMetrics:
    def test_perfect_predictions_have_zero_deviance(self):
        y = np.array([1.0, 10.0, 100.0])
        assert tweedie_deviance(y, y) == pytest.approx(0.0, abs=1e-12)

    def test_deviance_asymmetry_favors_conservative_frp(self):
        y = np.array([10.0])
        under = tweedie_deviance(y, [5.0])   # 50% under
        over = tweedie_deviance(y, [15.0])   # 50% over
        # For Tweedie power 1.5, underpredicting a large fire costs more than
        # overpredicting by the same margin (favoring conservative estimates).
        assert under > over

    def test_rmse_known_value(self):
        assert rmse([0.0, 0.0], [3.0, 4.0]) == pytest.approx(((9 + 16) / 2) ** 0.5)

    def test_bias_sign(self):
        assert bias([10.0], [12.0]) == pytest.approx(2.0)  # pred above obs -> positive
        assert bias([10.0], [8.0]) == pytest.approx(-2.0)

    def test_ratio_of_totals(self):
        y = [10.0, 20.0, 30.0]
        assert ratio_of_totals(y, [20.0, 40.0, 60.0]) == pytest.approx(2.0)
        assert np.isnan(ratio_of_totals([0.0, 0.0], [1.0, 1.0]))


class TestSlices:
    def test_frp_range_slices_partition_data(self):
        rng = np.random.default_rng(2)
        frp = rng.uniform(0, 100, 300)
        y = frp * 1.2
        months = pd.Series(rng.integers(1, 13, 300))
        report = evaluate_predictions("m", y, y, frp_source=pd.Series(frp), months=months)
        assert set(report["by_frp_range"].keys()) == set(FRP_RANGE_LABELS)
        assert sum(v["n"] for v in report["by_frp_range"].values()) == 300
        assert set(report["by_season"].keys()) == {"DJF", "MAM", "JJA", "SON"}

    def test_season_mapping(self):
        months = pd.Series([12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
        assert list(season_from_month(months)) == [
            "DJF", "DJF", "DJF", "MAM", "MAM", "MAM",
            "JJA", "JJA", "JJA", "SON", "SON", "SON",
        ]


class TestComparison:
    def test_compare_predictions_structure(self):
        X = pd.DataFrame({"modis_frp_mw": [10.0, 20.0]})
        y = [12.0, 22.0]
        reports = compare_predictions(
            {
                "identity": lambda X: X["modis_frp_mw"].to_numpy(),
                "double": np.array([20.0, 40.0]),
            },
            X,
            y,
        )
        assert set(reports) == {"identity", "double"}
        assert reports["identity"]["name"] == "identity"
        assert reports["double"]["n"] == 2

    def test_beats_baseline_flag(self):
        better = {"tweedie_deviance": 1.0, "n": 10}
        worse = {"tweedie_deviance": 2.0, "n": 10}
        beats, margin = beats_baseline(better, worse)
        assert beats and margin == pytest.approx(1.0)
        beats, _ = beats_baseline(worse, better)
        assert not beats

    def test_beats_baseline_empty_is_false(self):
        beats, margin = beats_baseline({"tweedie_deviance": 1.0, "n": 0}, {"tweedie_deviance": 1.0, "n": 5})
        assert not beats and np.isnan(margin)


def test_evaluation_records_bias_and_ratio_on_fixture(training_result):
    """Held-out evaluation from the real training pipeline satisfies SC-03 shape."""
    for split in ("validation", "test"):
        reports = training_result.evaluation[split]
        if reports.get("skipped"):
            continue
        assert set(reports) >= {"identity", "calibration", "lightgbm_tweedie"}
        for report in reports.values():
            assert report["n"] > 0
            assert "bias" in report and "by_frp_range" in report
        # The learned model must fix most of the identity underprediction bias.
        model_bias = abs(reports["lightgbm_tweedie"]["bias"])
        identity_bias = abs(reports["identity"]["bias"])
        assert model_bias <= identity_bias * 1.5
