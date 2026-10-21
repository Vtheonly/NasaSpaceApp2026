"""Scientific leakage tests (docs/TESTING.md pass 3; CHALLENGE_AND_REQUIREMENTS rules).

"No future information in training features or evaluation. Never randomly
shuffle temporal observations into train/test. Prevent matched events or
overlapping windows from leaking across partitions."
"""

import numpy as np
import pandas as pd
import pytest

from src.calendar.forecast import rolling_origin_backtest
from src.calendar.seasonal import SeasonalClimatology
from src.common.config import BacktestConfig, ForecastConfig, SeasonalConfig, SplitConfig
from src.geo.matching import build_matched_records, eligible_matches
from src.harmonization.train import SPLIT_TEST, SPLIT_TRAIN, SPLIT_VALIDATION, split_matched_records


class TestChronologicalSplits:
    def test_splits_are_disjoint_and_ordered(self, training_result):
        splits = training_result.splits
        train_dates = pd.to_datetime(splits[SPLIT_TRAIN]["match_date"])
        val_dates = pd.to_datetime(splits[SPLIT_VALIDATION]["match_date"])
        test_dates = pd.to_datetime(splits[SPLIT_TEST]["match_date"])
        config = training_result.config_snapshot["split"]
        assert train_dates.max() <= pd.Timestamp(config["train_end"])
        assert (
            val_dates.min() > pd.Timestamp(config["train_end"])
            if len(val_dates)
            else True
        )
        assert val_dates.max() <= pd.Timestamp(config["validation_end"]) if len(val_dates) else True
        assert (
            test_dates.min() > pd.Timestamp(config["validation_end"])
            if len(test_dates)
            else True
        )

    def test_no_match_row_spans_splits(self, training_result):
        """Each match carries a single timestamp, so it cannot appear twice."""
        all_splits = pd.concat(
            [
                training_result.splits[SPLIT_TRAIN],
                training_result.splits[SPLIT_VALIDATION],
                training_result.splits[SPLIT_TEST],
            ]
        )
        assert all_splits["match_id"].is_unique

    def test_split_boundaries_on_fixture_dates(self):
        matched = pd.DataFrame(
            {
                "match_id": ["m1", "m2", "m3", "m4"],
                "match_date": ["2021-12-31", "2022-01-01", "2023-12-31", "2024-01-01"],
            }
        )
        splits = split_matched_records(matched, SplitConfig())
        assert list(splits[SPLIT_TRAIN]["match_id"]) == ["m1"]
        assert list(splits[SPLIT_VALIDATION]["match_id"]) == ["m2", "m3"]
        assert list(splits[SPLIT_TEST]["match_id"]) == ["m4"]

    def test_training_features_contain_no_future_information(self, training_result):
        # Features allowed: MODIS-side + temporal context only (enforced in
        # features.py); here we additionally verify no split leakage in the
        # feature matrix: train X rows correspond to train rows by index.
        train = training_result.splits[SPLIT_TRAIN]
        val = training_result.splits[SPLIT_VALIDATION]
        assert train.index.is_unique and val.index.is_unique
        assert not (set(train.index) & set(val.index))


class TestRollingOriginLeakage:
    def test_forecast_dates_strictly_after_cutoff(self, synthetic_daily):
        captured = {}

        class SpyForecaster:
            name = "spy"
            version = "0"

            def __init__(self, config=None, **kw):
                pass

            def fit(self, daily):
                captured["fit_dates_max"] = pd.to_datetime(daily["date"]).max()
                captured["fit_frame"] = daily
                return self

            def predict(self, daily=None):
                return {
                    "model_name": "spy",
                    "model_version": "0",
                    "point": np.zeros(14),
                    "quantiles": np.zeros((14, 3)),
                    "quantile_levels": [0.1, 0.5, 0.9],
                    "horizon_dates": [
                        d.date().isoformat()
                        for d in pd.date_range(
                            captured["fit_dates_max"] + pd.Timedelta(days=1), periods=14
                        )
                    ],
                }

        records = rolling_origin_backtest(
            synthetic_daily,
            SpyForecaster,
            BacktestConfig(n_origins=3, origin_spacing_days=14, min_train_days=60),
            ForecastConfig(),
        )
        for record in records:
            cutoff = pd.Timestamp(record.training_cutoff)
            assert record.n_train_days_observed >= 60
            for horizon_date in record.horizon_dates:
                assert pd.Timestamp(horizon_date) > cutoff

    def test_training_history_excludes_all_data_after_cutoff(self, synthetic_daily):
        max_seen = {"value": None}

        class CaptureForecaster:
            name = "capture"
            version = "0"

            def __init__(self, config=None, **kw):
                pass

            def fit(self, daily):
                if max_seen["value"] is None or pd.to_datetime(daily["date"]).max() > max_seen["value"]:
                    max_seen["value"] = pd.to_datetime(daily["date"]).max()
                return self

            def predict(self, daily=None):
                return {
                    "model_name": "capture",
                    "model_version": "0",
                    "point": np.zeros(14),
                    "quantiles": np.zeros((14, 3)),
                    "quantile_levels": [0.1, 0.5, 0.9],
                    "horizon_dates": [],
                }

        rolling_origin_backtest(
            synthetic_daily,
            CaptureForecaster,
            BacktestConfig(n_origins=2, min_train_days=60),
            ForecastConfig(),
        )
        # The largest training cutoff must still be strictly before the last day.
        assert max_seen["value"] < pd.to_datetime(synthetic_daily["date"]).max()


class TestClimatologyLeakage:
    def test_baseline_uses_only_fitted_samples(self):
        # Fit on 2021 values only; a 2023 date must map to the 2021 climatology.
        dates_2021 = pd.date_range("2021-01-01", "2021-12-31", freq="D")
        values_2021 = pd.Series(50.0, index=range(len(dates_2021)))
        clim = SeasonalClimatology(SeasonalConfig(window_days=0, min_samples_per_day=1)).fit(
            pd.Series(dates_2021), values_2021
        )
        baselines, counts = clim.baseline(pd.Series(pd.to_datetime(["2023-07-15"])))
        assert baselines[0] == pytest.approx(50.0)
        assert counts[0] == 1

    def test_fit_date_range_is_recorded_via_sample_count(self):
        dates = pd.date_range("2021-01-01", periods=365, freq="D")
        clim = SeasonalClimatology(SeasonalConfig()).fit(pd.Series(dates), pd.Series(1.0, index=range(365)))
        assert clim.n_samples_ == 365
        # A day outside the fit distribution still gets the pooled-window mean,
        # never values that were not fitted.
        baselines, _ = clim.baseline(pd.Series(pd.to_datetime(["2030-03-01"])))
        assert baselines[0] == pytest.approx(1.0)


class TestMatchedEligibility:
    def test_ineligible_matches_excluded_from_training(self, gate_outcome, aoi, harmonization_config):
        from src.geo.aoi import filter_records

        frames = {p: filter_records(f, aoi) for p, f in gate_outcome.frames.items()}
        matched = build_matched_records(
            frames["modis"], frames["viirs"], harmonization_config.matching, aoi_id=aoi.aoi_id
        )
        elig = eligible_matches(matched)
        # Ineligible rows are exactly the no-coverage rows with NaN targets.
        ineligible = matched[~matched["eligible"].astype(bool)]
        assert ineligible["viirs_frp_sum_mw"].isna().all()
        assert elig["viirs_frp_sum_mw"].notna().all()
