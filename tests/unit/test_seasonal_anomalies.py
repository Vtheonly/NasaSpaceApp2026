"""Unit tests for seasonal climatology and anomaly scoring (SC-04, SC-05)."""

import numpy as np
import pandas as pd
import pytest

from src.calendar.anomalies import (
    STATUS_ANOMALY_HIGH,
    STATUS_ANOMALY_LOW,
    STATUS_INSUFFICIENT_BASELINE,
    STATUS_MISSING_COVERAGE,
    STATUS_NORMAL,
    score_anomalies,
)
from src.calendar.seasonal import BASELINE_VERSION, SeasonalClimatology
from src.common.config import AnomalyConfig, SeasonalConfig


class TestSeasonalClimatology:
    def test_pure_seasonal_series_recovered(self):
        # Two+ years of an exact annual cycle keyed by PSEUDO day-of-year (the
        # documented leap policy): the climatology must return the phase value.
        from src.data.schema import pseudo_day_of_year

        dates = pd.date_range("2019-01-01", "2022-12-31", freq="D")
        pseudo_doy = np.array([pseudo_day_of_year(d) for d in dates])
        values = 10.0 + 20.0 * np.sin(2 * np.pi * (pseudo_doy - 100) / 365.0)
        clim = SeasonalClimatology(SeasonalConfig(window_days=0)).fit(pd.Series(dates), pd.Series(values))
        baselines, counts = clim.baseline(pd.Series(dates))
        # window=0 compares exact pseudo-doy buckets (2-4 samples each)
        valid = ~np.isnan(baselines)
        assert valid.mean() > 0.95
        assert np.abs(baselines[valid] - values[valid]).max() < 1e-9

    def test_min_samples_yields_nan_baseline(self):
        dates = pd.Series(pd.to_datetime(["2020-07-01", "2021-07-01"]))
        clim = SeasonalClimatology(SeasonalConfig(min_samples_per_day=3, window_days=0)).fit(
            dates, pd.Series([5.0, 7.0])
        )
        baselines, counts = clim.baseline(pd.Series(pd.to_datetime(["2022-07-01"])))
        assert np.isnan(baselines[0])
        assert counts[0] == 2  # 2020 + 2021 samples pooled (circular window includes doy 182)

    def test_circular_window_across_year_boundary(self):
        # Late-Dec / early-Jan samples must share information across the boundary.
        dates = pd.Series(pd.to_datetime(["2020-12-31", "2021-01-02", "2021-12-30", "2022-01-01"]))
        clim = SeasonalClimatology(SeasonalConfig(window_days=5, min_samples_per_day=2)).fit(
            dates, pd.Series([10.0, 10.0, 10.0, 10.0])
        )
        baselines, counts = clim.baseline(pd.Series(pd.to_datetime(["2022-12-31"])))
        assert counts[0] >= 4  # all four samples are within +/- 5 days circularly

    def test_leap_day_pooled_with_feb28(self):
        dates = pd.Series(pd.to_datetime(["2020-02-28", "2020-02-29", "2021-02-28", "2022-02-28"]))
        clim = SeasonalClimatology(SeasonalConfig(window_days=0, min_samples_per_day=2)).fit(
            dates, pd.Series([8.0, 12.0, 8.0, 8.0])
        )
        baselines, counts = clim.baseline(pd.Series(pd.to_datetime(["2023-02-28"])))
        assert counts[0] == 4  # Feb 29 (2020) pooled into the Feb 28 bucket

    def test_baseline_before_fit_raises(self):
        clim = SeasonalClimatology()
        with pytest.raises(RuntimeError, match="before fit"):
            clim.baseline(pd.Series([pd.Timestamp("2022-01-01")]))

    def test_version_recorded(self):
        clim = SeasonalClimatology()
        assert clim.version == BASELINE_VERSION


class TestAnomalyScoring:
    @pytest.fixture()
    def scored(self, synthetic_daily):
        clim = SeasonalClimatology(SeasonalConfig()).fit(
            synthetic_daily["date"], synthetic_daily["value"]
        )
        return score_anomalies(synthetic_daily, clim, AnomalyConfig())

    def test_ordinary_days_are_normal(self, scored):
        normal = scored[scored["status"] == STATUS_NORMAL]
        assert len(normal) > 500
        ratios = normal["ratio"].dropna()
        assert ratios.between(0.5, 2.0).all()

    def test_known_spike_is_flagged_high(self, scored):
        spike_rows = scored[scored["date"] == "2022-08-15"]
        assert len(spike_rows) == 1
        assert spike_rows["status"].iloc[0] == STATUS_ANOMALY_HIGH
        assert spike_rows["ratio"].iloc[0] > 5.0

    def test_missing_coverage_days_carry_nan_ratio(self, scored):
        missing = scored[scored["status"] == STATUS_MISSING_COVERAGE]
        assert len(missing) > 0
        assert missing["ratio"].isna().all()  # never a numeric zero-activity claim
        assert missing["observed"].isna().all()

    def test_thresholds_are_respected(self, synthetic_daily):
        # Tighten thresholds: normal days should start flipping to anomaly_low/high.
        clim = SeasonalClimatology(SeasonalConfig()).fit(
            synthetic_daily["date"], synthetic_daily["value"]
        )
        strict = score_anomalies(synthetic_daily, clim, AnomalyConfig(high_threshold=1.05, low_threshold=0.95))
        assert (strict["status"] == STATUS_ANOMALY_HIGH).sum() > 0
        assert (strict["status"] == STATUS_ANOMALY_LOW).sum() > 0

    def test_insufficient_baseline_status(self, synthetic_daily):
        clim = SeasonalClimatology(SeasonalConfig(min_samples_per_day=500)).fit(
            synthetic_daily["date"], synthetic_daily["value"]
        )
        scored = score_anomalies(synthetic_daily, clim, AnomalyConfig())
        assert (scored["status"] == STATUS_INSUFFICIENT_BASELINE).sum() > 0

    def test_score_before_fit_raises(self, synthetic_daily):
        with pytest.raises(RuntimeError, match="fitted"):
            score_anomalies(synthetic_daily, SeasonalClimatology(), AnomalyConfig())

    def test_contract_columns_and_versions(self, scored):
        from src.calendar.anomalies import ANOMALY_COLUMNS

        missing = [c for c in ANOMALY_COLUMNS if c not in scored.columns]
        assert not missing
        assert (scored["baseline_version"] == BASELINE_VERSION).all()

    def test_anomaly_frame_passes_artifact_contract(self, scored):
        from src.artifacts.contracts import validate_anomaly_frame

        validate_anomaly_frame(scored)  # raises on violation


def test_zero_detection_day_with_coverage_scores_low(synthetic_daily):
    """Zero detections (with coverage) is a meaningful low-activity signal."""
    daily = synthetic_daily.copy()
    daily.loc[daily["date"] == "2022-06-10", "value"] = 0.0
    daily.loc[daily["date"] == "2022-06-10", "coverage_status"] = "observed"
    clim = SeasonalClimatology(SeasonalConfig()).fit(daily["date"], daily["value"])
    scored = score_anomalies(daily, clim, AnomalyConfig())
    row = scored[scored["date"] == "2022-06-10"].iloc[0]
    assert row["status"] in (STATUS_ANOMALY_LOW, STATUS_NORMAL)
    assert row["ratio"] < 0.1  # eps-stabilized ratio near zero
