"""Unit tests for daily AOI aggregation (SC-02: deterministic aggregation).

Core invariants tested here are re-checked in tests/scientific for random
data: FRP conservation per day and the missing-coverage vs zero-detection
distinction (docs: "Missing observations are not the same as zero fire
activity").
"""

import numpy as np
import pandas as pd
import pytest

from src.data.aggregation import (
    DAILY_AOI_COLUMNS,
    STATUS_MISSING,
    STATUS_OBSERVED,
    EmptyAOIError,
    best_observed_value,
    build_daily_frame,
    summarize_daily,
)
from tests.conftest import SAMPLE_DIR


def make_frame(dates, lats, lons, times, frp, product="modis"):
    return pd.DataFrame(
        {
            "latitude": lats,
            "longitude": lons,
            "acq_date": [d.date().isoformat() for d in dates],
            "acq_date_raw": [d.date().isoformat() for d in dates],
            "acq_time": times,
            "frp": frp,
            "acq_datetime": dates,
            "source_product": product,
            "source_file": "test.csv",
            "quarantine_reason": [None] * len(dates),
        }
    )


class TestConservationAndCounts:
    def test_daily_frp_sum_conserves_raw_total(self):
        dates = pd.to_datetime(
            ["2022-06-01 13:30", "2022-06-01 13:40", "2022-06-02 13:30", "2022-06-02 13:50"]
        )
        frame = make_frame(
            dates,
            [40.0] * 4,
            [-121.0] * 4,
            [1330, 1340, 1330, 1350],
            [10.0, 20.0, 30.0, 40.0],
        )
        daily = build_daily_frame({"modis": frame}, include_missing_days=True)
        modis_rows = daily[daily["modis_coverage"]]
        assert modis_rows["modis_frp_sum_mw"].sum() == pytest.approx(100.0)
        assert modis_rows["modis_active_pixels"].sum() == 4

    def test_per_day_counts(self):
        dates = pd.to_datetime(["2022-06-01 13:30", "2022-06-01 13:40", "2022-06-03 13:30"])
        frame = make_frame(dates, [40.0] * 3, [-121.0] * 3, [1330, 1340, 1330], [10.0, 20.0, 5.0])
        daily = build_daily_frame({"modis": frame})
        by_day = daily.set_index("date")
        assert by_day.loc["2022-06-01", "modis_active_pixels"] == 2
        assert by_day.loc["2022-06-03", "modis_active_pixels"] == 1

    def test_missing_days_are_explicit_not_zero_detections(self):
        dates = pd.to_datetime(["2022-06-01 13:30", "2022-06-05 13:30"])
        frame = make_frame(dates, [40.0] * 2, [-121.0] * 2, [1330, 1330], [10.0, 20.0])
        daily = build_daily_frame({"modis": frame})
        assert len(daily) == 5  # full calendar span reindexed
        gap = daily[(daily["date"] >= "2022-06-02") & (daily["date"] <= "2022-06-04")]
        assert (gap["coverage_status"] == STATUS_MISSING).all()
        assert (gap["modis_coverage"] == False).all()  # noqa: E712
        assert (gap["modis_active_pixels"] == 0).all()


class TestDeterminism:
    def test_same_input_same_output(self, daily_frame):
        from src.data.adapters import read_product_frames
        from src.geo.aoi import BBoxAOI, filter_records

        aoi = BBoxAOI(-123.5, 39.5, -120.5, 42.0)
        gate_frames = {
            product: filter_records(read_product_frames(str(SAMPLE_DIR), product), aoi)
            for product in ("modis", "viirs")
        }
        rebuilt = build_daily_frame(gate_frames, aoi_id=aoi.aoi_id)
        pd.testing.assert_frame_equal(rebuilt, daily_frame)

    def test_aggregation_is_pure(self, daily_frame):
        before = daily_frame.copy(deep=True)
        summarize_daily(daily_frame)
        pd.testing.assert_frame_equal(daily_frame, before)


class TestBestObservedValue:
    def test_viirs_preferred_over_modis(self):
        dates = pd.to_datetime(["2022-06-01 13:30"])
        modis = make_frame(dates, [40.0], [-121.0], [1330], [10.0], product="modis")
        viirs = make_frame(dates, [40.0], [-121.0], [1330], [99.0], product="viirs")
        daily = build_daily_frame({"modis": modis, "viirs": viirs})
        values = best_observed_value(daily)
        assert values.iloc[0] == pytest.approx(99.0)


class TestEmptyAOI:
    def test_empty_aoi_raises_clear_error(self):
        dates = pd.to_datetime(["2022-06-01 13:30"])
        frame = make_frame(dates, [40.0], [-121.0], [1330], [10.0])
        with pytest.raises(EmptyAOIError, match="No validated observations remain"):
            build_daily_frame(
                {"modis": frame},
                aoi_filter=lambda df: df.iloc[0:0],
            )


class TestSummary:
    def test_summary_counts(self, daily_frame):
        summary = summarize_daily(daily_frame)
        assert summary["days_total"] == len(daily_frame)
        assert (
            summary["days_observed"]
            + summary["days_missing_coverage"]
            == summary["days_total"]
        )
        assert summary["days_missing_coverage"] > 0  # fixture has coverage gaps


def test_daily_contract_columns_present(daily_frame):
    missing = [c for c in DAILY_AOI_COLUMNS if c not in daily_frame.columns]
    assert not missing
    assert set(daily_frame["coverage_status"]) <= {STATUS_OBSERVED, STATUS_MISSING}


def test_leap_day_exists_in_range():
    dates = pd.to_datetime(["2020-02-28 13:30", "2020-02-29 13:30", "2020-03-01 13:30"])
    frame = make_frame(dates, [40.0] * 3, [-121.0] * 3, [1330] * 3, [1.0, 2.0, 3.0])
    daily = build_daily_frame({"modis": frame})
    assert list(daily["date"].dt.strftime("%Y-%m-%d")) == [
        "2020-02-28",
        "2020-02-29",
        "2020-03-01",
    ]
    assert np.isclose(daily["modis_frp_sum_mw"].iloc[1], 2.0)
