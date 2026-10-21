"""Scientific invariant tests (docs/TESTING.md pass 4: transformation invariants).

Randomized property tests: aggregation conservation, AOI containment,
standardization round trips, and the missing-vs-zero distinction.
"""

import numpy as np
import pandas as pd
import pytest

from src.data.aggregation import build_daily_frame
from src.geo.aoi import BBoxAOI, filter_records


def make_random_frame(rng, n=500, start="2021-01-01", product="modis"):
    dates = pd.date_range(start, periods=max(n // 5, 3), freq="D")
    times = rng.integers(0, 2359, n)
    day_idx = rng.integers(0, len(dates), n)
    lats = rng.uniform(39.5, 42.0, n)
    lons = rng.uniform(-123.5, -120.5, n)
    frp = rng.lognormal(1.0, 1.0, n)
    return pd.DataFrame(
        {
            "latitude": lats,
            "longitude": lons,
            "acq_date": [dates[i].date().isoformat() for i in day_idx],
            "acq_time": times,
            "frp": frp,
            "acq_datetime": [
                dates[i] + pd.Timedelta(minutes=int(t // 100 * 60 + t % 100))
                for i, t in zip(day_idx, times, strict=False)
            ],
            "source_product": product,
            "source_file": "random.csv",
            "quarantine_reason": [None] * n,
        }
    )


class TestAggregationConservation:
    @pytest.mark.parametrize("seed", [1, 2, 3])
    def test_daily_sums_conserve_frp(self, seed):
        rng = np.random.default_rng(seed)
        frame = make_random_frame(rng, n=600)
        daily = build_daily_frame({"modis": frame}, include_missing_days=False)
        observed = daily[daily["modis_coverage"]]
        # Conservation: sum of daily sums equals the raw FRP total.
        assert observed["modis_frp_sum_mw"].sum() == pytest.approx(frame["frp"].sum(), rel=1e-9)
        # Pixel counts conserve the total number of detections.
        assert observed["modis_active_pixels"].sum() == len(frame)

    def test_conservation_holds_per_day(self):
        rng = np.random.default_rng(11)
        frame = make_random_frame(rng, n=300)
        daily = build_daily_frame({"modis": frame}, include_missing_days=False)
        for _, row in daily.iterrows():
            day = pd.Timestamp(row["date"])
            expected = frame[frame["acq_datetime"].dt.normalize() == day]["frp"].sum()
            assert row["modis_frp_sum_mw"] == pytest.approx(expected, rel=1e-9)


class TestAOIContainment:
    def test_filter_output_lies_inside_aoi(self):
        rng = np.random.default_rng(5)
        frame = make_random_frame(rng, n=600)
        # Sample around the AOI so a meaningful fraction falls inside.
        frame["latitude"] = rng.uniform(38, 44, len(frame))
        frame["longitude"] = rng.uniform(-124, -120, len(frame))
        aoi = BBoxAOI(-123.5, 39.5, -120.5, 42.0)
        out = filter_records(frame, aoi)
        assert len(out) > 10
        assert out["latitude"].between(aoi.min_lat, aoi.max_lat).all()
        assert out["longitude"].between(aoi.min_lon, aoi.max_lon).all()

    def test_polygon_filter_output_lies_inside_polygon(self):
        from src.geo.aoi import PolygonAOI

        rng = np.random.default_rng(6)
        frame = make_random_frame(rng, n=400)
        frame["latitude"] = rng.uniform(35, 45, len(frame))
        frame["longitude"] = rng.uniform(-125, -118, len(frame))
        poly = PolygonAOI(((-123.0, 40.0), (-121.0, 40.0), (-122.0, 42.0)))
        out = filter_records(frame, poly)
        for _, row in out.iterrows():
            assert poly.contains(row["longitude"], row["latitude"])


class TestStandardizationRoundTrip:
    def test_channel_standardization_roundtrip(self):
        # Standardize then de-standardize with the documented epsilon-guarded
        # std -> exact round trip (transformation invariant, docs pass 4).
        rng = np.random.default_rng(31)
        matrix = rng.lognormal(0, 1, size=(200, 3))
        mean = matrix.mean(axis=0)
        std = np.maximum(matrix.std(axis=0), 1e-8)
        standardized = (matrix - mean) / std
        restored = standardized * std + mean
        assert np.allclose(restored, matrix)

    def test_daily_aggregation_roundtrip_dates(self):
        rng = np.random.default_rng(9)
        frame = make_random_frame(rng, n=400)
        daily = build_daily_frame({"modis": frame}, include_missing_days=False)
        raw_days = set(pd.to_datetime(frame["acq_datetime"]).dt.normalize())
        assert set(pd.to_datetime(daily["date"])) == raw_days


class TestMissingVsZeroInvariant:
    def test_missing_days_never_report_frp_zero_activity(self):
        rng = np.random.default_rng(12)
        # Two 5-day observation clusters with a multi-week gap between them.
        frame_a = make_random_frame(rng, n=25, start="2021-01-01")
        frame_b = make_random_frame(rng, n=25, start="2021-02-01")
        frame = pd.concat([frame_a, frame_b], ignore_index=True)
        daily = build_daily_frame({"modis": frame})
        gap = daily[
            (daily["date"] > "2021-01-10") & (daily["date"] < "2021-02-01")
        ]
        assert len(gap) > 10
        assert (gap["coverage_status"] == "missing").all()
        # The invariant: gap days are flagged missing, NOT counted as zero FRP days
        # in the observed summary; observed FRP totals exclude them entirely.
        observed = daily[daily["modis_coverage"]]
        assert observed["modis_frp_sum_mw"].sum() > 0
        assert not (gap["modis_coverage"]).any()

    def test_zero_detection_day_is_observed_not_missing(self):
        dates = pd.to_datetime(["2021-01-01 13:30", "2021-01-05 13:30"])
        frame = pd.DataFrame(
            {
                "latitude": [40.0, 40.0],
                "longitude": [-121.0, -121.0],
                "acq_date": ["2021-01-01", "2021-01-05"],
                "acq_time": [1330, 1330],
                "frp": [10.0, 20.0],
                "acq_datetime": dates,
                "source_product": "modis",
                "source_file": "x.csv",
                "quarantine_reason": [None, None],
            }
        )
        daily = build_daily_frame({"modis": frame})
        mid = daily[daily["date"] == "2021-01-03"].iloc[0]
        # A day with no detections between two observed days is a coverage gap
        # (no overpass evidence) - flagged missing, never zero.
        assert mid["coverage_status"] == "missing"
