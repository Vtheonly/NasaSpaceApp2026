"""Unit tests for AOI validation and filtering (challenge requirement R-01)."""

import pandas as pd
import pytest

from src.geo.aoi import (
    AOIValidationError,
    BBoxAOI,
    PolygonAOI,
    filter_records,
    parse_aoi,
)


class TestBBoxValidation:
    def test_valid_bbox(self):
        box = BBoxAOI(-123.5, 39.5, -120.5, 42.0)
        assert box.contains(-122.0, 40.0)
        assert not box.contains(-100.0, 40.0)

    def test_lon_out_of_range(self):
        with pytest.raises(AOIValidationError, match="outside the valid range"):
            BBoxAOI(-200.0, 39.5, -120.5, 42.0)

    def test_lat_out_of_range(self):
        with pytest.raises(AOIValidationError, match="outside the valid range"):
            BBoxAOI(-123.5, -95.0, -120.5, 42.0)

    def test_min_greater_than_max(self):
        with pytest.raises(AOIValidationError, match="min coordinates must not exceed"):
            BBoxAOI(-120.5, 42.0, -123.5, 39.5)

    def test_nan_coordinate(self):
        with pytest.raises(AOIValidationError):
            BBoxAOI(float("nan"), 39.5, -120.5, 42.0)

    def test_aoi_id_is_stable_and_identifies_geometry(self):
        box = BBoxAOI(-123.5, 39.5, -120.5, 42.0)
        assert box.aoi_id == BBoxAOI(-123.5, 39.5, -120.5, 42.0).aoi_id
        assert box.aoi_id.startswith("bbox_")


class TestPolygon:
    def test_concave_polygon_contains(self):
        # "U" shape: concave polygon.
        poly = PolygonAOI(((0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (6.0, 10.0), (6.0, 4.0), (4.0, 4.0), (4.0, 10.0), (0.0, 10.0)))
        assert poly.contains(2.0, 8.0)   # left arm
        assert poly.contains(8.0, 8.0)   # right arm
        assert not poly.contains(5.0, 8.0)  # inside the notch

    def test_too_few_vertices(self):
        with pytest.raises(AOIValidationError, match="at least 3"):
            PolygonAOI(((0.0, 0.0), (1.0, 1.0)))

    def test_out_of_range_vertex(self):
        with pytest.raises(AOIValidationError, match="outside valid coordinate"):
            PolygonAOI(((0.0, 0.0), (1.0, 1.0), (200.0, 1.0)))


class TestParseAOI:
    def test_bbox_spec(self):
        aoi = parse_aoi("bbox:-123.5,39.5,-120.5,42.0")
        assert isinstance(aoi, BBoxAOI)
        assert aoi.min_lon == pytest.approx(-123.5)

    def test_polygon_spec(self):
        aoi = parse_aoi("polygon:0,0,1,0,1,1")
        assert isinstance(aoi, PolygonAOI)

    def test_missing_kind(self):
        with pytest.raises(AOIValidationError, match="must look like"):
            parse_aoi("-123.5,39.5,-120.5,42.0")

    def test_unknown_kind(self):
        with pytest.raises(AOIValidationError, match="Unknown AOI type"):
            parse_aoi("circle:0,0,1")

    def test_wrong_bbox_arity(self):
        with pytest.raises(AOIValidationError, match="exactly 4"):
            parse_aoi("bbox:1,2,3")

    def test_non_numeric(self):
        with pytest.raises(AOIValidationError, match="numeric"):
            parse_aoi("bbox:a,b,c,d")


class TestFilterRecords:
    @pytest.fixture()
    def frame(self):
        return pd.DataFrame(
            {
                "latitude": [40.0, 50.0, 39.4],
                "longitude": [-122.0, -121.0, -100.0],
                "frp": [1.0, 2.0, 3.0],
            }
        )

    def test_bbox_filter_keeps_inside_rows(self, frame):
        box = BBoxAOI(-123.5, 39.5, -120.5, 42.0)
        out = filter_records(frame, box)
        assert len(out) == 1
        assert out["latitude"].iloc[0] == 40.0

    def test_filter_does_not_mutate_input(self, frame):
        before = frame.copy(deep=True)
        filter_records(frame, BBoxAOI(-123.5, 39.5, -120.5, 42.0))
        pd.testing.assert_frame_equal(frame, before)

    def test_empty_result_is_explicit_not_error(self, frame):
        out = filter_records(frame, BBoxAOI(10.0, 10.0, 11.0, 11.0))
        assert len(out) == 0  # callers raise explicit EmptyAOIError downstream

    def test_polygon_filter_containment(self, frame):
        poly = PolygonAOI(((-123.0, 39.0), (-121.0, 39.0), (-121.0, 41.0), (-123.0, 41.0)))
        out = filter_records(frame, poly)
        assert len(out) == 1

    def test_empty_frame_passthrough(self):
        empty = pd.DataFrame(columns=["latitude", "longitude", "frp"])
        out = filter_records(empty, BBoxAOI(-123.5, 39.5, -120.5, 42.0))
        assert out.empty
