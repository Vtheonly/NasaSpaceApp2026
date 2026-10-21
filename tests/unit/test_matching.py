"""Unit tests for overlap-period sensor matching (docs/ARCHITECTURE.md Pillar 1).

Every matching rule is pinned: temporal tolerance, footprint box, multi-VIIRS
summation, duplicate policy, confidence filter, and the missing-coverage
ineligibility rule.
"""

import pandas as pd
import pytest

from src.common.config import MatchingConfig
from src.geo.matching import (
    REASON_NO_VIIRS_COVERAGE,
    TARGET_COLUMN,
    build_matched_records,
    eligible_matches,
)


def modis_row(ts, lat=40.0, lon=-121.0, frp=100.0, scan=0.02, track=0.02, **extra):
    row = {
        "latitude": lat,
        "longitude": lon,
        "acq_datetime": pd.Timestamp(ts),
        "frp": frp,
        "scan": scan,
        "track": track,
        "quarantine_reason": None,
        "source_file": "modis_test.csv",
    }
    row.update(extra)
    return row


def viirs_row(ts, lat=40.0, lon=-121.0, frp=50.0, **extra):
    row = {
        "latitude": lat,
        "longitude": lon,
        "acq_datetime": pd.Timestamp(ts),
        "frp": frp,
        "quarantine_reason": None,
        "source_file": "viirs_test.csv",
    }
    row.update(extra)
    return row


class TestTemporalTolerance:
    def test_within_tolerance_matches(self):
        modis = pd.DataFrame([modis_row("2022-06-01 13:30", frp=100.0)])
        viirs = pd.DataFrame([viirs_row("2022-06-01 14:45", frp=60.0)])  # 75 min later
        matched = build_matched_records(modis, viirs, MatchingConfig(temporal_tolerance_minutes=90))
        assert len(matched) == 1
        assert matched[TARGET_COLUMN].iloc[0] == pytest.approx(60.0)
        assert matched["viirs_matched_count"].iloc[0] == 1

    def test_beyond_tolerance_does_not_match_and_is_ineligible(self):
        modis = pd.DataFrame([modis_row("2022-06-01 13:30", frp=100.0)])
        viirs = pd.DataFrame([viirs_row("2022-06-01 16:00", frp=60.0)])  # 150 min later
        matched = build_matched_records(modis, viirs, MatchingConfig(temporal_tolerance_minutes=90))
        # No VIIRS observation within the temporal tolerance window: the match
        # is ineligible and the target is NaN (never a zero).
        assert not bool(matched["eligible"].iloc[0])
        assert matched["ineligibility_reason"].iloc[0] == REASON_NO_VIIRS_COVERAGE
        assert pd.isna(matched[TARGET_COLUMN].iloc[0])

    def test_boundary_is_inclusive(self):
        modis = pd.DataFrame([modis_row("2022-06-01 13:30", frp=10.0)])
        viirs = pd.DataFrame([viirs_row("2022-06-01 15:00", frp=5.0)])  # exactly 90 min
        matched = build_matched_records(modis, viirs, MatchingConfig(temporal_tolerance_minutes=90))
        assert matched["viirs_matched_count"].iloc[0] == 1

    def test_near_midnight_crosses_date_buckets(self):
        modis = pd.DataFrame([modis_row("2022-06-02 00:10", frp=10.0)])
        viirs = pd.DataFrame([viirs_row("2022-06-01 23:50", frp=4.0)])  # 20 min earlier, prior day
        matched = build_matched_records(modis, viirs, MatchingConfig(temporal_tolerance_minutes=90))
        assert matched["viirs_matched_count"].iloc[0] == 1


class TestSpatialFootprint:
    def test_multiple_viirs_inside_footprint_are_summed(self):
        modis = pd.DataFrame([modis_row("2022-06-01 13:30", frp=100.0, scan=0.2, track=0.2)])
        viirs = pd.DataFrame(
            [
                viirs_row("2022-06-01 13:35", lat=40.03, lon=-121.03, frp=10.0),
                viirs_row("2022-06-01 13:36", lat=39.99, lon=-120.99, frp=25.0),
                viirs_row("2022-06-01 13:37", lat=40.5, lon=-121.5, frp=999.0),  # far away
            ]
        )
        matched = build_matched_records(modis, viirs, MatchingConfig())
        assert matched["viirs_matched_count"].iloc[0] == 2
        assert matched[TARGET_COLUMN].iloc[0] == pytest.approx(35.0)

    def test_default_footprint_when_scan_track_missing(self):
        modis = pd.DataFrame([modis_row("2022-06-01 13:30", frp=100.0)])
        modis = modis.drop(columns=["scan", "track"])
        viirs = pd.DataFrame([viirs_row("2022-06-01 13:33", lat=40.005, lon=-121.005, frp=7.0)])
        matched = build_matched_records(modis, viirs, MatchingConfig())
        assert matched["viirs_matched_count"].iloc[0] == 1
        assert matched["footprint_radius_deg"].iloc[0] == pytest.approx(0.02)

    def test_footprint_radius_clipped_to_bounds(self):
        modis = pd.DataFrame([modis_row("2022-06-01 13:30", frp=100.0, scan=2.0, track=2.0)])
        viirs = pd.DataFrame([viirs_row("2022-06-01 13:33", lat=40.1, lon=-121.1, frp=5.0)])
        matched = build_matched_records(modis, viirs, MatchingConfig())
        assert matched["footprint_radius_deg"].iloc[0] <= MatchingConfig().max_footprint_radius_deg


class TestCoverageEligibility:
    def test_no_viirs_coverage_day_is_ineligible_with_nan_target(self):
        modis = pd.DataFrame([modis_row("2022-06-01 13:30", frp=100.0)])
        viirs = pd.DataFrame(
            [viirs_row("2022-05-30 13:30", frp=50.0)]  # different day entirely
        )
        matched = build_matched_records(modis, viirs, MatchingConfig(temporal_tolerance_minutes=90))
        assert not bool(matched["eligible"].iloc[0])
        assert matched["ineligibility_reason"].iloc[0] == REASON_NO_VIIRS_COVERAGE
        # Missing coverage is NEVER a zero target (docs rule).
        assert pd.isna(matched[TARGET_COLUMN].iloc[0])

    def test_eligible_matches_filters_ineligible(self):
        modis = pd.DataFrame(
            [
                modis_row("2022-06-01 13:30", frp=100.0),
                modis_row("2022-06-02 13:30", frp=80.0),
            ]
        )
        viirs = pd.DataFrame([viirs_row("2022-06-01 13:35", frp=40.0)])
        matched = build_matched_records(modis, viirs, MatchingConfig())
        elig = eligible_matches(matched)
        assert len(matched) == 2 and len(elig) == 1
        assert elig["viirs_matched_count"].iloc[0] == 1


class TestPolicies:
    def test_confidence_filter_excludes_low_confidence(self):
        modis = pd.DataFrame(
            [
                modis_row("2022-06-01 13:30", frp=100.0, confidence_ordinal=0),
                modis_row("2022-06-01 13:40", frp=90.0, confidence_ordinal=2),
            ]
        )
        viirs = pd.DataFrame(
            [
                viirs_row("2022-06-01 13:35", frp=40.0, confidence_ordinal=0),
                viirs_row("2022-06-01 13:45", frp=30.0, confidence_ordinal=2),
            ]
        )
        matched = build_matched_records(
            modis, viirs, MatchingConfig(min_confidence="n")
        )
        # Only the high-confidence MODIS detection remains (low-confidence dropped).
        assert len(matched) == 1
        assert matched["viirs_matched_count"].iloc[0] == 1  # matches only the 'n' VIIRS row

    def test_duplicate_modis_rows_deduplicated(self):
        modis = pd.DataFrame(
            [
                modis_row("2022-06-01 13:30", frp=100.0),
                modis_row("2022-06-01 13:30", frp=100.0),  # exact duplicate
            ]
        )
        viirs = pd.DataFrame([viirs_row("2022-06-01 13:35", frp=40.0)])
        matched = build_matched_records(modis, viirs, MatchingConfig())
        assert len(matched) == 1

    def test_match_ids_are_deterministic_and_ordered(self):
        modis = pd.DataFrame(
            [
                modis_row("2022-06-02 13:30", lat=40.5, frp=10.0),
                modis_row("2022-06-01 13:30", lat=41.0, frp=20.0),
            ]
        )
        viirs = pd.DataFrame([viirs_row("2022-06-01 13:35", frp=5.0)])
        first = build_matched_records(modis, viirs, MatchingConfig())
        second = build_matched_records(modis.copy(), viirs.copy(), MatchingConfig())
        assert list(first["match_id"]) == list(second["match_id"])
        assert list(first["match_date"]) == sorted(first["match_date"])

    def test_invalid_min_confidence_rejected(self):
        modis = pd.DataFrame(
            [modis_row("2022-06-01 13:30", frp=10.0, confidence_ordinal=1)]
        )
        viirs = pd.DataFrame(
            [viirs_row("2022-06-01 13:35", frp=5.0, confidence_ordinal=1)]
        )
        with pytest.raises(ValueError, match="min_confidence"):
            build_matched_records(modis, viirs, MatchingConfig(min_confidence="z"))


def test_fixture_matching_contract_columns(gate_outcome, aoi, harmonization_config):
    from src.geo.aoi import filter_records
    from src.geo.matching import MATCHED_COLUMNS

    frames = {p: filter_records(f, aoi) for p, f in gate_outcome.frames.items()}
    matched = build_matched_records(
        frames["modis"], frames["viirs"], harmonization_config.matching, aoi_id=aoi.aoi_id
    )
    missing = [c for c in MATCHED_COLUMNS if c not in matched.columns]
    assert not missing
    assert len(matched) > 1000
    # Synthetic fixture guarantees in-tolerance coverage for most detections.
    assert matched["eligible"].mean() > 0.8
