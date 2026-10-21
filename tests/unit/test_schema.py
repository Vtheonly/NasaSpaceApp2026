"""Unit tests for FIRMS schema helpers (docs/TESTING.md pass-1 foundations)."""

import pandas as pd
import pytest

from src.data.schema import (
    CONFIDENCE_ORDINAL,
    REQUIRED_COLUMNS,
    confidence_ordinal,
    parse_acq_datetime,
    pseudo_day_of_year,
)


class TestParseAcqDatetime:
    def test_valid_timestamp(self):
        ts = parse_acq_datetime("2022-07-15", 1330)
        assert ts == pd.Timestamp("2022-07-15 13:30:00")

    def test_early_morning(self):
        ts = parse_acq_datetime("2020-02-29", 130)
        assert ts == pd.Timestamp("2020-02-29 01:30:00")

    def test_leap_day_parsed(self):
        assert parse_acq_datetime("2020-02-29", 0) is not None

    def test_invalid_time_hour(self):
        assert parse_acq_datetime("2022-07-15", 2500) is None

    def test_invalid_time_minute(self):
        assert parse_acq_datetime("2022-07-15", 1370) is None

    def test_invalid_date(self):
        assert parse_acq_datetime("2021-13-45", 1330) is None

    def test_non_numeric_time(self):
        assert parse_acq_datetime("2022-07-15", "ab") is None

    def test_missing_time(self):
        assert parse_acq_datetime("2022-07-15", None) is None


class TestConfidenceOrdinal:
    @pytest.mark.parametrize("label,expected", [("l", 0), ("n", 1), ("h", 2), ("L", 0), ("H", 2)])
    def test_labels(self, label, expected):
        assert confidence_ordinal(label) == expected

    def test_numeric_0_100(self):
        assert confidence_ordinal(20) == 0
        assert confidence_ordinal(50) == 1
        assert confidence_ordinal(80) == 2

    def test_invalid(self):
        assert confidence_ordinal("weird") is None
        assert confidence_ordinal(None) is None

    def test_ordinal_map_matches(self):
        assert CONFIDENCE_ORDINAL == {"l": 0, "n": 1, "h": 2}


class TestPseudoDayOfYear:
    def test_jan1(self):
        assert pseudo_day_of_year(pd.Timestamp("2023-01-01")) == 1

    def test_feb28_non_leap(self):
        assert pseudo_day_of_year(pd.Timestamp("2023-02-28")) == 59

    def test_feb29_leap_pooled_with_feb28(self):
        # Documented leap policy: Feb 29 samples pool with Feb 28.
        assert pseudo_day_of_year(pd.Timestamp("2020-02-29")) == 59

    def test_mar1_leap_shifted(self):
        # In leap years, Mar 1 (doy 61) shifts down to keep month-day alignment.
        assert pseudo_day_of_year(pd.Timestamp("2020-03-01")) == 60
        assert pseudo_day_of_year(pd.Timestamp("2023-03-01")) == 60

    def test_dec31(self):
        assert pseudo_day_of_year(pd.Timestamp("2023-12-31")) == 365
        assert pseudo_day_of_year(pd.Timestamp("2020-12-31")) == 365


def test_required_columns_are_core_firms_fields():
    for column in ("latitude", "longitude", "acq_date", "acq_time", "frp"):
        assert column in REQUIRED_COLUMNS
    assert len(REQUIRED_COLUMNS) == 5
