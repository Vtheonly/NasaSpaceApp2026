"""Unit tests for harmonizer feature construction (leakage-free, availability-aware)."""

import numpy as np
import pandas as pd
import pytest

from src.harmonization.features import (
    BASE_FEATURES,
    assert_no_target_leakage,
    build_feature_matrix,
)


def base_records(n=8):
    dates = pd.date_range("2022-06-01 13:30", periods=n, freq="12h")
    return pd.DataFrame(
        {
            "modis_frp_mw": [10.0, 20.0, 5.0, 0.5, 80.0, 33.0, 7.0, 12.0][:n],
            "modis_acq_datetime": dates,
            "footprint_radius_deg": [0.02] * n,
        }
    )


class TestLeakageGuard:
    def test_no_viirs_or_target_features(self):
        X, names = build_feature_matrix(base_records())
        for name in names:
            assert "viirs" not in name.lower()
            assert "target" not in name.lower()

    def test_assert_no_target_leakage_raises(self):
        with pytest.raises(ValueError, match="forbidden"):
            assert_no_target_leakage(["viirs_frp_sum", "modis_frp_mw"])
        with pytest.raises(ValueError, match="forbidden"):
            assert_no_target_leakage(["target_amount"])


class TestBaseFeatures:
    def test_all_base_features_present(self):
        X, names = build_feature_matrix(base_records())
        for feature in BASE_FEATURES:
            assert feature in names

    def test_doy_encoding_values(self):
        records = base_records(n=2)
        records["modis_acq_datetime"] = pd.to_datetime(["2022-01-01 13:30", "2022-01-01 13:30"])
        X, _ = build_feature_matrix(records)
        expected_sin = np.sin(2 * np.pi * (1 - 1) / 365.0)
        expected_cos = np.cos(0.0)
        assert X["doy_sin"].iloc[0] == pytest.approx(expected_sin)
        assert X["doy_cos"].iloc[0] == pytest.approx(expected_cos)

    def test_log_frp(self):
        X, _ = build_feature_matrix(base_records(1))
        assert X["log_modis_frp"].iloc[0] == pytest.approx(np.log1p(10.0))

    def test_density_counts_same_day_detections(self):
        records = base_records(4)
        records["modis_acq_datetime"] = pd.to_datetime(
            [
                "2022-06-01 13:30",
                "2022-06-01 13:40",
                "2022-06-01 13:50",
                "2022-06-02 13:30",
            ]
        )
        X, _ = build_feature_matrix(records)
        assert X["modis_daily_active_pixels"].tolist()[:3] == [3.0, 3.0, 3.0]
        assert X["modis_daily_active_pixels"].iloc[3] == 1.0


class TestOptionalFeatures:
    def test_optional_columns_used_when_present(self):
        records = base_records(2)
        records["modis_brightness"] = [320.0, 330.0]
        records["modis_scan_deg"] = [0.02, 0.03]
        records["modis_track_deg"] = [0.02, 0.02]
        records["modis_daynight"] = ["D", "N"]
        X, names = build_feature_matrix(records)
        for feature in ("modis_brightness", "modis_scan_deg", "modis_track_deg", "is_night"):
            assert feature in names
        assert X["is_night"].tolist() == [0.0, 1.0]

    def test_absent_optional_columns_dropped(self):
        X, names = build_feature_matrix(base_records(2))
        for feature in ("modis_brightness", "modis_scan_deg", "is_night"):
            assert feature not in names

    def test_row_level_nan_preserved_for_lightgbm(self):
        records = base_records(2)
        records["modis_brightness"] = [320.0, np.nan]
        X, _ = build_feature_matrix(records)
        assert pd.isna(X["modis_brightness"].iloc[1])


def test_empty_records_rejected():
    with pytest.raises(ValueError, match="empty record set"):
        build_feature_matrix(pd.DataFrame(columns=["modis_frp_mw", "modis_acq_datetime", "footprint_radius_deg"]))


def test_inf_values_converted_to_nan():
    records = base_records(2)
    records.loc[0, "modis_frp_mw"] = np.inf
    X, _ = build_feature_matrix(records)
    assert np.isinf(X.to_numpy(dtype=float)).sum() == 0
