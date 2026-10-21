"""Unit tests for calendar/forecast record assembly and contract validation."""

import numpy as np
import pandas as pd
import pytest

from src.calendar.records import (
    CALENDAR_FORECAST_COLUMNS,
    QUALITY_STATES,
    VALUE_TYPE_FORECAST,
    build_calendar_forecast_records,
    validate_calendar_forecast_frame,
)
from src.calendar.seasonal import SeasonalClimatology
from src.common.config import SeasonalConfig


@pytest.fixture()
def climatology(synthetic_daily):
    return SeasonalClimatology(SeasonalConfig()).fit(
        synthetic_daily["date"], synthetic_daily["value"]
    )


def make_forecast(daily):
    last_value = daily["value"].dropna().iloc[-1]
    horizon_dates = pd.date_range(daily["date"].max() + pd.Timedelta(days=1), periods=14)
    return {
        "model_name": "patchtst",
        "model_version": "1.0.0",
        "point": np.full(14, last_value),
        "quantiles": np.tile([last_value * 0.8, last_value, last_value * 1.2], (14, 1)),
        "quantile_levels": [0.1, 0.5, 0.9],
        "horizon_dates": [d.date().isoformat() for d in horizon_dates],
        "calibration_status": "experimental_uncalibrated",
    }


class TestAssembly:
    def test_contract_columns_present(self, synthetic_daily, climatology):
        records = build_calendar_forecast_records(
            synthetic_daily, climatology, forecast=None
        )
        missing = [c for c in CALENDAR_FORECAST_COLUMNS if c not in records.columns]
        assert not missing

    def test_forecast_rows_appended_after_history(self, synthetic_daily, climatology):
        forecast = make_forecast(synthetic_daily)
        records = build_calendar_forecast_records(
            synthetic_daily,
            climatology,
            forecast=forecast,
            forecast_training_cutoff="2023-12-31",
            forecast_issue_date="2024-01-01",
        )
        future = records[records["value_type"] == VALUE_TYPE_FORECAST]
        assert len(future) == 14
        assert (future["quality_state"] == "forecast").all()
        assert (future["forecast_model"] == "patchtst").all()
        assert (future["forecast_training_cutoff"] == "2023-12-31").all()
        assert (future["date"] > synthetic_daily["date"].max()).all()
        # History keeps its own labels.
        history = records[records["value_type"] != VALUE_TYPE_FORECAST]
        assert history["date"].max() == synthetic_daily["date"].max()

    def test_no_forecast_is_an_explicit_state(self, synthetic_daily, climatology):
        records = build_calendar_forecast_records(synthetic_daily, climatology, forecast=None)
        assert records["forecast_model"].isna().all()
        validate_calendar_forecast_frame(records)  # contract holds without forecast

    def test_quality_states_follow_value_types(self, synthetic_daily, climatology):
        daily = synthetic_daily.copy()
        daily["value_type"] = "observed_viirs"
        forecast = make_forecast(synthetic_daily)
        records = build_calendar_forecast_records(daily, climatology, forecast=forecast)
        assert set(records["quality_state"]) <= QUALITY_STATES
        mapping = dict(
            observed_viirs="observed",
            harmonized_estimate="harmonized",
            missing_coverage="missing",
            forecast="forecast",
        )
        for value_type, quality in mapping.items():
            mask = records["value_type"] == value_type
            if mask.any():
                assert (records.loc[mask, "quality_state"] == quality).all()


class TestValidation:
    def test_missing_columns_rejected(self, synthetic_daily, climatology):
        records = build_calendar_forecast_records(synthetic_daily, climatology)
        broken = records.drop(columns=["seasonal_baseline"])
        with pytest.raises(Exception, match="missing columns"):
            validate_calendar_forecast_frame(broken)

    def test_invalid_quality_state_rejected(self, synthetic_daily, climatology):
        records = build_calendar_forecast_records(synthetic_daily, climatology)
        records.loc[0, "quality_state"] = "maybe"
        with pytest.raises(Exception, match="invalid quality states"):
            validate_calendar_forecast_frame(records)

    def test_forecast_rows_require_model_name(self, synthetic_daily, climatology):
        forecast = make_forecast(synthetic_daily)
        forecast["model_name"] = None
        records = build_calendar_forecast_records(
            synthetic_daily, climatology, forecast=forecast
        )
        # Assembly still assigns the label; simulate a corrupted frame instead.
        records.loc[records.index[-1], "forecast_model"] = None
        with pytest.raises(Exception, match="record the forecast model"):
            validate_calendar_forecast_frame(records)

    def test_sorted_by_date(self, synthetic_daily, climatology):
        records = build_calendar_forecast_records(
            synthetic_daily, climatology, forecast=make_forecast(synthetic_daily)
        )
        dates = pd.to_datetime(records["date"])
        assert dates.is_monotonic_increasing
