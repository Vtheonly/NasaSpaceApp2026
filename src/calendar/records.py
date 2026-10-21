"""Calendar + forecast record assembly (docs/ARCHITECTURE.md artifact contract).

"Calendar/forecast records: date, observed/harmonized values, seasonal
baseline, anomaly score, forecast quantiles, model version, training cutoff,
quality state." Observed, harmonized, and forecast values are labeled
distinctly; future days only exist when a forecast is available.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.calendar.seasonal import SeasonalClimatology
from src.common.config import AnomalyConfig
from src.data.aggregation import DATE_COLUMN

CALENDAR_FORECAST_COLUMNS: tuple[str, ...] = (
    DATE_COLUMN,
    "value",
    "value_type",
    "sensor_era",
    "quality_state",
    "seasonal_baseline",
    "anomaly_ratio",
    "anomaly_status",
    "forecast_q10",
    "forecast_median",
    "forecast_q90",
    "forecast_model",
    "forecast_model_version",
    "forecast_training_cutoff",
    "forecast_issue_date",
    "forecast_calibration_status",
)

QUALITY_STATES: frozenset[str] = frozenset(
    {"observed", "harmonized", "forecast", "missing"}
)

VALUE_TYPE_FORECAST = "forecast"


def build_calendar_forecast_records(
    harmonized_daily: pd.DataFrame,
    climatology: SeasonalClimatology,
    anomaly_config: AnomalyConfig | None = None,
    forecast: dict[str, Any] | None = None,
    forecast_training_cutoff: str | None = None,
    forecast_issue_date: str | None = None,
) -> pd.DataFrame:
    """Assemble the calendar/forecast record contract frame.

    ``forecast`` is a forecaster ``predict()`` result (point + quantiles + model
    provenance). When it is None the forecast columns stay NaN and the frame
    still satisfies the contract (explicit "no forecast available" state).
    """
    from src.calendar.anomalies import score_anomalies

    anomaly = score_anomalies(harmonized_daily, climatology, anomaly_config)
    out = anomaly.merge(
        harmonized_daily[[DATE_COLUMN, "value", "value_type", "sensor_era"]],
        on=DATE_COLUMN,
        how="left",
        suffixes=("", "_daily"),
    )
    out = out.rename(
        columns={
            "baseline": "seasonal_baseline",
            "ratio": "anomaly_ratio",
            "status": "anomaly_status",
        }
    )

    out["forecast_q10"] = np.nan
    out["forecast_median"] = np.nan
    out["forecast_q90"] = np.nan
    out["forecast_model"] = None
    out["forecast_model_version"] = None
    out["forecast_training_cutoff"] = None
    out["forecast_issue_date"] = None
    out["forecast_calibration_status"] = None

    if forecast is not None:
        quantile_levels = list(forecast.get("quantile_levels", []))
        quantiles = np.asarray(forecast.get("quantiles"), dtype=float)
        horizon_dates = list(forecast.get("horizon_dates", []))
        if len(horizon_dates) != len(quantiles):
            raise ValueError("forecast horizon_dates and quantiles length mismatch")
        q_low = min(quantile_levels) if quantile_levels else None
        q_high = max(quantile_levels) if quantile_levels else None
        for i, date in enumerate(horizon_dates):
            row_mask = out[DATE_COLUMN] == date
            out.loc[row_mask, "forecast_median"] = forecast["point"][i]
            if q_low in quantile_levels:
                out.loc[row_mask, "forecast_q10"] = quantiles[i][
                    quantile_levels.index(q_low)
                ]
            if 0.5 in quantile_levels:
                out.loc[row_mask, "forecast_median"] = quantiles[i][
                    quantile_levels.index(0.5)
                ]
            if q_high in quantile_levels:
                out.loc[row_mask, "forecast_q90"] = quantiles[i][
                    quantile_levels.index(q_high)
                ]
            out.loc[row_mask, "forecast_model"] = forecast.get("model_name")
            out.loc[row_mask, "forecast_model_version"] = forecast.get("model_version")
            out.loc[row_mask, "forecast_training_cutoff"] = forecast_training_cutoff
            out.loc[row_mask, "forecast_issue_date"] = forecast_issue_date
            out.loc[row_mask, "forecast_calibration_status"] = forecast.get(
                "calibration_status", "experimental_uncalibrated"
            )
            # A future forecast date may lie beyond the observed range: append it.
            if not row_mask.any():
                new_row = {
                    DATE_COLUMN: pd.Timestamp(date),
                    "value": float(forecast["point"][i]),
                    "value_type": VALUE_TYPE_FORECAST,
                }
                out = pd.concat([out, pd.DataFrame([new_row])], ignore_index=True)
                last_idx = out.index[-1]
                out.loc[last_idx, "forecast_median"] = float(forecast["point"][i])
                if len(quantiles[i]) == len(quantile_levels) and q_low in quantile_levels:
                    out.loc[last_idx, "forecast_q10"] = quantiles[i][quantile_levels.index(q_low)]
                    out.loc[last_idx, "forecast_q90"] = quantiles[i][quantile_levels.index(q_high)]
                out.loc[last_idx, "forecast_model"] = forecast.get("model_name")
                out.loc[last_idx, "forecast_model_version"] = forecast.get("model_version")
                out.loc[last_idx, "forecast_training_cutoff"] = forecast_training_cutoff
                out.loc[last_idx, "forecast_issue_date"] = forecast_issue_date
                out.loc[last_idx, "forecast_calibration_status"] = forecast.get(
                    "calibration_status", "experimental_uncalibrated"
                )

    # Distinct quality states per documented vocabulary.
    def _quality_state(row) -> str:
        value_type = str(row.get("value_type", ""))
        if value_type == VALUE_TYPE_FORECAST:
            return "forecast"
        if value_type == "observed_viirs":
            return "observed"
        if value_type == "harmonized_estimate":
            return "harmonized"
        return "missing"

    out["quality_state"] = out.apply(_quality_state, axis=1)

    columns = [c for c in CALENDAR_FORECAST_COLUMNS if c in out.columns]
    out = out.sort_values(DATE_COLUMN, kind="mergesort").reset_index(drop=True)
    return out[columns]


def validate_calendar_forecast_frame(frame: pd.DataFrame) -> None:
    from src.artifacts.contracts import ContractError

    missing = [c for c in CALENDAR_FORECAST_COLUMNS if c not in frame.columns]
    if missing:
        raise ContractError(f"calendar/forecast records missing columns: {missing}")
    bad = set(frame["quality_state"].astype(str)) - QUALITY_STATES
    if bad:
        raise ContractError(f"calendar/forecast records have invalid quality states: {bad}")
    forecast_rows = frame[frame["quality_state"] == "forecast"]
    if len(forecast_rows) and forecast_rows["forecast_model"].isna().any():
        raise ContractError("forecast rows must record the forecast model name")


__all__ = [
    "CALENDAR_FORECAST_COLUMNS",
    "QUALITY_STATES",
    "VALUE_TYPE_FORECAST",
    "build_calendar_forecast_records",
    "validate_calendar_forecast_frame",
]
