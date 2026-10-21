"""Integration tests: train -> save -> load -> inference roundtrips.

Exercises the complete harmonization and calendar pillars through the public
library API on the committed synthetic fixture, including artifact integrity
verification and the harmonized-series label contract.
"""

import numpy as np
import pandas as pd
import pytest

from src.artifacts.contracts import validate_harmonized_daily, validate_matched_frame
from src.artifacts.manifest import build_manifest
from src.artifacts.serialization import (
    load_harmonizer_bundle,
    save_harmonizer_bundle,
)
from src.calendar.records import build_calendar_forecast_records, validate_calendar_forecast_frame
from src.calendar.seasonal import SeasonalClimatology
from src.common.config import SeasonalConfig
from src.data.schema import PRODUCT_MODIS, PRODUCT_VIIRS
from src.geo.aoi import filter_records
from src.harmonization.inference import (
    VALUE_TYPE_HARMONIZED,
    VALUE_TYPE_MISSING,
    VALUE_TYPE_OBSERVED,
    build_harmonized_daily,
)


class TestHarmonizerRoundtrip:
    def test_save_load_predict_consistency(
        self, tmp_path, training_result, gate_outcome, aoi
    ):
        manifest = build_manifest(
            pipeline="harmonization",
            config=training_result.config_snapshot,
            source_files=[],
            model={
                "name": training_result.model.name,
                "version": training_result.model.version,
            },
            metrics=training_result.evaluation,
            code_commit="integration-test",
            limitations=["synthetic fixture evidence only"],
        )
        artifact_dir = tmp_path / "harmonizer_artifact"
        save_harmonizer_bundle(
            artifact_dir,
            {"model": training_result.model, "feature_names": training_result.feature_names},
            manifest,
        )
        loaded = load_harmonizer_bundle(artifact_dir)

        frames = {p: filter_records(f, aoi) for p, f in gate_outcome.frames.items()}
        daily = training_result.daily
        in_memory = build_harmonized_daily(
            daily, frames[PRODUCT_MODIS], frames[PRODUCT_VIIRS], training_result.model
        )
        from_disk = build_harmonized_daily(
            daily, frames[PRODUCT_MODIS], frames[PRODUCT_VIIRS], loaded["model"]
        )
        pd.testing.assert_frame_equal(in_memory, from_disk)

    def test_matched_frame_contract_on_real_pipeline(self, training_result):
        validate_matched_frame(training_result.matched)

    def test_harmonized_daily_labels_are_distinct(self, gate_outcome, training_result, aoi):
        frames = {p: filter_records(f, aoi) for p, f in gate_outcome.frames.items()}
        harmonized = build_harmonized_daily(
            training_result.daily,
            frames[PRODUCT_MODIS],
            frames[PRODUCT_VIIRS],
            training_result.model,
        )
        validate_harmonized_daily(harmonized)
        value_types = set(harmonized["value_type"])
        assert value_types <= {VALUE_TYPE_OBSERVED, VALUE_TYPE_HARMONIZED, VALUE_TYPE_MISSING}
        # The synthetic fixture guarantees all three states occur.
        assert value_types == {VALUE_TYPE_OBSERVED, VALUE_TYPE_HARMONIZED, VALUE_TYPE_MISSING}

    def test_harmonized_values_are_non_negative(self, gate_outcome, training_result, aoi):
        frames = {p: filter_records(f, aoi) for p, f in gate_outcome.frames.items()}
        harmonized = build_harmonized_daily(
            training_result.daily,
            frames[PRODUCT_MODIS],
            frames[PRODUCT_VIIRS],
            training_result.model,
        )
        assert (harmonized["value"].dropna() >= 0).all()

    def test_observed_days_prefer_native_viirs(self, gate_outcome, training_result, aoi):
        frames = {p: filter_records(f, aoi) for p, f in gate_outcome.frames.items()}
        harmonized = build_harmonized_daily(
            training_result.daily,
            frames[PRODUCT_MODIS],
            frames[PRODUCT_VIIRS],
            training_result.model,
        )
        viirs_days = harmonized[harmonized["value_type"] == VALUE_TYPE_OBSERVED]
        expected = training_result.daily[training_result.daily["viirs_coverage"]]
        assert np.allclose(
            viirs_days["value"].to_numpy(),
            expected["viirs_frp_sum_mw"].to_numpy(),
        )

    def test_gap_days_carry_missing_label_and_nan_value(
        self, gate_outcome, training_result, aoi
    ):
        frames = {p: filter_records(f, aoi) for p, f in gate_outcome.frames.items()}
        harmonized = build_harmonized_daily(
            training_result.daily,
            frames[PRODUCT_MODIS],
            frames[PRODUCT_VIIRS],
            training_result.model,
        )
        gaps = harmonized[harmonized["value_type"] == VALUE_TYPE_MISSING]
        assert len(gaps) > 0
        assert gaps["value"].isna().all()


class TestForecastRoundtrip:
    @pytest.fixture(scope="class")
    def fitted_forecaster(self):
        pytest.importorskip("torch")
        from src.models.patchtst import PatchTSTForecaster
        from tests.conftest import tiny_calendar_config

        config = tiny_calendar_config(min_train_days=120)
        daily = make_forecast_daily()
        forecaster = PatchTSTForecaster(config.forecast, seed=config.seed).fit(daily)
        return forecaster, daily

    def test_state_dict_json_serializable_and_restorable(self, fitted_forecaster, tmp_path):
        import json

        from src.models.patchtst import PatchTSTForecaster

        forecaster, daily = fitted_forecaster
        state = forecaster.state_dict()
        text = json.dumps(state, allow_nan=True)  # must be JSON-serializable
        restored = PatchTSTForecaster().load_state_dict(json.loads(text))
        assert np.allclose(
            forecaster.predict()["point"], restored.predict(daily)["point"], rtol=1e-9
        )
        _ = tmp_path

    def test_calendar_records_contract_with_real_forecast(self, fitted_forecaster):
        from src.calendar.records import CALENDAR_FORECAST_COLUMNS

        forecaster, daily = fitted_forecaster
        climatology = SeasonalClimatology(SeasonalConfig()).fit(daily["date"], daily["value"])
        forecast = forecaster.predict(daily)
        records = build_calendar_forecast_records(
            daily,
            climatology,
            forecast=forecast,
            forecast_training_cutoff=str(pd.to_datetime(daily["date"]).max().date()),
            forecast_issue_date="2030-01-01",
        )
        validate_calendar_forecast_frame(records)
        missing = [c for c in CALENDAR_FORECAST_COLUMNS if c not in records.columns]
        assert not missing
        assert (records["quality_state"] == "forecast").sum() == 14

    def test_forecast_quantiles_sorted_and_clipped(self, fitted_forecaster):
        forecaster, _ = fitted_forecaster
        result = forecaster.predict()
        assert (np.diff(result["quantiles"], axis=1) >= 0).all()
        assert (result["quantiles"] >= 0).all()


def make_forecast_daily():
    from tests.conftest import make_daily_series

    daily = make_daily_series(periods=400, seed=17, gap_fraction=0.0)
    rng = np.random.default_rng(17)
    weekly = 20.0 * np.sin(2 * np.pi * np.arange(len(daily)) / 7.0)
    daily["value"] = np.clip(daily["value"].to_numpy() + weekly + rng.normal(0, 2, len(daily)), 0, None)
    return daily
