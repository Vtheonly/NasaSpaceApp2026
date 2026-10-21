"""End-to-end test: AOI -> gate -> harmonization -> calendar -> forecast -> export.

Runs the complete documented workflow on the synthetic fixture in a single
process and verifies the final app-consumable artifact contract, the labeled
provenance, and end-to-end determinism (docs: "Test clean setup -> fixture
validation -> AOI aggregation -> artifact loading -> calendar/trend/anomaly/
forecast UI data").
"""


import pandas as pd
import pytest

from src.artifacts.contracts import (
    SCHEMA_VERSION,
    validate_daily_frame,
    validate_harmonized_daily,
)
from src.artifacts.manifest import build_manifest
from src.artifacts.serialization import save_forecaster_bundle, save_harmonizer_bundle
from src.calendar.anomalies import score_anomalies
from src.calendar.records import build_calendar_forecast_records, validate_calendar_forecast_frame
from src.calendar.seasonal import SeasonalClimatology
from src.common.config import HarmonizationConfig, resolve_code_commit, seed_everything
from src.data.aggregation import build_daily_frame
from src.data.gate import FivePassDataGate
from src.geo.aoi import filter_records, parse_aoi
from src.geo.matching import build_matched_records, eligible_matches
from src.harmonization.features import build_feature_matrix
from src.harmonization.inference import build_harmonized_daily
from src.harmonization.model import TweedieGBDTHarmonizer
from src.harmonization.train import SPLIT_TRAIN, split_matched_records
from tests.conftest import AOI_SPEC, SAMPLE_DIR, tiny_calendar_config


@pytest.fixture(scope="module")
def e2e(tmp_path_factory):
    """Run the entire workflow once and return every intermediate artifact."""
    out = tmp_path_factory.mktemp("e2e")
    config = HarmonizationConfig()
    seed_everything(config.seed)
    aoi = parse_aoi(AOI_SPEC)

    # Five-pass gate (fail closed).
    outcome = FivePassDataGate(config).run(str(SAMPLE_DIR))
    assert outcome.passed

    # AOI filter + daily aggregation.
    frames = {p: filter_records(f, aoi) for p, f in outcome.frames.items()}
    daily = build_daily_frame(frames, aoi_id=aoi.aoi_id)
    validate_daily_frame(daily)

    # Matching + features + chronological split.
    matched = eligible_matches(
        build_matched_records(frames["modis"], frames["viirs"], config.matching, aoi_id=aoi.aoi_id)
    )
    X, feature_names = build_feature_matrix(matched)
    splits = split_matched_records(matched, config.split)
    train = splits[SPLIT_TRAIN]

    # Train the harmonizer.
    model = TweedieGBDTHarmonizer(config.model, seed=config.seed).fit(
        X.loc[train.index],
        train["viirs_frp_sum_mw"].astype(float),
        stop_dates=pd.to_datetime(train["modis_acq_datetime"]),
    )

    # Versioned artifact + manifest.
    artifact_dir = out / "harmonizer_v1"
    manifest = build_manifest(
        pipeline="harmonization",
        config=config.to_dict(),
        source_files=outcome.inventory_manifest["products"]["modis"]["files"],
        model={"name": model.name, "version": model.version, "params": model.params},
        metrics={"e2e": True},
        code_commit=resolve_code_commit(),
        limitations=["synthetic fixture; e2e determinism check"],
        extra={"seed": config.seed, "aoi_id": aoi.aoi_id},
    )
    save_harmonizer_bundle(
        artifact_dir, {"model": model, "feature_names": feature_names}, manifest
    )

    # Inference: harmonized daily series.
    harmonized = build_harmonized_daily(
        daily, frames["modis"], frames["viirs"], model, config.matching
    )
    validate_harmonized_daily(harmonized)
    harmonized.to_csv(out / "harmonized_daily.csv", index=False)

    # Calendar: climatology + anomalies.
    climatology = SeasonalClimatology(tiny_calendar_config(min_train_days=120).seasonal).fit(
        harmonized["date"], harmonized["value"]
    )
    anomalies = score_anomalies(harmonized, climatology, tiny_calendar_config().anomaly)
    anomalies.to_csv(out / "anomaly_records.csv", index=False)

    # Forecast (PatchTST candidate when torch is available).
    forecast = None
    forecast_cutoff = None
    try:
        from src.models.patchtst import PatchTSTForecaster

        calendar_config = tiny_calendar_config(min_train_days=120)
        forecaster = PatchTSTForecaster(calendar_config.forecast, seed=calendar_config.seed).fit(
            harmonized
        )
        forecast = forecaster.predict(harmonized)
        forecast_cutoff = str(pd.to_datetime(harmonized["date"]).max().date())
        state = forecaster.state_dict()
        forecaster_manifest = build_manifest(
            pipeline="calendar_forecast",
            config=calendar_config.to_dict(),
            source_files=[{"path": "harmonized_daily.csv"}],
            model={
                "name": "patchtst",
                "version": forecaster.version,
                "horizon_days": 14,
                "quantiles": [0.1, 0.5, 0.9],
            },
            metrics={},
            code_commit=resolve_code_commit(),
            limitations=["experimental uncalibrated intervals"],
        )
        save_forecaster_bundle(out / "forecaster_v1", state, forecaster_manifest)
    except ImportError:
        forecast = None

    # Final app-consumable records.
    records = build_calendar_forecast_records(
        harmonized,
        climatology,
        tiny_calendar_config().anomaly,
        forecast=forecast,
        forecast_training_cutoff=forecast_cutoff,
        forecast_issue_date="2030-01-01",
    )
    records.to_csv(out / "calendar_forecast_records.csv", index=False)
    return {
        "daily": daily,
        "harmonized": harmonized,
        "anomalies": anomalies,
        "records": records,
        "forecast": forecast,
        "manifest": manifest,
        "out": out,
    }


class TestFullPipeline:
    def test_daily_aggregation_contract(self, e2e):
        validate_daily_frame(e2e["daily"])
        # The synthetic fixture's first and last ACTIVE day (deterministic, seed 2026).
        assert len(e2e["daily"]) == 2508
        assert e2e["daily"]["date"].min().strftime("%Y-%m-%d") == "2018-01-28"
        assert e2e["daily"]["date"].max().strftime("%Y-%m-%d") == "2024-12-09"

    def test_harmonized_series_contract(self, e2e):
        harmonized = e2e["harmonized"]
        value_types = set(harmonized["value_type"])
        assert value_types == {"observed_viirs", "harmonized_estimate", "missing_coverage"}
        # Observed VIIRS days dominate; MODIS-only days carry the harmonized
        # estimate (fixture reality: 449 observed / 39 harmonized / rest gaps).
        observed = (harmonized["value_type"] == "observed_viirs").sum()
        harmonized_days = (harmonized["value_type"] == "harmonized_estimate").sum()
        assert observed > 400 and harmonized_days > 0

    def test_anomaly_records(self, e2e):
        anomalies = e2e["anomalies"]
        assert anomalies["ratio"].notna().sum() > 200  # observed days carry scores
        assert "normal" in set(anomalies["status"])
        assert (anomalies["status"] == "missing_coverage").sum() > 100

    def test_forecast_available_and_contract(self, e2e):
        if e2e["forecast"] is None:
            pytest.skip("torch unavailable; forecast section is an explicit unavailable state")
        assert len(e2e["forecast"]["point"]) == 14
        forecast_rows = e2e["records"][e2e["records"]["quality_state"] == "forecast"]
        assert len(forecast_rows) == 14
        assert (forecast_rows["forecast_model"] == "patchtst").all()
        assert forecast_rows["forecast_training_cutoff"].notna().all()
        validate_calendar_forecast_frame(e2e["records"])

    def test_records_cover_history_plus_horizon(self, e2e):
        records = e2e["records"]
        history_days = len(e2e["harmonized"])
        expected = history_days + (14 if e2e["forecast"] is not None else 0)
        assert len(records) == expected
        dates = pd.to_datetime(records["date"])
        assert dates.is_monotonic_increasing

    def test_manifest_records_provenance(self, e2e):
        manifest = e2e["manifest"]
        assert manifest["schema_version"] == SCHEMA_VERSION
        assert manifest["pipeline"] == "harmonization"
        assert manifest["aoi_id"].startswith("bbox_")
        assert manifest["config"]["seed"] == HarmonizationConfig().seed
        assert len(manifest["source_files"]) >= 1
        assert all(len(f["sha256"]) == 64 for f in manifest["source_files"])

    def test_pipeline_determinism(self, e2e):
        """Re-run the harmonizer training with the same seed: identical output."""
        config = HarmonizationConfig()
        aoi = parse_aoi(AOI_SPEC)
        seed_everything(config.seed)
        outcome = FivePassDataGate(config).run(str(SAMPLE_DIR))
        frames = {p: filter_records(f, aoi) for p, f in outcome.frames.items()}
        matched = eligible_matches(
            build_matched_records(frames["modis"], frames["viirs"], config.matching, aoi_id=aoi.aoi_id)
        )
        X, _ = build_feature_matrix(matched)
        splits = split_matched_records(matched, config.split)
        train = splits[SPLIT_TRAIN]
        model = TweedieGBDTHarmonizer(config.model, seed=config.seed).fit(
            X.loc[train.index],
            train["viirs_frp_sum_mw"].astype(float),
            stop_dates=pd.to_datetime(train["modis_acq_datetime"]),
        )
        harmonized = build_harmonized_daily(
            build_daily_frame(frames, aoi_id=aoi.aoi_id),
            frames["modis"],
            frames["viirs"],
            model,
            config.matching,
        )
        pd.testing.assert_frame_equal(harmonized, e2e["harmonized"])

    def test_all_artifacts_written(self, e2e):
        out = e2e["out"]
        assert (out / "harmonized_daily.csv").is_file()
        assert (out / "anomaly_records.csv").is_file()
        assert (out / "calendar_forecast_records.csv").is_file()
        assert (out / "harmonizer_v1" / "manifest.json").is_file()
