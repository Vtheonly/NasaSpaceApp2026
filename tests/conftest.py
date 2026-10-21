"""Shared pytest fixtures for the hotspot-harmonization test suite."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures"
SAMPLE_DIR = FIXTURE_DIR / "sample_aoi"
DIRTY_DIR = FIXTURE_DIR / "raw_dirty"

# Deterministic fixture AOI (matches tests/fixtures/generate_fixtures.py).
AOI_SPEC = "bbox:-123.5,39.5,-120.5,42.0"


@pytest.fixture(scope="session")
def sample_dir() -> Path:
    return SAMPLE_DIR


@pytest.fixture(scope="session")
def dirty_dir() -> Path:
    return DIRTY_DIR


@pytest.fixture(scope="session")
def harmonization_config():
    from src.common.config import HarmonizationConfig

    return HarmonizationConfig()


@pytest.fixture(scope="session")
def calendar_config():
    from src.common.config import CalendarConfig

    return CalendarConfig()


@pytest.fixture(scope="session")
def aoi():
    from src.geo.aoi import parse_aoi

    return parse_aoi(AOI_SPEC)


@pytest.fixture(scope="session")
def gate_outcome(harmonization_config, sample_dir):
    from src.data.gate import FivePassDataGate

    return FivePassDataGate(harmonization_config).run(str(sample_dir))


@pytest.fixture(scope="session")
def training_result(gate_outcome, harmonization_config, aoi, sample_dir):
    from src.harmonization.train import run_harmonization_training

    return run_harmonization_training(
        str(sample_dir), config=harmonization_config, aoi=aoi, gate_outcome=gate_outcome
    )


@pytest.fixture(scope="session")
def daily_frame(gate_outcome, aoi):
    from src.data.aggregation import build_daily_frame
    from src.geo.aoi import filter_records

    frames = {p: filter_records(f, aoi) for p, f in gate_outcome.frames.items()}
    return build_daily_frame(frames, aoi_id=aoi.aoi_id)


def tiny_calendar_config(min_train_days: int = 90) -> object:
    """Fast PatchTST config for tests (tiny architecture, few epochs)."""
    from src.common.config import (
        BacktestConfig,
        CalendarConfig,
        ForecastConfig,
        PatchTSTParams,
        SeasonalConfig,
        TrainerConfig,
    )

    return CalendarConfig(
        seed=2026,
        seasonal=SeasonalConfig(),
        forecast=ForecastConfig(
            horizon_days=14,
            quantiles=(0.1, 0.5, 0.9),
            patchtst=PatchTSTParams(
                context_len=48,
                patch_len=8,
                stride=4,
                d_model=16,
                n_encoder_blocks=1,
                n_heads=2,
                dropout=0.0,
                dim_feedforward=32,
            ),
            trainer=TrainerConfig(
                epochs=8,
                batch_size=16,
                learning_rate=0.01,
                weight_decay=1e-4,
                patience=3,
                validation_fraction=0.2,
            ),
        ),
        backtest=BacktestConfig(
            n_origins=2,
            origin_spacing_days=14,
            min_train_days=min_train_days,
        ),
    )


@pytest.fixture()
def tiny_calendar():
    return tiny_calendar_config()


def make_daily_series(
    start: str = "2021-01-01",
    periods: int = 1096,
    seed: int = 7,
    spike_date: str = "2022-08-15",
    gap_fraction: float = 0.02,
) -> pd.DataFrame:
    """Deterministic synthetic daily series with seasonality, a spike, gaps.

    Mirrors the harmonized daily frame contract (date, value, value_type,
    sensor_era, coverage_status) so calendar/forecast code paths exercise
    the real column set.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=periods, freq="D")
    doy = dates.dayofyear.to_numpy()
    seasonal = 50.0 * (1.0 + 0.6 * np.sin(2.0 * np.pi * (doy - 100) / 365.0))
    values = seasonal + rng.normal(0.0, 4.0, size=len(dates))
    values = np.clip(values, 0.0, None)
    frame = pd.DataFrame(
        {
            "date": dates,
            "value": values,
            "coverage_status": "observed",
        }
    )
    spike = pd.Timestamp(spike_date)
    if spike in set(dates):
        frame.loc[frame["date"] == spike, "value"] *= 20.0
    # Deterministic coverage gaps: every ~50th day loses observation coverage.
    gap_mask = np.arange(len(frame)) % round(1.0 / max(gap_fraction, 1e-3)) == 0
    frame.loc[gap_mask, "value"] = np.nan
    frame.loc[gap_mask, "coverage_status"] = "missing"
    frame["value_type"] = np.where(
        frame["coverage_status"].eq("observed"), "observed_viirs", "missing_coverage"
    )
    frame["sensor_era"] = np.where(
        frame["coverage_status"].eq("observed"), "viirs_only", "no_coverage"
    )
    return frame


@pytest.fixture()
def synthetic_daily() -> pd.DataFrame:
    return make_daily_series()


@pytest.fixture()
def tmp_output(tmp_path) -> Path:
    out = tmp_path / "output"
    out.mkdir()
    return out


def run_script(name: str, *args: str, expect: int = 0) -> subprocess.CompletedProcess:
    """Run a repo CLI script in a subprocess and return the completed process."""
    import subprocess

    cmd = [sys.executable, str(REPO_ROOT / "scripts" / name), *args]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600, cwd=str(REPO_ROOT))
    if expect is not None and proc.returncode != expect:
        raise AssertionError(
            f"{name} exited {proc.returncode} (expected {expect}).\n"
            f"stdout: {proc.stdout[-2000:]}\nstderr: {proc.stderr[-2000:]}"
        )
    return proc
