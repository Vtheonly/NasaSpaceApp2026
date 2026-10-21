#!/usr/bin/env python3
"""Generate the deterministic synthetic FIRMS-like test fixtures.

IMPORTANT: these are SYNTHETIC, reproducible sample data (seed 2026) used only
for tests and offline demonstrations. They are NOT real NASA FIRMS observations
and must always be labeled as sample data (docs: "Label ... sample data
distinctly").

Outputs:
- tests/fixtures/sample_aoi/modis_active_fires.csv  (clean, gate-passing)
- tests/fixtures/sample_aoi/viirs_active_fires.csv  (clean, gate-passing)
- tests/fixtures/raw_dirty/*                        (gate-failure fixtures)

The VIIRS-per-MODIS relationship embeds a learnable, time-varying calibration
factor so the trained harmonizer can demonstrably beat the identity and global
calibration baselines on held-out chronological periods.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 2026
OUT_ROOT = Path(__file__).resolve().parent
SAMPLE_DIR = OUT_ROOT / "sample_aoi"
DIRTY_DIR = OUT_ROOT / "raw_dirty"

BBOX = (-123.5, 39.5, -120.5, 42.0)  # min_lon, min_lat, max_lon, max_lat (N. California)
START = pd.Timestamp("2018-01-01")
END = pd.Timestamp("2024-12-31")


def activity_probability(doy: int) -> float:
    """Seasonal burning probability with a summer/autumn peak (~doy 210)."""
    return 0.05 + 0.60 * float(np.exp(-(((doy - 210) / 50.0) ** 2)))


def calibration_factor(doy: int, night: bool) -> float:
    """Time-varying MODIS->VIIRS relationship the harmonizer must learn.

    Seasonal amplitude and day/night contrast are exaggerated relative to
    real sensor physics so the learned harmonizer can be shown to beat both
    the identity mapping and a single global calibration factor on held-out
    chronological periods (SC-03 evidence on synthetic data).
    """
    seasonal = 1.25 * (1.0 + 0.30 * np.sin(2.0 * np.pi * (doy - 40.0) / 365.0))
    diurnal = 1.18 if night else 0.90
    return seasonal * diurnal


def format_hhmm(hour: int, minute: int) -> int:
    return hour * 100 + minute


def generate_sample_aoi(rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    days = pd.date_range(START, END, freq="D")
    modis_rows: list[dict] = []
    viirs_rows: list[dict] = []
    viirs_gap_days = {
        days[i]
        for i in rng.choice(len(days), size=int(len(days) * 0.02), replace=False)
    }
    modis_gap_days = {
        days[i]
        for i in rng.choice(len(days), size=int(len(days) * 0.015), replace=False)
    }

    for day in days:
        doy = day.dayofyear
        if rng.random() > activity_probability(doy):
            continue
        night_pass = bool(rng.random() < 0.5)
        base_hour, base_minute = (1, 30) if night_pass else (13, 30)
        skip_modis = day in modis_gap_days
        skip_viirs = day in viirs_gap_days

        n_modis = int(rng.poisson(2.5)) + 1
        for i in range(n_modis):
            lon = float(rng.uniform(BBOX[0], BBOX[2]))
            lat = float(rng.uniform(BBOX[1], BBOX[3]))
            frp = float(np.clip(rng.lognormal(mean=2.2, sigma=0.9), 0.3, 3500.0))
            edge = float(rng.uniform()) ** 2
            scan = 0.011 + 0.035 * edge
            track = 0.010
            jitter = int(rng.integers(-10, 11))
            hh, mm = divmod(base_hour * 60 + base_minute + jitter, 60)
            conf_roll = rng.random()
            if frp > 60:
                confidence = "h" if conf_roll < 0.6 else "n"
            elif frp > 8:
                confidence = "n" if conf_roll < 0.75 else "h" if conf_roll < 0.9 else "l"
            else:
                confidence = "l" if conf_roll < 0.6 else "n"
            if not skip_modis:
                modis_rows.append(
                    {
                        "latitude": lat,
                        "longitude": lon,
                        "brightness": round(300.0 + frp / 40.0 + rng.normal(0, 8), 2),
                        "scan": round(scan, 4),
                        "track": track,
                        "acq_date": day.date().isoformat(),
                        "acq_time": format_hhmm(hh, mm),
                        "satellite": "Aqua" if night_pass else "Terra",
                        "instrument": "MODIS",
                        "confidence": confidence,
                        "version": "6.1",
                        "frp": round(frp, 2),
                        "daynight": "N" if night_pass else "D",
                    }
                )

            if skip_viirs:
                continue
            k = calibration_factor(doy, night_pass)
            # Small fires: VIIRS sometimes detects nothing inside the footprint.
            n_viirs = int(rng.poisson(0.4 + frp / 12.0))
            if frp < 4.0 and rng.random() < 0.5:
                n_viirs = 0
            n_viirs = min(n_viirs, 40)
            total_target = k * frp * float(rng.lognormal(mean=0.0, sigma=0.22))
            if n_viirs > 0:
                weights = rng.dirichlet(np.full(n_viirs, 2.0))
                radius = max(scan, track) / 2.0
                for w in weights:
                    delta = int(rng.integers(-45, 46))
                    vhh, vmm = divmod(base_hour * 60 + base_minute + delta, 60)
                    viirs_rows.append(
                        {
                            "latitude": round(lat + rng.uniform(-radius, radius), 4),
                            "longitude": round(lon + rng.uniform(-radius, radius), 4),
                            "bright_ti4": round(295.0 + frp / 45.0 + rng.normal(0, 9), 2),
                            "scan": 0.0007,
                            "track": 0.0007,
                            "acq_date": day.date().isoformat(),
                            "acq_time": format_hhmm(vhh, vmm),
                            "satellite": "S-NPP",
                            "instrument": "VIIRS",
                            "confidence": "h" if frp > 40 else "n",
                            "version": "2.0nt",
                            "frp": round(max(total_target * w, 0.1), 2),
                            "daynight": "N" if night_pass else "D",
                        }
                    )

        # VIIRS-only small fires MODIS misses (detection sensitivity difference).
        if not skip_viirs and rng.random() < 0.12:
            for _ in range(int(rng.integers(1, 3))):
                viirs_rows.append(
                    {
                        "latitude": round(float(rng.uniform(BBOX[1], BBOX[3])), 4),
                        "longitude": round(float(rng.uniform(BBOX[0], BBOX[2])), 4),
                        "bright_ti4": round(292.0 + rng.normal(0, 6), 2),
                        "scan": 0.0007,
                        "track": 0.0007,
                        "acq_date": day.date().isoformat(),
                        "acq_time": format_hhmm(
                            *divmod(base_hour * 60 + base_minute + 180, 60)
                        ),
                        "satellite": "S-NPP",
                        "instrument": "VIIRS",
                        "confidence": "n",
                        "version": "2.0nt",
                        "frp": round(float(rng.uniform(0.5, 5.0)), 2),
                        "daynight": "N" if night_pass else "D",
                    }
                )

    modis = pd.DataFrame(modis_rows).sort_values(["acq_date", "acq_time", "latitude", "longitude"])
    viirs = pd.DataFrame(viirs_rows).sort_values(["acq_date", "acq_time", "latitude", "longitude"])
    return modis.reset_index(drop=True), viirs.reset_index(drop=True)


def generate_dirty(modis: pd.DataFrame, viirs: pd.DataFrame) -> None:
    """Write gate-failure fixtures covering every quarantine/critical category."""
    DIRTY_DIR.mkdir(parents=True, exist_ok=True)

    modis.head(30).to_csv(DIRTY_DIR / "modis_nominal_head.csv", index=False)

    bad = modis.head(12).copy()
    bad.loc[bad.index[0], "latitude"] = 95.0
    bad.loc[bad.index[1], "latitude"] = -95.0
    bad.loc[bad.index[2], "longitude"] = 200.0
    bad.loc[bad.index[3], "frp"] = -5.0
    bad.loc[bad.index[4], "frp"] = -0.1
    bad.loc[bad.index[5], "frp"] = np.nan
    bad.loc[bad.index[6], "frp"] = np.inf
    bad.loc[bad.index[7], "acq_date"] = "2021-13-45"
    bad.loc[bad.index[8], "acq_time"] = 2500
    bad.loc[bad.index[9], "frp"] = 45000.0  # extreme-FRP review warning, kept
    dup_row = bad.iloc[10].copy()
    bad = pd.concat([bad, dup_row.to_frame().T], ignore_index=True)  # exact duplicate
    bad.to_csv(DIRTY_DIR / "modis_issues.csv", index=False)

    viirs.head(8).to_csv(DIRTY_DIR / "viirs_issues.csv", index=False)
    with pd.option_context("mode.chained_assignment", None):
        vi = pd.read_csv(DIRTY_DIR / "viirs_issues.csv")
    vi.loc[3, "frp"] = np.nan
    vi.loc[4, "latitude"] = np.nan
    vi = pd.concat([vi, vi.iloc[0:1]], ignore_index=True)  # duplicate row
    vi.to_csv(DIRTY_DIR / "viirs_issues.csv", index=False)

    # Empty file and missing-column file (critical structural failures).
    (DIRTY_DIR / "modis_empty.csv").write_text("", encoding="utf-8")
    modis.head(5).drop(columns=["frp"]).to_csv(
        DIRTY_DIR / "modis_missing_column.csv", index=False
    )


def main() -> int:
    rng = np.random.default_rng(SEED)
    modis, viirs = generate_sample_aoi(rng)
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    modis.to_csv(SAMPLE_DIR / "modis_active_fires.csv", index=False)
    viirs.to_csv(SAMPLE_DIR / "viirs_active_fires.csv", index=False)
    generate_dirty(modis, viirs)
    print(f"modis rows: {len(modis)}; viirs rows: {len(viirs)}")
    print(f"fixtures written under {OUT_ROOT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
