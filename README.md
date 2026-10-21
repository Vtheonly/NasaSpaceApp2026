# Harmonization of MODIS and VIIRS Hot Spots

**NASA Space Apps Challenge 2026 · Earth Science · Software**

Build an interactive web application that harmonizes MODIS and VIIRS active-fire records into a more consistent burning-activity calendar. Users select an area of interest (AOI), explore historical fire patterns, inspect unusual activity, and review probabilistic critical burning windows.

## Selected solution

1. **Sensor harmonization:** LightGBM gradient-boosted regression with a Tweedie objective as the primary candidate for mapping MODIS observations to a VIIRS-equivalent FRP scale using overlap-period matches and available physical/contextual features.
2. **Burning activity calendar:** daily AOI time series, seasonal baselines, anomaly scoring, and PatchTST as the selected temporal-model candidate for evaluating 14-day regional activity forecasts.
3. **Independent optical verification:** optional NASA–IBM Prithvi-EO 2.0 burn-scar segmentation using compatible HLS optical imagery. It is not an active-fire generator and is not part of the thermal harmonization model.
4. **Interactive web app:** AOI selection, hotspot map, 52-week-by-year calendar, historical trends, anomaly details, forecast uncertainty, provenance, and limitations.
5. **Offline compute:** stage dependencies, model weights, and processed data separately; verify all required components in an internet-disabled Kaggle environment before GPU-heavy runs. Run the user-facing app outside Kaggle.

## Core workflow

AOI selection → FIRMS ingestion and five-pass validation → daily aggregation → overlap-period MODIS/VIIRS matching → harmonization and evaluation → calendar and anomaly baseline → probabilistic 14-day forecast → versioned artifacts → interactive web application. Optional HLS/Prithvi verification is a separate branch.

## Scientific guardrails

- MODIS and VIIRS measurements are not directly interchangeable without evaluation; harmonized values are model estimates, not ground truth.
- Missing observations are not the same as zero fire activity.
- A burn scar is not equivalent to an active-fire detection or ignition timestamp.
- Forecast regional burning activity probabilistically; do not claim to predict exact ignition coordinates or times.
- Compare learned models with simple baselines and use chronological/rolling-origin evaluation.
- Treat performance figures in planning notes as targets until measured reproducibly.
- Label observed, harmonized, estimated, forecast, cached, and sample data distinctly.

## Compute and artifact separation

- **GitHub:** source code, docs, tests, small fixtures, configuration templates.
- **Offline wheelhouse dataset:** compatible Linux wheels, versions, and hashes.
- **Model/data datasets:** verified checkpoints and processed data, subject to their terms.
- **Kaggle working output:** reports, checkpoints, metrics, and exported artifacts.
- **Separate app host:** UI and query layer over exported artifacts.

The supplied design targets a Kaggle environment with 96 GB VRAM and 170 GB RAM, but the actual runtime must be inspected. Assume no network access during compute. Treat /kaggle/input as read-only and write generated outputs to /kaggle/working. Do not commit virtual environments, large datasets, model weights, or credentials to GitHub.

## Project documents

- [Project charter](docs/PROJECT_CHARTER.md)
- [Challenge requirements and traceability](docs/CHALLENGE_AND_REQUIREMENTS.md)
- [Architecture](docs/ARCHITECTURE.md)
- [Model contracts and scientific assumptions](docs/MODEL_CONTRACTS.md)
- [Repository map](docs/REPOSITORY_MAP.md)
- [Development and offline workflow](docs/DEVELOPMENT.md)
- [Testing and five-pass validation](docs/TESTING.md)
- [Research and data governance](docs/RESEARCH_AND_DATA.md)
- [Risk register](docs/RISK_REGISTER.md)
- [Architecture decisions](docs/decisions/README.md)
- [GitHub Issues](https://github.com/Vtheonly/NasaSpaceApp2026/issues)

## Status and definition of done

**Implemented and tested (offline ML pipeline):** the five-pass data gate with
fail-closed audit reports; AOI validation and overlap-period MODIS/VIIRS
matching; the LightGBM/Tweedie harmonizer with identity and calibration
baselines and held-out chronological evaluation; the seasonal calendar and
anomaly scoring; the PatchTST 14-day probabilistic forecast candidate with
seasonal-naive/persistence baselines and rolling-origin backtests; versioned,
hash-verified artifacts; CLI scripts; a ~294-test suite and CI (issue #12).
All evidence so far runs on deterministic synthetic fixtures — see
docs/MODEL_CONTRACTS.md for contracts, assumptions, and limitations.

**Still planned:** the Streamlit app, the six staging/offline notebooks, the
optional Prithvi optical audit, and validation on real FIRMS archives.

A component is complete only when its acceptance criteria, tests, data
provenance, and reproducible evidence are recorded. Final delivery requires a
clean AOI-to-calendar demo, validated harmonization and forecasting, a passing
offline quarantine test, visible uncertainty/limitations, and a verified
official submission checklist.

## Quickstart (offline pipeline)

```bash
pip install -e ".[dev,forecast]"   # torch CPU wheels: --index-url https://download.pytorch.org/whl/cpu

python scripts/validate_data.py --source-dir <firms_csv_dir> --output-dir out/gate
python scripts/train_harmonizer.py --source-dir <firms_csv_dir> --output-dir out \
    --aoi "bbox:-123.5,39.5,-120.5,42.0"
python scripts/run_inference.py --source-dir <firms_csv_dir> \
    --harmonizer-dir out/<harmonizer artifact> --output-dir out/inference \
    --aoi "bbox:-123.5,39.5,-120.5,42.0"
pytest tests    # full suite; ruff check src scripts tests for quality
```
