# Repository Map and Planned File/Folder Structure

This map reflects the **actual** tree as of the offline ML pipeline
implementation (five-pass gate, harmonization, calendar/forecast, artifacts,
CLI scripts, tests, CI). `app/` and `notebooks/` remain planned (see
docs/DEVELOPMENT.md) and are marked accordingly.

```text
.
├── README.md
├── CONTRIBUTING.md
├── SECURITY.md
├── pyproject.toml            # package metadata, pytest/ruff config, extras
├── .gitignore
├── .github/workflows/ci.yml  # lint + full tests (with and without PyTorch)
├── configs/
│   ├── harmonization.yaml    # split boundaries, matching tolerances, model params
│   └── calendar.yaml         # seasonal/anomaly/forecast/backtest settings
├── src/
│   ├── common/               # config.py (typed configs), logging.py
│   ├── data/                 # schema.py, adapters.py, gate.py (five-pass), aggregation.py
│   ├── geo/                  # aoi.py (bbox/polygon), matching.py (overlap matching)
│   ├── harmonization/        # features.py, baselines.py, model.py (LightGBM/Tweedie),
│   │                         # evaluate.py, train.py, inference.py
│   ├── calendar/             # seasonal.py, anomalies.py, forecast.py (baselines +
│   │                         # rolling-origin), evaluation.py, records.py
│   ├── models/               # patchtst.py (torch, lazy import), registry.py
│   └── artifacts/            # contracts.py, manifest.py, serialization.py
├── scripts/
│   ├── validate_data.py      # run the five-pass gate, write the audit reports
│   ├── train_harmonizer.py   # gate -> match -> train -> evaluate -> versioned artifact
│   ├── train_forecaster.py   # calendar/anomalies -> backtests -> forecast artifact
│   └── run_inference.py      # AOI -> gate -> harmonize -> calendar -> forecast export
├── tests/
│   ├── conftest.py           # shared fixtures (gate outcome, training result, tiny configs)
│   ├── fixtures/             # SYNTHETIC sample data (seed 2026) + raw_dirty gate fixtures
│   ├── unit/                 # schema, adapters, gate, aggregation, AOI, matching,
│   │                         # features, baselines, model, evaluate, seasonal/anomalies,
│   │                         # PatchTST, forecast, records, artifacts, config
│   ├── integration/          # train->save->load->inference roundtrips + CLI subprocesses
│   ├── scientific/           # leakage guards, transformation invariants, baseline comparisons
│   ├── offline/              # no-network imports, read-only sources, graceful failures
│   └── e2e/                  # full AOI -> gate -> train -> calendar -> forecast workflow
├── docs/                     # charter, requirements, architecture, model contracts,
│   └── decisions/            # research, testing, risks, ADRs (unchanged in this PR)
├── app/                      # PLANNED: Streamlit UI over exported artifacts
└── notebooks/                # PLANNED: staging, quarantine, validation, training,
                             # backtesting, export (docs/DEVELOPMENT.md)
```

## Responsibilities and contracts

| Area | Responsibility | Actual modules | Expected result |
|---|---|---|---|
| `src/data` | Read source products, validate schema, inventory, aggregate | `adapters`, `schema`, `gate`, `aggregation` | Validated observations, audit reports, daily AOI frame |
| `src/geo` | AOI validation/filtering, overlap-period matching | `aoi`, `matching` | Matched calibration records with explicit tolerances |
| `src/harmonization` | Baselines, features, Tweedie GBDT, evaluation, inference | `features`, `baselines`, `model`, `evaluate`, `train`, `inference` | Harmonizer, held-out metrics, harmonized daily series |
| `src/calendar` | Seasonal baselines, anomalies, forecasting, evaluation | `seasonal`, `anomalies`, `forecast`, `evaluation`, `records` | Calendar, anomaly and forecast records |
| `src/models` | Model architectures and registry | `patchtst` (lazy torch), `registry` | Validated PatchTST predictions, stable model names |
| `src/artifacts` | Stable schemas, manifests, hashes, serialization | `contracts`, `manifest`, `serialization` | Portable, integrity-verified artifacts |
| `scripts` | Runnable commands over the library API | 4 CLIs with argparse | Reproducible offline workflow + machine-readable evidence |
| `tests` | Software correctness and scientific invariants | 5 suites, ~294 tests, synthetic fixtures | Repeatable pass/fail evidence (CI runs with and without torch) |
| `app` | AOI interaction and visualization | planned | Map, calendar, trends, anomaly and forecast views |
| `notebooks` | Stage, quarantine, validate, train, backtest, export | planned | Reproducible logs and versioned outputs |

Detailed model/input/output contracts, units, tolerance values, error
behavior, and limitations are pinned in [docs/MODEL_CONTRACTS.md](MODEL_CONTRACTS.md).

Create modules only for real responsibilities. Keep reusable logic out of
notebook-only cells. Do not commit model weights, raw/processed datasets,
wheelhouses, virtual environments, notebook caches, or secrets. Update this map
to match the implementation rather than forcing code to match an aspirational
tree.
