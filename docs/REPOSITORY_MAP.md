# Repository Map and Planned File/Folder Structure

This is the intended structure for the selected MODIS/VIIRS harmonization solution. Planned files are not evidence that code exists; update this map as the actual tree grows.

```text
.
├── README.md
├── CONTRIBUTING.md
├── SECURITY.md
├── pyproject.toml
├── .gitignore
├── .env.example
├── configs/                 # source, harmonization, calendar, offline config
├── src/
│   ├── data/                # FIRMS adapters, schemas, inventory, validation, aggregation
│   ├── geo/                 # AOI validation, spatial matching, optional rasterization
│   ├── harmonization/       # features, baselines, training, evaluation, inference
│   ├── calendar/            # seasonal baseline, anomaly scoring, forecasting, evaluation
│   ├── models/              # PatchTST and optional Prithvi adapter
│   ├── artifacts/           # contracts, manifests, serialization, integrity checks
│   └── common/              # configuration and logging
├── app/                     # Streamlit entry point, map/calendar/trend components
├── notebooks/
│   ├── 01_stage_offline_assets.ipynb
│   ├── 02_airgapped_quarantine.ipynb
│   ├── 03_validate_and_inventory_data.ipynb
│   ├── 04_train_harmonizer.ipynb
│   ├── 05_train_and_backtest_calendar.ipynb
│   └── 06_export_artifacts.ipynb
├── scripts/                 # validation, aggregation, training, export commands
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── scientific/
│   ├── offline/
│   └── e2e/
└── docs/                    # charter, requirements, architecture, research, testing, risks, ADRs
```

## Responsibilities and contracts

| Area | Responsibility | Expected result |
|---|---|---|
| `src/data` | Read source products, validate schema, inventory and audit records | Validated observations and audit manifest |
| `src/geo` | AOI validation, matching, daily spatial aggregation | Matched pairs and AOI summaries |
| `src/harmonization` | Baselines, feature construction, training, evaluation, inference | Harmonizer, metrics, estimated VIIRS-equivalent series |
| `src/calendar` | Seasonal baselines, anomaly scores, forecast evaluation | Calendar, anomaly and forecast records |
| `src/models` | Model-specific input/output contracts | Validated PatchTST/Prithvi predictions |
| `src/artifacts` | Stable schemas, manifests, hashes and serialization | Portable app-ready artifacts |
| `app` | AOI interaction and visualization | Map, calendar, trends, anomaly and forecast views |
| `notebooks` | Stage, quarantine, validate, train, backtest, export | Reproducible logs and versioned outputs |
| `tests` | Software correctness and scientific invariants | Repeatable pass/fail evidence |

Create modules only for real responsibilities. Keep reusable logic out of notebook-only cells. Do not commit model weights, raw/processed datasets, wheelhouses, virtual environments, notebook caches, or secrets. Update this map to match the implementation rather than forcing code to match an aspirational tree.