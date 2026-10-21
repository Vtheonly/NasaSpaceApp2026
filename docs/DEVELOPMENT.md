# Development and Offline Execution Guide

## Environments

1. **Connected staging:** acquire approved wheels, checkpoints, and permitted data; generate hashes/manifests.
2. **Offline quarantine:** with internet disabled, verify package installation, imports, local checkpoint loading, and representative inference.
3. **Kaggle compute:** validate data, aggregate, train/evaluate, and export artifacts after quarantine passes.
4. **Application host:** run the user-facing web UI separately using exported artifacts.

## Runtime discovery

Record Python/platform, GPU/VRAM, CUDA/runtime, RAM, disk, internet state, mounted datasets, source revision, and config. The project design targets a 96 GB VRAM / 170 GB RAM Kaggle accelerator, but runtime inspection is authoritative.

## Workflow notebooks

All six are committed under `notebooks/` (issue #9); they call the CLI
scripts and read their artifacts - reusable logic stays out of cells
(docs/REPOSITORY_MAP.md).

- `01_stage_offline_assets.ipynb`: resolve compatible dependencies, stage wheels/checkpoints/data, validate terms, generate hashes/manifests (connected).
- `02_airgapped_quarantine.ipynb`: prove offline installation/import/local model loading and run a representative forward pass (internet disabled).
- `03_validate_and_inventory_data.ipynb`: inventory files and run the five-pass validation gate.
- `04_train_harmonizer.ipynb`: create overlap matches, train baselines and Tweedie GBDT, evaluate slices, export model/manifest.
- `05_train_and_backtest_calendar.ipynb`: build daily sequences, evaluate seasonal baselines and PatchTST using rolling-origin backtests.
- `06_export_artifacts.ipynb`: validate contracts, hash outputs, and produce an app-consumable export manifest.

For the full journey in one command use `scripts/run_pipeline.py` (stage
statuses + timings in `pipeline_run.json`) and `scripts/run_demo.py`
(reproducible demo report, docs/DEMO_GUIDE.md).

## Offline artifact strategy

Keep GitHub for source, docs, tests, and small fixtures. Store platform-compatible Linux wheels, model weights, and processed data in separate versioned datasets. Verify Python/platform/CUDA compatibility; top-level package downloads alone do not guarantee all transitive dependencies are available offline. Respect data/model terms.

Illustrative staging command, to be adjusted after dependencies are locked:

```bash
python -m pip download --only-binary=:all: --dest wheelhouse -r requirements-offline.txt
sha256sum wheelhouse/* > wheelhouse-SHA256SUMS.txt
```

In the offline notebook, install only from the local wheelhouse using `pip --no-index --find-links=...`; set appropriate offline flags; verify file hashes and run model smoke tests. A failed network request by itself does not prove the complete pipeline is offline-safe.

## Kaggle boundaries

Treat `/kaggle/input/` as read-only; write to `/kaggle/working/`. Do not rely on PyPI, Hugging Face, GitHub, REST APIs, or online model downloads during offline compute. Save checkpoints and reports before sessions end. Export outputs using supported Kaggle output/dataset workflows. Do not host the public app or depend on Docker/SSH/ngrok tunnels inside the isolated compute notebook.

## Application

The selected direction is a separate Streamlit-based web interface with map/chart components, using versioned exported artifacts. Pin exact dependencies only after implementation and clean-environment testing. The UI must label cached/sample data and must not silently fetch the network in offline/demo mode.

## Reproducibility

Every run should record code commit, source product/version and hashes, config, environment, split boundaries, random seeds where applicable, model parameters, metrics, artifact hashes, and limitations. Setup commands become authoritative only after a clean setup has actually passed.