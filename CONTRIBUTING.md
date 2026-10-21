# Contributing

## Project context

This repository builds an interactive application that harmonizes MODIS and VIIRS active-fire records into a historical burning calendar, anomaly exploration, and probabilistic regional critical-period forecasts. Read the README, challenge requirements, architecture, data governance, and testing documents before changing core behavior.

## Workflow

1. Choose an actionable issue and check dependencies.
2. Keep changes focused on one deliverable or closely related fix.
3. Preserve separation between source validation, harmonization, temporal modeling, optional optical verification, artifacts, and UI.
4. Add tests with implementation changes.
5. Update schemas, provenance, docs, repository map, and ADRs when behavior changes.
6. Run relevant checks and record exact commands and results.
7. Reference the issue in the commit/PR description.

## Scientific rules

- Do not assume MODIS and VIIRS measurements are directly interchangeable.
- Label observed, harmonized, estimated, forecast, cached, and sample values distinctly.
- Never use future information in training features or evaluation.
- Do not treat missing coverage as zero detections without an explicit policy.
- Compare learned models against simple baselines.
- Do not present a regional activity forecast as an exact ignition prediction.
- Keep Prithvi optical burn-scar inference separate from active-fire thermal harmonization.
- Treat performance estimates as targets until measured reproducibly.

## Offline rules

- No online package/model downloads in the offline compute path.
- Do not commit virtual environments, wheelhouses, model checkpoints, raw/processed datasets, notebook caches, or secrets.
- Verify local artifact hashes and keep Kaggle input datasets read-only.
- Do not depend on Docker or tunnels inside the offline Kaggle notebook.

## Change checklist

- [ ] Issue and acceptance criteria are clear.
- [ ] Tests cover expected and failure behavior.
- [ ] Units, schemas, provenance, and missingness semantics are preserved.
- [ ] No future-data leakage or fabricated output is introduced.
- [ ] No secrets or large generated artifacts are committed.
- [ ] Documentation matches the resulting code.
- [ ] Exact test commands/results are recorded.

Every issue should define an outcome, tasks, acceptance criteria, dependencies, and evidence needed to close it.