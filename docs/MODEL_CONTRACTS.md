# Model Contracts, Scientific Assumptions, and Error Behavior

This document pins the implemented contracts for the harmonization and
calendar/forecast models. It follows the challenge's scientific acceptance
rules (docs/CHALLENGE_AND_REQUIREMENTS.md) and the artifact contracts in
docs/ARCHITECTURE.md. Anything not measured on real FIRMS data is stated as an
assumption or limitation, not a result.

## 1. Source data contract (normalized FIRMS observations)

- Products: processed FIRMS **MODIS** (Collection 6.x) and **VIIRS** (S-NPP
  375 m) active-fire CSV exports. Files are attributed to a product when the
  filename contains `modis` / `viirs` (case-insensitive).
- Required columns (five-pass gate pass 1): `latitude`, `longitude`,
  `acq_date` (YYYY-MM-DD), `acq_time` (HHMM UTC), `frp`.
- Optional columns, used only when present (docs/RESEARCH_AND_DATA.md: never
  assume candidate fields exist): `brightness` (alias of `bright_ti4`),
  `brightness_t31` (alias of `bright_t31`/`bright_ti5`), `scan`, `track`,
  `confidence`, `daynight`, `satellite`, `instrument`, `version`.
- **Units:** FRP is megawatts (MW) in both products; the pipeline never
  converts units silently. All FRP columns and aggregates are suffixed `_mw`.
- `acq_datetime` = parsed UTC timestamp from `acq_date` + `acq_time`.
- Confidence ordinal mapping: `l=0`, `n=1`, `h=2`; numeric 0-100 confidence is
  scaled by thirds to the same ordinal.
- **Fixture caveat:** the committed tests/fixtures data is SYNTHETIC (seed
  2026). Real FIRMS schemas must be re-inventoried per issue #6 before any
  submission claim.

## 2. Five-pass data gate (issue #11; fail-closed)

Implemented in `src/data/gate.py`. Training and inference call
`assert_gate_passed` **before** any model is constructed.

1. **Structural:** file readability, required columns, duplicates, empty
   files, date coverage; SHA-256 hashes and row counts recorded in
   `inventory_manifest.json`.
2. **Semantic:** row-level quarantine with reason codes
   (`lat_out_of_bounds`, `lon_out_of_bounds`, `coords_missing`,
   `acq_date_invalid`, `acq_time_invalid`, `frp_missing`, `frp_nonfinite`,
   `frp_negative`, `duplicate_row`). Quarantined rows are excluded downstream
   but counted in the report. Extreme FRP (>5000 MW by default) raises a
   review warning and is **kept** - it is not proof of bad data.
3. **Leakage/split:** chronological boundaries `train_end=2021-12-31`,
   `validation_end=2023-12-31` (defaults, configurable). Boundary sanity and
   per-split coverage warnings are recorded; matched records carry a single
   timestamp so they can never span splits.
4. **Transformation invariants:** daily aggregation must conserve total FRP;
   gap days are never fabricated as zero-activity observations.
5. **Report:** `inventory_manifest.json` + `dataset_audit_report.json`
   (machine-readable); any critical error fails the gate closed.

## 3. Overlap-period matching contract (Pillar 1)

`src/geo/matching.py`; defaults in `configs/harmonization.yaml`:

- **Temporal tolerance:** VIIRS detections within 90 minutes of the MODIS
  overpass (inclusive), pooled across adjacent calendar days near midnight.
- **Spatial rule:** a VIIRS detection matches when its center lies inside the
  MODIS footprint box of half-width `max(scan, track)/2` degrees (clipped to
  [0.01, 0.25]); a fixed 0.02-degree radius is used when `scan`/`track` are
  absent. This is an axis-aligned approximation of the true swath geometry.
- **Target definition:** `viirs_frp_sum_mw` = **summed VIIRS FRP inside the
  MODIS footprint within tolerance** (the docs' research question choice).
  Multiple VIIRS detections inside one footprint are summed into one target.
- **Zero vs missing:** a MODIS detection with VIIRS observations inside its
  tolerance window but none inside the footprint gets target `0.0` (a true
  zero detection). A MODIS detection with **no VIIRS observation within the
  temporal tolerance** is marked ineligible with reason
  `no_viirs_coverage` and target NaN - missing observations are never treated
  as zero fire activity.
- **Duplicate policy:** exact duplicate MODIS rows are dropped before
  matching (the gate also quarantines them upstream).
- **Confidence filter:** records below `min_confidence` (default `l`, i.e. no
  filter) are excluded from matching entirely.
- **Determinism:** match IDs (`m000001`, ...) are assigned in sorted
  (datetime, latitude, longitude) order; identical inputs produce identical
  matches.
- **Selection bias (documented limitation):** matched calibration records can
  only describe fires both sensors could see. The regression cannot recover
  fires never detected by MODIS (docs: scientific acceptance rules).

## 4. Harmonizer model contract (Pillar 1)

`src/harmonization/`:

- **Feature contract (`features.py`):** MODIS-side and temporal-context
  features only: `modis_frp_mw`, `log_modis_frp`, `footprint_radius_deg`,
  `doy_sin`, `doy_cos`, `modis_daily_active_pixels`, plus optional
  `modis_brightness`, `modis_scan_deg`, `modis_track_deg`,
  `modis_confidence_ordinal`, `is_night` when the source columns exist.
  Anything VIIRS-derived is the target and is forbidden in the feature matrix
  (`assert_no_target_leakage` is enforced and unit-tested). No feature uses
  future information.
- **Model:** LightGBM `LGBMRegressor` with `objective="tweedie"`,
  `tweedie_variance_power=1.5` (selected primary candidate). Defaults are
  regularized for small matched samples (num_leaves=15,
  min_child_samples=40, lambda_l2=5.0, lr=0.03, max 2000 trees) and paired
  with **chronological early stopping**: the last 365 days of the training
  period form the internal stopping set; validation/test periods are never
  used for stopping. `best_iteration_` is recorded in the manifest.
- **Baselines (mandatory comparisons):** `identity` (no correction) and
  `calibration` (single global scale factor fitted through the origin on the
  training split). SC-03 evidence = held-out Tweedie deviance vs both
  baselines plus FRP-range and season slices.
- **Output:** per-detection VIIRS-equivalent FRP estimates, clipped at zero,
  labeled `harmonized_estimate`. Estimates are **model outputs, not ground
  truth**.
- **Reproducibility:** fixed seed (`random_state`), `deterministic=True`,
  `force_row_wise=True`; same seed and data produce identical predictions.

## 5. Seasonal baseline and anomaly contracts (Pillar 2)

- **Climatology (`calendar/seasonal.py`):** day-of-year climatology over
  observed days; pseudo day-of-year 1..365 with **Feb 29 pooled into Feb 28**
  (documented leap policy); circular +/-7-day window; a calendar day with
  fewer than `min_samples_per_day` (default 2) pooled samples yields a NaN
  baseline. Only observed-coverage days contribute samples.
- **Anomaly score (`calendar/anomalies.py`):**
  `ratio = (observed + eps) / (baseline + eps)` with `eps=1e-6`; statuses:
  `normal` (0.5 < ratio < 2.0 by default), `anomaly_high` (>= 2.0),
  `anomaly_low` (<= 0.5), `insufficient_baseline` (NaN baseline),
  `missing_coverage` (no sensor coverage; ratio is NaN, never a numeric
  claim). Thresholds are documented defaults that still require validation
  before any operational alerting (docs: "any alert threshold must be
  validated and documented").
- Anomaly statements are **relative-to-season** descriptions, never ignition
  claims.

## 6. Forecast contract (Pillar 2; R-08)

`src/models/patchtst.py` + `src/calendar/forecast.py`:

- **Architecture (PatchTST candidate):** channel-independent patching
  (defaults 16-day patches, stride 8, d_model 128, four encoder blocks, eight
  heads, dropout 0.1 - documented *tunable starting values*, not proven
  optimal); Transformer encoder (pre-norm, GELU); linear head producing
  horizon x quantiles. PyTorch is an **optional** dependency (`[forecast]`
  extra); without it the forecaster raises a clear, offline-friendly error and
  the seasonal-naive/persistence baselines remain fully usable.
- **Channels:** [value (target, gaps filled with 0.0), active pixel count,
  coverage indicator (1 observed / 0 gap)]. The coverage channel keeps
  "missing observation" distinct from "zero detections" instead of silently
  zero-filling gaps.
- **Horizon and quantiles:** 14 days; quantiles (0.1, 0.5, 0.9) trained with
  the pinball (quantile) loss; predictions are sorted per step and clipped at
  zero, so q10 <= q50 <= q90 always holds. Point forecast = q50.
- **Baselines:** `persistence` (last observed value) and `seasonal_naive`
  (day-of-year climatology, last-value fallback); baseline quantiles are
  empirical residual quantiles (uncalibrated).
- **Evaluation protocol:** rolling-origin backtests only; each origin's
  training history is strictly `<= origin - 1 day` (leakage-guarded and
  unit-tested); every forecast records its issue date, training cutoff, and
  model version. Metrics: mean pinball per quantile, interval coverage, RMSE,
  MAE, bias. **Intervals are labeled experimental/uncalibrated** until
  interval calibration is measured on real data.
- **Scientific boundary:** regional activity forecasts only - never exact
  ignition coordinates or times.

## 7. Artifact contracts

Versioned artifact directories contain `manifest.json` plus the model file
(`model.joblib` for the harmonizer, pure-JSON state `model.json` for the
forecaster). `schema_version` is `1.0.0` for all contracts.

- **Daily AOI records** (`src/data/aggregation.py::DAILY_AOI_COLUMNS`): date,
  aoi_id, coverage_status, per-sensor coverage flags, active pixel counts, FRP
  sums/means, mean confidence.
- **Matched calibration records** (`src/geo/matching.py::MATCHED_COLUMNS`):
  match id, tolerances, MODIS-side fields, matched count, target, eligibility.
- **Harmonized daily series**: value + `value_type` in {`observed_viirs`,
  `harmonized_estimate`, `missing_coverage`} + `sensor_era`.
- **Calendar/forecast records** (`src/calendar/records.py`): seasonal
  baseline, anomaly ratio/status, forecast quantiles, model name/version,
  training cutoff, issue date, calibration status, and `quality_state` in
  {observed, harmonized, forecast, missing}.
- **Manifest keys (validated on load):** schema_version, created_at,
  pipeline, code_commit, config, source_files (with SHA-256), model (name,
  version, sha256), metrics, limitations.
- **Integrity:** every bundle load verifies the recorded model SHA-256;
  mismatches raise `ArtifactIntegrityError` instead of silently using a
  corrupted model.

## 8. Error behavior (issue #13 graceful failure paths)

- Gate failure: training/inference abort with a non-zero exit **before** model
  initialization; reports are still written.
- Empty AOI: `run_inference.py` writes an explicit `{"status": "empty_aoi"}`
  state and exits with a dedicated code (2).
- Missing/corrupt artifacts: `ArtifactIntegrityError` with the offending file
  named; missing manifest keys are listed in `ContractError`.
- Missing optional dependency (PyTorch): clear `TorchDependencyError`
  mentioning the install path and that baselines remain available; the
  forecast section degrades to "unavailable" instead of failing the flow.
- Missing forecast artifact: the calendar export still succeeds; forecast
  columns stay empty (explicit "no forecast available" state).
- No source files / unreadable / empty CSV / missing columns: critical gate
  errors listing the offending file and column names.

## 9. Assumptions and limitations

1. Harmonized values are model estimates, not ground truth; they inherit
   MODIS detection-selection bias.
2. The axis-aligned footprint box and fixed default footprint radius are
   documented approximations of true swath geometry.
3. Region and land-cover/geometry error slices require auxiliary joins (MCD12
   land cover, elevation) that are not yet implemented.
4. PatchTST hyperparameters are starting values from the architecture
   document; no tuning claim is made. Forecast intervals are uncalibrated.
5. All committed fixture evidence is synthetic; performance on real FIRMS
   archives must be measured before any submission claim.
6. Prithvi-EO optical burn-scar verification (Pillar 3) is intentionally not
   part of this implementation; it is independent of the thermal
   harmonization path by design.

## 10. Requirement traceability (implemented coverage)

| Requirement | Implementation | Test evidence |
|---|---|---|
| R-01 AOI selection | `src/geo/aoi.py` (bbox/polygon, validated) | `tests/unit/test_aoi.py` |
| R-03 harmonization | matched records + LightGBM/Tweedie + baselines | `tests/scientific/test_beats_baselines.py` |
| R-04 continuous record | daily aggregation + harmonized series | `tests/unit/test_aggregation.py`, `tests/e2e` |
| R-07 unusual conditions | seasonal baseline + ratio score | `tests/unit/test_seasonal_anomalies.py` |
| R-08 critical periods | 14-day probabilistic forecast + rolling-origin backtests | `tests/unit/test_forecast.py`, `test_patchtst.py` |
| R-10 provenance/labels | manifests, value_type/quality_state labels | `tests/unit/test_artifacts.py`, `test_records.py` |
| R-12 offline behavior | no network imports, lazy torch, hash verification | `tests/offline/test_offline_safety.py` |
| R-14 reproducibility | seeds, versioned configs, hash-checked artifacts, determinism tests | `tests/e2e/test_full_pipeline.py` |
