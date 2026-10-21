# Testing and Scientific Validation

## Five-pass data gate

Training must stop before model/GPU initialization if any critical validation error remains.

1. **Structural integrity:** file readability, required columns/types, partitions, duplicates, empty files, date coverage; emit inventory manifest and SHA-256 hashes.
2. **Semantic/numerical validity:** field-specific null/NaN/Inf policies, coordinate bounds, timestamps, units, quality/confidence values, sensor-specific fields, suspicious FRP/geometry values; quarantine with reason codes rather than blindly deleting outliers.
3. **Leakage/split integrity:** chronological splits, lagged features only, matched events and overlapping windows prevented from leaking; proposed starting split is train 2000–2021, validation 2022–2023, test 2024 onward, subject to available data.
4. **Transformation invariants:** AOI boundaries, CRS assumptions, date bucketing, leap years, duplicates, missing days, units, aggregation totals where conservation is expected, normalization round trips.
5. **Report and gate:** emit `inventory_manifest.json` and `dataset_audit_report.json`; fail closed on critical errors and summarize warnings/exclusions.

Field schemas and outlier thresholds must be justified against the actual FIRMS product. Extreme FRP is a review flag, not automatic proof of bad data.

## Unit tests

Test source parsing, schema drift, AOI geometry/boundaries, time normalization, leap days, aggregation, duplicate policy, matching tolerances, feature generation, seasonal baselines, anomaly edge cases, quantile ordering/horizon, artifact schema/version/hash, and UI empty/error states.

## Harmonization evaluation

Compare Tweedie GBDT against identity and simple calibration baselines. Use held-out chronological evaluation. Report bias and error by FRP range, region, season, land cover/geometry where supported. Test matching-tolerance sensitivity, zero inflation, extreme observations, and selection bias from detections missed by MODIS.

## Calendar and forecast evaluation

Test missing coverage separately from zero detections. Validate anomaly false positives on ordinary periods and behavior on documented cases. Use rolling-origin backtests against seasonal-naive and persistence baselines; report suitable point metrics and quantile coverage/pinball loss. Record forecast issue time and training cutoff. Never randomly shuffle temporal observations into train/test.

## Optional Prithvi audit

Validate HLS band order/scaling, dimensions, nodata, CRS, date and checkpoint availability. Check output shape, finite values, alignment, and local checkpoint hash. This tests burn-scar segmentation, not active-fire timestamp accuracy.

## Offline quarantine

With internet disabled, install only from local wheels; verify imports, local paths and checkpoint hashes, run a representative forward pass, and confirm no required step attempts external downloads. Ensure code never writes to `/kaggle/input/`. Fail clearly when dependencies or artifacts are absent.

## End-to-end and release gate

Test clean setup → fixture validation → AOI aggregation → artifact loading → calendar/trend/anomaly/forecast UI. Empty AOIs, missing forecasts, unavailable optical imagery, corrupt artifacts, and disabled network must have explicit states. Record actual commands/results and environment. Never claim tests passed without preserved evidence.