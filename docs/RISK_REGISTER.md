# Risk Register

| ID | Risk | Mitigation / exit evidence |
|---|---|---|
| R-01 | Sensor differences reduced to an invalid global multiplier | Matched data, contextual features, baselines, subgroup diagnostics |
| R-02 | MODIS-missed fires cannot be recovered from matched detections alone | Define estimand and detection limits in model card |
| R-03 | Incorrect matching or temporal leakage | Explicit tolerances, pair audit, chronological/event-aware splits, leakage tests |
| R-04 | Missing coverage treated as zero activity | Carry coverage/missingness separately and test UI labels |
| R-05 | PatchTST adds complexity without forecast benefit | Compare against seasonal-naive and persistence baselines |
| R-06 | Anomaly threshold creates excessive false alarms | Backtest thresholds and document known-case review |
| R-07 | Forecast mistaken for exact ignition prediction | Regional probabilistic language and uncertainty display |
| R-08 | Offline wheels incompatible with Python/OS/CUDA | Lock platform, hash wheels, pass quarantine notebook |
| R-09 | Missing/corrupt checkpoint or data artifact | Separate datasets, hashes, local paths, explicit fallback |
| R-10 | Dataset volume exceeds sprint/session capacity | Inventory early, prototype on AOI/sample, measure storage and runtime |
| R-11 | Kaggle session ends before outputs persist | Save checkpoints/reports and test export/recovery |
| R-12 | Public app attempted inside offline Kaggle | Separate serving host; Kaggle is batch compute only |
| R-13 | Prithvi misused on thermal active-fire data | Keep HLS optical input contract separate and test it |
| R-14 | Forecast uncertainty is uncalibrated | Rolling-origin quantile evaluation; label experimental intervals |
| R-15 | Planning performance estimates presented as measured results | Require benchmark methodology and logs |
| R-16 | Product schema, archive coverage, or terms differ from assumptions | Official source registry and representative sample validation |
| R-17 | UI scope delays the core workflow | Prioritize AOI-to-calendar path; defer optional audit if needed |
| R-18 | AOI geometry/projection bugs | CRS-aware validation, boundary/dateline tests |
| R-19 | External imagery unavailable offline | Pre-stage permitted imagery or show unavailable state |
| R-20 | Secrets/large artifacts enter Git history | Ignore rules, secret scan, separate artifact storage |

## Stop-the-line conditions

Do not launch expensive training when critical data errors remain, required wheel/checkpoint hashes fail, offline quarantine has not passed, temporal leakage is suspected, source target/units/matching rules are undefined, or data/model terms prohibit the planned use.

Review risks after data inventory, first baseline evaluation, before GPU-heavy runs, before demo rehearsal, and before submission. Assign owners to high risks or explicitly record acceptance.