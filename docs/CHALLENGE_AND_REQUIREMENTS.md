# Challenge Requirements and Traceability

## Selected challenge

**Harmonization of MODIS and VIIRS Hot Spots** — NASA Space Apps Challenge 2026, Earth Science / Software.

The challenge calls for a web application that harmonizes active-fire hotspot records into a burning activity calendar, enabling early-warning use and exploration of historical patterns, unusual conditions, and critical periods within a user-selected area of interest (AOI).

## Requirements matrix

Status legend: **done** = implemented + tested + demo evidence; **partial** =
pipeline done, UI view pending (#14); **deferred** = tracked in
docs/POST_HACKATHON_BACKLOG.md; **event** = submission-time task
(docs/SUBMISSION_CHECKLIST.md).

| ID | Requirement | Implementation (actual modules) | Verification evidence (tests/demo) | Status |
|---|---|---|---|---|
| R-01 | Select an AOI | `src/geo/aoi.py` (bbox/polygon, validated); `--aoi` on every CLI | tests/unit/test_aoi.py; demo guide step 1 | done |
| R-02 | Explore historical active-fire observations | FIRMS ingestion (`src/data/adapters.py`), five-pass gate, daily aggregation; map view pending | tests/unit/test_adapters.py, test_gate.py, test_aggregation.py; gate reports in demo | partial (map in #14) |
| R-03 | Harmonize MODIS and VIIRS | matched overlap records + LightGBM/Tweedie vs identity/calibration baselines (`src/geo/matching.py`, `src/harmonization/`) | tests/scientific/test_beats_baselines.py, test_leakage.py; DEMO_REPORT section 2 | done |
| R-04 | Build a continuous historical activity record | daily AOI aggregation + harmonized series with explicit value_type labels (`src/data/aggregation.py`, `src/harmonization/inference.py`) | tests/unit/test_aggregation.py; DEMO_REPORT section 1 (sensor-era transition) | done |
| R-05 | Show a burning activity calendar | seasonal climatology + calendar records (`src/calendar/seasonal.py`, `records.py`); 52-week UI view pending | tests/unit/test_seasonal_anomalies.py, test_records.py; DEMO_REPORT section 3 | partial (calendar view in #14) |
| R-06 | Explore historical patterns | trends/sensor-era comparison in exported records; interactive filtering pending | pipeline_run.json + demo_summary.json comparisons | partial (#14) |
| R-07 | Identify unusual conditions | seasonal baseline + ratio anomaly score with explicit statuses (`src/calendar/anomalies.py`) | tests/unit/test_seasonal_anomalies.py; DEMO_REPORT section 3 counts | done |
| R-08 | Examine critical periods | 14-day probabilistic forecast, quantile bands, rolling-origin backtests (`src/calendar/forecast.py`, `src/models/patchtst.py`) | tests/unit/test_forecast.py, test_patchtst.py; DEMO_REPORT section 4 | done |
| R-09 | Avoid false ignition claims | regional-only wording enforced in docs/contracts/limitations block | MODEL_CONTRACTS section 6; DEMO_REPORT section 5; UI copy review at #14 | done (copy review with app) |
| R-10 | Expose provenance and uncertainty | manifests with hashes/code commit/config; value_type/quality_state/calibration_status labels (`src/artifacts/`) | tests/unit/test_artifacts.py, test_records.py; manifest in demo export | done |
| R-11 | Optional burn-scar verification | not implemented by design (separate optional branch; never blocks the core) | MODEL_CONTRACTS section 9 assumption 6; explicit unavailable state contract | deferred (backlog D4) |
| R-12 | Work offline in Kaggle | staged wheelhouse + quarantine notebook; no network imports; hash-verified artifacts | tests/offline/test_offline_safety.py; notebooks 01-02 | done |
| R-13 | Deliver a functional web application | app reads exported artifacts only (planned Streamlit host) | run_pipeline/run_demo provide the full offline journey today | deferred (backlog D3 / #14) |
| R-14 | Reproduce outputs | seeds, versioned configs, hash-checked artifacts, deterministic ordering | tests/e2e/test_full_pipeline.py; determinism tests; demo rerun | done |
| R-15 | Meet event submission rules | official checklist with sourced event facts (verified 2026-10-22) | docs/SUBMISSION_CHECKLIST.md | event (team-owned) |

Real-data status (verified 2026-10-20/21): official FIRMS NRT MODIS C6.1 and
VIIRS S-NPP files pass the five-pass gate with zero quarantined rows (see
docs/DEPENDENCY_AND_SECURITY_AUDIT.md); multi-year archive validation remains
open (issue #6) and no synthetic result is presented as a real measurement.

## Scientific acceptance rules

- Compare harmonization against identity/no-correction and simple calibration baselines.
- Document spatial matching, temporal tolerance, duplicate handling, FRP aggregation, and quality filtering.
- Use chronological splits and prevent matched events or overlapping windows from leaking across partitions.
- Report bias and error across FRP ranges, regions, seasons, and sensor geometry where sample size supports it.
- A regression model cannot recover fires never detected by MODIS unless that detection limitation is explicitly modeled and validated.
- Distinguish missing coverage from zero observed detections.
- Compare PatchTST with seasonal-naive and persistence forecasts using rolling-origin backtests.
- Report quantile coverage/calibration; label uncalibrated intervals as experimental.
- Prithvi burn-scar output is independent optical evidence, not an active-fire detection or proof of exact ignition time.

## AOI user workflow and measurable acceptance criteria (issue #2)

Primary journey (fire managers, emergency planners, researchers, evaluators):

1. **Select an AOI** - draw/enter a validated bbox or polygon.
   Measurable: invalid geometry is rejected with a message; a valid AOI yields
   an `aoi_id` recorded on every artifact row (tests/unit/test_aoi.py).
2. **Inspect historical activity** - daily series with sensor, date, coverage,
   and confidence context.
   Measurable: `days_observed`/`days_missing_coverage` counts in
   `summarize_daily`; missing days never carry values (aggregation tests;
   DEMO_REPORT section 1).
3. **Compare raw and harmonized series** - sensor-era transition marked;
   harmonized days labeled `harmonized_estimate`, never "observed".
   Measurable: value_type counts + transition date in demo_summary.json;
   `calibration_status` shown with every forecast.
4. **Read the calendar/anomalies** - seasonal baseline with explicit
   `insufficient_baseline` and `missing_coverage` states.
   Measurable: anomaly status counts; statuses are relative-to-season only
   (tests/unit/test_seasonal_anomalies.py).
5. **Review the 14-day probabilistic forecast** - quantile bands with issue
   date, training cutoff, and uncalibrated label; never ignition claims.
   Measurable: q10 <= q50 <= q90, horizon = 14 rows of quality_state
   'forecast' (tests/unit/test_forecast.py; DEMO_REPORT section 4).
6. **Check provenance/limitations** - manifest (hashes, code commit, config,
   metrics) reachable from every screen; limitations text shown in-app.
   Measurable: manifest keys validated on load (tests/unit/test_artifacts.py).

The interactive views (map, 52-week calendar, filters) inherit these exact
labels/limits from the exported records when the app lands (#14); the
product language must never claim deterministic ignition prediction.

## Minimum viable demo

1. Select a known AOI.
2. Show source observations with sensor, date, quality/confidence, and FRP context.
3. Compare raw and harmonized daily activity with the sensor transition marked.
4. Inspect the burning calendar and a historical season.
5. Select an anomaly and explain its seasonal baseline.
6. Show the probabilistic 14-day regional activity forecast and uncertainty.
7. Show optional optical verification only if compatible imagery and weights are available; otherwise explain why it is unavailable.
8. Present data provenance, validation evidence, and limitations.

## Official event checks (verified 2026-10-22; issues #1 and #17)

Verified from official materials and recorded with sources in
[SUBMISSION_CHECKLIST.md](SUBMISSION_CHECKLIST.md):

- 2026 event dates **November 14-15, 2026**; submission window closes
  **Nov 15, 11:59 PM local time** via the team page Project tab.
- Teams of **up to six** members, all in the same Local Event (or Universal
  Event) for global judging eligibility.
- Judging criteria: Impact, Creativity, Validity, Relevance, Presentation.
- Participation terms (updated Aug 7, 2026) apply; see the checklist for the
  full list and blockers.

The repo consistently names the selected challenge - **Harmonization of
MODIS and VIIRS Hot Spots** - and no task asks the team to switch. Challenge
statement wording is re-checked against the official 2026 challenge page
before submission (recorded as a checklist blocker, not assumed).
