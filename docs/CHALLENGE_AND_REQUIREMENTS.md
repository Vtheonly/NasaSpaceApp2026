# Challenge Requirements and Traceability

## Selected challenge

**Harmonization of MODIS and VIIRS Hot Spots** — NASA Space Apps Challenge 2026, Earth Science / Software.

The challenge calls for a web application that harmonizes active-fire hotspot records into a burning activity calendar, enabling early-warning use and exploration of historical patterns, unusual conditions, and critical periods within a user-selected area of interest (AOI).

## Requirements matrix

| ID | Requirement | Planned implementation | Verification evidence |
|---|---|---|---|
| R-01 | Select an AOI | Map drawing or validated bounding box/polygon | Geometry and UI tests |
| R-02 | Explore historical active-fire observations | FIRMS ingestion and sensor-aware map | Source/date labels and sample comparison |
| R-03 | Harmonize MODIS and VIIRS | Matched overlap-period records; Tweedie gradient-boosted model | Held-out metrics against simple baselines |
| R-04 | Build a continuous historical activity record | Daily AOI aggregation and harmonized series | Coverage, missingness, and aggregation tests |
| R-05 | Show a burning activity calendar | 52-week-by-year view and seasonal summaries | Calendar calculation tests and demo evidence |
| R-06 | Explore historical patterns | Trends, filtering, sensor-era comparison | End-to-end UI test |
| R-07 | Identify unusual conditions | Seasonal baseline and documented anomaly score | Known/synthetic case tests and false-positive review |
| R-08 | Examine critical periods | Probabilistic 14-day regional activity forecast | Rolling-origin backtest and uncertainty metrics |
| R-09 | Avoid false ignition claims | Regional activity wording and explicit scientific limits | UI/copy review |
| R-10 | Expose provenance and uncertainty | Source, date, coverage, model version, observed/estimated/forecast labels | Data-contract and UI tests |
| R-11 | Optional burn-scar verification | HLS optical imagery plus compatible Prithvi checkpoint | Band/input validation and inference smoke test |
| R-12 | Work offline in Kaggle | Staged wheels/checkpoints and air-gapped quarantine notebook | Offline install/import/forward-pass report |
| R-13 | Deliver a functional web application | Separate app host using exported artifacts | Clean setup and demo run |
| R-14 | Reproduce outputs | Versioned configs, source hashes, split manifests, model metrics | Re-run and artifact-integrity checks |
| R-15 | Meet event submission rules | Official final checklist | Verified submission evidence |

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

## Minimum viable demo

1. Select a known AOI.
2. Show source observations with sensor, date, quality/confidence, and FRP context.
3. Compare raw and harmonized daily activity with the sensor transition marked.
4. Inspect the burning calendar and a historical season.
5. Select an anomaly and explain its seasonal baseline.
6. Show the probabilistic 14-day regional activity forecast and uncertainty.
7. Show optional optical verification only if compatible imagery and weights are available; otherwise explain why it is unavailable.
8. Present data provenance, validation evidence, and limitations.

## Official event checks

The selected challenge is known from the project owner's supplied prompt. Before submission, independently verify current official event rules, deadline/time zone, team eligibility, required media and fields, judging criteria, and attribution requirements from official NASA Space Apps materials. Do not invent details absent from the supplied prompt.
