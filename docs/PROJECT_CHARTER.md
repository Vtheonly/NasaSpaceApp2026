# Project Charter

## Identity

- Challenge: Harmonization of MODIS and VIIRS Hot Spots
- Event: NASA Space Apps Challenge 2026
- Product: interactive web application for cross-sensor active-fire harmonization, historical burning calendar, anomaly exploration, and probabilistic regional critical-period forecasts.

## Problem

MODIS and VIIRS differ in pixel footprint, scan geometry, detection sensitivity, overpass timing, and radiometric behavior. Raw counts and FRP are not directly interchangeable. The app should help users explore historical burning in a selected area of interest (AOI), understand normal seasonal patterns, and identify unusual activity relative to a defensible baseline.

## Intended users

Fire and land managers, emergency-response planners, Earth-science researchers/educators, and evaluators. These are intended personas; user research remains a task, not a completed claim.

## Selected solution

1. **Harmonization:** LightGBM gradient-boosted regression with Tweedie objective as the primary candidate, trained on matched MODIS/VIIRS overlap-period records.
2. **Calendar and anomalies:** daily AOI aggregates, seasonal baselines, anomaly scoring, and PatchTST as the temporal-model candidate for probabilistic 14-day regional activity forecasts.
3. **Optical audit:** optional Prithvi-EO 2.0 burn-scar verification using compatible HLS imagery; separate from thermal active-fire harmonization.
4. **Web app:** AOI selection, hotspot map, 52-week-by-year calendar, historical trends, anomaly detail, uncertainty-aware forecast, source provenance, and limitations.
5. **Offline compute:** stage wheels, model weights, and data separately; run internet-disabled quarantine checks before training; serve the UI outside Kaggle.

## Scope

### P0
- FIRMS source inventory, data validation, AOI filtering, and daily aggregation.
- Matched overlap-period calibration data with explicit spatial/temporal rules.
- Baselines and evaluated Tweedie GBDT harmonization.
- Seasonal calendar, anomaly baseline, and forecast evaluation; add PatchTST only with leakage-safe evidence.
- End-to-end UI and versioned artifact contracts.
- Five-pass validation, offline quarantine, reproducible demo, documentation, and official submission review.

### P1
- Optional Prithvi/HLS audit; richer uncertainty display and exportable reports; performance optimization based on measurements.

### P2
- Global high-resolution processing, complex distributed infrastructure, continuous production ingestion, and unrelated features that do not improve the core demo.

## Success criteria

| ID | Outcome | Evidence |
|---|---|---|
| SC-01 | AOI observation exploration works | UI smoke test and source/date labels |
| SC-02 | Daily aggregation is deterministic | Unit tests and audit report |
| SC-03 | Harmonization beats or explains comparison to baselines | Held-out metrics and split manifest |
| SC-04 | Calendar exposes historical seasonality | Calculation tests and demo |
| SC-05 | Anomaly scores are interpretable | Threshold docs and test cases |
| SC-06 | Forecast is evaluated probabilistically | Rolling-origin report and uncertainty metrics |
| SC-07 | Offline dependencies and local model loading work | Quarantine notebook report |
| SC-08 | Clean setup reproduces core demo | Rehearsal checklist |
| SC-09 | Optical audit is kept distinct from active-fire observations | Contract tests and UI labels |

## Scientific boundaries

Do not predict exact ignition locations/times. Do not treat burn scars as active-fire detections. Do not claim a model recovers fires that were never observed unless explicitly modeled and validated. Distinguish missing coverage from zero detections. Planning performance estimates are not results until measured.

## Constraints

The supplied design targets an offline Kaggle accelerator with 96 GB VRAM and 170 GB RAM; inspect actual runtime capabilities. Treat `/kaggle/input/` as read-only and write outputs under `/kaggle/working/`. Keep large data, weights, wheelhouses, environments, and secrets out of GitHub. Verify all source/model terms and official event requirements.

## Delivery order

Freeze requirements and data contracts → validate sample data → implement aggregation → evaluate harmonization baselines/model → implement calendar/anomaly/forecast → connect UI → run offline quarantine and end-to-end tests → verify submission checklist.