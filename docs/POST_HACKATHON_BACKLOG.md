# Post-Hackathon Backlog (issue #20)

Deferred work is captured here so it never blocks the P0 demo or the official
submission checklist. Nothing below is required for the core journey
(AOI -> harmonized history -> calendar -> forecast); items move into GitHub
issues only when someone commits to owning them.

## Priorities

- **D1 (high value, low risk):** re-run the whole pipeline on a verified real
  FIRMS archive for the demo AOI, replacing synthetic-evidence caveats with
  measured results (issue #6 data inventory first).
- **D2 (high value, medium risk):** calibrate forecast intervals on real data
  (conformal or empirical coverage adjustment) and update
  `calibration_status` accordingly.
- **D3 (medium value):** land the Streamlit app (map, 52-week calendar,
  anomaly details, forecast bands) over exported artifacts on a separate host.
- **D4 (medium value):** Prithvi-EO 2.0 burn-scar audit once compatible HLS
  tiles, checkpoint, and local inference are validated (Pillar 3 contract in
  docs/ARCHITECTURE.md).
- **D5 (research):** XGBoost comparison against the LightGBM/Tweedie
  harmonizer on the same splits.
- **D6 (research):** auxiliary error slices via MCD12 land cover and elevation
  joins (docs/MODEL_CONTRACTS.md limitation 3).
- **D7 (research):** AOI-aware uncertainty and region-stratified baselines.

## Separating speculative from validated

| Track | Items | Rule |
|---|---|---|
| Validated direction | five-pass gate, LightGBM/Tweedie harmonizer vs mandatory baselines, seasonal calendar/anomalies, PatchTST candidate with rolling-origin backtests | already has test evidence; extend carefully with the same guards |
| Speculative research | D2 interval calibration, D5 model comparison, D6/D7 slices and uncertainty | require measured evidence before any claim enters docs or the demo |
| Operations | data refresh cadence, hosting costs, dependency updates | see maintenance costs below |

## Maintenance costs (honest accounting)

- **Data refresh:** FIRMS archives change; re-running the five-pass gate plus
  re-training the harmonizer on a mid-size AOI costs minutes of CPU (see
  `scripts/benchmark.py` measurements for the actual environment); interval
  recalibration adds a full backtest pass.
- **Hosting:** the app runs on any small Streamlit-capable host reading
  exported artifacts; no GPU needed. Cost is dominated by the host, not the
  pipeline.
- **Dependency updates:** light dependencies (numpy/pandas/LightGBM). Pin
  bumps require a fresh wheelhouse + quarantine run (notebooks 01-02) before
  adoption; torch is optional and can lag behind without affecting the
  baseline path.
- **Model governance:** every artifact carries hash, code commit, config and
  limitations, so an old artifact can be re-audited without re-training.

## Rules

1. No backlog item may alter the scientific guardrails (baselines before
   claims, leakage guards, explicit missing/estimate labels).
2. Anything measured on real data must update docs/MODEL_CONTRACTS.md and the
   demo evidence, not just the code.
3. A backlog item becomes a tracked issue only with an owner, acceptance
   criteria, and required evidence (see CONTRIBUTING.md issue rules).
