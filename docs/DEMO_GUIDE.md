# Scientific Demo Guide (issue #16)

This guide lets another person reproduce the core workflow end-to-end from a
clean setup, using only committed scripts and clearly labeled evidence. Every
claim below names the artifact that proves it; nothing is presented as a real
measurement unless it was produced from a verified FIRMS archive.

## 0. Clean setup

```bash
git clone https://github.com/Vtheonly/NasaSpaceApp2026.git
cd NasaSpaceApp2026
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,forecast]"   # torch CPU: --index-url https://download.pytorch.org/whl/cpu
pytest tests                       # must be green before any demo claim
```

CI runs the same suite with and without the optional `forecast` extra, so the
demo works on a machine without PyTorch (the forecast degrades to the
seasonal-naive baseline with a printed warning - never a silent failure).

## 1. One-command demo (rehearsal)

```bash
python scripts/run_demo.py --output-dir out/demo
```

This executes the full pipeline on the committed **SYNTHETIC** sample fixtures
(seed 2026) and writes:

| File | Content |
|---|---|
| `out/demo/DEMO_REPORT.md` | Human-readable demo script: raw vs harmonized series, sensor-era transition, harmonizer vs baselines, anomaly counts, forecast state, limitations |
| `out/demo/demo_summary.json` | Machine-readable version of every number above |
| `out/demo/pipeline/pipeline_run.json` | Stage-by-stage statuses, exit codes, timings |
| `out/demo/pipeline/**` | Gate reports, versioned model artifacts, calendar/forecast records |

The fixture run is the **fallback path**: it is always reproducible and is
explicitly labeled SYNTHETIC so it can never be mistaken for real evidence.

## 2. Demo on real FIRMS data

```bash
python scripts/run_demo.py \
    --source-dir <verified_firms_csv_dir> \
    --output-dir out/demo_real \
    --aoi "bbox:-123.5,39.5,-120.5,42.0"
```

The five-pass gate runs first and fails closed: any structural/semantic
critical error aborts the demo with machine-readable reports before a model is
ever trained. Only a gate-passing archive produces a demo report whose source
label reads "user source directory ... (five-pass gate applied)".

## 3. Narrated walkthrough (what to say, what to show)

1. **AOI and data** - "The user selects an area of interest; we ingest FIRMS
   MODIS C6.x and VIIRS 375 m active-fire records and validate them through a
   five-pass gate."
   Show: `pipeline/gate/inventory_manifest.json` (file hashes, row counts) and
   `dataset_audit_report.json` (quarantine reason codes).
2. **Raw vs harmonized** - "MODIS and VIIRS are not interchangeable; we train
   a LightGBM/Tweedie harmonizer on overlap-period matches so the pre-VIIRS
   era can be expressed on a VIIRS-equivalent scale. These are model
   estimates, not ground truth."
   Show: DEMO_REPORT section 1 (observed vs harmonized day counts, mean FRP)
   and the sensor-era transition date.
3. **Model quality, honestly** - "On held-out chronological splits the
   learned model is compared with identity and calibration baselines."
   Show: DEMO_REPORT section 2 table (evidence:
   `pipeline/harmonization/harmonization_evaluation.json`).
4. **Calendar and anomalies** - "Seasonal climatology defines the baseline;
   anomaly statuses are relative-to-season descriptions."
   Show: DEMO_REPORT section 3 counts; point out `missing_coverage` is never
   rendered as zero.
5. **Probabilistic forecast** - "A 14-day regional activity forecast with
   experimental quantile bands; intervals are uncalibrated until measured on
   real data, and we never claim ignition coordinates or times."
   Show: DEMO_REPORT section 4 and the forecast rows of
   `pipeline/inference/calendar_forecast_records.csv`.
6. **Limitations** - read section 5 verbatim before Q&A.

## 4. Failure rehearsal (do at least once before the live demo)

| Injected failure | Command | Expected explicit behavior |
|---|---|---|
| Dirty source data | copy a `tests/fixtures/raw_dirty/*.csv` into a source dir, run demo against it | Gate fails closed; reports written; no model trained; demo refuses to render a report |
| No forecast artifact | `--forecaster none` | Calendar export still succeeds; forecast columns stay NaN (explicit no-forecast state) |
| Corrupt model artifact | flip one byte in `model.joblib`, rerun inference | `ArtifactIntegrityError` naming the file; no silent use of a corrupted model |
| No PyTorch | uninstall torch (or skip the extra) | PatchTST skipped with a printed warning; seasonal-naive forecast still produced |

## 5. Optional Prithvi optical verification

Included only when HLS imagery, the correct Prithvi-EO 2.0 burn-scar
checkpoint (hash-verified), and local inference are all ready. It is a
separate branch of the workflow and is never required for the core demo; if
any input is unavailable the demo shows an explicit "unavailable" state
instead of pretending to verify anything.

## 6. Evidence checklist before presenting

- [ ] `pytest tests` green on this machine (record the count and command).
- [ ] DEMO_REPORT.md generated from this session and its numbers match
      `demo_summary.json`.
- [ ] Source label matches reality (SYNTHETIC fixture vs gate-passed archive).
- [ ] Every number you plan to say aloud appears in DEMO_REPORT.md or a file it
      links to.
- [ ] Limitations block read at least once during the talk.
