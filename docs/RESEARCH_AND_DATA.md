# Research, Data Sources, and Scientific Governance

## Research rules

Use official product documentation for source semantics, schema, quality flags, temporal coverage, licensing, and limitations. Record URLs, access date, version, sample checks, and which claims each source supports. The selected architecture comes from the supplied project specification; numerical performance estimates must be independently measured.

## Core source: NASA FIRMS - inspected products (issue #6)

The primary inputs are processed georeferenced FIRMS MODIS and VIIRS active-fire
records. The concrete product variants below were verified against the official
NASA Earthdata attribute documentation and the FIRMS archive download page on
**2026-10-22**; re-inspect the pages before relying on this section after any
FIRMS announcement.

- Attribute documentation (official):
  https://www.earthdata.nasa.gov/data/tools/firms/active-fire-data-attributes-modis-viirs
- Archive coverage + access (official): https://firms.modaps.eosdis.nasa.gov/download

### Inspected product: MODIS Collection 6.1 active fire (Terra/Aqua)

| Attribute | Units / semantics (as officially documented) |
|---|---|
| `latitude`, `longitude` | center of 1 km fire pixel (degrees); a detection does not pin the fire's true location |
| `brightness` | channel 21/22 brightness temperature (K) |
| `scan`, `track` | along-scan / along-track pixel size (degrees); pixels grow toward scan edge |
| `acq_date`, `acq_time` | acquisition date (YYYY-MM-DD) and UTC overpass time (HHMM) |
| `satellite` | `A` = Aqua, `T` = Terra |
| `instrument` | MODIS |
| `confidence` | **numeric 0-100%**, assigned low/nominal/high classes (NOT l/n/h labels) |
| `version` | collection + processing source, e.g. `6.1NRT` (near real-time), `6.1` (standard) |
| `bright_t31` | channel 31 brightness temperature (K) |
| `frp` | fire radiative power, **MW** |
| `type` | 0 = vegetation fire, 1 = volcano, 2 = other static land source, 3 = offshore; **MCD14ML standard-quality files only** |
| `daynight` | `D` daytime / `N` nighttime |

Archive coverage (official FIRMS download page, 2026-10-22): **1 November 2000 -
present**. NRT records are later replaced by standard science-quality records
with an approximately 5-month lag.

### Inspected product: VIIRS 375 m active fire (S-NPP / NOAA-20 / NOAA-21)

| Attribute | Units / semantics (as officially documented) |
|---|---|
| `latitude`, `longitude` | center of nominal 375 m fire pixel (degrees) |
| `bright_ti4` | I-4 channel brightness temperature (K); called `Brightness` in web services |
| `scan`, `track` | along-scan / along-track pixel size (degrees) |
| `acq_date`, `acq_time` | acquisition date and UTC overpass time (HHMM) |
| `satellite` | `N` = Suomi NPP, `N20` = NOAA-20, `N21` = NOAA-21 (renamed from `1`/`2`) |
| `instrument` | VIIRS |
| `confidence` | **low/nominal/high labels (`l`/`n`/`h`), NOT numeric**; low-confidence nighttime pixels over the South Atlantic Magnetic Anomaly region are removed from NRT distribution |
| `version` | collection + processing source, e.g. `1.0NRT`, `1.0` |
| `bright_ti5` | I-5 channel brightness temperature (K); called `Brightness_2` in web services |
| `frp` | fire radiative power, **MW**; hybrid 375 m + 750 m retrieval (M13 channel incorporated since 2015) |
| `daynight` | `D` daytime / `N` nighttime |

Archive coverage (official FIRMS download page, 2026-10-22): **S-NPP 20 January
2012 - present; NOAA-20 1 April 2018 - present; NOAA-21 17 January 2024 -
present.**

### Operational notices recorded with the inspection (2026-10-22)

1. **Suomi NPP delivery ends 1-2 November 2026.** The MODIS/VIIRS overlap
calibration in this project therefore relies on a CLOSED historical S-NPP
archive (2012-2026), not an open-ended stream; transition NOAA-20/NOAA-21 data
   in any live demo.
2. NRT records are replaced by standard science-quality data with an
   approximately 5-month lag; the five-pass gate must re-run whenever an
   archive is refreshed.
3. A Collection-2 Vxx14IMGML production bug set the `type` field inconsistently
   for product versions 1 and 2 (fixed May 2025, reprocessed to version 3);
   do not compare `type` distributions across those versions.
4. Archive downloads (CSV/Shapefile/JSON, country yearly summaries) require
   **Earthdata authentication**; the sandboxed CI cannot fetch them, so the
   committed sample evidence below is the reproducible baseline.

### License, attribution, and redistribution

NASA supports full and open sharing of FIRMS data; when providing the data to
third parties, use the following acknowledgment verbatim (official guidance):

> We acknowledge the use of data and/or imagery from NASA's Fire Information
> for Resource Management System (FIRMS) (https://earthdata.nasa.gov/firms),
> part of NASA's Land, Atmosphere Near real-time Capability for Earth
> observations (LANCE) (https://earthdata.nasa.gov/lance) and NASA's Earth
> Science Data and Information System (ESDIS).

FIRMS data are provided "as is"; users bear all responsibility for their use
(the official disclaimer applies). Specific value-added products may carry
Earthdata End User License Agreement conditions (e.g. peaceful,
non-commercial use), so check the EULA page before redistributing derived
products: https://www.earthdata.nasa.gov/engage/open-data-services-software/end-user-license-agreements

### Reproducible sample evidence

`scripts/inventory_firms.py` records what a CSV ACTUALLY contains - SHA-256
hash, row count, observed columns vs. the official attribute tables above,
missingness, temporal coverage, coordinate bounds, confidence semantics, and
FRP statistics - and cross-walks the observed schema against the pipeline
contracts in `src/data/schema.py`. Reports are always written, even when a
file fails (fail-closed, like the five-pass gate):

```bash
python3 scripts/inventory_firms.py \
  --source-dir <firms-archive-dir> \
  --output-dir firms_inventory \
  --accessed-date 2026-10-22
```

Committed evidence from the synthetic sample fixtures lives in
`docs/evidence/firms_inventory/` (`firms_inventory.json` machine-readable,
`firms_inventory.md` human-readable). Facts recorded there that correct easy
mistakes:

- The fixture files carry ALL pipeline-required columns
  (`latitude`, `longitude`, `acq_date`, `acq_time`, `frp`).
- The fixtures deliberately OMIT documented optional attributes (`bright_t31`,
  `type` for MODIS; `bright_ti5` for VIIRS) - no feature is assumed present.
- The synthetic MODIS fixture uses l/n/h LABEL confidence, while real MODIS
  NRT/standard files use NUMERIC 0-100 confidence. The adapter
  (`confidence_ordinal`) maps both, and the inventory reports the observed
  kind so the difference can never be silently ignored.

Do not assume every candidate feature (brightness temperature, scan angle,
view geometry, etc.) exists in a selected CSV. Use only observed fields or
join a documented auxiliary source.

## Matching and harmonization

Document the overlap-period sensors/platforms, temporal tolerance, spatial overlap method, multiple VIIRS detections within a MODIS footprint, duplicate policy, confidence filtering, missing coverage, FRP aggregation, and target units. Evaluate on held-out dates/regions. Matching is subject to selection bias: a matched dataset cannot directly describe fires never detected by one sensor.

## Source registry with verification status

| Source | Role | Verification status (2026-10-22) |
|---|---|---|
| NASA FIRMS MODIS C6.1 active fire archive | Historical observations | **INSPECTED** - attribute table, units, coverage (2000-11-01+), license, attribution recorded above |
| NASA FIRMS VIIRS 375 m archive (S-NPP/NOAA-20/NOAA-21) | Overlap matching and modern observations | **INSPECTED** - attribute table, coverage (2012-01-20+ / 2018-04-01+ / 2024-01-17+), S-NPP end-of-delivery notice recorded |
| MODIS land-cover product (e.g. MCD12Q1) | Optional context feature | CANDIDATE - not downloaded; requires Earthdata access; verify version, class mapping, temporal alignment, terms before use |
| Elevation data | Optional context feature | CANDIDATE - not downloaded; verify resolution, CRS, coverage, resampling |
| HLS Landsat/Sentinel-2 imagery | Optional burn-scar audit (Prithvi) | CANDIDATE - official NASA project page confirmed to exist (earthdata.nasa.gov), but per-AOI tile/date availability NOT verified here; requires Earthdata access. The burn-scar audit remains strictly optional and must not gate any deliverable. |
| Prithvi-EO 2.0 burn-scar checkpoint | Optional segmentation | CANDIDATE - official version, terms, local hash, and input contract must be recorded before any use |

"INSPECTED" means the official documentation was read and transcribed with
access dates; it is NOT proof that a real archive was downloaded. Real-archive
access requires Earthdata authentication and must run through
`scripts/inventory_firms.py` to regenerate `docs/evidence/firms_inventory/`.
Prefer processed FIRMS records over full global raw orbital-swath reconstruction
during the hackathon unless evidence shows the processed approach is insufficient.

## Data products

Keep source archives/manifests, validated normalized records, matched calibration records, daily AOI aggregates, harmonized output, calendar/anomaly/forecast output, and optional optical audit output distinct. Parquet partitioning is a candidate; measure storage/query behavior on a sample before deciding. Do not assume total data volume until inventoried.

## Research questions

- Which product variants and platforms cover the needed dates? **Answered for FIRMS: MODIS C6.1 (2000-11+), S-NPP VIIRS (2012-01+), NOAA-20 (2018-04+), NOAA-21 (2024-01+); S-NPP stream ends 2026-11 (see notices).**
- Is the target summed VIIRS FRP within a MODIS footprint, daily regional FRP, or another explicitly defined quantity? (Answered by the implementation: daily AOI-summed FRP in MW.)
- How are missing coverage and zero detections distinguished? (Answered by the pipeline: per-sensor daily coverage flags.)
- How much discrepancy comes from sensor geometry, thresholds, timing, or actual fire evolution?
- Does the model beat a simple baseline on held-out data? (Measured in committed evaluation artifacts; see docs/DEMO_GUIDE.md.)
- Does PatchTST beat seasonal-naive forecasting under rolling-origin evaluation? (Measured; seasonal-naive remains the documented fallback.)
- Are anomaly thresholds stable across sensor eras and AOI sizes?
- Is suitable HLS imagery available for optional audits? **OPEN - requires Earthdata access; the audit is optional and non-blocking.**

## Provenance

Every result records source version, retrieval identifier/date, hashes where practical, schema and units, filtering/quarantine counts, processing config, code commit, split boundaries, matching rules, model config, metrics, artifact hash, and known limitations. Verify exact scientific citations in the implementation/model report before submission. Planning estimates such as sub-100-ms inference or under-15-minute training are not facts until measured on the actual runtime.