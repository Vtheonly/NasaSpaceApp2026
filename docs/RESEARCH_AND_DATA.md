# Research, Data Sources, and Scientific Governance

## Research rules

Use official product documentation for source semantics, schema, quality flags, temporal coverage, licensing, and limitations. Record URLs, access date, version, sample checks, and which claims each source supports. The selected architecture comes from the supplied project specification; numerical performance estimates must be independently measured.

## Core source: NASA FIRMS

The intended primary inputs are processed georeferenced FIRMS MODIS and VIIRS active-fire records, including historical archives where available. Before selecting concrete product variants, verify the current official FIRMS catalog and archive access. Record sensor/platform, time range, spatial footprint caveats, columns, units, quality/confidence flags, missing-value conventions, formats, coverage, access limits, terms, and attribution.

Do not assume every candidate feature (brightness temperature, scan angle, view geometry, etc.) exists in a selected CSV. Use only observed fields or join a documented auxiliary source.

## Matching and harmonization

Document the overlap-period sensors/platforms, temporal tolerance, spatial overlap method, multiple VIIRS detections within a MODIS footprint, duplicate policy, confidence filtering, missing coverage, FRP aggregation, and target units. Evaluate on held-out dates/regions. Matching is subject to selection bias: a matched dataset cannot directly describe fires never detected by one sensor.

## Candidate source registry

| Source | Role | Validation |
|---|---|---|
| NASA FIRMS MODIS archive | Historical observations | Product/platform, schema, dates, quality flags, terms |
| NASA FIRMS VIIRS archive | Overlap matching and modern observations | Product variant, coverage, schema, terms |
| MODIS land-cover product (e.g. MCD12Q1) | Optional context feature | Version, class mapping, temporal alignment, terms |
| Elevation data | Optional context feature | Resolution, CRS, coverage, resampling |
| HLS Landsat/Sentinel-2 imagery | Optional burn-scar audit | Bands, date availability, nodata, license |
| Prithvi-EO 2.0 burn-scar checkpoint | Optional segmentation | Official version, terms, local hash, input contract |

These are candidates to validate, not proof that the data have been downloaded. Prefer processed FIRMS records over full global raw orbital-swath reconstruction during the hackathon unless evidence shows the processed approach is insufficient.

## Data products

Keep source archives/manifests, validated normalized records, matched calibration records, daily AOI aggregates, harmonized output, calendar/anomaly/forecast output, and optional optical audit output distinct. Parquet partitioning is a candidate; measure storage/query behavior on a sample before deciding. Do not assume total data volume until inventoried.

## Research questions

- Which product variants and platforms cover the needed dates?
- Is the target summed VIIRS FRP within a MODIS footprint, daily regional FRP, or another explicitly defined quantity?
- How are missing coverage and zero detections distinguished?
- How much discrepancy comes from sensor geometry, thresholds, timing, or actual fire evolution?
- Does the model beat a simple baseline on held-out data?
- Does PatchTST beat seasonal-naive forecasting under rolling-origin evaluation?
- Are anomaly thresholds stable across sensor eras and AOI sizes?
- Is suitable HLS imagery available for optional audits?

## Provenance

Every result records source version, retrieval identifier/date, hashes where practical, schema and units, filtering/quarantine counts, processing config, code commit, split boundaries, matching rules, model config, metrics, artifact hash, and known limitations. Verify exact scientific citations in the implementation/model report before submission. Planning estimates such as sub-100-ms inference or under-15-minute training are not facts until measured on the actual runtime.