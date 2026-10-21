# Research, Data Sources, and Governance

## 1. Research standard

Use evidence to validate the problem, solution, and claims. Prefer original sources and authoritative publications. For each important claim, capture the source, publication/update date, retrieval date, relevant excerpt or finding, confidence, and how it affects a design decision.

## 2. Data source registry

Create one entry per dataset, API, map layer, model, paper, or other external resource.

| Field | Required record |
|---|---|
| Source ID | Stable identifier, e.g. SRC-001 |
| Name and publisher | Original source/provider |
| URL and access date | Direct source link and retrieval date |
| Purpose | Requirement or feature supported |
| Coverage | Geography, time range, population, resolution, or domain |
| Format and schema | Types, fields, units, CRS, timestamps |
| Update cadence | Static, periodic, or live |
| License/terms | Exact terms and redistribution constraints |
| Attribution | Required credit or citation wording |
| Access | Public, registration, API key, quota, or paid |
| Quality caveats | Missingness, bias, uncertainty, known limitations |
| Processing | Cleaning, filtering, conversion, aggregation |
| Version/provenance | Snapshot, release, query parameters, checksum if practical |
| Fallback | Behavior if the source is unavailable |
| Owner/status | Responsible person and verification status |

Never assume that a publicly downloadable dataset is free to redistribute or use without attribution.

## 3. Data pipeline record

For every nontrivial pipeline, document:
1. Original source and version.
2. Acquisition method and query/filters.
3. Raw versus processed artifact locations.
4. Schema and validation rules.
5. Unit, timezone, and coordinate transformations.
6. Missing values, duplicates, and outlier handling.
7. Any manual corrections and their evidence.
8. Output schema and reproducibility instructions.
9. Tests and known limitations.

Do not commit large raw datasets, personally identifiable information, access tokens, or material with unclear rights. Provide a download/preparation script and checksum or a small permitted fixture when practical.

## 4. NASA and scientific sources

Select sources based on the actual challenge. Potential source categories may include NASA open-data portals, mission datasets, Earth observation products, climate/environmental observations, scientific publications, or government geospatial datasets. These are possibilities, not selected dependencies. Confirm that the source is relevant, accessible, sufficiently documented, and permitted by its terms.

## 5. Research log

For each experiment or research question, record:
- Question and hypothesis.
- Method and sources.
- Environment and versions.
- Result and evidence.
- Interpretation and uncertainty.
- Decision made or next action.
- Reproduction steps.

## 6. Responsible claims

Distinguish measured outcomes from hypotheses and illustrative examples. Disclose resolution limits, uncertainty, geographic/time coverage, model limitations, and where expert review is needed. Do not imply that a hackathon prototype replaces official emergency, medical, scientific, or policy decision-making.
