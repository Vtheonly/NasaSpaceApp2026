# FIRMS sample inventory (issue #6 evidence)

- Generated: 2026-10-21T23:01:27+00:00 (UTC)
- Schema source: https://www.earthdata.nasa.gov/data/tools/firms/active-fire-data-attributes-modis-viirs (accessed 2026-10-22)
- Coverage source: https://firms.modaps.eosdis.nasa.gov/download (accessed 2026-10-22)
- Files inventoried: 2

## modis_active_fires.csv (modis)

| Check | Value |
|---|---|
| SHA-256 | `eaa68829768eebbebd970a1d0e9eb30e412b7dfd4a7b890e1f2bfa30227e4db4` |
| Size | 163133 bytes |
| Rows | 1678 (0 duplicate) |
| Acquisition dates | 2018-01-28 .. 2024-12-09 (0 unparseable datetimes) |
| Coordinates | lat 39.502075..41.99938, lon -123.498213..-120.503097 (0 out of bounds) |
| Confidence | label (l/n/h) (h: 146, l: 528, n: 1004) |
| FRP (MW) | min 0.47, median 9.44, mean 13.94907, max 131.03 (0 negative) |
| Documented attributes absent | bright_t31, type |
| Undocumented extra columns | none |
| Pipeline required columns present | yes |

## viirs_active_fires.csv (viirs)

| Check | Value |
|---|---|
| SHA-256 | `633f4f55a343889d46d639a7b80e2e3a43ca5d4e01b8b4b54fab152060a2322c` |
| Size | 212348 bytes |
| Rows | 2607 (0 duplicate) |
| Acquisition dates | 2018-01-28 .. 2024-12-09 (0 unparseable datetimes) |
| Coordinates | lat 39.4991..42.0089, lon -123.4994..-120.4996 (0 out of bounds) |
| Confidence | label (l/n/h) (h: 550, n: 2057) |
| FRP (MW) | min 0.1, median 7.91, mean 10.519582, max 80.67 (0 negative) |
| Documented attributes absent | bright_ti5 |
| Undocumented extra columns | none |
| Pipeline required columns present | yes |

## Official notices recorded with this inventory

- Delivery of Suomi NPP satellite observations will cease on 1-2 November 2026; users should transition to NOAA-20/NOAA-21 products (official FIRMS notification, firms.modaps.eosdis.nasa.gov, inspected 2026-10-22). The MODIS/VIIRS overlap calibration in this project therefore relies on a historical S-NPP archive that is closed, not open-ended.
- NRT data are replaced with standard science-quality data as they become available, with an approximately 5-month lag (official FIRMS archive download page, inspected 2026-10-22). Archive downloads require Earthdata authentication.
- A Collection-2 Vxx14IMGML monthly fire-location production bug set the 'type' field inconsistently for product versions 1 and 2; corrected in May 2025 and reprocessed to version 3 (official FIRMS what's-new notice).

## Attribution

We acknowledge the use of data and/or imagery from NASA's Fire Information for Resource Management System (FIRMS) (https://earthdata.nasa.gov/firms), part of NASA's Land, Atmosphere Near real-time Capability for Earth observations (LANCE) (https://earthdata.nasa.gov/lance) and NASA's Earth Science Data and Information System (ESDIS).
