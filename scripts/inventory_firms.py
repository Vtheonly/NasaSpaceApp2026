#!/usr/bin/env python3
"""Inventory FIRMS active-fire CSVs against the OFFICIAL product schemas (issue #6).

For every MODIS/VIIRS CSV this tool records what the file ACTUALLY contains -
SHA-256 hash, row counts, observed columns vs. the officially documented
attributes, per-column missingness, temporal coverage, coordinate bounds,
confidence semantics, FRP statistics, satellite/version/daynight values -
and cross-walks the observed schema against the pipeline contracts
(``src/data/schema.py``). Outputs:

- ``firms_inventory.json``  machine-readable inventory (issue #6: "record
  hashes, row counts, and schema");
- ``firms_inventory.md``    human-readable evidence table grounded in the
  official attribute documentation.

The official attribute tables, archive coverage statements, and
license/attribution text below were transcribed from:

- NASA Earthdata, "Active Fire Data Attributes for MODIS and VIIRS"
  https://www.earthdata.nasa.gov/data/tools/firms/active-fire-data-attributes-modis-viirs
- NASA FIRMS archive download page (temporal coverage + Earthdata login)
  https://firms.modaps.eosdis.nasa.gov/download
- NASA Earthdata FIRMS acknowledgment guidance
  https://www.earthdata.nasa.gov/data/tools/firms

Record the access date in the produced inventory; do not silently update
these tables without re-inspecting the pages (issue #6 acceptance: "No
feature is assumed to exist if the chosen source does not provide it").
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from datetime import UTC, date, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd  # noqa: E402

from src.artifacts.serialization import save_json  # noqa: E402
from src.common.logging import configure_logging, get_logger  # noqa: E402
from src.data.schema import (  # noqa: E402
    CONFIDENCE_ORDINAL,
    OPTIONAL_COLUMNS,
    REQUIRED_COLUMNS,
    parse_acq_datetime,
)

LOGGER = get_logger(__name__)

INVENTORY_SCHEMA_VERSION = 1

#: Official NRT MODIS active-fire attributes (Earthdata attribute page).
OFFICIAL_MODIS_ATTRIBUTES: dict[str, str] = {
    "latitude": "center of 1 km fire pixel (degrees)",
    "longitude": "center of 1 km fire pixel (degrees)",
    "brightness": "channel 21/22 brightness temperature (K)",
    "scan": "along-scan pixel size (degrees)",
    "track": "along-track pixel size (degrees)",
    "acq_date": "acquisition date (YYYY-MM-DD)",
    "acq_time": "acquisition time (HHMM UTC)",
    "satellite": "A = Aqua, T = Terra",
    "instrument": "MODIS",
    "confidence": "0-100% numeric (low/nominal/high classes)",
    "version": "collection + processing source, e.g. '6.1NRT', '6.1'",
    "bright_t31": "channel 31 brightness temperature (K)",
    "frp": "fire radiative power (MW)",
    "daynight": "D = daytime fire, N = nighttime fire",
    "type": "0=vegetation fire, 1=volcano, 2=other static land source, 3=offshore; "
    "MCD14ML standard-quality data ONLY",
}

#: Official NRT VIIRS 375 m active-fire attributes (Earthdata attribute page).
OFFICIAL_VIIRS_ATTRIBUTES: dict[str, str] = {
    "latitude": "center of nominal 375 m fire pixel (degrees)",
    "longitude": "center of nominal 375 m fire pixel (degrees)",
    "bright_ti4": "VIIRS I-4 channel brightness temperature (K); "
    "called 'Brightness' in web services",
    "scan": "along-scan pixel size (degrees)",
    "track": "along-track pixel size (degrees)",
    "acq_date": "acquisition date (YYYY-MM-DD)",
    "acq_time": "acquisition time (HHMM UTC)",
    "satellite": "N = Suomi NPP, N20 = NOAA-20, N21 = NOAA-21",
    "instrument": "VIIRS",
    "confidence": "low/nominal/high labels (l/n/h); NOT numeric",
    "version": "collection + processing source, e.g. '1.0NRT', '1.0'",
    "bright_ti5": "VIIRS I-5 channel brightness temperature (K); "
    "called 'Brightness_2' in web services",
    "frp": "fire radiative power (MW); hybrid 375 m + 750 m retrieval",
    "daynight": "D = daytime fire, N = nighttime fire",
}

#: Archive coverage per product, as stated on the official FIRMS download page
#: (https://firms.modaps.eosdis.nasa.gov/download, inspected 2026-10-22).
OFFICIAL_ARCHIVE_COVERAGE: dict[str, dict[str, str]] = {
    "modis": {
        "product": "MODIS Collection 6.1 (Terra/Aqua) active fire archive",
        "temporal_coverage": "1 November 2000 - present",
        "confidence_semantics": "numeric 0-100% assigned low/nominal/high classes",
    },
    "viirs": {
        "product": "VIIRS 375 m active fire archive (S-NPP / NOAA-20 / NOAA-21)",
        "temporal_coverage": "S-NPP: 20 January 2012 - present; "
        "NOAA-20: 1 April 2018 - present; NOAA-21: 17 January 2024 - present",
        "confidence_semantics": "low/nominal/high labels; low-confidence "
        "nighttime pixels over the South Atlantic Magnetic Anomaly region are "
        "removed from NRT distribution",
    },
}

#: Operationally critical official notices that affect harmonization planning.
OFFICIAL_NOTICES: list[str] = [
    "Delivery of Suomi NPP satellite observations will cease on 1-2 November "
    "2026; users should transition to NOAA-20/NOAA-21 products (official FIRMS "
    "notification, firms.modaps.eosdis.nasa.gov, inspected 2026-10-22). The "
    "MODIS/VIIRS overlap calibration in this project therefore relies on a "
    "historical S-NPP archive that is closed, not open-ended.",
    "NRT data are replaced with standard science-quality data as they become "
    "available, with an approximately 5-month lag (official FIRMS archive "
    "download page, inspected 2026-10-22). Archive downloads require Earthdata "
    "authentication.",
    "A Collection-2 Vxx14IMGML monthly fire-location production bug set the "
    "'type' field inconsistently for product versions 1 and 2; corrected in "
    "May 2025 and reprocessed to version 3 (official FIRMS what's-new notice).",
]

#: Official attribution required when FIRMS data is shared with third parties
#: (verbatim from https://www.earthdata.nasa.gov/data/tools/firms).
OFFICIAL_ACKNOWLEDGMENT = (
    "We acknowledge the use of data and/or imagery from NASA's Fire "
    "Information for Resource Management System (FIRMS) "
    "(https://earthdata.nasa.gov/firms), part of NASA's Land, Atmosphere Near "
    "real-time Capability for Earth observations (LANCE) "
    "(https://earthdata.nasa.gov/lance) and NASA's Earth Science Data and "
    "Information System (ESDIS)."
)

SCHEMA_SOURCE_URL = (
    "https://www.earthdata.nasa.gov/data/tools/firms/"
    "active-fire-data-attributes-modis-viirs"
)
COVERAGE_SOURCE_URL = "https://firms.modaps.eosdis.nasa.gov/download"

#: Filename hints for product detection (case-insensitive substring).
_FILENAME_HINTS = {"modis": ("modis", "mcd14"), "viirs": ("viirs", "vnp14")}

_TEXT_COLUMNS = (
    "satellite",
    "instrument",
    "confidence",
    "version",
    "daynight",
)


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    """Stream-hash a file so multi-GB archives never load into memory."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def detect_product(path: Path, columns: list[str]) -> str | None:
    """Detect MODIS vs VIIRS from the filename, then column fingerprints.

    Returns None when neither signal is conclusive (caller must decide whether
    to abort - we never guess silently).
    """
    name = path.name.lower()
    for product, hints in _FILENAME_HINTS.items():
        if any(h in name for h in hints):
            return product
    lower = {c.lower() for c in columns}
    if "bright_ti4" in lower or "bright_ti5" in lower:
        return "viirs"
    if "brightness" in lower and ("bright_t31" in lower or "type" in lower):
        return "modis"
    if "brightness" in lower:
        return "modis"  # `brightness` (not bright_ti4) is the MODIS column name
    return None


def _column_profile(series: pd.Series) -> dict:
    """Missingness + dtype + value stats for one column."""
    missing = int(series.isna().sum())
    profile: dict = {
        "dtype": str(series.dtype),
        "present_count": int(series.notna().sum()),
        "missing_count": missing,
        "missing_rate": round(float(missing) / len(series) if len(series) else 0.0, 6),
    }
    if series.name in _TEXT_COLUMNS or series.dtype == object:
        values = series.dropna().astype(str)
        counts = values.value_counts()
        profile["value_counts"] = {str(k): int(v) for k, v in counts.items()}
    else:
        numeric = pd.to_numeric(series, errors="coerce")
        profile.update(
            {
                "min": _num(numeric.min()),
                "median": _num(numeric.median()),
                "mean": _round(float(numeric.mean())) if numeric.notna().any() else None,
                "max": _num(numeric.max()),
                "non_numeric_count": int(series.notna().sum() - numeric.notna().sum()),
            }
        )
    return profile


def _num(value) -> float | None:
    if value is None or pd.isna(value):
        return None
    return _round(float(value))


def _round(value: float) -> float:
    return round(float(value), 6)


def _confidence_kind(series: pd.Series) -> str:
    values = series.dropna()
    if values.empty:
        return "missing"
    labels = values.astype(str).str.strip().str.lower().isin(CONFIDENCE_ORDINAL)
    if labels.all():
        return "label (l/n/h)"
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().all() and numeric.between(0, 100).all():
        return "numeric (0-100)"
    return "mixed/unmapped"


def inventory_file(path: Path) -> dict:
    """Profile one FIRMS CSV: hash, rows, schema vs official, data quality."""
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    columns = list(frame.columns)
    product = detect_product(path, columns)
    if product is None:
        raise ValueError(
            f"Could not detect MODIS vs VIIRS for {path.name} (no filename hint "
            "and no bright_ti4/brightness fingerprint); refusing to guess."
        )
    official = (
        OFFICIAL_MODIS_ATTRIBUTES if product == "modis" else OFFICIAL_VIIRS_ATTRIBUTES
    )
    observed_lower = {c.lower(): c for c in columns}
    documented_absent = [c for c in official if c not in observed_lower]
    undocumented_present = [
        c for c in columns if c.lower() not in official and c.lower() not in ("type",)
    ]

    inventory: dict = {
        "file": path.name,
        "path": str(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "product": product,
        "row_count": int(len(frame)),
        "duplicate_row_count": int(frame.duplicated().sum()),
        "columns": {
            "observed": columns,
            "documented_but_absent": documented_absent,
            "undocumented_but_present": undocumented_present,
            "profiles": {
                c: _column_profile(frame[c]) for c in columns
            },
        },
    }

    # Temporal coverage + coordinate bounds (use the pipeline's own parser so
    # the inventory cannot drift from what the five-pass gate accepts).
    if "acq_date" in frame.columns and "acq_time" in frame.columns:
        datetimes = [
            parse_acq_datetime(d, t)
            for d, t in zip(frame["acq_date"], frame["acq_time"], strict=False)
        ]
    else:
        datetimes = []
    valid_ts = [t for t in datetimes if t is not None]
    inventory["temporal"] = {
        "acq_date_min": str(min(frame["acq_date"].dropna())) if "acq_date" in frame.columns and frame["acq_date"].notna().any() else None,
        "acq_date_max": str(max(frame["acq_date"].dropna())) if "acq_date" in frame.columns and frame["acq_date"].notna().any() else None,
        "parseable_datetime_count": len(valid_ts),
        "unparseable_datetime_count": len(datetimes) - len(valid_ts),
    }

    if "latitude" in frame.columns and "longitude" in frame.columns:
        lat = pd.to_numeric(frame["latitude"], errors="coerce")
        lon = pd.to_numeric(frame["longitude"], errors="coerce")
        inventory["coordinates"] = {
            "lat_min": _num(lat.min()),
            "lat_max": _num(lat.max()),
            "lon_min": _num(lon.min()),
            "lon_max": _num(lon.max()),
            "lat_out_of_bounds_count": int(((lat < -90) | (lat > 90)).sum()),
            "lon_out_of_bounds_count": int(((lon < -180) | (lon > 180)).sum()),
            "missing_count": int(lat.isna().sum() + lon.isna().sum()),
        }

    if "confidence" in frame.columns:
        inventory["confidence"] = {
            "kind": _confidence_kind(frame["confidence"]),
            "distribution": {
                str(k): int(v)
                for k, v in frame["confidence"].value_counts(dropna=False).items()
            },
        }

    if "frp" in frame.columns:
        frp = pd.to_numeric(frame["frp"], errors="coerce")
        inventory["frp_mw"] = {
            "missing_count": int(frp.isna().sum()),
            "negative_count": int((frp < 0).sum()),
            "min": _num(frp.min()),
            "median": _num(frp.median()),
            "mean": _round(float(frp.mean())) if frp.notna().any() else None,
            "max": _num(frp.max()),
        }

    # Cross-walk against pipeline contracts so training can never assume a
    # feature this archive does not actually provide.
    inventory["pipeline_contract"] = {
        "required_columns": {
            col: col in observed_lower for col in REQUIRED_COLUMNS
        },
        "optional_columns": {
            col: col in observed_lower for col in OPTIONAL_COLUMNS
        },
        "all_required_present": all(
            col in observed_lower for col in REQUIRED_COLUMNS
        ),
    }
    return inventory


def render_markdown(inventory: dict) -> str:
    """Human-readable evidence table for docs/RESEARCH_AND_DATA.md."""
    lines = [
        "# FIRMS sample inventory (issue #6 evidence)",
        "",
        f"- Generated: {inventory['generated_at']} (UTC)",
        f"- Schema source: {SCHEMA_SOURCE_URL} (accessed {inventory['official_sources']['schema_access_date']})",
        f"- Coverage source: {COVERAGE_SOURCE_URL} (accessed {inventory['official_sources']['schema_access_date']})",
        f"- Files inventoried: {len(inventory['files'])}",
        "",
    ]
    for file_inv in inventory["files"]:
        lines += [
            f"## {file_inv['file']} ({file_inv['product']})",
            "",
            "| Check | Value |",
            "|---|---|",
            f"| SHA-256 | `{file_inv['sha256']}` |",
            f"| Size | {file_inv['size_bytes']} bytes |",
            f"| Rows | {file_inv['row_count']} ({file_inv['duplicate_row_count']} duplicate) |",
            f"| Acquisition dates | {file_inv['temporal']['acq_date_min']} .. "
            f"{file_inv['temporal']['acq_date_max']} "
            f"({file_inv['temporal']['unparseable_datetime_count']} unparseable datetimes) |",
        ]
        if "coordinates" in file_inv:
            c = file_inv["coordinates"]
            lines.append(
                f"| Coordinates | lat {c['lat_min']}..{c['lat_max']}, "
                f"lon {c['lon_min']}..{c['lon_max']} "
                f"({c['lat_out_of_bounds_count'] + c['lon_out_of_bounds_count']} "
                "out of bounds) |"
            )
        if "confidence" in file_inv:
            dist = ", ".join(
                f"{k}: {v}"
                for k, v in sorted(file_inv["confidence"]["distribution"].items())
            )
            lines.append(
                f"| Confidence | {file_inv['confidence']['kind']} ({dist}) |"
            )
        if "frp_mw" in file_inv:
            f = file_inv["frp_mw"]
            lines.append(
                f"| FRP (MW) | min {f['min']}, median {f['median']}, "
                f"mean {f['mean']}, max {f['max']} ({f['negative_count']} negative) |"
            )
        contract = file_inv["pipeline_contract"]
        lines += [
            f"| Documented attributes absent | "
            f"{', '.join(file_inv['columns']['documented_but_absent']) or 'none'} |",
            f"| Undocumented extra columns | "
            f"{', '.join(file_inv['columns']['undocumented_but_present']) or 'none'} |",
            f"| Pipeline required columns present | "
            f"{'yes' if contract['all_required_present'] else 'NO'} |",
            "",
        ]
    lines += [
        "## Official notices recorded with this inventory",
        "",
        *(f"- {notice}" for notice in OFFICIAL_NOTICES),
        "",
        "## Attribution",
        "",
        OFFICIAL_ACKNOWLEDGMENT,
        "",
    ]
    return "\n".join(lines)


def collect_files(source_dir: str | None, csvs: list[str]) -> list[Path]:
    """Resolve --source-dir / --csv into a de-duplicated, sorted file list."""
    files: list[Path] = []
    if source_dir:
        files.extend(sorted(Path(source_dir).glob("*.csv")))
    for raw in csvs:
        candidate = Path(raw)
        if candidate not in files:
            files.append(candidate)
    return files


def build_inventory(files: list[Path], accessed_date: str) -> dict:
    """Inventory every file; failures are recorded, never silently dropped."""
    file_payloads: list[dict] = []
    failures: list[dict] = []
    for path in files:
        try:
            file_payloads.append(inventory_file(path))
        except Exception as exc:  # noqa: BLE001
            LOGGER.error("inventory failed for %s: %s", path, exc)
            failures.append({"file": str(path), "error": str(exc)})
    return {
        "schema_version": INVENTORY_SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "official_sources": {
            "schema": SCHEMA_SOURCE_URL,
            "coverage": COVERAGE_SOURCE_URL,
            "schema_access_date": accessed_date,
        },
        "official_archive_coverage": OFFICIAL_ARCHIVE_COVERAGE,
        "official_notices": OFFICIAL_NOTICES,
        "official_attribution": OFFICIAL_ACKNOWLEDGMENT,
        "files": file_payloads,
        "failures": failures,
        "all_files_passed": not failures,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        default=None,
        help="Directory scanned (non-recursively) for FIRMS CSV files",
    )
    parser.add_argument(
        "--csv",
        action="append",
        default=None,
        help="Explicit CSV path (repeatable); combined with --source-dir",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Where firms_inventory.json / firms_inventory.md are written",
    )
    parser.add_argument(
        "--accessed-date",
        default=datetime.now(UTC).date().isoformat(),
        help="Date the official schema pages were last inspected (YYYY-MM-DD)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    args = build_parser().parse_args(argv)
    if not args.source_dir and not args.csv:
        print("Provide --source-dir and/or at least one --csv", file=sys.stderr)
        return 2
    files = collect_files(args.source_dir, args.csv or [])
    if not files:
        print("No CSV files found to inventory", file=sys.stderr)
        return 2

    # Fail-closed like the five-pass data gate: every missing/unreadable file
    # becomes a recorded failure entry, and the reports are ALWAYS written.
    inventory = build_inventory(files, args.accessed_date)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "firms_inventory.json", inventory)
    (output_dir / "firms_inventory.md").write_text(
        render_markdown(inventory), encoding="utf-8"
    )

    print(f"Inventoried {len(inventory['files'])} file(s) -> {output_dir}")
    if not inventory["all_files_passed"]:
        for failure in inventory["failures"]:
            print(f"FAILED: {failure['file']}: {failure['error']}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
