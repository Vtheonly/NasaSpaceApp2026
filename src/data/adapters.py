"""FIRMS adapters: read processed active-fire CSV exports into normalized frames."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pandas as pd

from src.common.logging import get_logger
from src.data.schema import (
    COL_ACQ_DATE,
    COL_ACQ_TIME,
    COL_LATITUDE,
    COL_LONGITUDE,
    COLUMN_ALIASES,
    CONFIDENCE_ORDINAL,
    OPTIONAL_COLUMNS,
    PRODUCT_MODIS,
    PRODUCT_VIIRS,
    QUARANTINE_COLUMN,
    REQUIRED_COLUMNS,
    SUPPORTED_PRODUCTS,
    confidence_ordinal,
    parse_acq_datetime,
)

LOGGER = get_logger(__name__)


class SourceDataError(RuntimeError):
    """Raised when a source file cannot be read into the expected product shape."""


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """Compute the SHA-256 hex digest of a file (streamed)."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def list_source_files(source_dir: str | Path, product: str) -> list[Path]:
    """Return CSV files for *product* under *source_dir*.

    File discovery is filename-based: a file belongs to product ``modis`` when
    its name contains ``modis`` (case-insensitive), likewise for ``viirs``.
    """
    if product not in SUPPORTED_PRODUCTS:
        raise ValueError(f"Unsupported product {product!r}; expected one of {SUPPORTED_PRODUCTS}")
    root = Path(source_dir)
    if not root.is_dir():
        raise SourceDataError(f"Source directory does not exist: {root}")
    return sorted(p for p in root.glob("*.csv") if product in p.name.lower())


def read_observations_csv(path: str | Path, product: str) -> pd.DataFrame:
    """Read one product CSV into the normalized observation frame.

    The adapter applies documented aliases (e.g. ``bright_ti4`` -> ``brightness``
    for VIIRS), keeps optional columns only when present, parses
    ``acq_datetime`` (UTC) and ``confidence_ordinal``, adds provenance columns
    (``source_product``, ``source_file``), and initializes the quarantine
    column used by the five-pass gate. No rows are dropped here: the gate
    decides quarantine policy.
    """
    if product not in SUPPORTED_PRODUCTS:
        raise ValueError(f"Unsupported product {product!r}; expected one of {SUPPORTED_PRODUCTS}")
    path = Path(path)
    if not path.is_file():
        raise SourceDataError(f"Source file does not exist: {path}")
    if os.path.getsize(path) == 0:
        raise SourceDataError(f"Source file is empty: {path}")

    try:
        frame = pd.read_csv(path)
    except Exception as exc:  # noqa: BLE001 - surfaced as a single clear error
        raise SourceDataError(f"Could not read CSV {path}: {exc}") from exc

    missing = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
    if missing:
        raise SourceDataError(f"{path.name} is missing required columns: {missing}")

    frame = frame.rename(columns=COLUMN_ALIASES[product])
    keep = list(REQUIRED_COLUMNS) + [c for c in OPTIONAL_COLUMNS if c in frame.columns]
    frame = frame[keep].copy()

    frame["acq_datetime"] = [
        parse_acq_datetime(d, t) for d, t in zip(frame[COL_ACQ_DATE], frame[COL_ACQ_TIME], strict=False)
    ]
    if "confidence" in frame.columns:
        frame["confidence_ordinal"] = frame["confidence"].map(confidence_ordinal)
    frame["source_product"] = product
    frame["source_file"] = path.name
    frame[QUARANTINE_COLUMN] = None
    return frame


def read_product_frames(source_dir: str | Path, product: str) -> pd.DataFrame:
    """Read and concatenate every product CSV in *source_dir*.

    Emits a warning when no files are found; the gate turns that into a
    critical structural failure (fail-closed) when appropriate.
    """
    files = list_source_files(source_dir, product)
    if not files:
        LOGGER.warning("No %s source files found under %s", product, source_dir)
        return pd.DataFrame(columns=list(REQUIRED_COLUMNS) + ["acq_datetime", "source_product", "source_file", QUARANTINE_COLUMN])
    frames = [read_observations_csv(path, product) for path in files]
    out = pd.concat(frames, ignore_index=True)
    # Stable ordering for deterministic downstream processing.
    return out.sort_values(
        [COL_ACQ_DATE, COL_ACQ_TIME, COL_LATITUDE, COL_LONGITUDE], kind="mergesort"
    ).reset_index(drop=True)


__all__ = [
    "PRODUCT_MODIS",
    "PRODUCT_VIIRS",
    "SourceDataError",
    "CONFIDENCE_ORDINAL",
    "read_observations_csv",
    "read_product_frames",
    "list_source_files",
    "sha256_file",
]
