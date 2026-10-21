"""Unit tests for FIRMS CSV adapters (source parsing and schema drift)."""

import pandas as pd
import pytest

from src.data.adapters import (
    SourceDataError,
    list_source_files,
    read_observations_csv,
    read_product_frames,
    sha256_file,
)
from src.data.schema import QUARANTINE_COLUMN


def test_list_source_files_finds_product_files(sample_dir):
    modis_files = list_source_files(sample_dir, "modis")
    viirs_files = list_source_files(sample_dir, "viirs")
    assert len(modis_files) == 1 and "modis" in modis_files[0].name.lower()
    assert len(viirs_files) == 1 and "viirs" in viirs_files[0].name.lower()


def test_list_source_files_rejects_unknown_product(sample_dir):
    with pytest.raises(ValueError, match="Unsupported product"):
        list_source_files(sample_dir, "goes")


def test_read_clean_modis_csv(sample_dir):
    frame = read_observations_csv(sample_dir / "modis_active_fires.csv", "modis")
    assert len(frame) > 500
    assert {"acq_datetime", "source_product", "source_file", QUARANTINE_COLUMN}.issubset(
        frame.columns
    )
    assert (frame["source_product"] == "modis").all()
    assert frame["acq_datetime"].notna().all()


def test_read_viirs_csv_aliases_bright_ti4(sample_dir):
    frame = read_observations_csv(sample_dir / "viirs_active_fires.csv", "viirs")
    assert "brightness" in frame.columns  # bright_ti4 -> brightness alias
    assert "confidence_ordinal" in frame.columns


def test_missing_required_column_raises(tmp_path):
    bad = tmp_path / "modis_bad.csv"
    pd.DataFrame({"latitude": [1.0], "longitude": [2.0], "acq_date": ["2022-01-01"]}).to_csv(
        bad, index=False
    )
    with pytest.raises(SourceDataError, match="missing required columns"):
        read_observations_csv(bad, "modis")


def test_empty_file_raises(tmp_path):
    empty = tmp_path / "modis_empty.csv"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(SourceDataError, match="empty"):
        read_observations_csv(empty, "modis")


def test_missing_file_raises(tmp_path):
    with pytest.raises(SourceDataError, match="does not exist"):
        read_observations_csv(tmp_path / "nope.csv", "modis")


def test_unreadable_csv_raises(tmp_path):
    broken = tmp_path / "modis_broken.csv"
    # Non-UTF-8 binary payload -> guaranteed hard decode failure on read.
    broken.write_bytes(b"latitude,longitude,acq_date,acq_time,frp\n1.0,2.0,\xff\xfe\xfa\xfb\n")
    with pytest.raises(SourceDataError):
        read_observations_csv(broken, "modis")


def test_read_product_frames_concatenates_and_sorts(tmp_path):
    part1 = pd.DataFrame(
        {
            "latitude": [40.0, 41.0],
            "longitude": [-122.0, -121.0],
            "acq_date": ["2022-06-02", "2022-06-01"],
            "acq_time": [1330, 1330],
            "frp": [10.0, 20.0],
        }
    )
    part1.to_csv(tmp_path / "viirs_a.csv", index=False)
    part2 = pd.DataFrame(
        {
            "latitude": [40.5],
            "longitude": [-121.5],
            "acq_date": ["2022-06-01"],
            "acq_time": [1310],
            "frp": [5.0],
        }
    )
    part2.to_csv(tmp_path / "viirs_b.csv", index=False)
    combined = read_product_frames(tmp_path, "viirs")
    assert len(combined) == 3
    dates = combined["acq_datetime"].tolist()
    assert dates == sorted(dates)  # deterministic chronological ordering


def test_read_product_frames_empty_dir_warns_and_returns_empty(tmp_path):
    frame = read_product_frames(tmp_path, "viirs")
    assert frame.empty


def test_sha256_file_is_stable(tmp_path):
    f = tmp_path / "data.bin"
    f.write_bytes(b"hello hotspot")
    first = sha256_file(f)
    assert first == sha256_file(f)  # stable across reads
    f.write_bytes(b"hello hotspot!")
    assert sha256_file(f) != first  # different content, different hash
