"""Heavy tests for scripts/inventory_firms.py (issue #6).

The inventory tool is the evidence generator behind docs/RESEARCH_AND_DATA.md:
it must never overstate what a source provides ("No feature is assumed to
exist if the chosen source does not provide it"), must record stable hashes
and row counts, and must surface - not hide - dirty data.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
for entry in (REPO_ROOT, SCRIPTS_DIR):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from scripts.inventory_firms import (  # noqa: E402
    OFFICIAL_ACKNOWLEDGMENT,
    OFFICIAL_NOTICES,
    build_inventory,
    collect_files,
    detect_product,
    inventory_file,
    main,
    render_markdown,
)

FIXTURES = REPO_ROOT / "tests" / "fixtures"
SAMPLE_AOI = FIXTURES / "sample_aoi"
MODIS_CSV = SAMPLE_AOI / "modis_active_fires.csv"
VIIRS_CSV = SAMPLE_AOI / "viirs_active_fires.csv"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class TestCommittedFixtureInventory:
    """End-to-end runs on the committed synthetic sample (the demo evidence)."""

    def test_main_on_sample_dir_exits_zero_and_writes_both_outputs(
        self, tmp_path: Path
    ) -> None:
        rc = main(
            [
                "--source-dir",
                str(SAMPLE_AOI),
                "--output-dir",
                str(tmp_path),
                "--accessed-date",
                "2026-10-22",
            ]
        )
        assert rc == 0
        payload = json.loads((tmp_path / "firms_inventory.json").read_text())
        assert payload["all_files_passed"] is True
        assert len(payload["files"]) == 2
        assert (tmp_path / "firms_inventory.md").is_file()

    def test_hashes_and_row_counts_are_exact(self, tmp_path: Path) -> None:
        main(["--source-dir", str(SAMPLE_AOI), "--output-dir", str(tmp_path)])
        payload = json.loads((tmp_path / "firms_inventory.json").read_text())
        by_name = {f["file"]: f for f in payload["files"]}
        modis = by_name["modis_active_fires.csv"]
        viirs = by_name["viirs_active_fires.csv"]
        assert modis["sha256"] == _sha256(MODIS_CSV)
        assert viirs["sha256"] == _sha256(VIIRS_CSV)
        # wc -l is 1679 / 2608 lines INCLUDING the header row.
        assert modis["row_count"] == 1678
        assert viirs["row_count"] == 2607

    def test_products_detected_correctly(self, tmp_path: Path) -> None:
        main(["--source-dir", str(SAMPLE_AOI), "--output-dir", str(tmp_path)])
        payload = json.loads((tmp_path / "firms_inventory.json").read_text())
        products = {f["file"]: f["product"] for f in payload["files"]}
        assert products["modis_active_fires.csv"] == "modis"
        assert products["viirs_active_fires.csv"] == "viirs"

    def test_temporal_coverage_matches_fixture_dates(self) -> None:
        modis = inventory_file(MODIS_CSV)
        assert modis["temporal"]["acq_date_min"] == "2018-01-28"
        assert modis["temporal"]["acq_date_max"] == "2024-12-09"
        assert modis["temporal"]["unparseable_datetime_count"] == 0

    def test_pipeline_required_columns_present_in_sample(self) -> None:
        for csv in (MODIS_CSV, VIIRS_CSV):
            inv = inventory_file(csv)
            assert inv["pipeline_contract"]["all_required_present"] is True

    def test_optional_official_fields_reported_absent_not_assumed(self) -> None:
        """The fixtures lack bright_t31/bright_ti5; the tool must SAY so."""
        modis = inventory_file(MODIS_CSV)
        viirs = inventory_file(VIIRS_CSV)
        assert "bright_t31" in modis["columns"]["documented_but_absent"]
        assert "bright_ti5" in viirs["columns"]["documented_but_absent"]
        # And nothing undocumented may be invented for either file.
        assert modis["columns"]["undocumented_but_present"] == []
        assert viirs["columns"]["undocumented_but_present"] == []

    def test_confidence_kind_matches_observed_values(self) -> None:
        """Synthetic fixtures use l/n/h labels even for MODIS (real MODIS NRT
        is 0-100 numeric); the inventory must report what the FILE contains."""
        modis = inventory_file(MODIS_CSV)
        viirs = inventory_file(VIIRS_CSV)
        assert modis["confidence"]["kind"] == "label (l/n/h)"
        assert viirs["confidence"]["kind"] == "label (l/n/h)"
        labels = set(modis["confidence"]["distribution"])
        assert labels <= {"l", "n", "h"}

    def test_no_duplicate_rows_in_committed_sample(self) -> None:
        for csv in (MODIS_CSV, VIIRS_CSV):
            assert inventory_file(csv)["duplicate_row_count"] == 0


class TestProductDetection:
    def test_filename_hints(self) -> None:
        assert detect_product(Path("MCD14DL_2020.csv"), []) == "modis"
        assert detect_product(Path("VNP14IMG.csv"), []) == "viirs"

    def test_column_fingerprints(self) -> None:
        assert detect_product(Path("mystery.csv"), ["bright_ti4"]) == "viirs"
        assert (
            detect_product(Path("mystery.csv"), ["brightness", "bright_t31"])
            == "modis"
        )

    def test_ambiguous_refuses_to_guess(self, tmp_path: Path) -> None:
        assert detect_product(Path("mystery.csv"), ["latitude", "frp"]) is None
        path = tmp_path / "mystery.csv"
        pd.DataFrame({"latitude": [1.0], "longitude": [2.0], "frp": [3.0]}).to_csv(
            path, index=False
        )
        with pytest.raises(ValueError, match="refusing to guess"):
            inventory_file(path)


class TestDirtyDataSurfaced:
    """Dirty inputs must be PROFILED, not hidden - issue #6 inventories
    actual missingness rather than assuming clean archives."""

    def _dirty_csv(self, tmp_path: Path) -> Path:
        path = tmp_path / "modis_dirty.csv"
        pd.DataFrame(
            {
                "latitude": [10.0, 999.0, None, 0.0],  # one out-of-bounds, one missing
                "longitude": [20.0, 20.5, 21.0, -500.0],  # one out-of-bounds
                "brightness": [300.0, None, 310.0, 320.0],
                "scan": [1.0, 1.1, 1.2, 1.3],
                "track": [1.0, 1.1, 1.2, 1.3],
                "acq_date": ["2020-01-01", "not-a-date", "2020-01-03", "2020-01-04"],
                "acq_time": [100, 9999, 200, 300],  # 9999 is an invalid HHMM
                "satellite": ["T", "A", "T", "A"],
                "instrument": ["MODIS"] * 4,
                "confidence": [80, 55, 20, 100],
                "version": ["6.1"] * 4,
                "bright_t31": [290.0, 291.0, 292.0, 293.0],
                "frp": [5.0, -3.0, None, 12.0],  # one negative, one missing
                "daynight": ["D", "N", "D", "N"],
            }
        ).to_csv(path, index=False)
        return path

    def test_out_of_bounds_and_missing_counts(self, tmp_path: Path) -> None:
        inv = inventory_file(self._dirty_csv(tmp_path))
        coords = inv["coordinates"]
        assert coords["lat_out_of_bounds_count"] == 1
        assert coords["lon_out_of_bounds_count"] == 1
        assert coords["missing_count"] == 1
        brightness = inv["columns"]["profiles"]["brightness"]
        assert brightness["missing_count"] == 1

    def test_invalid_datetimes_counted(self, tmp_path: Path) -> None:
        inv = inventory_file(self._dirty_csv(tmp_path))
        # Row 2 carries BOTH the bad date and the invalid HHMM 9999 -> one
        # unparseable row; every other row parses.
        assert inv["temporal"]["unparseable_datetime_count"] == 1
        assert inv["temporal"]["parseable_datetime_count"] == 3

    def test_negative_frp_counted(self, tmp_path: Path) -> None:
        inv = inventory_file(self._dirty_csv(tmp_path))
        assert inv["frp_mw"]["negative_count"] == 1
        assert inv["frp_mw"]["missing_count"] == 1

    def test_numeric_modis_confidence_detected(self, tmp_path: Path) -> None:
        inv = inventory_file(self._dirty_csv(tmp_path))
        assert inv["confidence"]["kind"] == "numeric (0-100)"

    def test_duplicate_rows_counted(self, tmp_path: Path) -> None:
        path = tmp_path / "viirs_dup.csv"
        pd.DataFrame(
            {
                "latitude": [1.0, 1.0, 2.0],
                "longitude": [2.0, 2.0, 3.0],
                "bright_ti4": [300.0, 300.0, 310.0],
                "scan": [0.1, 0.1, 0.1],
                "track": [0.1, 0.1, 0.1],
                "acq_date": ["2020-01-01"] * 3,
                "acq_time": [100, 100, 200],
                "satellite": ["N", "N", "N"],
                "instrument": ["VIIRS"] * 3,
                "confidence": ["l", "l", "h"],
                "version": ["1.0NRT"] * 3,
                "frp": [1.0, 1.0, 2.0],
                "daynight": ["D", "D", "N"],
            }
        ).to_csv(path, index=False)
        inv = inventory_file(path)
        assert inv["duplicate_row_count"] == 1

    def test_failure_recorded_never_dropped(self, tmp_path: Path) -> None:
        bad = tmp_path / "ambiguous.csv"
        pd.DataFrame({"latitude": [1.0], "frp": [2.0]}).to_csv(bad, index=False)
        inventory = build_inventory([bad], accessed_date="2026-10-22")
        assert inventory["all_files_passed"] is False
        assert len(inventory["failures"]) == 1
        assert "refusing to guess" in inventory["failures"][0]["error"]


class TestCliContracts:
    def test_no_inputs_is_usage_error(self, tmp_path: Path) -> None:
        assert main(["--output-dir", str(tmp_path)]) == 2

    def test_empty_source_dir_is_usage_error(self, tmp_path: Path) -> None:
        assert main(["--source-dir", str(tmp_path), "--output-dir", str(tmp_path)]) == 2

    def test_missing_file_exits_one(self, tmp_path: Path) -> None:
        rc = main(
            [
                "--csv",
                str(tmp_path / "nope.csv"),
                "--output-dir",
                str(tmp_path / "out"),
            ]
        )
        assert rc == 1

    def test_missing_file_failure_writes_reports_anyway(self, tmp_path: Path) -> None:
        """Fail-closed LIKE the data gate: evidence is always written."""
        out = tmp_path / "out"
        assert main(["--csv", str(tmp_path / "nope.csv"), "--output-dir", str(out)]) == 1
        payload = json.loads((out / "firms_inventory.json").read_text())
        assert payload["failures"][0]["file"].endswith("nope.csv")
        assert (out / "firms_inventory.md").is_file()

    def test_collect_files_dedupes_and_sorts(self, tmp_path: Path) -> None:
        a = tmp_path / "a_modis.csv"
        b = tmp_path / "b_viirs.csv"
        a.write_text("x\n")
        b.write_text("x\n")
        files = collect_files(str(tmp_path), [str(a)])
        assert files == sorted(files)
        assert len(files) == 2  # a listed twice -> kept once


class TestDeterminismAndProvenance:
    def test_file_payloads_deterministic_across_runs(self) -> None:
        first = build_inventory([MODIS_CSV, VIIRS_CSV], "2026-10-22")
        second = build_inventory([MODIS_CSV, VIIRS_CSV], "2026-10-22")
        assert first["files"] == second["files"]
        # Hash stability is the backbone of the recorded evidence.
        assert first["files"][0]["sha256"] == _sha256(MODIS_CSV)

    def test_official_sources_and_access_date_recorded(self, tmp_path: Path) -> None:
        main(
            [
                "--source-dir",
                str(SAMPLE_AOI),
                "--output-dir",
                str(tmp_path),
                "--accessed-date",
                "2026-10-22",
            ]
        )
        payload = json.loads((tmp_path / "firms_inventory.json").read_text())
        assert payload["official_sources"]["schema_access_date"] == "2026-10-22"
        assert "active-fire-data-attributes-modis-viirs" in payload["official_sources"]["schema"]

    def test_operationally_critical_notices_cannot_be_lost(self) -> None:
        inventory = build_inventory([MODIS_CSV], "2026-10-22")
        joined = " ".join(inventory["official_notices"])
        # S-NPP end-of-delivery drives the project's overlap-window strategy.
        assert "Suomi NPP" in joined and "November" in joined
        assert "5-month lag" in joined.replace("approximately 5-month", "approximately 5-month")
        assert inventory["official_attribution"] == OFFICIAL_ACKNOWLEDGMENT

    def test_markdown_evidence_renders(self) -> None:
        inventory = build_inventory([MODIS_CSV, VIIRS_CSV], "2026-10-22")
        md = render_markdown(inventory)
        assert md.startswith("# FIRMS sample inventory")
        assert _sha256(MODIS_CSV) in md
        assert "2018-01-28 .. 2024-12-09" in md
        assert OFFICIAL_ACKNOWLEDGMENT in md
        assert all(notice[:40] in md for notice in OFFICIAL_NOTICES)


class TestRawDirtyFixtureIntegration:
    """The raw_dirty fixtures intentionally violate the contract; the
    inventory must describe them without crashing or hiding the violations."""

    @pytest.mark.parametrize(
        "name",
        [
            "modis_issues.csv",
            "modis_missing_column.csv",
            "viirs_issues.csv",
        ],
    )
    def test_dirty_fixture_profiles_complete(self, name: str) -> None:
        inv = inventory_file(FIXTURES / "raw_dirty" / name)
        assert inv["row_count"] > 0
        assert "confidence" in inv or "frp_mw" in inv
