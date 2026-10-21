"""Unit tests for the five-pass data gate (issue #11; docs/TESTING.md).

The dirty fixture exercises every quarantine category and critical structural
failure; the gate must fail closed on critical errors while still emitting
machine-readable reports.
"""

import json

import pytest

from src.common.config import HarmonizationConfig
from src.data.gate import (
    PASS_LEAKAGE,
    PASS_REPORT,
    PASS_SEMANTIC,
    PASS_STRUCTURAL,
    DataGateError,
    FivePassDataGate,
    assert_gate_passed,
)


class TestCleanFixturePasses:
    def test_gate_passes(self, gate_outcome):
        assert gate_outcome.passed

    def test_report_has_five_passes(self, gate_outcome):
        names = [p["name"] for p in gate_outcome.audit_report["passes"]]
        assert PASS_STRUCTURAL in names and PASS_SEMANTIC in names
        assert PASS_LEAKAGE in names and PASS_REPORT in names
        assert gate_outcome.audit_report["gate"] == "pass"

    def test_inventory_records_hashes_and_rows(self, gate_outcome, sample_dir):
        products = gate_outcome.inventory_manifest["products"]
        assert products["modis"]["total_rows"] > 500
        assert products["viirs"]["total_rows"] > 500
        for product in ("modis", "viirs"):
            for entry in products[product]["files"]:
                assert len(entry["sha256"]) == 64
        assert products["modis"]["date_min"].startswith("2018-")
        assert products["modis"]["date_max"].startswith("2024-")

    def test_clean_frames_have_no_quarantine(self, gate_outcome):
        for frame in gate_outcome.frames.values():
            assert len(frame) > 0
            assert frame["quarantine_reason"].isna().all()

    def test_leakage_pass_records_split_counts(self, gate_outcome):
        details = gate_outcome.audit_report["passes"]  # find pass 3 details
        pass3 = next(p for p in details if p["name"] == PASS_LEAKAGE)
        assert pass3["details"]["modis"]["rows_train"] > 0
        assert pass3["details"]["modis"]["rows_validation"] > 0
        assert pass3["details"]["modis"]["rows_test"] > 0


class TestDirtyFixtureFailsClosed:
    @pytest.fixture(scope="class")
    def dirty_outcome(self, dirty_dir):
        return FivePassDataGate(HarmonizationConfig()).run(str(dirty_dir))

    def test_gate_fails(self, dirty_outcome):
        assert not dirty_outcome.passed
        assert dirty_outcome.audit_report["gate"] == "fail"

    def test_structural_failures_recorded(self, dirty_outcome):
        errors = " ".join(
            str(e) for p in dirty_outcome.audit_report["passes"] if p["status"] == "fail" for e in p["errors"]
        )
        # Empty file + missing required column are critical structural errors.
        assert "empty" in errors.lower()
        assert "missing required columns" in errors.lower()

    def test_quarantine_reason_codes(self, dirty_outcome):
        reasons = dirty_outcome.audit_report["quarantine"]
        modis_reasons = reasons.get("modis", {})
        assert modis_reasons.get("lat_out_of_bounds", 0) >= 1
        assert modis_reasons.get("lon_out_of_bounds", 0) >= 1
        assert modis_reasons.get("frp_negative", 0) >= 1
        assert modis_reasons.get("frp_missing", 0) >= 1
        assert modis_reasons.get("frp_nonfinite", 0) >= 1
        assert modis_reasons.get("duplicate_row", 0) >= 1
        assert modis_reasons.get("acq_time_invalid", 0) >= 1 or modis_reasons.get(
            "acq_date_invalid", 0
        ) >= 1

    def test_extreme_frp_is_review_warning_not_critical(self, dirty_outcome):
        warnings = " ".join(
            str(w)
            for p in dirty_outcome.audit_report["passes"]
            for w in p["warnings"]
        )
        # Docs: extreme FRP is a review flag, not proof of bad data.
        assert "extreme-FRP review threshold" in warnings
        quarantined_reasons = dirty_outcome.audit_report["quarantine"]["modis"]
        assert "frp_extreme" not in quarantined_reasons

    def test_pass2_semantic_status(self, dirty_outcome):
        pass2 = next(p for p in dirty_outcome.audit_report["passes"] if p["name"] == PASS_SEMANTIC)
        assert pass2["status"] in ("pass", "warn")

    def test_assert_gate_passed_raises_with_report(self, dirty_outcome):
        with pytest.raises(DataGateError) as excinfo:
            assert_gate_passed(dirty_outcome)
        assert "aborted before model initialization" in str(excinfo.value)
        assert excinfo.value.report["gate"] == "fail"


class TestGateEdgeCases:
    def test_missing_product_dir_fails(self, tmp_path):
        outcome = FivePassDataGate(HarmonizationConfig()).run(tmp_path)
        assert not outcome.passed
        errors = " ".join(
            str(e) for p in outcome.audit_report["passes"] if p["status"] == "fail" for e in p["errors"]
        )
        assert "No CSV source files found" in errors

    def test_bad_split_config_fails_pass3(self, tmp_path, sample_dir):
        config = HarmonizationConfig.from_dict(
            {
                "split": {
                    "train_end": "2023-12-31",
                    "validation_end": "2021-12-31",
                }
            }
        )
        outcome = FivePassDataGate(config).run(str(sample_dir))
        assert not outcome.passed
        pass3 = next(p for p in outcome.audit_report["passes"] if p["name"] == PASS_LEAKAGE)
        assert any("Split boundaries" in str(e) for e in pass3["errors"])

    def test_report_is_json_serializable(self, gate_outcome):
        json.dumps(gate_outcome.audit_report)
        json.dumps(gate_outcome.inventory_manifest)


def test_gate_is_deterministic(sample_dir, harmonization_config):
    first = FivePassDataGate(harmonization_config).run(str(sample_dir))
    second = FivePassDataGate(harmonization_config).run(str(sample_dir))
    assert first.passed == second.passed
    assert first.audit_report["quarantine"] == second.audit_report["quarantine"]
    for product in ("modis", "viirs"):
        assert len(first.frames[product]) == len(second.frames[product])
