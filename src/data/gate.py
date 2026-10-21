"""Five-pass data gate and scientific validation (docs/TESTING.md; issue #11).

Passes
-----
1. **Structural integrity:** file readability, required columns/types, duplicates,
   empty files, date coverage; emits an inventory manifest with SHA-256 hashes.
2. **Semantic/numerical validity:** field-specific null/NaN/Inf policies, coordinate
   bounds, timestamps, negative FRP; quarantines rows with reason codes instead of
   deleting them. Extreme FRP raises a *review warning*, never an automatic error.
3. **Leakage/split integrity:** chronological split boundaries and coverage.
4. **Transformation invariants:** aggregation conservation, deterministic date
   bucketing, no fabricated observations on missing days.
5. **Report and gate:** machine-readable ``inventory_manifest`` +
   ``dataset_audit_report``; the gate **fails closed** on critical errors, so
   training stops before model initialization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.common.config import HarmonizationConfig
from src.common.logging import get_logger
from src.data.adapters import (
    SourceDataError,
    list_source_files,
    read_observations_csv,
    sha256_file,
)
from src.data.schema import (
    COL_ACQ_DATE,
    COL_FRP,
    COL_LATITUDE,
    COL_LONGITUDE,
    LAT_MAX,
    LAT_MIN,
    LON_MAX,
    LON_MIN,
    QUARANTINE_COLUMN,
    REASON_COORDS_MISSING,
    REASON_DATE_INVALID,
    REASON_DUPLICATE_ROW,
    REASON_FRP_MISSING,
    REASON_FRP_NEGATIVE,
    REASON_FRP_NONFINITE,
    REASON_LAT_OUT_OF_BOUNDS,
    REASON_LON_OUT_OF_BOUNDS,
    REASON_TIME_INVALID,
    REQUIRED_COLUMNS,
    SUPPORTED_PRODUCTS,
)

LOGGER = get_logger(__name__)

PASS_STRUCTURAL = "pass_1_structural_integrity"
PASS_SEMANTIC = "pass_2_semantic_validity"
PASS_LEAKAGE = "pass_3_leakage_split_integrity"
PASS_INVARIANTS = "pass_4_transformation_invariants"
PASS_REPORT = "pass_5_report_and_gate"

GATE_SCHEMA_VERSION = "1.0.0"
#: Fraction of quarantined rows above which pass 2 becomes a critical failure.
MAX_QUARANTINE_FRACTION = 0.5


class DataGateError(RuntimeError):
    """Raised by :func:`assert_gate_passed` when the gate fails closed."""

    def __init__(self, message: str, report: dict[str, Any]):
        super().__init__(message)
        self.report = report


@dataclass
class PassResult:
    name: str
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def status(self) -> str:
        if self.errors:
            return "fail"
        return "warn" if self.warnings else "pass"

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "details": self.details,
        }


@dataclass
class GateOutcome:
    frames: dict[str, pd.DataFrame]
    quarantined_frames: dict[str, pd.DataFrame]
    inventory_manifest: dict[str, Any]
    audit_report: dict[str, Any]
    passed: bool

    @property
    def report(self) -> dict[str, Any]:
        return self.audit_report


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class FivePassDataGate:
    """Run the five-pass validation gate over a directory of FIRMS CSV files."""

    def __init__(self, config: HarmonizationConfig | None = None):
        self.config = config or HarmonizationConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, source_dir: str | Path) -> GateOutcome:
        """Run all five passes. Returns a :class:`GateOutcome` (never raises on
        gate failure; callers decide via :func:`assert_gate_passed`)."""
        source_dir = Path(source_dir)
        results: dict[str, PassResult] = {}
        inventory: dict[str, Any] = {
            "schema_version": GATE_SCHEMA_VERSION,
            "generated_at": _now_iso(),
            "source_dir": str(source_dir),
            "products": {},
        }
        frames: dict[str, pd.DataFrame] = {}
        quarantined: dict[str, pd.DataFrame] = {}

        raw = self._pass_1_structural(source_dir, results, inventory)
        self._pass_2_semantic(raw, results, frames, quarantined)
        self._pass_3_leakage(frames, results)
        self._pass_4_invariants(frames, results)
        passed = all(not r.errors for r in results.values())
        report = self._pass_5_report(results, inventory, passed)
        return GateOutcome(
            frames=frames,
            quarantined_frames=quarantined,
            inventory_manifest=inventory,
            audit_report=report,
            passed=passed,
        )

    # ------------------------------------------------------------------
    # Pass 1 — structural integrity (inventory + hashes)
    # ------------------------------------------------------------------

    def _pass_1_structural(
        self,
        source_dir: Path,
        results: dict[str, PassResult],
        inventory: dict[str, Any],
    ) -> dict[str, pd.DataFrame]:
        result = PassResult(PASS_STRUCTURAL)
        raw: dict[str, pd.DataFrame] = {}

        for product in SUPPORTED_PRODUCTS:
            files = list_source_files(source_dir, product)
            product_entry: dict[str, Any] = {"files": [], "total_rows": 0}
            product_frames: list[pd.DataFrame] = []

            if not files:
                result.errors.append(
                    f"No CSV source files found for required product '{product}' under {source_dir}"
                )
                inventory["products"][product] = product_entry
                continue

            for path in files:
                file_entry: dict[str, Any] = {
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
                try:
                    frame = read_observations_csv(path, product)
                except SourceDataError as exc:
                    result.errors.append(f"[{product}] {exc}")
                    product_entry["files"].append(file_entry)
                    continue
                file_entry["rows"] = int(len(frame))
                file_entry["columns"] = [c for c in frame.columns if c != QUARANTINE_COLUMN]
                product_entry["files"].append(file_entry)
                product_frames.append(frame)

            if product_frames:
                combined = pd.concat(product_frames, ignore_index=True)
                dates = pd.to_datetime(combined[COL_ACQ_DATE], errors="coerce")
                product_entry["total_rows"] = int(len(combined))
                product_entry["date_min"] = (
                    str(dates.min().date()) if not dates.isna().all() else None
                )
                product_entry["date_max"] = (
                    str(dates.max().date()) if not dates.isna().all() else None
                )
                raw[product] = combined
            inventory["products"][product] = product_entry

        for product, frame in raw.items():
            n_dup = int(frame.duplicated(subset=list(REQUIRED_COLUMNS) + ["source_file"]).sum())
            if n_dup:
                result.warnings.append(
                    f"[{product}] {n_dup} duplicate detection rows found (quarantined in pass 2)"
                )

        results[result.name] = result
        return raw

    # ------------------------------------------------------------------
    # Pass 2 — semantic/numerical validity and quarantine
    # ------------------------------------------------------------------

    def _pass_2_semantic(
        self,
        raw: dict[str, pd.DataFrame],
        results: dict[str, PassResult],
        frames: dict[str, pd.DataFrame],
        quarantined: dict[str, pd.DataFrame],
    ) -> None:
        result = PassResult(PASS_SEMANTIC)
        review_threshold = self.config.extreme_frp_review_threshold_mw

        for product, frame in raw.items():
            work = frame.copy()
            reasons = pd.Series(None, index=work.index, dtype=object)

            # Exact duplicates on data columns (keep first; documented duplicate policy).
            dup_mask = work.duplicated(
                subset=[c for c in REQUIRED_COLUMNS]
                + [c for c in ("brightness", "scan", "track") if c in work.columns]
            )
            reasons[dup_mask] = REASON_DUPLICATE_ROW

            # Invalid acq_datetime (acq_date or acq_time failed to parse).
            bad_time = work["acq_datetime"].isna()
            free = reasons.isna()
            reasons[free & bad_time] = REASON_TIME_INVALID

            lat = pd.to_numeric(work[COL_LATITUDE], errors="coerce")
            lon = pd.to_numeric(work[COL_LONGITUDE], errors="coerce")
            coords_missing = lat.isna() | lon.isna()
            free = reasons.isna()
            reasons[free & coords_missing] = REASON_COORDS_MISSING

            lat_bad = ((lat < LAT_MIN) | (lat > LAT_MAX)) & ~lat.isna()
            lon_bad = ((lon < LON_MIN) | (lon > LON_MAX)) & ~lon.isna()
            free = reasons.isna()
            reasons[free & lat_bad] = REASON_LAT_OUT_OF_BOUNDS
            free = reasons.isna()
            reasons[free & lon_bad & ~lat_bad] = REASON_LON_OUT_OF_BOUNDS
            free = reasons.isna()

            frp = pd.to_numeric(work[COL_FRP], errors="coerce")
            reasons[free & frp.isna()] = REASON_FRP_MISSING
            free = reasons.isna()
            reasons[free & np.isinf(frp.fillna(0.0))] = REASON_FRP_NONFINITE
            free = reasons.isna()
            reasons[free & (frp < 0)] = REASON_FRP_NEGATIVE
            free = reasons.isna()
            invalid_date = pd.to_datetime(
                work[COL_ACQ_DATE], format="%Y-%m-%d", errors="coerce"
            ).isna()
            reasons[free & invalid_date] = REASON_DATE_INVALID
            reasons[free & frp.isna()] = REASON_FRP_MISSING
            free = reasons.isna()
            reasons[free & np.isinf(frp.fillna(0.0))] = REASON_FRP_NONFINITE
            free = reasons.isna()
            reasons[free & (frp < 0)] = REASON_FRP_NEGATIVE
            free = reasons.isna()
            invalid_date = pd.to_datetime(work[COL_ACQ_DATE], format="%Y-%m-%d", errors="coerce").isna()
            reasons[free & invalid_date] = REASON_DATE_INVALID

            work[QUARANTINE_COLUMN] = reasons

            n_extreme = int((frp > review_threshold).sum())
            if n_extreme:
                # Extreme FRP is a review flag, not proof of bad data (docs/TESTING.md).
                result.warnings.append(
                    f"[{product}] {n_extreme} records exceed the extreme-FRP review "
                    f"threshold ({review_threshold} {self.config.frp_units}); flagged for review, kept."
                )

            quarantined[product] = work[work[QUARANTINE_COLUMN].notna()].copy()
            frames[product] = work[work[QUARANTINE_COLUMN].isna()].copy().reset_index(drop=True)

            n_total = len(work)
            n_quarantined = len(quarantined[product])
            result.details[product] = {
                "rows_total": int(n_total),
                "rows_quarantined": int(n_quarantined),
                "quarantine_reasons": (
                    quarantined[product][QUARANTINE_COLUMN].value_counts().to_dict()
                ),
            }
            if n_total and n_quarantined / n_total > MAX_QUARANTINE_FRACTION:
                result.errors.append(
                    f"[{product}] {n_quarantined}/{n_total} rows quarantined "
                    f"(>{MAX_QUARANTINE_FRACTION:.0%}); source data unusable"
                )

        results[result.name] = result

    # ------------------------------------------------------------------
    # Pass 3 — leakage / split integrity
    # ------------------------------------------------------------------

    def _pass_3_leakage(self, frames: dict[str, pd.DataFrame], results: dict[str, PassResult]) -> None:
        result = PassResult(PASS_LEAKAGE)
        split = self.config.split
        try:
            train_end = pd.Timestamp(split.train_end)
            validation_end = pd.Timestamp(split.validation_end)
        except ValueError as exc:
            result.errors.append(f"Invalid split boundary configuration: {exc}")
            results[result.name] = result
            return
        if validation_end <= train_end:
            result.errors.append(
                f"Split boundaries must satisfy validation_end ({split.validation_end}) > "
                f"train_end ({split.train_end})"
            )

        all_dates: list[pd.Timestamp] = []
        for product, frame in frames.items():
            if frame.empty:
                continue
            dates = pd.to_datetime(frame["acq_datetime"]).dt.normalize()
            all_dates.extend(dates.tolist())
            n_train = int((dates <= train_end).sum())
            n_val = int(((dates > train_end) & (dates <= validation_end)).sum())
            n_test = int((dates > validation_end).sum())
            result.details[product] = {
                "rows_train": n_train,
                "rows_validation": n_val,
                "rows_test": n_test,
            }
            if n_train == 0:
                result.warnings.append(f"[{product}] no observations in the training period")
            if n_val == 0:
                result.warnings.append(
                    f"[{product}] no observations in the validation period ({split.train_end}..{split.validation_end})"
                )
            if n_test == 0:
                result.warnings.append(
                    f"[{product}] no observations in the test period (after {split.validation_end})"
                )

        if all_dates and len(set(all_dates)) <= 1:
            result.warnings.append(
                "only a single distinct observation date found; chronological splits cannot be evaluated"
            )
        results[result.name] = result

    # ------------------------------------------------------------------
    # Pass 4 — transformation invariants
    # ------------------------------------------------------------------

    def _pass_4_invariants(self, frames: dict[str, pd.DataFrame], results: dict[str, PassResult]) -> None:
        result = PassResult(PASS_INVARIANTS)

        for product, frame in frames.items():
            if frame.empty:
                continue
            frp = pd.to_numeric(frame[COL_FRP], errors="coerce")
            dates = pd.to_datetime(frame["acq_datetime"]).dt.normalize()
            daily_sum = frp.groupby(dates).sum()
            total_raw = float(frp.sum())
            total_daily = float(daily_sum.sum())
            if not np.isclose(total_raw, total_daily, rtol=1e-9, atol=1e-6):
                result.errors.append(
                    f"[{product}] daily aggregation does not conserve total FRP "
                    f"(raw={total_raw:.6f}, daily={total_daily:.6f})"
                )
            n_unique_days = int(dates.nunique())
            result.details[product] = {
                "total_frp": total_raw,
                "unique_observed_days": n_unique_days,
                "frp_min": float(frp.min()),
                "frp_max": float(frp.max()),
            }
            # Missing days are never fabricated as zero-activity observations:
            # coverage flags in the daily frame carry the distinction.
            if n_unique_days > 0:
                span_days = (dates.max() - dates.min()).days + 1
                result.details[product]["span_days"] = span_days
                result.details[product]["missing_days"] = span_days - n_unique_days
        results[result.name] = result

    # ------------------------------------------------------------------
    # Pass 5 — report and gate decision
    # ------------------------------------------------------------------

    def _pass_5_report(
        self,
        results: dict[str, PassResult],
        inventory: dict[str, Any],
        passed: bool,
    ) -> dict[str, Any]:
        result = PassResult(PASS_REPORT)
        result.details["gate"] = "pass" if passed else "fail"
        results[result.name] = result  # registered BEFORE building the pass list
        report: dict[str, Any] = {
            "schema_version": GATE_SCHEMA_VERSION,
            "generated_at": _now_iso(),
            "split_boundaries": {
                "train_end": self.config.split.train_end,
                "validation_end": self.config.split.validation_end,
                "test_start": self.config.split.validation_end,
            },
            "passes": [results[name].as_dict() for name in self._ordered_pass_names(results)],
            "quarantine": {
                product: results[PASS_SEMANTIC].details.get(product, {}).get("quarantine_reasons", {})
                for product in SUPPORTED_PRODUCTS
            },
            "gate_passed": passed,
            "gate": "pass" if passed else "fail",
        }
        return report

    @staticmethod
    def _ordered_pass_names(results: dict[str, PassResult]) -> list[str]:
        order = [PASS_STRUCTURAL, PASS_SEMANTIC, PASS_LEAKAGE, PASS_INVARIANTS, PASS_REPORT]
        return [name for name in order if name in results] + [
            name for name in results if name not in order
        ]


def assert_gate_passed(outcome: GateOutcome) -> None:
    """Fail closed: raise :class:`DataGateError` when any critical error remains.

    Training entry points must call this **before** any model/GPU initialization
    (docs/TESTING.md: "Training must stop before model/GPU initialization if
    any critical validation error remains").
    """
    if not outcome.passed:
        failing = [
            p["name"]
            for p in outcome.audit_report["passes"]
            if p["status"] == "fail"
        ]
        raise DataGateError(
            "Five-pass data gate FAILED ("
            + ", ".join(failing)
            + "); see dataset_audit_report.json. Training aborted before model initialization.",
            report=outcome.audit_report,
        )


__all__ = [
    "DataGateError",
    "FivePassDataGate",
    "GateOutcome",
    "PassResult",
    "assert_gate_passed",
    "PASS_STRUCTURAL",
    "PASS_SEMANTIC",
    "PASS_LEAKAGE",
    "PASS_INVARIANTS",
    "PASS_REPORT",
]
