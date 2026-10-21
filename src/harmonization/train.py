"""Training orchestration for the harmonization pillar.

Order of operations enforced by the docs:

1. five-pass data gate (fail-closed **before** any model is constructed),
2. AOI filtering and deterministic daily aggregation,
3. overlap-period matching with explicit rules,
4. chronological split (train <= train_end < validation <= validation_end < test),
5. fit identity + calibration baselines and the LightGBM/Tweedie harmonizer,
6. held-out evaluation with FRP-range and season slices + baseline comparison.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from src.common.config import HarmonizationConfig, SplitConfig, seed_everything
from src.common.logging import get_logger
from src.data.aggregation import build_daily_frame
from src.data.gate import FivePassDataGate, GateOutcome, assert_gate_passed
from src.data.schema import PRODUCT_MODIS, PRODUCT_VIIRS
from src.geo.aoi import BBoxAOI, PolygonAOI, filter_records
from src.geo.matching import (
    MATCHED_COUNT_COLUMN,
    TARGET_COLUMN,
    build_matched_records,
    eligible_matches,
)
from src.harmonization.baselines import CalibrationBaseline, IdentityBaseline
from src.harmonization.evaluate import beats_baseline, compare_predictions
from src.harmonization.features import build_feature_matrix
from src.harmonization.model import TweedieGBDTHarmonizer

LOGGER = get_logger(__name__)

SPLIT_TRAIN = "train"
SPLIT_VALIDATION = "validation"
SPLIT_TEST = "test"


class TrainingInputError(RuntimeError):
    """Raised when training cannot proceed with a clear, actionable message."""


def split_matched_records(
    matched: pd.DataFrame, split: SplitConfig
) -> dict[str, pd.DataFrame]:
    """Chronologically split matched records by match date (never shuffled).

    A match group carries a single MODIS timestamp, so it can never span
    partitions; the scientific test suite asserts this invariant. Indexes of
    the source frame are PRESERVED so feature matrices built from the full
    matched set can be aligned with ``X.loc[split.index]``.
    """
    dates = pd.to_datetime(matched["match_date"])
    train_end = pd.Timestamp(split.train_end)
    validation_end = pd.Timestamp(split.validation_end)
    train = matched[dates <= train_end]
    validation = matched[(dates > train_end) & (dates <= validation_end)]
    test = matched[dates > validation_end]
    return {SPLIT_TRAIN: train, SPLIT_VALIDATION: validation, SPLIT_TEST: test}


@dataclass
class HarmonizationTrainingResult:
    model: TweedieGBDTHarmonizer
    baselines: dict[str, Any]
    feature_names: list[str]
    matched: pd.DataFrame
    splits: dict[str, pd.DataFrame]
    evaluation: dict[str, Any]
    baseline_comparison: dict[str, Any]
    daily: pd.DataFrame
    aoi_id: str
    config_snapshot: dict[str, Any]
    notes: list[str] = field(default_factory=list)

    @property
    def metrics_summary(self) -> dict[str, Any]:
        return self.evaluation


def run_harmonization_training(
    source_dir: str,
    config: HarmonizationConfig | None = None,
    aoi: BBoxAOI | PolygonAOI | None = None,
    gate_outcome: GateOutcome | None = None,
) -> HarmonizationTrainingResult:
    """Run the full harmonization training pipeline (gate -> match -> train -> eval).

    ``gate_outcome`` lets a caller that already ran the five-pass gate (e.g. the
    CLI, which must persist the gate reports even on failure) inject the
    precomputed outcome instead of re-reading the source directory.
    """
    config = config or HarmonizationConfig()
    seed_everything(config.seed)
    notes: list[str] = []

    # 1. Five-pass gate: fail closed BEFORE model initialization.
    if gate_outcome is None:
        gate = FivePassDataGate(config)
        gate_outcome = gate.run(source_dir)
    assert_gate_passed(gate_outcome)
    frames = gate_outcome.frames

    # 2. AOI filtering + deterministic daily aggregation.
    if aoi is not None:
        frames = {product: filter_records(frame, aoi) for product, frame in frames.items()}
        aoi_id = aoi.aoi_id
    else:
        aoi_id = "global"
    if frames.get(PRODUCT_MODIS, pd.DataFrame()).empty or frames.get(
        PRODUCT_VIIRS, pd.DataFrame()
    ).empty:
        raise TrainingInputError(
            "Both MODIS and VIIRS observations are required to train the harmonizer; "
            "check the AOI filter and source files."
        )
    daily = build_daily_frame(frames, aoi_id=aoi_id)

    # 3. Overlap matching with explicit rules.
    matched = build_matched_records(
        frames[PRODUCT_MODIS], frames[PRODUCT_VIIRS], config.matching, aoi_id=aoi_id
    )
    elig = eligible_matches(matched)
    if elig.empty:
        raise TrainingInputError(
            "No eligible matched records (VIIRS coverage required); cannot train. "
            "Check matching tolerances and source coverage."
        )

    # 4. Features + chronological split.
    X_all, feature_names = build_feature_matrix(elig)
    splits = split_matched_records(elig, config.split)
    train = splits[SPLIT_TRAIN]
    if train.empty:
        raise TrainingInputError(
            f"No matched training records on or before {config.split.train_end}; "
            "adjust split boundaries to the available overlap period."
        )
    X_train = X_all.loc[train.index]

    # 5. Fit baselines and the selected model (early stopping on the
    #    chronological train tail; validation/test periods stay untouched).
    y_train = train[TARGET_COLUMN].astype(float)
    identity = IdentityBaseline().fit(X_train, y_train)
    calibration = CalibrationBaseline().fit(X_train, y_train)
    model = TweedieGBDTHarmonizer(config.model, seed=config.seed).fit(
        X_train,
        y_train,
        stop_dates=pd.to_datetime(train["modis_acq_datetime"]),
        stopping_days=config.model.early_stopping_days,
    )
    baselines = {identity.name: identity, calibration.name: calibration}
    if calibration.scale_factor_ == 1.0:
        notes.append("calibration scale factor degenerated to 1.0 (zero source energy)")

    # 6. Held-out chronological evaluation with slices.
    evaluation: dict[str, Any] = {}
    comparison: dict[str, Any] = {}
    predictions = {
        identity.name: identity.predict,
        calibration.name: calibration.predict,
        model.name: model.predict,
    }
    for split_name in (SPLIT_VALIDATION, SPLIT_TEST):
        frame = splits[split_name]
        if frame.empty:
            notes.append(f"no records available for the {split_name} split; evaluation skipped")
            evaluation[split_name] = {"skipped": True, "reason": "empty split"}
            comparison[split_name] = {"skipped": True}
            continue
        X_split = X_all.loc[frame.index]
        y_split = frame[TARGET_COLUMN].astype(float)
        months = pd.to_datetime(frame["match_date"]).dt.month
        reports = compare_predictions(
            predictions, X_split, y_split, frp_source=X_split["modis_frp_mw"], months=months
        )
        evaluation[split_name] = reports
        comparison[split_name] = {}
        for baseline_name in (identity.name, calibration.name):
            beats, margin = beats_baseline(reports[model.name], reports[baseline_name])
            comparison[split_name][baseline_name] = {
                "beats": bool(beats),
                "metric": "tweedie_deviance",
                "margin": margin,
            }

    zero_fraction = float((train[TARGET_COLUMN] == 0).mean())
    notes.append(
        f"train zero-target fraction {zero_fraction:.3f} "
        f"({int((train[MATCHED_COUNT_COLUMN] == 0).sum())} MODIS detections with no VIIRS match)"
    )

    return HarmonizationTrainingResult(
        model=model,
        baselines=baselines,
        feature_names=feature_names,
        matched=matched,
        splits=splits,
        evaluation=evaluation,
        baseline_comparison=comparison,
        daily=daily,
        aoi_id=aoi_id,
        config_snapshot=config.to_dict(),
        notes=notes,
    )
