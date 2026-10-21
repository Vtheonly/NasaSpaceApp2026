"""Pillar 1 - sensor harmonization (docs/ARCHITECTURE.md).

Features, identity/calibration baselines, the selected LightGBM Tweedie
gradient-boosted harmonizer, held-out chronological evaluation, training
orchestration, and inference into harmonized daily series.
"""

from src.harmonization.baselines import CalibrationBaseline, IdentityBaseline
from src.harmonization.model import TweedieGBDTHarmonizer

__all__ = [
    "IdentityBaseline",
    "CalibrationBaseline",
    "TweedieGBDTHarmonizer",
]
