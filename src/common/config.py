"""Typed configuration objects for the harmonization and calendar pipelines.

Every configuration value that has a scientific meaning (matching tolerances,
split boundaries, anomaly thresholds, PatchTST starting values, quantiles) is
defined here so it can be serialized into artifact manifests and reproduced.

Split boundaries follow docs/TESTING.md pass 3: train 2000-2021, validation
2022-2023, test 2024 onward (proposed starting split, subject to available data).
PatchTST starting values follow docs/ARCHITECTURE.md Pillar 2 (16-day patches,
stride 8, dimension 128, four encoder blocks, eight heads, dropout 0.1) and are
tunable, not proven optimal.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import numpy as np

try:  # PyYAML is a core dependency; keep the import lazy-friendly for lint.
    import yaml
except ImportError:  # pragma: no cover - exercised only in broken environments
    yaml = None

from src.common.logging import get_logger

LOGGER = get_logger(__name__)


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def _take(data: dict[str, Any], key: str, default: Any) -> Any:
    """Return *data[key]* when present else *default*; warn about unknown keys."""
    if key in data:
        return data[key]
    return default


def _warn_unknown(data: dict[str, Any], known: set[str], context: str) -> None:
    unknown = set(data) - known
    if unknown:
        LOGGER.warning("Ignoring unknown configuration keys in %s: %s", context, sorted(unknown))


def seed_everything(seed: int) -> None:
    """Seed Python and NumPy RNGs (PyTorch is seeded inside src.models.patchtst)."""
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))


def resolve_code_commit(repo_root: str | Path | None = None) -> str:
    """Return the current git commit hash, or ``"unknown"`` when git is unavailable."""
    import subprocess

    root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[2]
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Harmonization configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SplitConfig:
    """Chronological split boundaries (inclusive end dates; docs/TESTING.md pass 3)."""

    train_end: str = "2021-12-31"
    validation_end: str = "2023-12-31"

    @staticmethod
    def from_dict(data: dict[str, Any]) -> SplitConfig:
        keys = {f.name for f in fields(SplitConfig)}
        _warn_unknown(data, keys, "split")
        return SplitConfig(
            train_end=_take(data, "train_end", SplitConfig.train_end),
            validation_end=_take(data, "validation_end", SplitConfig.validation_end),
        )


@dataclass(frozen=True)
class MatchingConfig:
    """Sensor-matching rules (docs/ARCHITECTURE.md Pillar 1).

    The MODIS footprint half-width is derived from the ``scan``/``track`` pixel
    size (degrees) when those columns exist, otherwise ``default_footprint_deg``
    is used. The candidate VIIRS pixels must lie inside the footprint box and
    within ``temporal_tolerance_minutes`` of the MODIS overpass.
    """

    temporal_tolerance_minutes: int = 90
    min_footprint_radius_deg: float = 0.01
    max_footprint_radius_deg: float = 0.25
    default_footprint_radius_deg: float = 0.02
    min_confidence: str = "l"
    deduplicate_modis: bool = True

    @staticmethod
    def from_dict(data: dict[str, Any]) -> MatchingConfig:
        keys = {f.name for f in fields(MatchingConfig)}
        _warn_unknown(data, keys, "matching")
        return MatchingConfig(
            temporal_tolerance_minutes=int(
                _take(data, "temporal_tolerance_minutes", MatchingConfig.temporal_tolerance_minutes)
            ),
            min_footprint_radius_deg=float(
                _take(data, "min_footprint_radius_deg", MatchingConfig.min_footprint_radius_deg)
            ),
            max_footprint_radius_deg=float(
                _take(data, "max_footprint_radius_deg", MatchingConfig.max_footprint_radius_deg)
            ),
            default_footprint_radius_deg=float(
                _take(
                    data, "default_footprint_radius_deg", MatchingConfig.default_footprint_radius_deg
                )
            ),
            min_confidence=str(_take(data, "min_confidence", MatchingConfig.min_confidence)),
            deduplicate_modis=bool(_take(data, "deduplicate_modis", True)),
        )


@dataclass(frozen=True)
class HarmonizerModelConfig:
    """LightGBM Tweedie gradient-boosted regressor settings (selected candidate).

    Defaults are regularized for small matched-sample sizes and paired with
    chronological early stopping (see TweedieGBDTHarmonizer.fit); they are
    documented starting values, not tuned truths (docs: planning estimates
    are not evidence until measured).
    """

    tweedie_variance_power: float = 1.5
    n_estimators: int = 2000
    learning_rate: float = 0.03
    num_leaves: int = 15
    min_child_samples: int = 40
    feature_fraction: float = 0.8
    bagging_fraction: float = 0.8
    bagging_freq: int = 1
    lambda_l2: float = 5.0
    early_stopping_rounds: int = 50
    early_stopping_days: int = 365

    @staticmethod
    def from_dict(data: dict[str, Any]) -> HarmonizerModelConfig:
        keys = {f.name for f in fields(HarmonizerModelConfig)}
        _warn_unknown(data, keys, "harmonizer model")
        defaults = HarmonizerModelConfig()
        return HarmonizerModelConfig(
            tweedie_variance_power=float(
                _take(data, "tweedie_variance_power", defaults.tweedie_variance_power)
            ),
            n_estimators=int(_take(data, "n_estimators", defaults.n_estimators)),
            learning_rate=float(_take(data, "learning_rate", defaults.learning_rate)),
            num_leaves=int(_take(data, "num_leaves", defaults.num_leaves)),
            min_child_samples=int(
                _take(data, "min_child_samples", defaults.min_child_samples)
            ),
            feature_fraction=float(_take(data, "feature_fraction", defaults.feature_fraction)),
            bagging_fraction=float(_take(data, "bagging_fraction", defaults.bagging_fraction)),
            bagging_freq=int(_take(data, "bagging_freq", defaults.bagging_freq)),
            lambda_l2=float(_take(data, "lambda_l2", defaults.lambda_l2)),
            early_stopping_rounds=int(
                _take(data, "early_stopping_rounds", defaults.early_stopping_rounds)
            ),
            early_stopping_days=int(
                _take(data, "early_stopping_days", defaults.early_stopping_days)
            ),
        )


@dataclass(frozen=True)
class HarmonizationConfig:
    """Top-level configuration for pillar 1 (sensor harmonization)."""

    seed: int = 2026
    frp_units: str = "MW"
    extreme_frp_review_threshold_mw: float = 5000.0
    split: SplitConfig = field(default_factory=SplitConfig)
    matching: MatchingConfig = field(default_factory=MatchingConfig)
    model: HarmonizerModelConfig = field(default_factory=HarmonizerModelConfig)

    @staticmethod
    def from_dict(data: dict[str, Any]) -> HarmonizationConfig:
        keys = {f.name for f in fields(HarmonizationConfig)}
        _warn_unknown(data, keys, "harmonization")
        defaults = HarmonizationConfig()
        return HarmonizationConfig(
            seed=int(_take(data, "seed", defaults.seed)),
            frp_units=str(_take(data, "frp_units", defaults.frp_units)),
            extreme_frp_review_threshold_mw=float(
                _take(
                    data,
                    "extreme_frp_review_threshold_mw",
                    defaults.extreme_frp_review_threshold_mw,
                )
            ),
            split=SplitConfig.from_dict(_take(data, "split", {}) or {}),
            matching=MatchingConfig.from_dict(_take(data, "matching", {}) or {}),
            model=HarmonizerModelConfig.from_dict(_take(data, "model", {}) or {}),
        )

    @staticmethod
    def from_yaml(path: str | Path) -> HarmonizationConfig:
        return HarmonizationConfig.from_dict(load_yaml_file(path))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Calendar / forecast configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SeasonalConfig:
    """Day-of-year climatology settings (docs/ARCHITECTURE.md Pillar 2)."""

    window_days: int = 7
    min_samples_per_day: int = 2
    # Leap-year policy: Feb 29 samples are pooled with Feb 28 for baseline
    # estimation so that the seasonal baseline never depends on a single day.
    leap_policy: str = "pool_feb29_with_feb28"

    @staticmethod
    def from_dict(data: dict[str, Any]) -> SeasonalConfig:
        keys = {f.name for f in fields(SeasonalConfig)}
        _warn_unknown(data, keys, "seasonal")
        defaults = SeasonalConfig()
        return SeasonalConfig(
            window_days=int(_take(data, "window_days", defaults.window_days)),
            min_samples_per_day=int(
                _take(data, "min_samples_per_day", defaults.min_samples_per_day)
            ),
            leap_policy=str(_take(data, "leap_policy", defaults.leap_policy)),
        )


@dataclass(frozen=True)
class AnomalyConfig:
    """Anomaly-score definition: ratio of observed to seasonal baseline.

    The epsilon stabilizes days with zero activity. Thresholds are documented
    defaults that must be validated before operational use (docs: any alert
    threshold must be validated and documented).
    """

    ratio_epsilon: float = 1e-6
    high_threshold: float = 2.0
    low_threshold: float = 0.5

    @staticmethod
    def from_dict(data: dict[str, Any]) -> AnomalyConfig:
        keys = {f.name for f in fields(AnomalyConfig)}
        _warn_unknown(data, keys, "anomaly")
        defaults = AnomalyConfig()
        return AnomalyConfig(
            ratio_epsilon=float(_take(data, "ratio_epsilon", defaults.ratio_epsilon)),
            high_threshold=float(_take(data, "high_threshold", defaults.high_threshold)),
            low_threshold=float(_take(data, "low_threshold", defaults.low_threshold)),
        )


@dataclass(frozen=True)
class PatchTSTParams:
    """PatchTST starting values (tunable, per docs/ARCHITECTURE.md Pillar 2)."""

    context_len: int = 90
    patch_len: int = 16
    stride: int = 8
    d_model: int = 128
    n_encoder_blocks: int = 4
    n_heads: int = 8
    dropout: float = 0.1
    dim_feedforward: int = 512

    @staticmethod
    def from_dict(data: dict[str, Any]) -> PatchTSTParams:
        keys = {f.name for f in fields(PatchTSTParams)}
        _warn_unknown(data, keys, "patchtst")
        defaults = PatchTSTParams()
        return PatchTSTParams(
            context_len=int(_take(data, "context_len", defaults.context_len)),
            patch_len=int(_take(data, "patch_len", defaults.patch_len)),
            stride=int(_take(data, "stride", defaults.stride)),
            d_model=int(_take(data, "d_model", defaults.d_model)),
            n_encoder_blocks=int(_take(data, "n_encoder_blocks", defaults.n_encoder_blocks)),
            n_heads=int(_take(data, "n_heads", defaults.n_heads)),
            dropout=float(_take(data, "dropout", defaults.dropout)),
            dim_feedforward=int(_take(data, "dim_feedforward", defaults.dim_feedforward)),
        )

    @property
    def n_patches(self) -> int:
        """Number of patches produced by ``unfold`` for a full context window."""
        if self.patch_len > self.context_len:
            raise ValueError(
                f"patch_len ({self.patch_len}) must not exceed context_len ({self.context_len})"
            )
        return 1 + (self.context_len - self.patch_len) // max(self.stride, 1)


@dataclass(frozen=True)
class TrainerConfig:
    """PatchTST training-loop settings."""

    epochs: int = 60
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 6
    validation_fraction: float = 0.2

    @staticmethod
    def from_dict(data: dict[str, Any]) -> TrainerConfig:
        keys = {f.name for f in fields(TrainerConfig)}
        _warn_unknown(data, keys, "trainer")
        defaults = TrainerConfig()
        return TrainerConfig(
            epochs=int(_take(data, "epochs", defaults.epochs)),
            batch_size=int(_take(data, "batch_size", defaults.batch_size)),
            learning_rate=float(_take(data, "learning_rate", defaults.learning_rate)),
            weight_decay=float(_take(data, "weight_decay", defaults.weight_decay)),
            patience=int(_take(data, "patience", defaults.patience)),
            validation_fraction=float(
                _take(data, "validation_fraction", defaults.validation_fraction)
            ),
        )


@dataclass(frozen=True)
class ForecastConfig:
    """Probabilistic 14-day regional activity forecast settings (docs: never
    forecast exact ignition coordinates/times; quantiles are experimental until
    calibrated)."""

    horizon_days: int = 14
    quantiles: tuple[float, ...] = (0.1, 0.5, 0.9)
    patchtst: PatchTSTParams = field(default_factory=PatchTSTParams)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)

    @staticmethod
    def from_dict(data: dict[str, Any]) -> ForecastConfig:
        defaults = ForecastConfig()
        quantiles = _take(data, "quantiles", list(defaults.quantiles))
        quantiles = tuple(float(q) for q in quantiles)
        if not quantiles or not all(0.0 < q < 1.0 for q in quantiles):
            raise ValueError(f"quantiles must be within (0, 1); got {quantiles}")
        if 0.5 not in quantiles:
            raise ValueError("quantiles must include the median 0.5 for point forecasts")
        return ForecastConfig(
            horizon_days=int(_take(data, "horizon_days", defaults.horizon_days)),
            quantiles=quantiles,
            patchtst=PatchTSTParams.from_dict(_take(data, "patchtst", {}) or {}),
            trainer=TrainerConfig.from_dict(_take(data, "trainer", {}) or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BacktestConfig:
    """Rolling-origin backtest protocol (docs/TESTING.md: rolling-origin backtests
    against seasonal-naive and persistence baselines; never shuffle temporally)."""

    n_origins: int = 4
    origin_spacing_days: int = 14
    min_train_days: int = 90

    @staticmethod
    def from_dict(data: dict[str, Any]) -> BacktestConfig:
        keys = {f.name for f in fields(BacktestConfig)}
        _warn_unknown(data, keys, "backtest")
        defaults = BacktestConfig()
        return BacktestConfig(
            n_origins=int(_take(data, "n_origins", defaults.n_origins)),
            origin_spacing_days=int(
                _take(data, "origin_spacing_days", defaults.origin_spacing_days)
            ),
            min_train_days=int(_take(data, "min_train_days", defaults.min_train_days)),
        )


@dataclass(frozen=True)
class CalendarConfig:
    """Top-level configuration for pillar 2 (calendar, anomalies, forecast)."""

    seed: int = 2026
    seasonal: SeasonalConfig = field(default_factory=SeasonalConfig)
    anomaly: AnomalyConfig = field(default_factory=AnomalyConfig)
    forecast: ForecastConfig = field(default_factory=ForecastConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)

    @staticmethod
    def from_dict(data: dict[str, Any]) -> CalendarConfig:
        keys = {f.name for f in fields(CalendarConfig)}
        _warn_unknown(data, keys, "calendar")
        defaults = CalendarConfig()
        return CalendarConfig(
            seed=int(_take(data, "seed", defaults.seed)),
            seasonal=SeasonalConfig.from_dict(_take(data, "seasonal", {}) or {}),
            anomaly=AnomalyConfig.from_dict(_take(data, "anomaly", {}) or {}),
            forecast=ForecastConfig.from_dict(_take(data, "forecast", {}) or {}),
            backtest=BacktestConfig.from_dict(_take(data, "backtest", {}) or {}),
        )

    @staticmethod
    def from_yaml(path: str | Path) -> CalendarConfig:
        return CalendarConfig.from_dict(load_yaml_file(path))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_yaml_file(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file into a plain dictionary ({} when empty)."""
    if yaml is None:  # pragma: no cover
        raise RuntimeError(
            "PyYAML is required to read configuration files; install project dependencies."
        )
    with open(path, encoding="utf-8") as fh:
        loaded = yaml.safe_load(fh)
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Configuration file must contain a mapping at top level: {path}")
    return loaded
