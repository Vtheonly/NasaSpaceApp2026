"""Model registry: stable names -> factories for harmonizer and forecaster models.

Names are recorded in artifact manifests; changing a name changes the artifact
contract and must be documented (docs/ARCHITECTURE.md artifact contracts).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from src.common.logging import get_logger

LOGGER = get_logger(__name__)


class UnknownModelError(KeyError):
    """Raised when a model name is absent from the registry."""


class RegistryKindError(TypeError):
    """Raised when a model is requested with the wrong kind."""


class ModelRegistry:
    def __init__(self) -> None:
        self._entries: dict[str, dict[str, Any]] = {}

    def register(
        self,
        name: str,
        kind: str,
        factory: Callable[..., Any],
        version: str,
        requires: str | None = None,
    ) -> None:
        if name in self._entries:
            raise ValueError(f"Model {name!r} is already registered")
        if kind not in ("harmonizer", "forecaster"):
            raise ValueError(f"Unknown kind {kind!r}; expected harmonizer or forecaster")
        self._entries[name] = {
            "kind": kind,
            "factory": factory,
            "version": version,
            "requires": requires,
        }

    def get(self, name: str, kind: str | None = None, **kwargs) -> Any:
        if name not in self._entries:
            raise UnknownModelError(
                f"Unknown model {name!r}; registered models: {sorted(self._entries)}"
            )
        entry = self._entries[name]
        if kind is not None and entry["kind"] != kind:
            raise RegistryKindError(
                f"Model {name!r} is a {entry['kind']}, not a {kind}"
            )
        return entry["factory"](**kwargs)

    def describe(self, kind: str | None = None) -> dict[str, dict[str, Any]]:
        return {
            name: {"kind": e["kind"], "version": e["version"], "requires": e["requires"]}
            for name, e in self._entries.items()
            if kind is None or e["kind"] == kind
        }

    def __contains__(self, name: str) -> bool:
        return name in self._entries


MODEL_REGISTRY = ModelRegistry()


def register_model(name: str, kind: str, factory: Callable[..., Any], version: str, requires: str | None = None) -> None:
    MODEL_REGISTRY.register(name, kind, factory, version, requires)


def get_model(name: str, kind: str | None = None, **kwargs) -> Any:
    return MODEL_REGISTRY.get(name, kind=kind, **kwargs)


def _populate() -> None:
    # Harmonizer models (pillar 1)
    from src.harmonization.baselines import CalibrationBaseline, IdentityBaseline
    from src.harmonization.model import TweedieGBDTHarmonizer

    MODEL_REGISTRY.register("identity", "harmonizer", lambda **kw: IdentityBaseline(), "1.0.0")
    MODEL_REGISTRY.register(
        "calibration", "harmonizer", lambda **kw: CalibrationBaseline(), "1.0.0"
    )
    MODEL_REGISTRY.register(
        "lightgbm_tweedie",
        "harmonizer",
        lambda config=None, seed=2026, **kw: TweedieGBDTHarmonizer(config, seed=seed),
        "1.0.0",
        requires="lightgbm",
    )

    # Forecaster models (pillar 2)
    from src.calendar.forecast import PersistenceForecaster, SeasonalNaiveForecaster
    from src.models.patchtst import PatchTSTForecaster

    MODEL_REGISTRY.register(
        "seasonal_naive",
        "forecaster",
        lambda config=None, seed=2026, **kw: SeasonalNaiveForecaster(config),
        "1.0.0",
    )
    MODEL_REGISTRY.register(
        "persistence",
        "forecaster",
        lambda config=None, seed=2026, **kw: PersistenceForecaster(config),
        "1.0.0",
    )
    MODEL_REGISTRY.register(
        "patchtst",
        "forecaster",
        lambda config=None, seed=2026, **kw: PatchTSTForecaster(config, seed=seed),
        "1.0.0",
        requires="torch",
    )


_populate()


__all__ = ["MODEL_REGISTRY", "register_model", "get_model", "UnknownModelError", "RegistryKindError"]
