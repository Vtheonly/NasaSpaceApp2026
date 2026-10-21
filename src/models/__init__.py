"""Model architectures and the model registry.

PatchTST is the selected temporal-model candidate for the 14-day regional
activity forecast (docs/ARCHITECTURE.md Pillar 2). PyTorch is an optional
dependency: importing this package never fails, but constructing or training
the PatchTST forecaster without torch raises a clear, offline-friendly error
while the seasonal-naive/persistence baselines stay fully functional.
"""

from src.models.registry import MODEL_REGISTRY, get_model, register_model

__all__ = ["MODEL_REGISTRY", "register_model", "get_model"]
