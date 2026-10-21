"""MODIS/VIIRS hotspot harmonization pipeline (NASA Space Apps Challenge 2026).

Package layout follows docs/ARCHITECTURE.md "Planned source boundaries":

- ``src.data``      FIRMS adapters, schemas, five-pass validation gate, daily aggregation
- ``src.geo``       AOI validation/filtering and overlap-period sensor matching
- ``src.harmonization`` features, baselines, LightGBM/Tweedie model, evaluation, inference
- ``src.calendar``  seasonal baselines, anomaly scoring, forecasting, backtests
- ``src.models``    PatchTST architecture and the model registry
- ``src.artifacts`` contracts, manifests, hashing, and serialization
- ``src.common``    configuration and logging helpers
"""

__version__ = "0.1.0"
