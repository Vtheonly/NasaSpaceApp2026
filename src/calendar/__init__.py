"""Pillar 2 - burning-activity calendar (docs/ARCHITECTURE.md).

Seasonal day-of-year climatology, anomaly scoring against the documented
ratio threshold, forecast baselines + PatchTST with rolling-origin backtests,
and probabilistic forecast evaluation (pinball loss, quantile coverage).
"""

from src.calendar.anomalies import STATUS_ANOMALY_HIGH, score_anomalies
from src.calendar.seasonal import SeasonalClimatology

__all__ = ["SeasonalClimatology", "score_anomalies", "STATUS_ANOMALY_HIGH"]
