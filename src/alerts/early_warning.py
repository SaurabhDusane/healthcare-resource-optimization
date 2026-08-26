"""
Early-Warning System
====================

Turns a demand forecast into actionable surge alerts for hospital staffing.

Each forecasted day is compared against a historical baseline (mean + k*std).
The number of standard deviations above the mean determines a severity level:

    z < watch_sigma                -> none
    watch_sigma <= z < warn_sigma  -> watch
    warn_sigma  <= z < crit_sigma  -> warning
    z >= crit_sigma                -> critical

Thresholds are configurable; defaults are deliberately conservative.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SEVERITY_ORDER = ["none", "watch", "warning", "critical"]


@dataclass
class Alert:
    """A single-day surge alert."""

    date: str
    predicted_visits: float
    baseline_mean: float
    z_score: float
    severity: str
    expected_excess: float  # predicted - baseline_mean, floored at 0

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


class EarlyWarningSystem:
    """Derive surge alerts from a forecast relative to a historical baseline."""

    def __init__(
        self,
        watch_sigma: float = 1.0,
        warn_sigma: float = 2.0,
        crit_sigma: float = 3.0,
    ):
        if not watch_sigma < warn_sigma < crit_sigma:
            raise ValueError("thresholds must satisfy watch < warn < crit")
        self.watch_sigma = watch_sigma
        self.warn_sigma = warn_sigma
        self.crit_sigma = crit_sigma
        self.logger = logger

    def _severity(self, z: float) -> str:
        if z >= self.crit_sigma:
            return "critical"
        if z >= self.warn_sigma:
            return "warning"
        if z >= self.watch_sigma:
            return "watch"
        return "none"

    def evaluate(
        self,
        forecast_values: Sequence[float],
        baseline_mean: float,
        baseline_std: float,
        dates: Optional[Sequence[str]] = None,
    ) -> List[Alert]:
        """
        Score each forecast point and return one Alert per day.

        Args:
            forecast_values: Predicted daily visit counts.
            baseline_mean: Historical mean daily visits.
            baseline_std: Historical std of daily visits (>0).
            dates: Optional ISO date strings aligned with ``forecast_values``.
        """
        values = np.asarray(forecast_values, dtype=float)
        # Guard a degenerate baseline so we never divide by zero.
        std = float(baseline_std) if baseline_std and baseline_std > 0 else 1.0
        if dates is None:
            dates = [f"t+{i + 1}" for i in range(len(values))]

        alerts: List[Alert] = []
        for date, value in zip(dates, values):
            z = (value - baseline_mean) / std
            alerts.append(
                Alert(
                    date=str(date),
                    predicted_visits=round(float(value), 2),
                    baseline_mean=round(float(baseline_mean), 2),
                    z_score=round(float(z), 3),
                    severity=self._severity(z),
                    expected_excess=round(max(float(value - baseline_mean), 0.0), 2),
                )
            )
        return alerts

    @staticmethod
    def highest_severity(alerts: Sequence[Alert]) -> str:
        """Return the most severe level across a set of alerts."""
        worst = "none"
        for alert in alerts:
            if SEVERITY_ORDER.index(alert.severity) > SEVERITY_ORDER.index(worst):
                worst = alert.severity
        return worst

    @staticmethod
    def actionable(alerts: Sequence[Alert]) -> List[Alert]:
        """Filter to alerts that warrant attention (severity above 'none')."""
        return [a for a in alerts if a.severity != "none"]
