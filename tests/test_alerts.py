"""Tests for the early-warning system and notifier."""

import numpy as np

from src.alerts.early_warning import EarlyWarningSystem
from src.alerts.notifier import AlertNotifier, LoggingChannel, format_alerts


def test_severity_thresholds():
    ews = EarlyWarningSystem(watch_sigma=1, warn_sigma=2, crit_sigma=3)
    # baseline mean 100, std 10 -> known z-scores.
    alerts = ews.evaluate([100, 115, 125, 135], baseline_mean=100, baseline_std=10)
    severities = [a.severity for a in alerts]
    assert severities == ["none", "watch", "warning", "critical"]


def test_highest_severity_and_actionable():
    ews = EarlyWarningSystem()
    alerts = ews.evaluate([100, 130], baseline_mean=100, baseline_std=10)
    assert ews.highest_severity(alerts) == "critical"
    assert len(ews.actionable(alerts)) == 1


def test_zero_std_does_not_crash():
    ews = EarlyWarningSystem()
    alerts = ews.evaluate([100, 101], baseline_mean=100, baseline_std=0)
    assert all(np.isfinite(a.z_score) for a in alerts)


def test_expected_excess_floored_at_zero():
    ews = EarlyWarningSystem()
    alerts = ews.evaluate([80], baseline_mean=100, baseline_std=10)
    assert alerts[0].expected_excess == 0.0


def test_invalid_thresholds_rejected():
    try:
        EarlyWarningSystem(watch_sigma=3, warn_sigma=2, crit_sigma=1)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_format_and_notify():
    ews = EarlyWarningSystem()
    quiet = ews.evaluate([100], baseline_mean=100, baseline_std=10)
    assert "No ER demand surge" in format_alerts(quiet)

    surge = ews.evaluate(
        [140], baseline_mean=100, baseline_std=10, dates=["2024-01-01"]
    )
    msg = format_alerts(surge)
    assert "CRITICAL" in msg

    results = AlertNotifier(channels=[LoggingChannel()]).notify(surge)
    assert results == [True]
