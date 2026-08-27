"""Tests for the data-drift monitor."""

import numpy as np
import pandas as pd

from src.monitoring.drift import DriftMonitor


def _frames(shift=0.0, seed=0, n=800):
    rng = np.random.default_rng(seed)
    ref = pd.DataFrame({"a": rng.normal(0, 1, n), "b": rng.normal(5, 2, n)})
    cur = pd.DataFrame({"a": rng.normal(0, 1, n), "b": rng.normal(5 + shift, 2, n)})
    return ref, cur


def test_no_drift_when_distributions_match():
    ref, cur = _frames(shift=0.0)
    report = DriftMonitor().compare(ref, cur)
    assert report.n_drifted == 0
    assert report.drift_share == 0.0


def test_detects_shifted_feature():
    ref, cur = _frames(shift=4.0)
    report = DriftMonitor().compare(ref, cur)
    assert "b" in report.drifted_features
    assert "a" not in report.drifted_features


def test_psi_zero_for_identical_samples():
    x = np.linspace(0, 10, 500)
    assert DriftMonitor.population_stability_index(x, x) == 0.0


def test_psi_positive_for_shift():
    rng = np.random.default_rng(1)
    ref = rng.normal(0, 1, 1000)
    cur = rng.normal(3, 1, 1000)
    assert DriftMonitor.population_stability_index(ref, cur) > 0.2


def test_profile_roundtrip(tmp_path):
    ref, cur = _frames(shift=4.0)
    monitor = DriftMonitor()
    path = str(tmp_path / "profile.json")
    monitor.save_profile(ref, path)
    report = monitor.compare_to_profile(cur, path)
    assert "b" in report.drifted_features


def test_to_dict_is_json_serializable():
    import json

    ref, cur = _frames(shift=2.0)
    report = DriftMonitor().compare(ref, cur)
    json.dumps(report.to_dict())
