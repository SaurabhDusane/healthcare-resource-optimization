"""Tests for the A/B testing framework."""

import numpy as np

from src.analysis.ab_testing import ABTest


def test_proportion_detects_real_difference():
    rng = np.random.default_rng(0)
    control = (rng.random(3000) < 0.30).astype(int)
    treatment = (rng.random(3000) < 0.20).astype(int)
    r = ABTest().proportion_test(control, treatment)
    assert r.significant is True
    assert r.absolute_effect < 0  # treatment lowered the rate
    assert "decrease" in r.decision


def test_proportion_no_difference_not_significant():
    rng = np.random.default_rng(1)
    control = (rng.random(2000) < 0.25).astype(int)
    treatment = (rng.random(2000) < 0.25).astype(int)
    r = ABTest().proportion_test(control, treatment)
    assert r.significant is False
    assert "No significant" in r.decision


def test_mean_test_detects_shift_and_effect_size():
    rng = np.random.default_rng(2)
    control = rng.normal(60, 15, 800)
    treatment = rng.normal(52, 15, 800)
    r = ABTest().mean_test(control, treatment)
    assert r.test_type == "mean"
    assert r.significant is True
    assert r.effect_size < 0  # negative Cohen's d


def test_mean_test_no_difference():
    rng = np.random.default_rng(3)
    control = rng.normal(50, 10, 500)
    treatment = rng.normal(50, 10, 500)
    r = ABTest().mean_test(control, treatment)
    assert r.significant is False


def test_result_is_json_serializable():
    import json

    rng = np.random.default_rng(4)
    r = ABTest().proportion_test(
        (rng.random(500) < 0.3).astype(int), (rng.random(500) < 0.2).astype(int)
    )
    json.dumps(r.to_dict())
    assert set(["p_value", "effect_size", "decision"]).issubset(r.to_dict())
