"""Tests for the MLP neural forecaster."""

import numpy as np
import pandas as pd
import pytest

from src.modeling.neural_forecaster import MLPForecaster


def _series(n=120, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-01", periods=n, freq="D")
    values = 20 + 5 * np.sin(2 * np.pi * np.arange(n) / 7) + rng.normal(0, 0.5, n)
    return pd.Series(values, index=idx)


def test_fit_and_forecast_shape():
    f = MLPForecaster(n_lags=7, max_iter=200).fit(_series())
    preds = f.forecast(periods=10)
    assert preds.shape == (10,)
    assert np.all(np.isfinite(preds))


def test_forecast_is_reasonable_range():
    s = _series()
    preds = MLPForecaster(n_lags=7, max_iter=300).fit(s).forecast(14)
    # Predictions should stay in a sane neighbourhood of the training range.
    assert preds.min() > s.min() - 3 * s.std()
    assert preds.max() < s.max() + 3 * s.std()


def test_too_short_series_raises():
    with pytest.raises(ValueError):
        MLPForecaster(n_lags=14).fit(_series(n=10))


def test_forecast_before_fit_raises():
    with pytest.raises(RuntimeError):
        MLPForecaster().forecast(5)


def test_deterministic_with_seed():
    s = _series()
    a = MLPForecaster(n_lags=7, max_iter=200, random_state=1).fit(s).forecast(5)
    b = MLPForecaster(n_lags=7, max_iter=200, random_state=1).fit(s).forecast(5)
    np.testing.assert_allclose(a, b)
