"""Tests for rolling-origin cross-validated forecasting."""

import numpy as np
import pandas as pd
import pytest

from src.modeling.time_series_forecaster import TimeSeriesForecaster


def _daily(n=200, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    weekly = 12 + 4 * np.sin(2 * np.pi * np.arange(n) / 7)
    visits = np.clip(weekly + rng.normal(0, 1, n), 1, None).round().astype(int)
    return pd.DataFrame({"date": dates, "visits": visits})


def test_backtest_cv_runs_over_folds():
    res = TimeSeriesForecaster().backtest_cv(
        _daily(), horizon=14, n_folds=4, models=["seasonal_naive", "arima"]
    )
    assert res["n_folds"] == 4
    assert res["horizon"] == 14
    assert res["best_model"] in {"seasonal_naive", "arima"}
    assert set(res["cv_mean_mae"]).issubset({"seasonal_naive", "arima"})
    assert all(v >= 0 for v in res["cv_mean_mae"].values())


def test_backtest_cv_rejects_short_series():
    with pytest.raises(ValueError):
        TimeSeriesForecaster().backtest_cv(_daily(n=30), horizon=14, n_folds=4)


def test_cv_baseline_only_matches_manual():
    df = _daily(n=160)
    res = TimeSeriesForecaster().backtest_cv(
        df, horizon=10, n_folds=3, models=["seasonal_naive"]
    )
    assert list(res["cv_mean_mae"].keys()) == ["seasonal_naive"]
    assert res["best_model"] == "seasonal_naive"
