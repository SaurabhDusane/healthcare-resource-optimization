"""Tests for the time-series forecaster."""

import numpy as np
import pandas as pd

from src.modeling.time_series_forecaster import TimeSeriesForecaster


def _daily_series(n_days=180, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-01", periods=n_days, freq="D")
    # Weekly seasonality + slight trend + noise.
    weekly = 10 + 4 * np.sin(2 * np.pi * np.arange(n_days) / 7)
    trend = np.arange(n_days) * 0.02
    visits = np.clip(weekly + trend + rng.normal(0, 1.0, n_days), 1, None)
    return pd.DataFrame({"date": dates, "visits": visits.round().astype(int)})


def test_prepare_series_is_daily_and_indexed():
    f = TimeSeriesForecaster()
    s = f.prepare_series(_daily_series())
    assert isinstance(s.index, pd.DatetimeIndex)
    assert s.index.freqstr == "D"


def test_train_test_split_is_chronological():
    f = TimeSeriesForecaster()
    s = f.prepare_series(_daily_series(n_days=100))
    train, test = f.train_test_split(s, test_size=20)
    assert len(test) == 20
    assert train.index.max() < test.index.min()


def test_seasonal_naive_length_and_period():
    f = TimeSeriesForecaster(season_length=7)
    s = f.prepare_series(_daily_series())
    train, _ = f.train_test_split(s, test_size=14)
    pred = f.seasonal_naive_forecast(train, periods=14)
    assert len(pred) == 14
    # Second week repeats the first week of the forecast.
    np.testing.assert_array_equal(pred[:7], pred[7:14])


def test_backtest_ranks_models_and_reports_window():
    f = TimeSeriesForecaster()
    result = f.backtest(
        _daily_series(n_days=150), test_size=21, models=["seasonal_naive", "arima"]
    )
    assert result["best_model"] in {"seasonal_naive", "arima"}
    assert result["n_test"] == 21
    assert "MAE" in result["best_metrics"]
    assert len(result["test_window"]) == 2


def test_evaluate_forecast_perfect_prediction():
    y = np.array([10.0, 12.0, 8.0])
    metrics = TimeSeriesForecaster.evaluate_forecast(y, y)
    assert metrics["MAE"] == 0
    assert metrics["MAPE"] == 0
