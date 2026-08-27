"""Tests for exogenous (SARIMAX) forecasting."""

import numpy as np
import pandas as pd

from src.modeling.time_series_forecaster import TimeSeriesForecaster


def _daily_with_exog(n=180, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    signal = rng.normal(0, 1, n)
    weekly = 12 + 3 * np.sin(2 * np.pi * np.arange(n) / 7)
    # Visits depend on the exogenous signal, so exog should help.
    visits = np.clip(weekly + 2.5 * signal + rng.normal(0, 0.5, n), 1, None)
    daily = pd.DataFrame({"date": dates, "visits": visits.round().astype(int)})
    exog = pd.DataFrame({"news_mentions": signal}, index=dates)
    return daily, exog


def test_backtest_exog_returns_expected_keys():
    daily, exog = _daily_with_exog()
    res = TimeSeriesForecaster().backtest_exog(daily, exog, test_size=21)
    for key in [
        "univariate_arima_mae",
        "exog_sarimax_mae",
        "exog_helps",
        "exog_features",
    ]:
        assert key in res
    assert res["exog_features"] == ["news_mentions"]
    assert res["n_test"] == 21


def test_exog_improves_when_signal_is_predictive():
    daily, exog = _daily_with_exog(seed=1)
    res = TimeSeriesForecaster().backtest_exog(daily, exog, test_size=21)
    # With a genuinely predictive exogenous signal, SARIMAX should not be worse
    # than univariate ARIMA by a meaningful margin.
    assert res["exog_sarimax_mae"] is not None
    assert res["exog_sarimax_mae"] <= res["univariate_arima_mae"] * 1.1


def test_train_and_forecast_sarimax_direct():
    daily, exog = _daily_with_exog(n=120)
    f = TimeSeriesForecaster()
    series = f.prepare_series(daily)
    exog.index = pd.to_datetime(exog.index)
    exog = exog.reindex(series.index).ffill().fillna(0.0)
    train_exog, test_exog = exog.iloc[:-10], exog.iloc[-10:]
    f.train_sarimax(series.iloc[:-10], train_exog)
    preds = f.forecast_sarimax(test_exog, 10)
    assert len(preds) == 10
    assert np.all(np.isfinite(preds))
