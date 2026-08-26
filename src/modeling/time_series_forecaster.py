"""
Time Series Forecasting Module
==============================

Forecasts daily ER-visit demand with three interchangeable approaches and a
chronological backtest so metrics are honest (no future leakage):

  * ``seasonal_naive``  - repeats the last observed weekly cycle (baseline)
  * ``arima``           - statsmodels ARIMA / SARIMA
  * ``prophet``         - Facebook Prophet (optional; used only if installed)

The baseline and ARIMA depend only on the core stack; Prophet is imported
lazily so its absence never breaks a run.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TimeSeriesForecaster:
    """Daily-demand forecasting with baseline, ARIMA and optional Prophet."""

    def __init__(self, season_length: int = 7):
        self.logger = logger
        self.season_length = season_length
        self.model = None
        self.forecast = None

    # ------------------------------------------------------------------ #
    # Data preparation                                                   #
    # ------------------------------------------------------------------ #
    def prepare_series(
        self, df: pd.DataFrame, target_col: str = "visits", date_col: str = "date"
    ) -> pd.Series:
        """Return a date-indexed, daily-frequency, gap-filled target series."""
        s = df.copy()
        s[date_col] = pd.to_datetime(s[date_col])
        s = s.set_index(date_col).sort_index()[target_col].astype(float)
        # Ensure a regular daily frequency; forward-fill any missing days.
        s = s.asfreq("D").ffill()
        return s

    @staticmethod
    def train_test_split(
        series: pd.Series, test_size: int = 30
    ) -> Tuple[pd.Series, pd.Series]:
        """Chronological split: the last ``test_size`` points are the holdout."""
        if test_size >= len(series):
            raise ValueError("test_size must be smaller than the series length")
        return series.iloc[:-test_size], series.iloc[-test_size:]

    # ------------------------------------------------------------------ #
    # Models                                                             #
    # ------------------------------------------------------------------ #
    def seasonal_naive_forecast(self, train: pd.Series, periods: int) -> np.ndarray:
        """Repeat the last full seasonal cycle to cover ``periods`` steps."""
        season = self.season_length
        last_cycle = train.iloc[-season:].to_numpy()
        reps = int(np.ceil(periods / season))
        return np.tile(last_cycle, reps)[:periods]

    def train_arima(self, train: pd.Series, order: Tuple[int, int, int] = (5, 1, 1)):
        """Fit an ARIMA model (statsmodels)."""
        from statsmodels.tsa.arima.model import ARIMA

        self.logger.info("Training ARIMA%s ...", order)
        self.model = ARIMA(train, order=order).fit()
        return self.model

    def forecast_arima(self, periods: int = 30) -> np.ndarray:
        """Forecast ``periods`` steps from a fitted ARIMA model."""
        if self.model is None:
            raise RuntimeError("ARIMA model not trained")
        return np.asarray(self.model.forecast(steps=periods))

    def train_prophet(self, train: pd.Series):
        """Fit a Prophet model (optional dependency)."""
        try:
            from prophet import Prophet
        except ImportError:
            self.logger.warning("Prophet not installed; skipping Prophet model")
            return None

        df_prophet = pd.DataFrame({"ds": train.index, "y": train.to_numpy()})
        model = Prophet(
            weekly_seasonality=True,
            yearly_seasonality=True,
            daily_seasonality=False,
        )
        model.fit(df_prophet)
        self.model = model
        return model

    def forecast_prophet(self, periods: int = 30) -> Optional[np.ndarray]:
        """Forecast ``periods`` steps from a fitted Prophet model."""
        if self.model is None:
            return None
        future = self.model.make_future_dataframe(periods=periods)
        forecast = self.model.predict(future)
        self.forecast = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]
        return forecast["yhat"].to_numpy()[-periods:]

    # ------------------------------------------------------------------ #
    # Evaluation & backtest                                              #
    # ------------------------------------------------------------------ #
    @staticmethod
    def evaluate_forecast(y_true, y_pred) -> Dict[str, float]:
        """Compute MAE, RMSE, MAPE and a 100-MAPE 'accuracy' figure."""
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        # Guard against divide-by-zero in MAPE.
        nonzero = y_true != 0
        mape = (
            float(
                np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero]))
                * 100
            )
            if nonzero.any()
            else float("nan")
        )
        return {
            "MAE": round(mae, 4),
            "RMSE": round(rmse, 4),
            "MAPE": round(mape, 4),
            "Accuracy": round(100 - mape, 4) if not np.isnan(mape) else float("nan"),
        }

    def backtest(
        self,
        df: pd.DataFrame,
        target_col: str = "visits",
        date_col: str = "date",
        test_size: int = 30,
        arima_order: Tuple[int, int, int] = (5, 1, 1),
        models: Optional[List[str]] = None,
    ) -> Dict[str, object]:
        """
        Chronologically backtest each model on a holdout and rank by MAE.

        Returns a dict with per-model metrics, the best model name, and the
        holdout window, suitable for logging to an experiment tracker.
        """
        models = models or ["seasonal_naive", "arima", "prophet"]
        series = self.prepare_series(df, target_col=target_col, date_col=date_col)
        train, test = self.train_test_split(series, test_size=test_size)
        horizon = len(test)

        results: Dict[str, Dict[str, float]] = {}

        if "seasonal_naive" in models:
            pred = self.seasonal_naive_forecast(train, horizon)
            results["seasonal_naive"] = self.evaluate_forecast(test.to_numpy(), pred)

        if "arima" in models:
            try:
                self.train_arima(train, order=arima_order)
                pred = self.forecast_arima(horizon)
                results["arima"] = self.evaluate_forecast(test.to_numpy(), pred)
            except Exception as exc:  # pragma: no cover - numerical edge cases
                self.logger.warning("ARIMA backtest failed: %s", exc)

        if "prophet" in models:
            model = self.train_prophet(train)
            if model is not None:
                pred = self.forecast_prophet(horizon)
                if pred is not None:
                    results["prophet"] = self.evaluate_forecast(test.to_numpy(), pred)

        if not results:
            raise RuntimeError("No forecasting model produced a result")

        best = min(results, key=lambda m: results[m]["MAE"])
        return {
            "metrics_by_model": results,
            "best_model": best,
            "best_metrics": results[best],
            "n_train": int(len(train)),
            "n_test": int(len(test)),
            "test_window": [
                str(test.index.min().date()),
                str(test.index.max().date()),
            ],
            "arima_order": list(arima_order),
        }


if __name__ == "__main__":
    print("TimeSeriesForecaster module loaded successfully")
