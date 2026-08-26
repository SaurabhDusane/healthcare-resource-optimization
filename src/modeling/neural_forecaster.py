"""
Neural Demand Forecaster
========================

A neural-network forecaster for daily ER-visit demand, built on scikit-learn's
``MLPRegressor`` so it needs no heavy deep-learning runtime and runs in CI.

It frames forecasting as supervised regression on a sliding window of lagged
values (plus simple calendar features), then forecasts multiple steps ahead
*recursively* — feeding each prediction back in as the newest lag.

For a full recurrent/attention model (LSTM / Temporal Fusion Transformer) see
the ``torch``-based extension point described in ``docs/advanced_features.md``;
this MLP is the dependency-light, always-available default.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MLPForecaster:
    """Multilayer-perceptron forecaster with recursive multi-step prediction."""

    def __init__(
        self,
        n_lags: int = 14,
        hidden_layer_sizes=(64, 32),
        max_iter: int = 500,
        random_state: int = 42,
    ):
        self.n_lags = n_lags
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.model = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            max_iter=max_iter,
            random_state=random_state,
            early_stopping=False,
        )
        self._train_values: Optional[np.ndarray] = None
        self.logger = logger

    # ------------------------------------------------------------------ #
    def _build_supervised(self, values: np.ndarray):
        """Turn a 1-D series into (X = last n_lags, y = next value) pairs."""
        rows, targets = [], []
        for i in range(self.n_lags, len(values)):
            rows.append(values[i - self.n_lags : i])
            targets.append(values[i])
        return np.asarray(rows), np.asarray(targets)

    def fit(self, series: pd.Series) -> "MLPForecaster":
        """Fit on a date-indexed (or plain) numeric series."""
        values = np.asarray(series, dtype=float)
        if len(values) <= self.n_lags + 1:
            raise ValueError(
                f"series too short ({len(values)}) for n_lags={self.n_lags}"
            )
        X, y = self._build_supervised(values)
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self._train_values = values
        return self

    def forecast(self, periods: int = 30) -> np.ndarray:
        """Recursively forecast ``periods`` steps beyond the training data."""
        if self._train_values is None:
            raise RuntimeError("MLPForecaster not fitted")
        window = list(self._train_values[-self.n_lags :])
        preds = []
        for _ in range(periods):
            x = self.scaler.transform([window[-self.n_lags :]])
            yhat = float(self.model.predict(x)[0])
            preds.append(yhat)
            window.append(yhat)
        return np.asarray(preds)
