"""
FastAPI Service
===============

Serves the trained artifacts as a small API plus a minimal HTML dashboard:

    GET  /                 HTML dashboard (KPIs, forecast, alerts)
    GET  /health           liveness + what artifacts are loaded
    GET  /metrics          the latest pipeline_metrics.json
    POST /predict/acuity   probability a visit is high-acuity
    GET  /forecast?days=N  N-day demand forecast (ARIMA on daily visits)
    GET  /alerts?days=N    early-warning surge alerts over the forecast

Artifacts are loaded lazily and defensively, so the app always imports and
starts even before `python main.py` has produced a model — endpoints that need
a missing artifact return a clear 503 instead of crashing the server.

Run locally:  uvicorn src.api.app:app --reload
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from src.alerts.early_warning import EarlyWarningSystem
from src.modeling.time_series_forecaster import TimeSeriesForecaster

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api")


@dataclass
class Paths:
    """Filesystem locations of the artifacts the API serves."""

    models_dir: str = os.environ.get("MODELS_DIR", "models")
    processed_dir: str = os.environ.get("PROCESSED_DIR", "data/processed")
    reports_dir: str = os.environ.get("REPORTS_DIR", "reports")

    @property
    def model(self) -> str:
        return os.path.join(self.models_dir, "acuity_model.joblib")

    @property
    def features(self) -> str:
        return os.path.join(self.models_dir, "acuity_features.json")

    @property
    def daily_visits(self) -> str:
        return os.path.join(self.processed_dir, "daily_visits.csv")

    @property
    def metrics(self) -> str:
        return os.path.join(self.reports_dir, "pipeline_metrics.json")


class AcuityRequest(BaseModel):
    """Feature payload for an acuity prediction."""

    features: Dict[str, float]


class ArtifactStore:
    """Lazily loads and caches the model, feature list and daily series."""

    def __init__(self, paths: Optional[Paths] = None):
        self.paths = paths or Paths()
        self._model = None
        self._features: Optional[List[str]] = None

    def model(self):
        if self._model is None:
            if not os.path.exists(self.paths.model):
                raise HTTPException(
                    status_code=503, detail="Model not trained; run `python main.py`."
                )
            import joblib

            self._model = joblib.load(self.paths.model)
        return self._model

    def feature_names(self) -> List[str]:
        if self._features is None:
            if not os.path.exists(self.paths.features):
                raise HTTPException(
                    status_code=503,
                    detail="Feature list missing; run `python main.py`.",
                )
            with open(self.paths.features, encoding="utf-8") as fh:
                self._features = json.load(fh)
        return self._features

    def daily_visits(self) -> pd.DataFrame:
        if not os.path.exists(self.paths.daily_visits):
            raise HTTPException(
                status_code=503,
                detail="Daily-visits series missing; run `python main.py`.",
            )
        return pd.read_csv(self.paths.daily_visits)

    def metrics(self) -> Dict:
        if not os.path.exists(self.paths.metrics):
            raise HTTPException(
                status_code=503, detail="Metrics missing; run `python main.py`."
            )
        with open(self.paths.metrics, encoding="utf-8") as fh:
            return json.load(fh)


def _forecast_frame(store: ArtifactStore, days: int) -> pd.DataFrame:
    """Fit ARIMA on the daily series and return a dated forecast frame."""
    daily = store.daily_visits()
    forecaster = TimeSeriesForecaster()
    series = forecaster.prepare_series(daily)
    forecaster.train_arima(series)
    preds = forecaster.forecast_arima(days)
    future_dates = pd.date_range(
        series.index.max() + pd.Timedelta(days=1), periods=days, freq="D"
    )
    return pd.DataFrame(
        {"date": future_dates.strftime("%Y-%m-%d"), "predicted_visits": preds.round(2)}
    )


def create_app(paths: Optional[Paths] = None) -> FastAPI:
    """Application factory (used by tests and by the module-level ``app``)."""
    app = FastAPI(
        title="Healthcare Resource Optimization API",
        description="Forecasts, acuity predictions and early-warning alerts.",
        version="1.0.0",
    )
    store = ArtifactStore(paths)

    @app.get("/health")
    def health() -> Dict[str, object]:
        return {
            "status": "ok",
            "model_available": os.path.exists(store.paths.model),
            "daily_visits_available": os.path.exists(store.paths.daily_visits),
            "metrics_available": os.path.exists(store.paths.metrics),
        }

    @app.get("/metrics")
    def metrics() -> Dict:
        return store.metrics()

    @app.post("/predict/acuity")
    def predict_acuity(request: AcuityRequest) -> Dict[str, object]:
        model = store.model()
        feature_names = store.feature_names()
        row = {name: float(request.features.get(name, 0.0)) for name in feature_names}
        frame = pd.DataFrame([row], columns=feature_names)
        proba = float(model.predict_proba(frame)[0, 1])
        return {
            "probability_high_acuity": round(proba, 4),
            "predicted_label": "high" if proba >= 0.5 else "low",
            "features_used": feature_names,
        }

    @app.get("/forecast")
    def forecast(days: int = 14) -> Dict[str, object]:
        days = max(1, min(days, 90))
        frame = _forecast_frame(store, days)
        return {"days": days, "forecast": frame.to_dict(orient="records")}

    @app.get("/alerts")
    def alerts(days: int = 14) -> Dict[str, object]:
        days = max(1, min(days, 90))
        daily = store.daily_visits()
        series = TimeSeriesForecaster().prepare_series(daily)
        frame = _forecast_frame(store, days)
        ews = EarlyWarningSystem()
        alert_objs = ews.evaluate(
            frame["predicted_visits"].to_numpy(),
            baseline_mean=float(series.mean()),
            baseline_std=float(series.std()),
            dates=frame["date"].tolist(),
        )
        return {
            "highest_severity": ews.highest_severity(alert_objs),
            "n_actionable": len(ews.actionable(alert_objs)),
            "alerts": [a.to_dict() for a in alert_objs],
        }

    @app.get("/", response_class=HTMLResponse)
    def dashboard() -> str:
        return _render_dashboard(store)

    return app


def _render_dashboard(store: ArtifactStore) -> str:
    """Server-render a minimal, dependency-free HTML dashboard."""
    try:
        metrics = store.metrics()
    except HTTPException:
        return (
            "<html><body style='font-family:sans-serif;padding:2rem'>"
            "<h1>Healthcare Resource Optimization</h1>"
            "<p>No run found yet. Generate artifacts with "
            "<code>python main.py</code>, then refresh.</p></body></html>"
        )

    acuity = metrics.get("acuity_model", {}).get("metrics", {})
    ts = metrics.get("timeseries", {})
    forecast = metrics.get("forecast", {})
    best = forecast.get("best_metrics", {})

    def kpi(label: str, value: object) -> str:
        return (
            "<div style='flex:1;min-width:160px;background:#f5f7fa;border-radius:10px;"
            "padding:1rem;margin:.4rem'>"
            f"<div style='font-size:.8rem;color:#5b6472'>{label}</div>"
            f"<div style='font-size:1.6rem;font-weight:700;color:#1f2933'>{value}</div></div>"
        )

    cards = "".join(
        [
            kpi("Acuity accuracy", acuity.get("accuracy", "—")),
            kpi("Acuity ROC-AUC", acuity.get("roc_auc", "—")),
            kpi("Mean daily visits", ts.get("mean_daily_visits", "—")),
            kpi("Best forecaster", forecast.get("best_model", "—")),
            kpi("Forecast MAE", best.get("MAE", "—")),
            kpi("Forecast accuracy", best.get("Accuracy", "—")),
        ]
    )

    return f"""<html><head><title>Healthcare Resource Optimization</title></head>
<body style='font-family:sans-serif;max-width:900px;margin:2rem auto;color:#1f2933'>
  <h1>Healthcare Resource Optimization</h1>
  <p style='color:#5b6472'>Live view of the latest pipeline run (synthetic data).</p>
  <div style='display:flex;flex-wrap:wrap'>{cards}</div>
  <h2>API</h2>
  <ul>
    <li><a href='/forecast?days=14'>/forecast?days=14</a> - 14-day demand forecast</li>
    <li><a href='/alerts?days=14'>/alerts?days=14</a> - early-warning surge alerts</li>
    <li><a href='/metrics'>/metrics</a> - full run metrics (JSON)</li>
    <li><a href='/health'>/health</a> - service status</li>
    <li><code>POST /predict/acuity</code> - acuity probability for a visit</li>
  </ul>
  <p style='color:#8a94a6;font-size:.85rem'>Metrics are illustrative, from synthetic data.</p>
</body></html>"""


# Module-level app for `uvicorn src.api.app:app`.
app = create_app()
