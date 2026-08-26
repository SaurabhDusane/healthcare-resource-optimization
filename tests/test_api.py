"""Tests for the FastAPI service (TestClient, no network)."""

import json

import pytest
from fastapi.testclient import TestClient

from src.api.app import Paths, create_app
from src.pipeline import Pipeline, PipelineConfig


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    """Run a small pipeline into a temp workspace and serve it."""
    root = tmp_path_factory.mktemp("api_ws")
    cfg = PipelineConfig(
        n_visits=1500,
        model_type="random_forest",
        raw_dir=str(root / "raw"),
        processed_dir=str(root / "processed"),
        models_dir=str(root / "models"),
        reports_dir=str(root / "reports"),
        visualizations_dir=str(root / "viz"),
        dirs_to_create=[
            str(root / p) for p in ("raw", "processed", "models", "reports", "viz")
        ],
    )
    Pipeline(cfg).run()
    paths = Paths(
        models_dir=str(root / "models"),
        processed_dir=str(root / "processed"),
        reports_dir=str(root / "reports"),
    )
    return TestClient(create_app(paths))


def test_health_reports_artifacts(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["model_available"] is True
    assert body["daily_visits_available"] is True


def test_metrics_endpoint(client):
    body = client.get("/metrics").json()
    assert "acuity_model" in body
    assert "forecast" in body


def test_forecast_endpoint(client):
    body = client.get("/forecast?days=10").json()
    assert body["days"] == 10
    assert len(body["forecast"]) == 10
    assert {"date", "predicted_visits"} <= set(body["forecast"][0])


def test_forecast_days_clamped(client):
    assert client.get("/forecast?days=1000").json()["days"] == 90


def test_alerts_endpoint(client):
    body = client.get("/alerts?days=14").json()
    assert body["highest_severity"] in {"none", "watch", "warning", "critical"}
    assert len(body["alerts"]) == 14


def test_predict_acuity(client):
    feats = client.get("/metrics").json()["acuity_model"]["features_used"]
    payload = {"features": {f: 1.0 for f in feats}}
    body = client.post("/predict/acuity", json=payload).json()
    assert 0.0 <= body["probability_high_acuity"] <= 1.0
    assert body["predicted_label"] in {"high", "low"}


def test_root_dashboard_html(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "Healthcare Resource Optimization" in resp.text


def test_missing_artifacts_return_503(tmp_path):
    empty = Paths(
        models_dir=str(tmp_path / "m"),
        processed_dir=str(tmp_path / "p"),
        reports_dir=str(tmp_path / "r"),
    )
    c = TestClient(create_app(empty))
    assert c.get("/health").json()["model_available"] is False
    assert c.get("/forecast").status_code == 503
    assert c.get("/metrics").status_code == 503
