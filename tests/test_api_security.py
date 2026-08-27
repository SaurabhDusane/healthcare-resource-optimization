"""Tests for API auth, rate limiting and input validation."""

import pytest
from fastapi.testclient import TestClient

from src.api.app import Paths, create_app
from src.config import Settings
from src.pipeline import Pipeline, PipelineConfig


@pytest.fixture(scope="module")
def workspace(tmp_path_factory):
    root = tmp_path_factory.mktemp("api_sec")
    cfg = PipelineConfig(
        n_visits=1200,
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
    return Paths(
        models_dir=str(root / "models"),
        processed_dir=str(root / "processed"),
        reports_dir=str(root / "reports"),
    )


def _client(paths, **settings_kwargs):
    settings = Settings(rate_limit_per_minute=0, **settings_kwargs)
    return TestClient(create_app(paths=paths, settings=settings))


# ----------------------------- auth ----------------------------- #
def test_open_when_no_api_key(workspace):
    client = _client(workspace, api_key=None)
    assert client.get("/forecast?days=3").status_code == 200
    assert client.get("/health").json()["auth_enabled"] is False


def test_requires_key_when_configured(workspace):
    client = _client(workspace, api_key="s3cret")
    assert client.get("/forecast?days=3").status_code == 401
    assert (
        client.get("/forecast?days=3", headers={"X-API-Key": "wrong"}).status_code
        == 401
    )
    assert (
        client.get("/forecast?days=3", headers={"X-API-Key": "s3cret"}).status_code
        == 200
    )


def test_health_always_open(workspace):
    client = _client(workspace, api_key="s3cret")
    assert client.get("/health").status_code == 200
    assert client.get("/health").json()["auth_enabled"] is True


# -------------------------- validation -------------------------- #
def test_predict_rejects_empty_features(workspace):
    client = _client(workspace, api_key=None)
    assert client.post("/predict/acuity", json={"features": {}}).status_code == 422


def test_predict_rejects_too_many_features(workspace):
    client = _client(workspace, api_key=None)
    payload = {"features": {f"f{i}": 1.0 for i in range(201)}}
    assert client.post("/predict/acuity", json=payload).status_code == 422


def test_predict_accepts_valid_payload(workspace):
    client = _client(workspace, api_key=None)
    feats = client.get("/metrics").json()["acuity_model"]["features_used"]
    payload = {"features": {f: 1.0 for f in feats}}
    body = client.post("/predict/acuity", json=payload).json()
    assert 0.0 <= body["probability_high_acuity"] <= 1.0


# ------------------------ rate limiting ------------------------- #
def test_rate_limit_returns_429(workspace):
    settings = Settings(api_key=None, rate_limit_per_minute=3)
    client = TestClient(create_app(paths=workspace, settings=settings))
    codes = [client.get("/forecast?days=2").status_code for _ in range(5)]
    assert codes.count(200) == 3
    assert codes.count(429) == 2


def test_health_not_rate_limited(workspace):
    settings = Settings(api_key=None, rate_limit_per_minute=2)
    client = TestClient(create_app(paths=workspace, settings=settings))
    assert all(client.get("/health").status_code == 200 for _ in range(6))
