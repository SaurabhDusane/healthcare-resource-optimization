"""Tests for the enriched (inline-SVG) API dashboard."""

import pytest
from fastapi.testclient import TestClient

from src.api.app import Paths, _svg_bar_chart, _svg_line_chart, create_app
from src.pipeline import Pipeline, PipelineConfig


def test_line_chart_renders_polyline():
    svg = _svg_line_chart([1.0, 3.0, 2.0, 5.0], "Visits")
    assert "<polyline" in svg
    assert "Visits" in svg


def test_line_chart_handles_insufficient_data():
    svg = _svg_line_chart([1.0], "Visits")
    assert "not enough data" in svg
    assert "<svg" not in svg


def test_bar_chart_renders_bars():
    items = [
        {"feature": "AGE", "importance": 0.4},
        {"feature": "SEX", "importance": 0.1},
    ]
    svg = _svg_bar_chart(items, "Predictors")
    assert svg.count("<rect") == 2
    assert "AGE" in svg


def test_bar_chart_handles_empty():
    assert "not available" in _svg_bar_chart([], "Predictors")


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    root = tmp_path_factory.mktemp("dash")
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
    paths = Paths(
        models_dir=str(root / "models"),
        processed_dir=str(root / "processed"),
        reports_dir=str(root / "reports"),
    )
    return TestClient(create_app(paths=paths))


def test_dashboard_includes_two_charts(client):
    html = client.get("/").text
    assert html.count("<svg") == 2
    assert "Daily ER visits" in html
    assert "Top acuity predictors" in html
