"""End-to-end smoke test for the analytics pipeline."""

import json
import os

from src.pipeline import Pipeline, PipelineConfig


def test_pipeline_runs_end_to_end(tmp_path):
    config = PipelineConfig(
        n_visits=1500,
        model_type="random_forest",
        raw_dir=str(tmp_path / "raw"),
        processed_dir=str(tmp_path / "processed"),
        models_dir=str(tmp_path / "models"),
        reports_dir=str(tmp_path / "reports"),
        dirs_to_create=[
            str(tmp_path / "raw"),
            str(tmp_path / "processed"),
            str(tmp_path / "models"),
            str(tmp_path / "reports"),
        ],
    )
    metrics = Pipeline(config).run()

    # Artifacts exist.
    assert os.path.exists(tmp_path / "models" / "acuity_model.joblib")
    assert os.path.exists(tmp_path / "reports" / "pipeline_metrics.json")
    assert os.path.exists(tmp_path / "processed" / "daily_visits.csv")

    # Metrics are well-formed.
    assert "acuity_model" in metrics
    assert 0.0 <= metrics["acuity_model"]["metrics"]["accuracy"] <= 1.0
    assert metrics["timeseries"]["n_days"] > 0

    # Metrics file is valid JSON matching the returned dict.
    with open(tmp_path / "reports" / "pipeline_metrics.json", encoding="utf-8") as fh:
        on_disk = json.load(fh)
    assert on_disk["seed"] == metrics["seed"]
