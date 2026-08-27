"""Tests for the dependency-free orchestration flow."""

from src.orchestration.flow import PipelineFlow, StepResult
from src.pipeline import PipelineConfig


def _tmp_config(root):
    return PipelineConfig(
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


def test_flow_runs_pipeline(tmp_path):
    summary = PipelineFlow(config=_tmp_config(tmp_path)).run()
    assert summary["ok"] is True
    assert "acuity_model" in summary["metrics"]
    assert summary["steps"][0]["name"] == "pipeline"
    assert summary["steps"][0]["ok"] is True


def test_step_retries_then_succeeds():
    flow = PipelineFlow(max_retries=3, retry_backoff_s=0)
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 2:
            raise RuntimeError("transient")

    result = flow._run_step("flaky", flaky)
    assert isinstance(result, StepResult)
    assert result.ok is True
    assert result.attempts == 2


def test_step_reports_failure():
    flow = PipelineFlow(max_retries=1, retry_backoff_s=0)

    def always_fails():
        raise ValueError("boom")

    result = flow._run_step("bad", always_fails)
    assert result.ok is False
    assert "boom" in result.error
    assert result.attempts == 2
