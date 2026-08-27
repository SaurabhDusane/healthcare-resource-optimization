"""
Pipeline Orchestration
======================

A small, dependency-free flow runner that sequences the pipeline as discrete,
retried steps with structured logging — the shape a scheduler (Airflow /
Prefect / cron) would drive in production.

``PipelineFlow`` runs anywhere with no extra dependencies. If ``prefect`` is
installed, :func:`build_prefect_flow` adapts the same steps into a real Prefect
flow (tasks + retries + scheduling) without changing the core logic. This keeps
the heavy orchestration dependency optional and out of the default/CI path.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from src.pipeline import Pipeline, PipelineConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("orchestration")


@dataclass
class StepResult:
    """Outcome of a single flow step."""

    name: str
    ok: bool
    attempts: int
    duration_s: float
    error: Optional[str] = None


@dataclass
class PipelineFlow:
    """Run the analytics pipeline as retried, logged steps."""

    config: PipelineConfig = field(default_factory=PipelineConfig)
    max_retries: int = 2
    retry_backoff_s: float = 2.0

    def __post_init__(self):
        self.logger = logger
        self.results: List[StepResult] = []
        self.metrics: Dict[str, object] = {}

    # ------------------------------------------------------------------ #
    def _run_step(self, name: str, fn: Callable[[], object]) -> StepResult:
        """Execute one step with retries and exponential backoff."""
        attempt = 0
        start = time.time()
        last_error: Optional[str] = None
        while attempt <= self.max_retries:
            attempt += 1
            try:
                fn()
                result = StepResult(name, True, attempt, round(time.time() - start, 3))
                self.logger.info("step '%s' ok (attempt %d)", name, attempt)
                self.results.append(result)
                return result
            except Exception as exc:  # noqa: BLE001 - surfaced via StepResult
                last_error = str(exc)
                self.logger.warning(
                    "step '%s' failed (attempt %d): %s", name, attempt, exc
                )
                if attempt <= self.max_retries:
                    time.sleep(self.retry_backoff_s * attempt)
        result = StepResult(
            name, False, attempt, round(time.time() - start, 3), error=last_error
        )
        self.results.append(result)
        return result

    def run(self) -> Dict[str, object]:
        """Run the whole pipeline as a single orchestrated step.

        The pipeline is internally staged; here we treat it as one retried unit
        so a transient failure (disk, numeric) is retried end-to-end. Returns a
        summary dict with per-step results and the pipeline metrics.
        """
        pipeline = Pipeline(self.config)

        def _run_pipeline():
            self.metrics = pipeline.run()

        self._run_step("pipeline", _run_pipeline)
        ok = all(r.ok for r in self.results)
        return {
            "ok": ok,
            "steps": [r.__dict__ for r in self.results],
            "metrics": self.metrics,
        }


def run_flow(config: Optional[PipelineConfig] = None) -> Dict[str, object]:
    """Convenience entry point for the dependency-free flow."""
    return PipelineFlow(config or PipelineConfig()).run()


def build_prefect_flow():  # pragma: no cover - exercised only when prefect present
    """
    Adapt the pipeline into a Prefect flow, if Prefect is installed.

    Usage::

        from src.orchestration.flow import build_prefect_flow
        flow = build_prefect_flow()
        flow()  # or deploy on a schedule

    Returns the Prefect flow callable, or raises ImportError if Prefect is
    unavailable (install with ``pip install prefect``).
    """
    from prefect import flow, task  # type: ignore

    @task(retries=2, retry_delay_seconds=5)
    def run_pipeline_task(config: Optional[PipelineConfig] = None):
        return Pipeline(config or PipelineConfig()).run()

    @flow(name="healthcare-resource-optimization")
    def healthcare_flow(config: Optional[PipelineConfig] = None):
        return run_pipeline_task(config)

    return healthcare_flow


if __name__ == "__main__":
    summary = run_flow()
    print("Flow ok:", summary["ok"])
