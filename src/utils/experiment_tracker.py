"""
Lightweight Experiment Tracker
==============================

A dependency-free alternative to MLflow for this project. Each call to
:meth:`ExperimentTracker.log` appends one JSON record capturing the run's
parameters, metrics, a timestamp, and the current git commit, so results are
reproducible and comparable across runs without any external service.

Records are written to ``reports/experiments/`` (one file per run) and can be
collated into a single table with :meth:`ExperimentTracker.load_history`.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
from datetime import datetime
from typing import Dict, List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _git_sha() -> Optional[str]:
    """Return the short git SHA of the working tree, or None if unavailable."""
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        )
        return sha.decode().strip()
    except (subprocess.SubprocessError, FileNotFoundError):  # pragma: no cover
        return None


class ExperimentTracker:
    """Append-only JSON experiment logger."""

    def __init__(self, experiments_dir: str = "reports/experiments"):
        self.experiments_dir = experiments_dir
        self.logger = logger

    def log(
        self,
        run_name: str,
        params: Dict[str, object],
        metrics: Dict[str, object],
        tags: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Persist a single experiment record and return its file path.

        Args:
            run_name: Short identifier for the run (e.g. ``"acuity_xgboost"``).
            params: Hyperparameters / configuration used.
            metrics: Metric name -> value.
            tags: Optional free-form metadata.
        """
        os.makedirs(self.experiments_dir, exist_ok=True)
        timestamp = datetime.utcnow()
        record = {
            "run_name": run_name,
            "timestamp": timestamp.isoformat() + "Z",
            "git_sha": _git_sha(),
            "params": params,
            "metrics": metrics,
            "tags": tags or {},
        }
        fname = f"{timestamp.strftime('%Y%m%dT%H%M%S')}_{run_name}.json"
        path = os.path.join(self.experiments_dir, fname)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(record, fh, indent=2, default=str)
        self.logger.info("Logged experiment '%s' -> %s", run_name, path)
        return path

    def load_history(self) -> List[Dict[str, object]]:
        """Load every logged run, newest first."""
        if not os.path.isdir(self.experiments_dir):
            return []
        records: List[Dict[str, object]] = []
        for name in sorted(os.listdir(self.experiments_dir), reverse=True):
            if name.endswith(".json"):
                with open(
                    os.path.join(self.experiments_dir, name), encoding="utf-8"
                ) as fh:
                    records.append(json.load(fh))
        return records
