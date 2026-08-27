"""
Model Registry
==============

A minimal, filesystem-backed model registry so trained models are versioned
rather than overwritten in place. Each ``register`` call writes a new version
under ``models/registry/<name>/v<N>/`` with the serialized estimator and a
``meta.json`` (metrics, params, timestamp, git SHA). A per-model ``index.json``
tracks the version list and a ``production`` pointer, enabling promotion and
rollback.

Usage::

    reg = ModelRegistry()
    version = reg.register(estimator, "acuity", metrics={...}, params={...})
    reg.promote("acuity", version)         # mark as production
    model, meta = reg.load("acuity")       # loads the production version
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _git_sha() -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        )
        return out.decode().strip()
    except (subprocess.SubprocessError, FileNotFoundError):  # pragma: no cover
        return None


class ModelRegistry:
    """Filesystem-backed, versioned model registry."""

    def __init__(self, base_dir: str = "models/registry"):
        self.base_dir = base_dir
        self.logger = logger

    # ------------------------------------------------------------------ #
    def _model_dir(self, name: str) -> str:
        return os.path.join(self.base_dir, name)

    def _index_path(self, name: str) -> str:
        return os.path.join(self._model_dir(name), "index.json")

    def _read_index(self, name: str) -> Dict[str, object]:
        path = self._index_path(name)
        if not os.path.exists(path):
            return {"name": name, "versions": [], "production": None}
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)

    def _write_index(self, name: str, index: Dict[str, object]) -> None:
        os.makedirs(self._model_dir(name), exist_ok=True)
        with open(self._index_path(name), "w", encoding="utf-8") as fh:
            json.dump(index, fh, indent=2, default=str)

    # ------------------------------------------------------------------ #
    def register(
        self,
        estimator,
        name: str,
        metrics: Optional[Dict[str, object]] = None,
        params: Optional[Dict[str, object]] = None,
        promote: bool = True,
    ) -> int:
        """Save a new version of ``estimator`` and return its version number."""
        index = self._read_index(name)
        version = (max(index["versions"], default=0) + 1) if index["versions"] else 1
        version_dir = os.path.join(self._model_dir(name), f"v{version}")
        os.makedirs(version_dir, exist_ok=True)

        joblib.dump(estimator, os.path.join(version_dir, "model.joblib"))
        meta = {
            "name": name,
            "version": version,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "git_sha": _git_sha(),
            "metrics": metrics or {},
            "params": params or {},
        }
        with open(os.path.join(version_dir, "meta.json"), "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2, default=str)

        index["versions"].append(version)
        if promote or index["production"] is None:
            index["production"] = version
        self._write_index(name, index)
        self.logger.info(
            "Registered %s v%d (production=%s)", name, version, index["production"]
        )
        return version

    def _resolve_version(self, name: str, version) -> int:
        index = self._read_index(name)
        if not index["versions"]:
            raise FileNotFoundError(f"No versions registered for model '{name}'")
        if version in (None, "production"):
            return int(index["production"] or max(index["versions"]))
        if version == "latest":
            return int(max(index["versions"]))
        if int(version) not in index["versions"]:
            raise FileNotFoundError(f"{name} has no version {version}")
        return int(version)

    def load(self, name: str, version="production") -> Tuple[object, Dict[str, object]]:
        """Load an estimator and its metadata (default: the production version)."""
        v = self._resolve_version(name, version)
        version_dir = os.path.join(self._model_dir(name), f"v{v}")
        estimator = joblib.load(os.path.join(version_dir, "model.joblib"))
        with open(os.path.join(version_dir, "meta.json"), encoding="utf-8") as fh:
            meta = json.load(fh)
        return estimator, meta

    def promote(self, name: str, version: int) -> None:
        """Point ``production`` at an existing version (promotion or rollback)."""
        index = self._read_index(name)
        if int(version) not in index["versions"]:
            raise FileNotFoundError(f"{name} has no version {version}")
        index["production"] = int(version)
        self._write_index(name, index)
        self.logger.info("Promoted %s v%d to production", name, version)

    def list_versions(self, name: str) -> List[int]:
        return list(self._read_index(name)["versions"])

    def production_version(self, name: str) -> Optional[int]:
        return self._read_index(name)["production"]
