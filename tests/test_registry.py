"""Tests for the versioned model registry."""

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from src.modeling.registry import ModelRegistry


def _fitted_estimator(seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(60, 2))
    y = (X[:, 0] > 0).astype(int)
    return LogisticRegression().fit(X, y)


def test_register_increments_versions(tmp_path):
    reg = ModelRegistry(base_dir=str(tmp_path))
    v1 = reg.register(_fitted_estimator(), "acuity", metrics={"accuracy": 0.8})
    v2 = reg.register(_fitted_estimator(1), "acuity", metrics={"accuracy": 0.9})
    assert v1 == 1 and v2 == 2
    assert reg.list_versions("acuity") == [1, 2]
    assert reg.production_version("acuity") == 2  # promoted by default


def test_load_production_and_specific_version(tmp_path):
    reg = ModelRegistry(base_dir=str(tmp_path))
    reg.register(_fitted_estimator(), "acuity", metrics={"accuracy": 0.8})
    reg.register(_fitted_estimator(1), "acuity", metrics={"accuracy": 0.9})

    _, meta_prod = reg.load("acuity")  # production == v2
    assert meta_prod["version"] == 2
    _, meta_v1 = reg.load("acuity", version=1)
    assert meta_v1["version"] == 1
    assert meta_v1["metrics"]["accuracy"] == 0.8


def test_promote_enables_rollback(tmp_path):
    reg = ModelRegistry(base_dir=str(tmp_path))
    reg.register(_fitted_estimator(), "acuity")
    reg.register(_fitted_estimator(1), "acuity")
    assert reg.production_version("acuity") == 2
    reg.promote("acuity", 1)  # roll back
    assert reg.production_version("acuity") == 1
    _, meta = reg.load("acuity")
    assert meta["version"] == 1


def test_register_without_promote_keeps_production(tmp_path):
    reg = ModelRegistry(base_dir=str(tmp_path))
    reg.register(_fitted_estimator(), "acuity")  # v1 -> production
    reg.register(_fitted_estimator(1), "acuity", promote=False)  # v2, not promoted
    assert reg.production_version("acuity") == 1


def test_load_missing_model_raises(tmp_path):
    reg = ModelRegistry(base_dir=str(tmp_path))
    with pytest.raises(FileNotFoundError):
        reg.load("does_not_exist")


def test_promote_unknown_version_raises(tmp_path):
    reg = ModelRegistry(base_dir=str(tmp_path))
    reg.register(_fitted_estimator(), "acuity")
    with pytest.raises(FileNotFoundError):
        reg.promote("acuity", 99)


def test_loaded_estimator_predicts(tmp_path):
    reg = ModelRegistry(base_dir=str(tmp_path))
    reg.register(_fitted_estimator(), "acuity")
    est, _ = reg.load("acuity")
    preds = est.predict(np.zeros((3, 2)))
    assert len(preds) == 3
