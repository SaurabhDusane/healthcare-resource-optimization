"""Tests for classifier hyperparameter tuning."""

import numpy as np
import pandas as pd

from src.modeling.classification_model import ClassificationModel


def _dataset(n=300, seed=0):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = ((x1 + 0.5 * x2 + rng.normal(0, 0.5, n)) > 0).astype(int)
    return pd.DataFrame({"x1": x1, "x2": x2}), pd.Series(y)


def test_tune_returns_best_params_and_fits():
    X, y = _dataset()
    model = ClassificationModel(model_type="random_forest")
    result = model.tune(X, y, n_iter=4, cv=3)
    assert result["tuned"] is True
    assert result["best_params"]  # non-empty
    assert 0.0 <= result["cv_roc_auc"] <= 1.0
    # Model is fitted to the best estimator and can predict.
    preds = model.predict(X)
    assert len(preds) == len(y)


def test_tune_logistic_search_space():
    X, y = _dataset()
    model = ClassificationModel(model_type="logistic")
    result = model.tune(X, y, n_iter=3, cv=3)
    assert result["tuned"] is True
    assert "C" in result["best_params"]


def test_tuned_model_is_usable_for_proba():
    X, y = _dataset()
    model = ClassificationModel(model_type="gradient_boost")
    model.tune(X, y, n_iter=3, cv=3)
    proba = model.predict_proba(X)
    assert proba.shape == (len(y), 2)
