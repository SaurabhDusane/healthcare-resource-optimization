"""Tests for the classification model wrapper."""

import numpy as np
import pandas as pd

from src.modeling.classification_model import ClassificationModel


def _dataset(n=400):
    rng = np.random.default_rng(0)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    # Learnable target so metrics are meaningfully above chance.
    y = ((x1 + x2) > 0).astype(int)
    X = pd.DataFrame({"x1": x1, "x2": x2})
    return X, pd.Series(y)


def test_train_predict_and_evaluate():
    X, y = _dataset()
    model = ClassificationModel(model_type="random_forest")
    model.train(X, y)
    preds = model.predict(X)
    assert len(preds) == len(y)

    metrics = model.evaluate(X, y)
    assert set(["accuracy", "precision", "recall", "f1"]).issubset(metrics)
    assert metrics["accuracy"] > 0.8  # separable data


def test_feature_importance_available_for_tree_model():
    X, y = _dataset()
    model = ClassificationModel(model_type="random_forest")
    model.train(X, y)
    importance = model.get_feature_importance()
    assert importance is not None
    assert set(importance["feature"]) == {"x1", "x2"}


def test_save_and_load_roundtrip(tmp_path):
    X, y = _dataset()
    model = ClassificationModel(model_type="logistic")
    model.train(X, y)
    path = tmp_path / "model.joblib"
    model.save_model(str(path))

    reloaded = ClassificationModel(model_type="logistic")
    reloaded.load_model(str(path))
    np.testing.assert_array_equal(model.predict(X), reloaded.predict(X))
