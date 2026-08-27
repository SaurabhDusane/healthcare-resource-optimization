"""Tests for the model evaluator (headless-safe)."""

import json

import numpy as np
import pandas as pd

from src.modeling.model_evaluator import ModelEvaluator


def _labels():
    y_true = np.array([0, 0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 1, 0, 0])
    y_proba = np.array([0.2, 0.6, 0.8, 0.7, 0.1, 0.4])
    return y_true, y_pred, y_proba


def test_evaluate_classification_is_json_serializable():
    y_true, y_pred, y_proba = _labels()
    metrics = ModelEvaluator().evaluate_classification(y_true, y_pred, y_proba)
    for key in ["accuracy", "precision", "recall", "f1", "roc_auc", "confusion_matrix"]:
        assert key in metrics
    # Must round-trip through JSON (no numpy types leaking through).
    json.dumps(metrics)
    assert 0.0 <= metrics["accuracy"] <= 1.0


def test_save_report_writes_json(tmp_path):
    y_true, y_pred, y_proba = _labels()
    ev = ModelEvaluator()
    metrics = ev.evaluate_classification(y_true, y_pred, y_proba)
    path = ev.save_report(metrics, str(tmp_path), "report")
    with open(path, encoding="utf-8") as fh:
        assert json.load(fh)["accuracy"] == metrics["accuracy"]


def test_plots_save_without_display(tmp_path):
    y_true, y_pred, y_proba = _labels()
    ev = ModelEvaluator()
    cm_path = tmp_path / "cm.png"
    roc_path = tmp_path / "roc.png"
    ev.plot_confusion_matrix(
        y_true, y_pred, labels=["low", "high"], save_path=str(cm_path)
    )
    ev.plot_roc_curve(y_true, y_proba, save_path=str(roc_path))

    importance = pd.DataFrame({"feature": ["a", "b"], "importance": [0.7, 0.3]})
    fi_path = tmp_path / "fi.png"
    ev.plot_feature_importance(importance, save_path=str(fi_path))

    assert cm_path.exists() and roc_path.exists() and fi_path.exists()
