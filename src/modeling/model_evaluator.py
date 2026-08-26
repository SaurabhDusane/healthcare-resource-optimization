"""
Model Evaluation Module
=======================

Headless-safe evaluation and reporting for the classification models. The
Matplotlib ``Agg`` backend is selected on import so figures render in CI and
batch runs without a display; plotting methods save to disk instead of calling
``plt.show()``.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")  # headless-safe; must precede pyplot import

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Evaluate, report and visualize classification model performance."""

    def __init__(self):
        self.logger = logger

    # ------------------------------------------------------------------ #
    # Metrics                                                            #
    # ------------------------------------------------------------------ #
    def evaluate_classification(
        self, y_true, y_pred, y_proba=None
    ) -> Dict[str, object]:
        """Return a JSON-serializable dict of headline metrics + confusion matrix."""
        metrics: Dict[str, object] = {
            "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
            "precision": round(
                float(
                    precision_score(y_true, y_pred, average="weighted", zero_division=0)
                ),
                4,
            ),
            "recall": round(
                float(
                    recall_score(y_true, y_pred, average="weighted", zero_division=0)
                ),
                4,
            ),
            "f1": round(
                float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 4
            ),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        }
        if y_proba is not None:
            try:
                metrics["roc_auc"] = round(float(roc_auc_score(y_true, y_proba)), 4)
            except ValueError:  # single-class holdout, etc.
                metrics["roc_auc"] = None
        return metrics

    def get_classification_report(
        self, y_true, y_pred, target_names=None
    ) -> pd.DataFrame:
        """Detailed per-class precision/recall/F1 as a DataFrame."""
        report = classification_report(
            y_true, y_pred, target_names=target_names, output_dict=True, zero_division=0
        )
        return pd.DataFrame(report).transpose()

    def save_report(self, metrics: Dict[str, object], out_dir: str, name: str) -> str:
        """Write a metrics dict to ``out_dir/name.json`` and return the path."""
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"{name}.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(metrics, fh, indent=2, default=str)
        self.logger.info("Saved evaluation report -> %s", path)
        return path

    # ------------------------------------------------------------------ #
    # Plots (saved to disk, never shown)                                 #
    # ------------------------------------------------------------------ #
    def plot_confusion_matrix(self, y_true, y_pred, labels=None, save_path=None):
        """Render a confusion-matrix heatmap; save if ``save_path`` given."""
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
            ax=ax,
        )
        ax.set_title("Confusion Matrix")
        ax.set_ylabel("True Label")
        ax.set_xlabel("Predicted Label")
        self._save_and_close(fig, save_path)
        return cm

    def plot_roc_curve(self, y_true, y_proba, save_path=None):
        """Render a ROC curve; save if ``save_path`` given. Returns AUC."""
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(
            fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})"
        )
        ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("Receiver Operating Characteristic (ROC) Curve")
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        self._save_and_close(fig, save_path)
        return roc_auc

    def plot_feature_importance(self, feature_importance_df, top_n=20, save_path=None):
        """Render a horizontal feature-importance bar chart; save if given."""
        top_features = feature_importance_df.head(top_n)
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(range(len(top_features)), top_features["importance"])
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features["feature"])
        ax.set_xlabel("Importance")
        ax.set_title(f"Top {top_n} Feature Importances")
        ax.invert_yaxis()
        self._save_and_close(fig, save_path)

    def compare_models(self, results_dict: Dict[str, Dict]) -> pd.DataFrame:
        """Tabulate metrics for multiple models."""
        return pd.DataFrame(results_dict).T

    # ------------------------------------------------------------------ #
    def _save_and_close(self, fig, save_path: Optional[str]) -> None:
        if save_path:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            self.logger.info("Saved figure -> %s", save_path)
        plt.close(fig)
