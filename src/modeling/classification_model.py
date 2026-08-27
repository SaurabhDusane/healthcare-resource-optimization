"""Classification Model Module"""

import pandas as pd
import numpy as np
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
import joblib
import logging

# Randomized-search spaces per model type (kept small so tuning stays fast).
PARAM_DISTRIBUTIONS = {
    "xgboost": {
        "n_estimators": [100, 150, 200, 300],
        "max_depth": [3, 4, 6, 8],
        "learning_rate": [0.03, 0.05, 0.1, 0.2],
        "subsample": [0.7, 0.85, 1.0],
    },
    "random_forest": {
        "n_estimators": [100, 200, 300],
        "max_depth": [6, 10, 16, None],
        "min_samples_leaf": [1, 2, 5],
    },
    "gradient_boost": {
        "n_estimators": [100, 150, 200],
        "max_depth": [2, 3, 4],
        "learning_rate": [0.03, 0.05, 0.1],
    },
    "logistic": {
        "C": [0.1, 0.3, 1.0, 3.0, 10.0],
    },
}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ClassificationModel:
    """Classification models for ER acuity prediction."""

    def __init__(self, model_type="xgboost"):
        self.logger = logger
        self.model_type = model_type
        self.model = None
        self.feature_names = None

    def initialize_model(self):
        """Initialize the selected model."""
        if self.model_type == "xgboost":
            self.model = XGBClassifier(
                n_estimators=150, max_depth=6, learning_rate=0.1, random_state=42
            )
        elif self.model_type == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=200, max_depth=10, random_state=42
            )
        elif self.model_type == "logistic":
            self.model = LogisticRegression(max_iter=1000, random_state=42)
        elif self.model_type == "gradient_boost":
            self.model = GradientBoostingClassifier(
                n_estimators=100, learning_rate=0.1, random_state=42
            )

        return self.model

    def train(self, X, y):
        """Train the model."""
        self.logger.info(f"Training {self.model_type} model...")

        if self.model is None:
            self.initialize_model()

        self.feature_names = X.columns.tolist() if isinstance(X, pd.DataFrame) else None

        self.model.fit(X, y)
        self.logger.info("Training complete")

        return self.model

    def tune(self, X, y, n_iter=15, cv=3, random_state=42):
        """
        Randomized hyperparameter search with stratified cross-validation.

        Fits ``self.model`` to the best estimator found (ranked by ROC-AUC) and
        returns a summary dict with the best params and CV score.
        """
        if self.model is None:
            self.initialize_model()
        self.feature_names = X.columns.tolist() if isinstance(X, pd.DataFrame) else None

        distributions = PARAM_DISTRIBUTIONS.get(self.model_type, {})
        if not distributions:
            self.logger.warning(
                "No search space for %s; training defaults", self.model_type
            )
            self.model.fit(X, y)
            return {"tuned": False, "best_params": {}, "cv_roc_auc": None}

        self.logger.info(
            "Tuning %s (%d iters, %d-fold CV)...", self.model_type, n_iter, cv
        )
        search = RandomizedSearchCV(
            estimator=self.model,
            param_distributions=distributions,
            n_iter=n_iter,
            scoring="roc_auc",
            cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state),
            random_state=random_state,
            n_jobs=-1,
        )
        search.fit(X, y)
        self.model = search.best_estimator_
        self.logger.info(
            "Best %s CV ROC-AUC=%.4f params=%s",
            self.model_type,
            search.best_score_,
            search.best_params_,
        )
        return {
            "tuned": True,
            "best_params": search.best_params_,
            "cv_roc_auc": round(float(search.best_score_), 4),
        }

    def predict(self, X):
        """Make predictions."""
        return self.model.predict(X)

    def predict_proba(self, X):
        """Predict probabilities."""
        return self.model.predict_proba(X)

    def evaluate(self, X, y):
        """Evaluate model performance."""
        y_pred = self.predict(X)
        y_proba = (
            self.predict_proba(X)[:, 1]
            if hasattr(self.model, "predict_proba")
            else None
        )

        metrics = {
            "accuracy": accuracy_score(y, y_pred),
            "precision": precision_score(y, y_pred, average="weighted"),
            "recall": recall_score(y, y_pred, average="weighted"),
            "f1": f1_score(y, y_pred, average="weighted"),
        }

        if y_proba is not None:
            metrics["roc_auc"] = roc_auc_score(y, y_proba)

        return metrics

    def get_feature_importance(self):
        """Get feature importance scores."""
        if hasattr(self.model, "feature_importances_"):
            importance_df = pd.DataFrame(
                {
                    "feature": self.feature_names,
                    "importance": self.model.feature_importances_,
                }
            ).sort_values("importance", ascending=False)

            return importance_df

        return None

    def save_model(self, filepath):
        """Save trained model."""
        joblib.dump(self.model, filepath)
        self.logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath):
        """Load trained model."""
        self.model = joblib.load(filepath)
        self.logger.info(f"Model loaded from {filepath}")
        return self.model
