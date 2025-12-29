"""Classification Model Module"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ClassificationModel:
    """Classification models for ER acuity prediction."""
    
    def __init__(self, model_type='xgboost'):
        self.logger = logger
        self.model_type = model_type
        self.model = None
        self.feature_names = None
    
    def initialize_model(self):
        """Initialize the selected model."""
        if self.model_type == 'xgboost':
            self.model = XGBClassifier(
                n_estimators=150,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
        elif self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                random_state=42
            )
        elif self.model_type == 'logistic':
            self.model = LogisticRegression(
                max_iter=1000,
                random_state=42
            )
        elif self.model_type == 'gradient_boost':
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                random_state=42
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
    
    def predict(self, X):
        """Make predictions."""
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Predict probabilities."""
        return self.model.predict_proba(X)
    
    def evaluate(self, X, y):
        """Evaluate model performance."""
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)[:, 1] if hasattr(self.model, 'predict_proba') else None
        
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, average='weighted'),
            'recall': recall_score(y, y_pred, average='weighted'),
            'f1': f1_score(y, y_pred, average='weighted')
        }
        
        if y_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y, y_proba)
        
        return metrics
    
    def get_feature_importance(self):
        """Get feature importance scores."""
        if hasattr(self.model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
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
