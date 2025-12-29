"""Model Evaluation Module"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Evaluate and visualize model performance."""
    
    def __init__(self):
        self.logger = logger
    
    def plot_confusion_matrix(self, y_true, y_pred, labels=None, save_path=None):
        """Plot confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=labels, yticklabels=labels)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
        return cm
    
    def plot_roc_curve(self, y_true, y_proba, save_path=None):
        """Plot ROC curve."""
        fpr, tpr, thresholds = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
        return roc_auc
    
    def plot_feature_importance(self, feature_importance_df, top_n=20, save_path=None):
        """Plot feature importance."""
        top_features = feature_importance_df.head(top_n)
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Importance')
        plt.title(f'Top {top_n} Feature Importances')
        plt.gca().invert_yaxis()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def get_classification_report(self, y_true, y_pred, target_names=None):
        """Get detailed classification report."""
        report = classification_report(y_true, y_pred, 
                                      target_names=target_names, 
                                      output_dict=True)
        
        return pd.DataFrame(report).transpose()
    
    def compare_models(self, results_dict):
        """Compare multiple model results."""
        comparison_df = pd.DataFrame(results_dict).T
        
        print("\nModel Comparison:")
        print(comparison_df)
        
        return comparison_df
