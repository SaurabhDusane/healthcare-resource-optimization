"""Exploratory Data Analysis Functions"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EDAAnalyzer:
    """Perform exploratory data analysis."""
    
    def __init__(self):
        self.logger = logger
        sns.set_style("whitegrid")
    
    def get_summary_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Get comprehensive summary statistics."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        summary = df[numeric_cols].describe().T
        summary['missing'] = df[numeric_cols].isnull().sum()
        summary['missing_pct'] = (df[numeric_cols].isnull().sum() / len(df)) * 100
        
        return summary
    
    def analyze_distributions(self, df: pd.DataFrame, columns: list):
        """Analyze distributions of specified columns."""
        results = {}
        
        for col in columns:
            if col not in df.columns:
                continue
            
            results[col] = {
                'mean': df[col].mean(),
                'median': df[col].median(),
                'std': df[col].std(),
                'skewness': df[col].skew(),
                'kurtosis': df[col].kurtosis(),
                'min': df[col].min(),
                'max': df[col].max()
            }
        
        return results
    
    def find_correlations(self, df: pd.DataFrame, threshold=0.5):
        """Find strong correlations."""
        numeric_df = df.select_dtypes(include=[np.number])
        corr_matrix = numeric_df.corr()
        
        strong_corrs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) > threshold:
                    strong_corrs.append({
                        'var1': corr_matrix.columns[i],
                        'var2': corr_matrix.columns[j],
                        'correlation': corr_matrix.iloc[i, j]
                    })
        
        return pd.DataFrame(strong_corrs)
    
    def detect_outliers(self, df: pd.DataFrame, column: str):
        """Detect outliers using IQR method."""
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
        
        return {
            'n_outliers': len(outliers),
            'pct_outliers': len(outliers) / len(df) * 100,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'outlier_indices': outliers.index.tolist()
        }
