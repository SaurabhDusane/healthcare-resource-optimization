"""
Data Validation Module
Validates data quality and integrity.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataValidator:
    """Data validation and quality checks."""
    
    def __init__(self):
        self.logger = logger
        self.validation_report = {}
    
    def check_missing_values(self, df: pd.DataFrame, threshold: float = 0.5) -> Dict:
        """
        Check for missing values.
        
        Args:
            df: Input DataFrame
            threshold: Maximum allowed missing ratio
            
        Returns:
            Dictionary with missing value statistics
        """
        self.logger.info("Checking missing values...")
        
        missing_counts = df.isnull().sum()
        missing_ratios = df.isnull().mean()
        
        high_missing_cols = missing_ratios[missing_ratios > threshold].index.tolist()
        
        report = {
            'total_missing': missing_counts.sum(),
            'columns_with_missing': missing_counts[missing_counts > 0].to_dict(),
            'missing_ratios': missing_ratios[missing_ratios > 0].to_dict(),
            'high_missing_columns': high_missing_cols
        }
        
        if high_missing_cols:
            self.logger.warning(f"Columns with >{threshold*100}% missing: {high_missing_cols}")
        
        self.validation_report['missing_values'] = report
        return report
    
    def check_duplicates(self, df: pd.DataFrame, subset: List[str] = None) -> Dict:
        """
        Check for duplicate rows.
        
        Args:
            df: Input DataFrame
            subset: Columns to check for duplicates
            
        Returns:
            Dictionary with duplicate statistics
        """
        self.logger.info("Checking duplicates...")
        
        if subset:
            dup_count = df.duplicated(subset=subset).sum()
        else:
            dup_count = df.duplicated().sum()
        
        report = {
            'duplicate_count': int(dup_count),
            'duplicate_ratio': float(dup_count / len(df)),
            'checked_columns': subset if subset else 'all'
        }
        
        if dup_count > 0:
            self.logger.warning(f"Found {dup_count} duplicate rows")
        
        self.validation_report['duplicates'] = report
        return report
    
    def check_data_types(self, df: pd.DataFrame) -> Dict:
        """
        Check data types consistency.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary with data type information
        """
        self.logger.info("Checking data types...")
        
        dtypes = df.dtypes.value_counts().to_dict()
        dtypes_str = {str(k): v for k, v in dtypes.items()}
        
        report = {
            'data_types': dtypes_str,
            'columns_by_type': {}
        }
        
        for dtype in df.dtypes.unique():
            cols = df.select_dtypes(include=[dtype]).columns.tolist()
            report['columns_by_type'][str(dtype)] = cols
        
        self.validation_report['data_types'] = report
        return report
    
    def check_numeric_ranges(self, df: pd.DataFrame, 
                            expected_ranges: Dict[str, Tuple[float, float]] = None) -> Dict:
        """
        Check numeric columns for outliers and range violations.
        
        Args:
            df: Input DataFrame
            expected_ranges: Dict of column: (min, max) tuples
            
        Returns:
            Dictionary with range check results
        """
        self.logger.info("Checking numeric ranges...")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        report = {
            'outliers': {},
            'range_violations': {},
            'statistics': {}
        }
        
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            
            report['outliers'][col] = {
                'count': len(outliers),
                'ratio': len(outliers) / len(df),
                'lower_bound': float(lower_bound),
                'upper_bound': float(upper_bound)
            }
            
            report['statistics'][col] = {
                'min': float(df[col].min()),
                'max': float(df[col].max()),
                'mean': float(df[col].mean()),
                'median': float(df[col].median()),
                'std': float(df[col].std())
            }
            
            if expected_ranges and col in expected_ranges:
                exp_min, exp_max = expected_ranges[col]
                violations = df[(df[col] < exp_min) | (df[col] > exp_max)]
                report['range_violations'][col] = len(violations)
        
        self.validation_report['numeric_ranges'] = report
        return report
    
    def check_categorical_distribution(self, df: pd.DataFrame, 
                                      min_categories: int = 2,
                                      max_categories: int = 100) -> Dict:
        """
        Check categorical columns for unusual distributions.
        
        Args:
            df: Input DataFrame
            min_categories: Minimum number of unique values
            max_categories: Maximum number of unique values
            
        Returns:
            Dictionary with categorical statistics
        """
        self.logger.info("Checking categorical distributions...")
        
        cat_cols = df.select_dtypes(include=['object', 'category']).columns
        
        report = {
            'distributions': {},
            'warnings': []
        }
        
        for col in cat_cols:
            value_counts = df[col].value_counts()
            n_unique = df[col].nunique()
            
            report['distributions'][col] = {
                'n_unique': int(n_unique),
                'top_5_values': value_counts.head(5).to_dict(),
                'null_count': int(df[col].isnull().sum())
            }
            
            if n_unique < min_categories:
                msg = f"{col} has only {n_unique} unique values"
                report['warnings'].append(msg)
                self.logger.warning(msg)
            
            if n_unique > max_categories:
                msg = f"{col} has {n_unique} unique values (high cardinality)"
                report['warnings'].append(msg)
                self.logger.warning(msg)
        
        self.validation_report['categorical_distribution'] = report
        return report
    
    def check_date_consistency(self, df: pd.DataFrame, date_cols: List[str]) -> Dict:
        """
        Check date columns for consistency and validity.
        
        Args:
            df: Input DataFrame
            date_cols: List of date column names
            
        Returns:
            Dictionary with date validation results
        """
        self.logger.info("Checking date consistency...")
        
        report = {}
        
        for col in date_cols:
            if col not in df.columns:
                continue
            
            date_series = pd.to_datetime(df[col], errors='coerce')
            
            report[col] = {
                'null_count': int(date_series.isnull().sum()),
                'min_date': str(date_series.min()),
                'max_date': str(date_series.max()),
                'date_range_days': (date_series.max() - date_series.min()).days if not date_series.isnull().all() else 0
            }
        
        self.validation_report['date_consistency'] = report
        return report
    
    def generate_full_report(self, df: pd.DataFrame,
                            expected_ranges: Dict = None,
                            date_cols: List[str] = None) -> Dict:
        """
        Generate comprehensive validation report.
        
        Args:
            df: Input DataFrame
            expected_ranges: Expected ranges for numeric columns
            date_cols: Date columns to check
            
        Returns:
            Complete validation report
        """
        self.logger.info("Generating comprehensive validation report...")
        
        self.validation_report = {
            'dataset_info': {
                'n_rows': len(df),
                'n_columns': len(df.columns),
                'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2
            }
        }
        
        self.check_missing_values(df)
        self.check_duplicates(df)
        self.check_data_types(df)
        self.check_numeric_ranges(df, expected_ranges)
        self.check_categorical_distribution(df)
        
        if date_cols:
            self.check_date_consistency(df, date_cols)
        
        self.logger.info("Validation report complete")
        
        return self.validation_report
    
    def print_report(self):
        """Print formatted validation report."""
        print("\n" + "="*60)
        print("DATA VALIDATION REPORT")
        print("="*60)
        
        if 'dataset_info' in self.validation_report:
            info = self.validation_report['dataset_info']
            print(f"\nDataset Info:")
            print(f"  Rows: {info['n_rows']:,}")
            print(f"  Columns: {info['n_columns']}")
            print(f"  Memory: {info['memory_usage_mb']:.2f} MB")
        
        if 'missing_values' in self.validation_report:
            mv = self.validation_report['missing_values']
            print(f"\nMissing Values:")
            print(f"  Total: {mv['total_missing']:,}")
            if mv['high_missing_columns']:
                print(f"  High missing columns: {mv['high_missing_columns']}")
        
        if 'duplicates' in self.validation_report:
            dup = self.validation_report['duplicates']
            print(f"\nDuplicates:")
            print(f"  Count: {dup['duplicate_count']}")
            print(f"  Ratio: {dup['duplicate_ratio']:.2%}")
        
        print("\n" + "="*60)

if __name__ == "__main__":
    validator = DataValidator()
    print("DataValidator module loaded successfully")
