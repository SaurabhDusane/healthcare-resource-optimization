"""
Feature Engineering Module
Creates advanced features for machine learning models.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import logging
from typing import List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureEngineer:
    """Feature engineering for healthcare data."""
    
    def __init__(self):
        self.logger = logger
    
    def create_temporal_features(self, df: pd.DataFrame, date_col: str = 'visit_date') -> pd.DataFrame:
        """
        Create time-based features.
        
        Args:
            df: Input DataFrame
            date_col: Name of date column
            
        Returns:
            DataFrame with temporal features
        """
        self.logger.info("Creating temporal features...")
        
        df_feat = df.copy()
        
        if date_col not in df_feat.columns:
            self.logger.warning(f"Date column '{date_col}' not found")
            return df_feat
        
        df_feat[date_col] = pd.to_datetime(df_feat[date_col])
        
        df_feat['year'] = df_feat[date_col].dt.year
        df_feat['month'] = df_feat[date_col].dt.month
        df_feat['day'] = df_feat[date_col].dt.day
        df_feat['day_of_week'] = df_feat[date_col].dt.dayofweek
        df_feat['week_of_year'] = df_feat[date_col].dt.isocalendar().week
        df_feat['quarter'] = df_feat[date_col].dt.quarter
        
        df_feat['is_weekend'] = (df_feat['day_of_week'] >= 5).astype(int)
        df_feat['is_monday'] = (df_feat['day_of_week'] == 0).astype(int)
        df_feat['is_friday'] = (df_feat['day_of_week'] == 4).astype(int)
        
        df_feat['is_month_start'] = df_feat[date_col].dt.is_month_start.astype(int)
        df_feat['is_month_end'] = df_feat[date_col].dt.is_month_end.astype(int)
        
        holidays = ['2023-01-01', '2023-07-04', '2023-11-23', '2023-12-25']
        holiday_dates = pd.to_datetime(holidays)
        df_feat['is_holiday'] = df_feat[date_col].isin(holiday_dates).astype(int)
        
        df_feat['days_since_epoch'] = (df_feat[date_col] - pd.Timestamp('2020-01-01')).dt.days
        
        self.logger.info(f"Created {df_feat.shape[1] - df.shape[1]} temporal features")
        
        return df_feat
    
    def create_cyclical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create cyclical encodings for temporal features.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with cyclical features
        """
        self.logger.info("Creating cyclical features...")
        
        df_feat = df.copy()
        
        if 'day_of_week' in df_feat.columns:
            df_feat['day_of_week_sin'] = np.sin(2 * np.pi * df_feat['day_of_week'] / 7)
            df_feat['day_of_week_cos'] = np.cos(2 * np.pi * df_feat['day_of_week'] / 7)
        
        if 'month' in df_feat.columns:
            df_feat['month_sin'] = np.sin(2 * np.pi * df_feat['month'] / 12)
            df_feat['month_cos'] = np.cos(2 * np.pi * df_feat['month'] / 12)
        
        if 'arrival_hour' in df_feat.columns:
            df_feat['hour_sin'] = np.sin(2 * np.pi * df_feat['arrival_hour'] / 24)
            df_feat['hour_cos'] = np.cos(2 * np.pi * df_feat['arrival_hour'] / 24)
        
        self.logger.info("Cyclical features created")
        
        return df_feat
    
    def create_lagged_features(self, df: pd.DataFrame, 
                              value_cols: List[str],
                              lags: List[int] = [1, 3, 7, 14]) -> pd.DataFrame:
        """
        Create lagged features for time series.
        
        Args:
            df: Input DataFrame
            value_cols: Columns to create lags for
            lags: List of lag periods
            
        Returns:
            DataFrame with lagged features
        """
        self.logger.info(f"Creating lagged features for {value_cols}...")
        
        df_feat = df.copy()
        
        for col in value_cols:
            if col in df_feat.columns:
                for lag in lags:
                    df_feat[f'{col}_lag{lag}'] = df_feat[col].shift(lag)
        
        self.logger.info(f"Created lagged features with lags: {lags}")
        
        return df_feat
    
    def create_rolling_features(self, df: pd.DataFrame,
                               value_cols: List[str],
                               windows: List[int] = [3, 7, 14]) -> pd.DataFrame:
        """
        Create rolling window statistics.
        
        Args:
            df: Input DataFrame
            value_cols: Columns to calculate rolling stats for
            windows: List of window sizes
            
        Returns:
            DataFrame with rolling features
        """
        self.logger.info(f"Creating rolling features for {value_cols}...")
        
        df_feat = df.copy()
        
        for col in value_cols:
            if col in df_feat.columns:
                for window in windows:
                    df_feat[f'{col}_rolling_mean_{window}d'] = df_feat[col].rolling(window).mean()
                    df_feat[f'{col}_rolling_std_{window}d'] = df_feat[col].rolling(window).std()
                    df_feat[f'{col}_rolling_max_{window}d'] = df_feat[col].rolling(window).max()
                    df_feat[f'{col}_rolling_min_{window}d'] = df_feat[col].rolling(window).min()
        
        self.logger.info(f"Created rolling features with windows: {windows}")
        
        return df_feat
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with interaction features
        """
        self.logger.info("Creating interaction features...")
        
        df_feat = df.copy()
        
        if 'is_weekend' in df_feat.columns and 'arrival_hour' in df_feat.columns:
            df_feat['weekend_evening'] = (
                (df_feat['is_weekend'] == 1) & 
                (df_feat['arrival_hour'].between(18, 23))
            ).astype(int)
        
        if 'age_group' in df_feat.columns and 'has_insurance' in df_feat.columns:
            df_feat['senior_uninsured'] = (
                (df_feat['age_group'] == '65+') & 
                (df_feat['has_insurance'] == 0)
            ).astype(int)
        
        if 'is_weekend' in df_feat.columns and 'high_acuity' in df_feat.columns:
            df_feat['weekend_high_acuity'] = (
                df_feat['is_weekend'] * df_feat['high_acuity']
            )
        
        if 'month' in df_feat.columns:
            df_feat['is_flu_season'] = df_feat['month'].isin([10, 11, 12, 1, 2, 3]).astype(int)
        
        self.logger.info("Interaction features created")
        
        return df_feat
    
    def create_aggregated_features(self, df: pd.DataFrame, 
                                  group_cols: List[str],
                                  agg_col: str) -> pd.DataFrame:
        """
        Create aggregated features by groups.
        
        Args:
            df: Input DataFrame
            group_cols: Columns to group by
            agg_col: Column to aggregate
            
        Returns:
            DataFrame with aggregated features
        """
        self.logger.info(f"Creating aggregated features for {group_cols}...")
        
        df_feat = df.copy()
        
        for group_col in group_cols:
            if group_col in df_feat.columns and agg_col in df_feat.columns:
                group_stats = df_feat.groupby(group_col)[agg_col].agg([
                    'mean', 'std', 'count', 'min', 'max'
                ]).add_prefix(f'{group_col}_')
                
                df_feat = df_feat.merge(group_stats, left_on=group_col, right_index=True, how='left')
        
        self.logger.info("Aggregated features created")
        
        return df_feat
    
    def create_all_features(self, df: pd.DataFrame, 
                           date_col: str = 'visit_date',
                           value_cols: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Create all feature types.
        
        Args:
            df: Input DataFrame
            date_col: Name of date column
            value_cols: Columns for lagged/rolling features
            
        Returns:
            DataFrame with all features
        """
        self.logger.info("Creating comprehensive feature set...")
        
        df_feat = df.copy()
        
        df_feat = self.create_temporal_features(df_feat, date_col)
        
        df_feat = self.create_cyclical_features(df_feat)
        
        if value_cols:
            df_feat = self.create_lagged_features(df_feat, value_cols)
            df_feat = self.create_rolling_features(df_feat, value_cols)
        
        df_feat = self.create_interaction_features(df_feat)
        
        initial_cols = len(df.columns)
        final_cols = len(df_feat.columns)
        self.logger.info(f"Feature engineering complete: {final_cols - initial_cols} new features created")
        
        return df_feat

if __name__ == "__main__":
    engineer = FeatureEngineer()
    print("FeatureEngineer module loaded successfully")
