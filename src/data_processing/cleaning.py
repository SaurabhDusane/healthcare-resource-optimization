"""
Data Cleaning Module
Comprehensive cleaning for NHAMCS and scraped data.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import logging
from typing import Dict, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataCleaner:
    """Main data cleaning class."""
    
    def __init__(self):
        self.logger = logger
    
    def clean_nhamcs_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean NHAMCS emergency room data.
        
        Args:
            df: Raw NHAMCS DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        self.logger.info(f"Cleaning NHAMCS data: {len(df)} records")
        
        df_clean = df.copy()
        
        df_clean = self._handle_missing_values(df_clean)
        
        initial_len = len(df_clean)
        df_clean = df_clean.drop_duplicates()
        self.logger.info(f"Removed {initial_len - len(df_clean)} duplicates")
        
        if 'VDATE' in df_clean.columns:
            df_clean['visit_date'] = pd.to_datetime(df_clean['VDATE'], errors='coerce')
        
        if 'AGE' in df_clean.columns:
            df_clean['age_group'] = pd.cut(
                df_clean['AGE'],
                bins=[0, 18, 45, 65, 120],
                labels=['0-17', '18-44', '45-64', '65+'],
                right=False
            )
        
        if 'ARRTIME' in df_clean.columns:
            df_clean['arrival_hour'] = pd.to_numeric(df_clean['ARRTIME'], errors='coerce') // 100
            df_clean['time_of_day'] = pd.cut(
                df_clean['arrival_hour'],
                bins=[0, 6, 12, 18, 24],
                labels=['Night', 'Morning', 'Afternoon', 'Evening'],
                right=False
            )
        
        if 'IMMEDR' in df_clean.columns:
            df_clean['high_acuity'] = df_clean['IMMEDR'].isin([1, 2]).astype(int)
        
        if 'PAYTYPER' in df_clean.columns:
            df_clean['has_insurance'] = (~df_clean['PAYTYPER'].isin([5, 6])).astype(int)
        
        diag_cols = [col for col in df_clean.columns if 'DIAG' in col]
        for col in diag_cols:
            df_clean[col] = df_clean[col].astype(str).str.strip()
        
        if 'visit_date' in df_clean.columns:
            df_clean['day_of_week'] = df_clean['visit_date'].dt.dayofweek
            df_clean['is_weekend'] = (df_clean['day_of_week'] >= 5).astype(int)
            df_clean['month'] = df_clean['visit_date'].dt.month
        
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df_clean[col].std() > 0:
                upper_bound = df_clean[col].quantile(0.99)
                df_clean[col] = df_clean[col].clip(upper=upper_bound)
        
        self.logger.info(f"Cleaning complete: {len(df_clean)} records, {len(df_clean.columns)} columns")
        
        return df_clean
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values with appropriate strategies."""
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().sum() > 0:
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
        
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            if df[col].isnull().sum() > 0:
                mode_val = df[col].mode()[0] if not df[col].mode().empty else 'Unknown'
                df[col] = df[col].fillna(mode_val)
        
        return df
    
    def clean_scraped_data(self, df: pd.DataFrame, source: str) -> pd.DataFrame:
        """
        Clean scraped data (CDC, Reddit, Twitter).
        
        Args:
            df: Raw scraped DataFrame
            source: Data source ('cdc', 'reddit', 'twitter')
            
        Returns:
            Cleaned DataFrame
        """
        self.logger.info(f"Cleaning {source} data: {len(df)} records")
        
        df_clean = df.copy()
        
        if 'text' in df_clean.columns:
            df_clean = df_clean[df_clean['text'].notna()]
            df_clean = df_clean[df_clean['text'].str.strip() != '']
        
        if 'content' in df_clean.columns:
            df_clean = df_clean[df_clean['content'].notna()]
            df_clean = df_clean[df_clean['content'].str.strip() != '']
        
        if 'date' in df_clean.columns:
            df_clean['date'] = pd.to_datetime(df_clean['date'], errors='coerce')
            df_clean = df_clean.dropna(subset=['date'])
        
        text_cols = ['text', 'title', 'content', 'clean_text']
        for col in text_cols:
            if col in df_clean.columns:
                df_clean[f'{col}_lower'] = df_clean[col].str.lower()
                df_clean[col] = df_clean[col].str.strip()
        
        if 'sentiment_polarity' in df_clean.columns:
            df_clean = df_clean[
                (df_clean['sentiment_polarity'] >= -1) & 
                (df_clean['sentiment_polarity'] <= 1)
            ]
        
        duplicate_cols = ['date', 'title'] if 'title' in df_clean.columns else ['date', 'text']
        duplicate_cols = [col for col in duplicate_cols if col in df_clean.columns]
        if duplicate_cols:
            initial_len = len(df_clean)
            df_clean = df_clean.drop_duplicates(subset=duplicate_cols)
            self.logger.info(f"Removed {initial_len - len(df_clean)} duplicates")
        
        if 'date' in df_clean.columns:
            df_clean['day_of_week'] = df_clean['date'].dt.dayofweek
            df_clean['month'] = df_clean['date'].dt.month
            df_clean['year'] = df_clean['date'].dt.year
        
        if 'date' in df_clean.columns:
            df_clean = df_clean.sort_values('date').reset_index(drop=True)
        
        self.logger.info(f"Cleaning complete: {len(df_clean)} records")
        
        return df_clean
    
    def merge_datasets(self, 
                      nhamcs_df: pd.DataFrame,
                      cdc_df: Optional[pd.DataFrame] = None,
                      reddit_df: Optional[pd.DataFrame] = None,
                      twitter_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Merge NHAMCS data with scraped data sources.
        
        Args:
            nhamcs_df: Cleaned NHAMCS data
            cdc_df: Cleaned CDC news data
            reddit_df: Cleaned Reddit data
            twitter_df: Cleaned Twitter data
            
        Returns:
            Merged DataFrame
        """
        self.logger.info("Merging datasets...")
        
        merged_df = nhamcs_df.copy()
        
        if cdc_df is not None and not cdc_df.empty:
            cdc_agg = cdc_df.groupby(cdc_df['date'].dt.date).agg({
                'title': 'count',
                'keywords': lambda x: ', '.join(x)
            }).rename(columns={'title': 'news_mentions'})
            cdc_agg.index = pd.to_datetime(cdc_agg.index)
            
            for lag in [1, 3, 5, 7]:
                cdc_agg[f'news_mentions_lag{lag}'] = cdc_agg['news_mentions'].shift(lag)
            
            if 'visit_date' in merged_df.columns:
                merged_df = merged_df.merge(
                    cdc_agg,
                    left_on=merged_df['visit_date'].dt.date,
                    right_index=True,
                    how='left'
                )
        
        if reddit_df is not None and not reddit_df.empty:
            reddit_agg = reddit_df.groupby(reddit_df['date'].dt.date).agg({
                'post_id': 'count',
                'sentiment_polarity': 'mean',
                'symptoms_mentioned': lambda x: ', '.join(x)
            }).rename(columns={'post_id': 'reddit_posts', 'sentiment_polarity': 'reddit_sentiment'})
            reddit_agg.index = pd.to_datetime(reddit_agg.index)
            
            reddit_agg['reddit_sentiment_7d'] = reddit_agg['reddit_sentiment'].rolling(7).mean()
            
            if 'visit_date' in merged_df.columns:
                merged_df = merged_df.merge(
                    reddit_agg,
                    left_on=merged_df['visit_date'].dt.date,
                    right_index=True,
                    how='left'
                )
        
        if twitter_df is not None and not twitter_df.empty:
            twitter_agg = twitter_df.groupby(twitter_df['date'].dt.date).agg({
                'tweet_id': 'count',
                'sentiment_polarity': 'mean',
                'likes': 'sum',
                'retweets': 'sum'
            }).rename(columns={
                'tweet_id': 'tweet_count',
                'sentiment_polarity': 'twitter_sentiment'
            })
            twitter_agg.index = pd.to_datetime(twitter_agg.index)
            
            if 'visit_date' in merged_df.columns:
                merged_df = merged_df.merge(
                    twitter_agg,
                    left_on=merged_df['visit_date'].dt.date,
                    right_index=True,
                    how='left'
                )
        
        scraped_cols = [col for col in merged_df.columns if any(
            x in col for x in ['news', 'reddit', 'twitter', 'sentiment']
        )]
        merged_df[scraped_cols] = merged_df[scraped_cols].fillna(0)
        
        self.logger.info(f"Merge complete: {len(merged_df)} records, {len(merged_df.columns)} columns")
        
        return merged_df

if __name__ == "__main__":
    cleaner = DataCleaner()
    print("DataCleaner module loaded successfully")
