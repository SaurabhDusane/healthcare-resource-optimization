"""
Data Merger Module
Combines multiple data sources into unified datasets.
"""

import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataMerger:
    """Merge and combine multiple data sources."""
    
    def __init__(self):
        self.logger = logger
    
    def merge_all_sources(self, nhamcs_df, cdc_df=None, reddit_df=None, twitter_df=None):
        """Merge all data sources."""
        self.logger.info("Merging all data sources...")
        merged_df = nhamcs_df.copy()
        
        if cdc_df is not None and not cdc_df.empty:
            cdc_agg = cdc_df.groupby(cdc_df['date'].dt.date).size().to_frame('cdc_count')
            cdc_agg.index = pd.to_datetime(cdc_agg.index)
            merged_df = merged_df.merge(cdc_agg, left_on='visit_date', right_index=True, how='left')
        
        if reddit_df is not None and not reddit_df.empty:
            reddit_agg = reddit_df.groupby(reddit_df['date'].dt.date).agg({
                'sentiment_polarity': 'mean'
            }).rename(columns={'sentiment_polarity': 'reddit_sentiment'})
            reddit_agg.index = pd.to_datetime(reddit_agg.index)
            merged_df = merged_df.merge(reddit_agg, left_on='visit_date', right_index=True, how='left')
        
        if twitter_df is not None and not twitter_df.empty:
            twitter_agg = twitter_df.groupby(twitter_df['date'].dt.date).size().to_frame('twitter_count')
            twitter_agg.index = pd.to_datetime(twitter_agg.index)
            merged_df = merged_df.merge(twitter_agg, left_on='visit_date', right_index=True, how='left')
        
        merged_df = merged_df.fillna(0)
        self.logger.info(f"Merge complete: {len(merged_df)} records")
        return merged_df
