"""Sentiment Analysis Module"""

import pandas as pd
from textblob import TextBlob
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    """Perform sentiment analysis on text data."""
    
    def __init__(self):
        self.logger = logger
    
    def analyze_text(self, text: str) -> dict:
        """Analyze sentiment of single text."""
        try:
            blob = TextBlob(str(text))
            return {
                'polarity': blob.sentiment.polarity,
                'subjectivity': blob.sentiment.subjectivity,
                'sentiment_label': self._get_sentiment_label(blob.sentiment.polarity)
            }
        except:
            return {'polarity': 0.0, 'subjectivity': 0.0, 'sentiment_label': 'neutral'}
    
    def _get_sentiment_label(self, polarity: float) -> str:
        """Convert polarity score to label."""
        if polarity > 0.3:
            return 'positive'
        elif polarity < -0.3:
            return 'negative'
        return 'neutral'
    
    def analyze_dataframe(self, df: pd.DataFrame, text_col: str) -> pd.DataFrame:
        """Analyze sentiment for entire DataFrame."""
        self.logger.info(f"Analyzing sentiment for {len(df)} texts...")
        
        df_result = df.copy()
        sentiments = df_result[text_col].apply(self.analyze_text)
        
        df_result['sentiment_polarity'] = sentiments.apply(lambda x: x['polarity'])
        df_result['sentiment_subjectivity'] = sentiments.apply(lambda x: x['subjectivity'])
        df_result['sentiment_label'] = sentiments.apply(lambda x: x['sentiment_label'])
        
        self.logger.info("Sentiment analysis complete")
        return df_result
    
    def get_sentiment_summary(self, df: pd.DataFrame) -> dict:
        """Get summary statistics of sentiment."""
        return {
            'mean_polarity': df['sentiment_polarity'].mean(),
            'std_polarity': df['sentiment_polarity'].std(),
            'positive_count': (df['sentiment_label'] == 'positive').sum(),
            'negative_count': (df['sentiment_label'] == 'negative').sum(),
            'neutral_count': (df['sentiment_label'] == 'neutral').sum()
        }
