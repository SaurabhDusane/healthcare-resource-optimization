"""
Twitter/X Health Trends Scraper
Scrapes health-related tweets for sentiment and trend analysis.
"""

import snscrape.modules.twitter as sntwitter
import pandas as pd
from datetime import datetime, timedelta
from textblob import TextBlob
from typing import List, Dict
import re
from .base_scraper import BaseScraper

class TwitterScraper(BaseScraper):
    """Scraper for Twitter health trends."""
    
    def __init__(self):
        super().__init__(name='twitter_health', rate_limit_seconds=1)
        
        self.hashtags = ['#fluseason', '#ERwait', '#sicktoday', '#healthanxiety']
        self.keywords = [
            'emergency room', 'urgent care', 'feel sick',
            'hospital wait', 'flu symptoms', 'covid symptoms'
        ]
        
        self.location = 'Phoenix'
        
        self.lookback_days = 7
        self.max_tweets_per_query = 500
    
    def clean_tweet_text(self, text: str) -> str:
        """
        Clean tweet text.
        
        Args:
            text: Raw tweet text
            
        Returns:
            Cleaned text
        """
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#(\w+)', r'\1', text)
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """Perform sentiment analysis."""
        try:
            blob = TextBlob(text)
            return {
                'polarity': blob.sentiment.polarity,
                'subjectivity': blob.sentiment.subjectivity
            }
        except:
            return {'polarity': 0.0, 'subjectivity': 0.0}
    
    def scrape_query(self, query: str, max_tweets: int = 500) -> List[Dict]:
        """
        Scrape tweets for a specific query.
        
        Args:
            query: Search query
            max_tweets: Maximum tweets to retrieve
            
        Returns:
            List of tweet dictionaries
        """
        tweets_data = []
        since_date = (datetime.now() - timedelta(days=self.lookback_days)).strftime('%Y-%m-%d')
        
        full_query = f"{query} since:{since_date}"
        
        try:
            self.logger.info(f"Searching for: {full_query}")
            
            for i, tweet in enumerate(sntwitter.TwitterSearchScraper(full_query).get_items()):
                if i >= max_tweets:
                    break
                
                clean_text = self.clean_tweet_text(tweet.content)
                
                sentiment = self.analyze_sentiment(clean_text)
                
                health_keywords = self._extract_health_keywords(clean_text)
                
                tweets_data.append({
                    'date': tweet.date,
                    'tweet_id': tweet.id,
                    'username': tweet.user.username,
                    'text': tweet.content[:280],
                    'clean_text': clean_text[:280],
                    'likes': tweet.likeCount,
                    'retweets': tweet.retweetCount,
                    'replies': tweet.replyCount,
                    'sentiment_polarity': sentiment['polarity'],
                    'sentiment_subjectivity': sentiment['subjectivity'],
                    'health_keywords': health_keywords,
                    'query': query,
                    'url': tweet.url,
                    'scraped_at': datetime.now()
                })
                
                if (i + 1) % 100 == 0:
                    self.logger.info(f"Collected {i + 1} tweets for query: {query}")
            
            self.logger.info(f"Total tweets for '{query}': {len(tweets_data)}")
            
        except Exception as e:
            self.logger.error(f"Error scraping query '{query}': {str(e)}")
        
        return tweets_data
    
    def _extract_health_keywords(self, text: str) -> str:
        """Extract health-related keywords."""
        keywords = [
            'flu', 'fever', 'cough', 'sick', 'symptoms', 'hospital',
            'doctor', 'emergency', 'pain', 'covid', 'test', 'positive'
        ]
        text_lower = text.lower()
        found = [kw for kw in keywords if kw in text_lower]
        return ', '.join(found) if found else 'none'
    
    def scrape(self) -> pd.DataFrame:
        """
        Scrape tweets for all configured queries.
        
        Returns:
            DataFrame with all tweets
        """
        all_tweets = []
        
        for hashtag in self.hashtags:
            tweets = self.scrape_query(hashtag, self.max_tweets_per_query)
            all_tweets.extend(tweets)
        
        for keyword in self.keywords:
            tweets = self.scrape_query(f'"{keyword}"', self.max_tweets_per_query // 2)
            all_tweets.extend(tweets)
        
        df = pd.DataFrame(all_tweets)
        
        if not df.empty:
            df = df.drop_duplicates(subset=['tweet_id'])
            
            df['sentiment_category'] = pd.cut(
                df['sentiment_polarity'],
                bins=[-1, -0.3, 0.3, 1],
                labels=['Negative', 'Neutral', 'Positive']
            )
            
            required_cols = ['date', 'text', 'sentiment_polarity']
            if self.validate_data(df, required_cols):
                self.logger.info(f"Successfully scraped {len(df)} unique tweets")
                return df
        
        return pd.DataFrame()

if __name__ == "__main__":
    scraper = TwitterScraper()
    data = scraper.run()
    
    if data is not None:
        print(f"\nScraped {len(data)} tweets")
        print(f"\nSentiment distribution:")
        print(data['sentiment_category'].value_counts())
        print(f"\nTop queries:")
        print(data['query'].value_counts().head())
        print(f"\nEngagement stats:")
        print(f"Total likes: {data['likes'].sum()}")
        print(f"Total retweets: {data['retweets'].sum()}")
