"""
Reddit Health Community Scraper
Scrapes health-related subreddits for symptom trends and sentiment analysis.
"""

import praw
import pandas as pd
from datetime import datetime, timedelta
from textblob import TextBlob
from typing import List, Dict
import re
import os
from dotenv import load_dotenv
from .base_scraper import BaseScraper

load_dotenv()

class RedditScraper(BaseScraper):
    """Scraper for Reddit health communities."""
    
    def __init__(self):
        super().__init__(name='reddit_health', rate_limit_seconds=1)
        
        self.reddit = praw.Reddit(
            client_id=os.getenv('REDDIT_CLIENT_ID', 'YOUR_CLIENT_ID'),
            client_secret=os.getenv('REDDIT_CLIENT_SECRET', 'YOUR_CLIENT_SECRET'),
            user_agent=os.getenv('REDDIT_USER_AGENT', 'HealthTrendAnalyzer/1.0')
        )
        
        self.subreddits = ['AskDocs', 'HealthAnxiety', 'phoenix', 'arizona']
        
        self.symptoms = [
            'fever', 'cough', 'shortness of breath', 'chest pain',
            'nausea', 'vomiting', 'headache', 'fatigue', 'sore throat',
            'body aches', 'chills', 'dizziness', 'congestion',
            'difficulty breathing', 'stomach pain', 'diarrhea'
        ]
        
        self.lookback_days = 30
    
    def detect_symptoms(self, text: str) -> str:
        """
        Detect symptom keywords in text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Comma-separated string of found symptoms
        """
        text_lower = text.lower()
        found_symptoms = [s for s in self.symptoms if s in text_lower]
        return ', '.join(found_symptoms) if found_symptoms else 'none'
    
    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """
        Perform sentiment analysis on text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with polarity and subjectivity scores
        """
        try:
            blob = TextBlob(text)
            return {
                'polarity': blob.sentiment.polarity,
                'subjectivity': blob.sentiment.subjectivity
            }
        except:
            return {'polarity': 0.0, 'subjectivity': 0.0}
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'[^\w\s\']', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def scrape_subreddit(self, subreddit_name: str, limit: int = 500) -> List[Dict]:
        """
        Scrape posts from a single subreddit.
        
        Args:
            subreddit_name: Name of subreddit
            limit: Maximum posts to retrieve
            
        Returns:
            List of post dictionaries
        """
        posts_data = []
        cutoff_date = datetime.now() - timedelta(days=self.lookback_days)
        
        try:
            subreddit = self.reddit.subreddit(subreddit_name)
            self.logger.info(f"Scraping r/{subreddit_name}...")
            
            for post in subreddit.new(limit=limit):
                post_date = datetime.fromtimestamp(post.created_utc)
                
                if post_date < cutoff_date:
                    continue
                
                full_text = f"{post.title} {post.selftext}"
                clean_full_text = self.clean_text(full_text)
                
                sentiment = self.analyze_sentiment(clean_full_text)
                
                symptoms = self.detect_symptoms(clean_full_text)
                
                is_health_related = any(
                    kw in clean_full_text.lower() 
                    for kw in ['sick', 'pain', 'doctor', 'hospital', 'symptoms', 'health']
                )
                
                posts_data.append({
                    'date': post_date,
                    'subreddit': subreddit_name,
                    'post_id': post.id,
                    'title': post.title,
                    'text': post.selftext[:1000],
                    'full_text_clean': clean_full_text[:1000],
                    'score': post.score,
                    'num_comments': post.num_comments,
                    'upvote_ratio': post.upvote_ratio,
                    'sentiment_polarity': sentiment['polarity'],
                    'sentiment_subjectivity': sentiment['subjectivity'],
                    'symptoms_mentioned': symptoms,
                    'is_health_related': is_health_related,
                    'url': f"https://reddit.com{post.permalink}",
                    'scraped_at': datetime.now()
                })
                
                self.rate_limit_delay()
            
            self.logger.info(f"Collected {len(posts_data)} posts from r/{subreddit_name}")
            
        except Exception as e:
            self.logger.error(f"Error scraping r/{subreddit_name}: {str(e)}")
        
        return posts_data
    
    def scrape(self) -> pd.DataFrame:
        """
        Scrape all configured subreddits.
        
        Returns:
            DataFrame with all posts
        """
        all_posts = []
        
        for subreddit in self.subreddits:
            posts = self.scrape_subreddit(subreddit)
            all_posts.extend(posts)
            self.logger.info(f"Total posts collected so far: {len(all_posts)}")
        
        df = pd.DataFrame(all_posts)
        
        if not df.empty:
            required_cols = ['date', 'subreddit', 'title', 'sentiment_polarity']
            if self.validate_data(df, required_cols):
                df['sentiment_category'] = pd.cut(
                    df['sentiment_polarity'],
                    bins=[-1, -0.3, 0.3, 1],
                    labels=['Negative', 'Neutral', 'Positive']
                )
                
                self.logger.info(f"Successfully scraped {len(df)} Reddit posts")
                return df
            
        return pd.DataFrame()

if __name__ == "__main__":
    scraper = RedditScraper()
    data = scraper.run()
    
    if data is not None:
        print(f"\nScraped {len(data)} posts")
        print(f"\nPosts by subreddit:")
        print(data['subreddit'].value_counts())
        print(f"\nSentiment distribution:")
        print(data['sentiment_category'].value_counts())
        print(f"\nHealth-related posts: {data['is_health_related'].sum()}")
        print(f"\nTop symptoms mentioned:")
        symptoms_exploded = data['symptoms_mentioned'].str.split(', ').explode()
        print(symptoms_exploded[symptoms_exploded != 'none'].value_counts().head(10))
