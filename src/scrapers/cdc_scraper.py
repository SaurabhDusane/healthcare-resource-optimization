"""
CDC News Scraper
Scrapes CDC newsroom for health alerts and outbreak announcements.
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
from typing import List, Dict
import re
from .base_scraper import BaseScraper

class CDCScraper(BaseScraper):
    """Scraper for CDC news and health alerts."""
    
    def __init__(self):
        super().__init__(name='cdc_news', rate_limit_seconds=2)
        self.base_url = "https://www.cdc.gov"
        self.news_url = f"{self.base_url}/media/releases/"
        
        self.keywords = [
            'flu', 'influenza', 'covid', 'coronavirus', 'rsv',
            'outbreak', 'emergency', 'hospitalization', 'respiratory',
            'epidemic', 'pandemic', 'vaccine', 'infection'
        ]
    
    def extract_health_keywords(self, text: str) -> str:
        """
        Extract relevant health keywords from text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Comma-separated string of found keywords
        """
        text_lower = text.lower()
        found_keywords = [kw for kw in self.keywords if kw in text_lower]
        return ', '.join(found_keywords) if found_keywords else 'none'
    
    def scrape_article_content(self, article_url: str) -> str:
        """
        Scrape full text from individual article page.
        
        Args:
            article_url: URL of article to scrape
            
        Returns:
            Article text content
        """
        try:
            self.rate_limit_delay()
            response = self.session.get(article_url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            content_selectors = [
                'div.syndicate',
                'div.content',
                'article',
                'div.col-md-12'
            ]
            
            for selector in content_selectors:
                content_div = soup.select_one(selector)
                if content_div:
                    text = content_div.get_text(separator=' ', strip=True)
                    text = re.sub(r'\s+', ' ', text)
                    return text[:5000]
            
            return "Content not found"
            
        except Exception as e:
            self.logger.warning(f"Error scraping article {article_url}: {str(e)}")
            return "Error retrieving content"
    
    def scrape(self) -> pd.DataFrame:
        """
        Scrape CDC news releases.
        
        Returns:
            DataFrame with columns: date, title, content, url, keywords, category
        """
        articles = []
        
        try:
            self.logger.info(f"Fetching CDC news from {self.news_url}")
            response = self.session.get(self.news_url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            article_cards = soup.find_all('div', class_='card-body', limit=100)
            
            self.logger.info(f"Found {len(article_cards)} articles to process")
            
            for idx, card in enumerate(article_cards, 1):
                try:
                    title_elem = card.find('h3')
                    if not title_elem:
                        continue
                    
                    title = title_elem.get_text(strip=True)
                    
                    link_elem = card.find('a', href=True)
                    if not link_elem:
                        continue
                    
                    article_link = link_elem['href']
                    if not article_link.startswith('http'):
                        article_link = f"{self.base_url}{article_link}"
                    
                    date_elem = card.find('time')
                    if date_elem and date_elem.get('datetime'):
                        date_str = date_elem['datetime']
                        date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                    else:
                        date = datetime.now()
                    
                    self.logger.info(f"Scraping article {idx}/{len(article_cards)}: {title[:50]}...")
                    content = self.scrape_article_content(article_link)
                    
                    keywords = self.extract_health_keywords(f"{title} {content}")
                    
                    category = self._categorize_article(title, content)
                    
                    articles.append({
                        'date': date,
                        'title': title,
                        'content': content,
                        'url': article_link,
                        'keywords': keywords,
                        'category': category,
                        'source': 'CDC',
                        'scraped_at': datetime.now()
                    })
                    
                except Exception as e:
                    self.logger.warning(f"Error processing article {idx}: {str(e)}")
                    continue
            
            df = pd.DataFrame(articles)
            
            required_cols = ['date', 'title', 'content', 'url', 'keywords']
            if self.validate_data(df, required_cols):
                self.logger.info(f"Successfully scraped {len(df)} CDC articles")
                return df
            else:
                self.logger.error("Data validation failed")
                return pd.DataFrame()
                
        except Exception as e:
            self.logger.error(f"Error in main scraping loop: {str(e)}")
            return pd.DataFrame()
    
    def _categorize_article(self, title: str, content: str) -> str:
        """Categorize article based on keywords."""
        text = f"{title} {content}".lower()
        
        if any(word in text for word in ['outbreak', 'epidemic', 'surge']):
            return 'Outbreak'
        elif any(word in text for word in ['vaccine', 'vaccination', 'immunization']):
            return 'Vaccination'
        elif any(word in text for word in ['advisory', 'alert', 'warning']):
            return 'Alert'
        elif any(word in text for word in ['flu', 'influenza', 'respiratory']):
            return 'Respiratory Illness'
        else:
            return 'General Health'

if __name__ == "__main__":
    scraper = CDCScraper()
    data = scraper.run()
    
    if data is not None:
        print(f"\nScraped {len(data)} articles")
        print(f"\nSample data:")
        print(data[['date', 'title', 'keywords', 'category']].head())
        print(f"\nKeyword distribution:")
        print(data['keywords'].value_counts().head(10))
