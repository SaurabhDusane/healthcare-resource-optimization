"""
Base scraper class with common functionality for all web scrapers.
Implements error handling, rate limiting, and logging.
"""

import logging
import time
import os
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import pandas as pd
from datetime import datetime
import requests

class BaseScraper(ABC):
    """Abstract base class for web scrapers."""
    
    def __init__(self, name: str, rate_limit_seconds: int = 2):
        """
        Initialize base scraper.
        
        Args:
            name: Scraper identifier for logging
            rate_limit_seconds: Delay between requests
        """
        self.name = name
        self.rate_limit = rate_limit_seconds
        self.logger = self._setup_logger()
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
    def _setup_logger(self) -> logging.Logger:
        """Configure logger for scraper."""
        logger = logging.getLogger(self.name)
        logger.setLevel(logging.INFO)
        
        os.makedirs('logs', exist_ok=True)
        
        fh = logging.FileHandler(f'logs/{self.name}.log')
        fh.setLevel(logging.INFO)
        
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        logger.addHandler(fh)
        logger.addHandler(ch)
        
        return logger
    
    def rate_limit_delay(self):
        """Implement rate limiting."""
        time.sleep(self.rate_limit)
    
    @abstractmethod
    def scrape(self) -> pd.DataFrame:
        """
        Main scraping method to be implemented by subclasses.
        
        Returns:
            DataFrame with scraped data
        """
        pass
    
    def save_data(self, df: pd.DataFrame, filename: str) -> None:
        """
        Save scraped data to CSV.
        
        Args:
            df: DataFrame to save
            filename: Output filename
        """
        try:
            filepath = f"data/raw/{self.name}/{filename}"
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            df.to_csv(filepath, index=False)
            self.logger.info(f"Saved {len(df)} records to {filepath}")
        except Exception as e:
            self.logger.error(f"Error saving data: {str(e)}")
            raise
    
    def validate_data(self, df: pd.DataFrame, required_columns: list) -> bool:
        """
        Validate scraped data.
        
        Args:
            df: DataFrame to validate
            required_columns: List of required column names
            
        Returns:
            True if valid, False otherwise
        """
        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            self.logger.error(f"Missing required columns: {missing_cols}")
            return False
        
        if df.empty:
            self.logger.warning("DataFrame is empty")
            return False
        
        null_counts = df[required_columns].isnull().sum()
        if null_counts.any():
            self.logger.warning(f"Null values found: {null_counts[null_counts > 0]}")
        
        return True
    
    def run(self) -> Optional[pd.DataFrame]:
        """
        Execute scraping with error handling.
        
        Returns:
            Scraped DataFrame or None if error
        """
        try:
            self.logger.info(f"Starting {self.name} scraper...")
            df = self.scrape()
            
            if df is not None and not df.empty:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                self.save_data(df, f"{self.name}_{timestamp}.csv")
                self.logger.info(f"Scraping completed successfully: {len(df)} records")
                return df
            else:
                self.logger.warning("No data scraped")
                return None
                
        except Exception as e:
            self.logger.error(f"Scraping failed: {str(e)}", exc_info=True)
            return None
