"""
Unit tests for web scrapers
"""

import unittest
import pandas as pd
from datetime import datetime
import sys
sys.path.append('..')

from src.scrapers.base_scraper import BaseScraper
from src.scrapers.cdc_scraper import CDCScraper

class TestBaseScraper(unittest.TestCase):
    """Test BaseScraper functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.scraper = CDCScraper()
    
    def test_rate_limit_delay(self):
        """Test rate limiting works."""
        start_time = datetime.now()
        self.scraper.rate_limit_delay()
        end_time = datetime.now()
        
        elapsed = (end_time - start_time).total_seconds()
        self.assertGreaterEqual(elapsed, self.scraper.rate_limit)
    
    def test_validate_data_success(self):
        """Test data validation with valid data."""
        df = pd.DataFrame({
            'date': [datetime.now()],
            'title': ['Test'],
            'content': ['Test content']
        })
        
        result = self.scraper.validate_data(df, ['date', 'title', 'content'])
        self.assertTrue(result)
    
    def test_validate_data_missing_columns(self):
        """Test data validation with missing columns."""
        df = pd.DataFrame({
            'date': [datetime.now()],
            'title': ['Test']
        })
        
        result = self.scraper.validate_data(df, ['date', 'title', 'content'])
        self.assertFalse(result)
    
    def test_validate_data_empty(self):
        """Test data validation with empty DataFrame."""
        df = pd.DataFrame()
        
        result = self.scraper.validate_data(df, ['date', 'title'])
        self.assertFalse(result)

class TestCDCScraper(unittest.TestCase):
    """Test CDC scraper functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.scraper = CDCScraper()
    
    def test_extract_health_keywords(self):
        """Test keyword extraction."""
        text = "The flu outbreak has caused many hospitalizations this season."
        keywords = self.scraper.extract_health_keywords(text)
        
        self.assertIn('flu', keywords)
        self.assertIn('outbreak', keywords)
        self.assertIn('hospitalization', keywords)
    
    def test_extract_health_keywords_none(self):
        """Test keyword extraction with no matches."""
        text = "The weather is nice today."
        keywords = self.scraper.extract_health_keywords(text)
        
        self.assertEqual(keywords, 'none')
    
    def test_categorize_article_outbreak(self):
        """Test article categorization for outbreak."""
        title = "New Flu Outbreak Detected"
        content = "Health officials report a surge in cases."
        
        category = self.scraper._categorize_article(title, content)
        self.assertEqual(category, 'Outbreak')
    
    def test_categorize_article_vaccination(self):
        """Test article categorization for vaccination."""
        title = "New Vaccine Approved"
        content = "The FDA approved a new immunization today."
        
        category = self.scraper._categorize_article(title, content)
        self.assertEqual(category, 'Vaccination')

if __name__ == '__main__':
    unittest.main()
