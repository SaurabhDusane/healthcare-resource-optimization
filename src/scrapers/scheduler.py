"""
Automated Scraper Scheduler
Runs all scrapers on a daily schedule.
"""

import schedule
import time
from datetime import datetime
import logging
from .cdc_scraper import CDCScraper
from .reddit_scraper import RedditScraper
from .twitter_scraper import TwitterScraper

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/scheduler.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('Scheduler')

def run_all_scrapers():
    """Execute all scrapers sequentially."""
    logger.info("="*60)
    logger.info("Starting daily scraping job")
    logger.info("="*60)
    
    results = {}
    
    try:
        logger.info("Running CDC scraper...")
        cdc_scraper = CDCScraper()
        cdc_data = cdc_scraper.run()
        results['CDC'] = len(cdc_data) if cdc_data is not None else 0
    except Exception as e:
        logger.error(f"CDC scraper failed: {str(e)}")
        results['CDC'] = 0
    
    try:
        logger.info("Running Reddit scraper...")
        reddit_scraper = RedditScraper()
        reddit_data = reddit_scraper.run()
        results['Reddit'] = len(reddit_data) if reddit_data is not None else 0
    except Exception as e:
        logger.error(f"Reddit scraper failed: {str(e)}")
        results['Reddit'] = 0
    
    try:
        logger.info("Running Twitter scraper...")
        twitter_scraper = TwitterScraper()
        twitter_data = twitter_scraper.run()
        results['Twitter'] = len(twitter_data) if twitter_data is not None else 0
    except Exception as e:
        logger.error(f"Twitter scraper failed: {str(e)}")
        results['Twitter'] = 0
    
    logger.info("="*60)
    logger.info("Scraping job completed")
    logger.info(f"Results: {results}")
    logger.info(f"Total records: {sum(results.values())}")
    logger.info("="*60)

def main():
    """Main scheduler function."""
    logger.info("Scheduler started")
    logger.info("Daily scraping scheduled for 06:00 AM")
    
    schedule.every().day.at("06:00").do(run_all_scrapers)
    
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--now':
        logger.info("Running scrapers immediately (--now flag)")
        run_all_scrapers()
    
    while True:
        schedule.run_pending()
        time.sleep(60)

if __name__ == "__main__":
    main()
