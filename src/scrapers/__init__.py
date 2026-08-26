"""
Web scrapers package.

``BaseScraper`` and ``CDCScraper`` depend only on the core stack. ``RedditScraper``
and ``TwitterScraper`` require optional third-party clients (``praw`` / ``snscrape``);
they are imported defensively so that importing this package — e.g. during testing
or when running the pipeline on synthetic data — never fails just because those
optional dependencies are absent.
"""

from .base_scraper import BaseScraper
from .cdc_scraper import CDCScraper

__all__ = ["BaseScraper", "CDCScraper"]

try:  # optional: requires `praw`
    from .reddit_scraper import RedditScraper

    __all__.append("RedditScraper")
except ImportError:  # pragma: no cover - exercised only without praw installed
    RedditScraper = None

try:  # optional: requires `snscrape`
    from .twitter_scraper import TwitterScraper

    __all__.append("TwitterScraper")
except ImportError:  # pragma: no cover - exercised only without snscrape installed
    TwitterScraper = None
