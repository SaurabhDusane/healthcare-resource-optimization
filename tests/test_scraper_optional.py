"""
Tests that the scrapers package degrades gracefully without optional deps.

``RedditScraper`` (praw) and ``TwitterScraper`` (snscrape) depend on optional,
network-only clients. Importing the package must never fail when they are
absent, and the base scraper's offline helpers must work standalone.
"""

import pandas as pd

import src.scrapers as scrapers
from src.scrapers.base_scraper import BaseScraper


def test_package_imports_without_optional_deps():
    # CDCScraper is always available; the optional ones are either a class or None.
    assert scrapers.CDCScraper is not None
    assert hasattr(scrapers, "RedditScraper")
    assert hasattr(scrapers, "TwitterScraper")


class _DummyScraper(BaseScraper):
    """Minimal concrete scraper for exercising base-class behavior offline."""

    def scrape(self) -> pd.DataFrame:
        return pd.DataFrame({"date": [pd.Timestamp("2024-01-01")], "text": ["hello"]})


def test_base_validate_data():
    s = _DummyScraper(name="dummy", rate_limit_seconds=0)
    good = pd.DataFrame({"date": [1], "text": ["x"]})
    assert s.validate_data(good, ["date", "text"]) is True
    assert s.validate_data(good, ["date", "missing"]) is False
    assert s.validate_data(pd.DataFrame(), ["date"]) is False


def test_base_save_data(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    s = _DummyScraper(name="dummy", rate_limit_seconds=0)
    df = s.scrape()
    s.save_data(df, "out.csv")
    assert (tmp_path / "data" / "raw" / "dummy" / "out.csv").exists()
