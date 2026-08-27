"""
Offline tests for the CDC scraper.

Network access is fully mocked — no HTTP request is ever made — so these run in
CI and document the parsing contract the scraper relies on.
"""

from datetime import datetime
from unittest.mock import MagicMock

import pandas as pd

from src.scrapers.cdc_scraper import CDCScraper

NEWS_HTML = b"""
<html><body>
  <div class="card-body">
    <h3>Flu Outbreak Detected in Region</h3>
    <a href="/media/releases/2024/flu-outbreak.html">read more</a>
    <time datetime="2024-01-15T00:00:00Z">Jan 15, 2024</time>
  </div>
  <div class="card-body">
    <h3>New Vaccine Approved</h3>
    <a href="https://www.cdc.gov/media/releases/2024/vaccine.html">read more</a>
    <time datetime="2024-02-01T00:00:00Z">Feb 1, 2024</time>
  </div>
  <div class="card-body">
    <h3>Missing Link Card</h3>
  </div>
</body></html>
"""

ARTICLE_HTML = b"""
<html><body>
  <div class="syndicate">Health officials report a respiratory illness surge.</div>
</body></html>
"""


def _mock_response(content: bytes):
    resp = MagicMock()
    resp.content = content
    resp.raise_for_status = MagicMock()
    return resp


def _make_scraper():
    scraper = CDCScraper()
    scraper.rate_limit = 0  # no real delay during tests

    def fake_get(url, *args, **kwargs):
        if "releases/2024" in url:  # an individual article page
            return _mock_response(ARTICLE_HTML)
        return _mock_response(NEWS_HTML)  # the news index

    scraper.session.get = MagicMock(side_effect=fake_get)
    return scraper


def test_scrape_parses_articles_offline():
    scraper = _make_scraper()
    df = scraper.scrape()
    assert isinstance(df, pd.DataFrame)
    # Two well-formed cards; the third (no link) is skipped.
    assert len(df) == 2
    assert {"date", "title", "content", "url", "keywords", "category"}.issubset(
        df.columns
    )
    assert scraper.session.get.called


def test_scrape_builds_absolute_urls():
    df = _make_scraper().scrape()
    assert df.iloc[0]["url"].startswith("https://www.cdc.gov/")
    assert df.iloc[1]["url"].startswith("https://www.cdc.gov/")


def test_scrape_extracts_keywords_and_category():
    df = _make_scraper().scrape()
    outbreak_row = df[df["title"].str.contains("Outbreak")].iloc[0]
    assert "outbreak" in outbreak_row["keywords"]
    assert outbreak_row["category"] == "Outbreak"


def test_scrape_article_content_offline():
    scraper = _make_scraper()
    text = scraper.scrape_article_content(
        "https://www.cdc.gov/media/releases/2024/x.html"
    )
    assert "respiratory illness surge" in text.lower()


def test_scrape_article_content_handles_errors():
    scraper = CDCScraper()
    scraper.rate_limit = 0
    scraper.session.get = MagicMock(side_effect=RuntimeError("network down"))
    assert scraper.scrape_article_content("http://x") == "Error retrieving content"


def test_extract_health_keywords_none():
    assert CDCScraper().extract_health_keywords("the weather is nice") == "none"
