"""Tests for cleaning and feature-engineering modules."""

import pandas as pd

from src.data.generate_synthetic_data import SyntheticDataGenerator
from src.data_processing.cleaning import DataCleaner
from src.data_processing.feature_engineering import FeatureEngineer


def _raw():
    gen = SyntheticDataGenerator(seed=3)
    return gen.generate_all(n_visits=1000)


def test_clean_nhamcs_adds_derived_columns():
    raw = _raw()
    cleaned = DataCleaner().clean_nhamcs_data(raw["er_visits"])
    for col in [
        "visit_date",
        "age_group",
        "arrival_hour",
        "high_acuity",
        "has_insurance",
    ]:
        assert col in cleaned.columns
    assert cleaned["high_acuity"].isin([0, 1]).all()
    assert cleaned["has_insurance"].isin([0, 1]).all()


def test_clean_scraped_data_parses_dates():
    raw = _raw()
    cleaned = DataCleaner().clean_scraped_data(raw["cdc_news"], "cdc")
    assert pd.api.types.is_datetime64_any_dtype(cleaned["date"])
    assert cleaned["date"].notna().all()


def test_merge_datasets_no_key_leak_and_adds_signals():
    cleaner = DataCleaner()
    raw = _raw()
    nhamcs = cleaner.clean_nhamcs_data(raw["er_visits"])
    cdc = cleaner.clean_scraped_data(raw["cdc_news"], "cdc")
    reddit = cleaner.clean_scraped_data(raw["reddit_posts"], "reddit")
    twitter = cleaner.clean_scraped_data(raw["twitter_posts"], "twitter")

    merged = cleaner.merge_datasets(
        nhamcs, cdc_df=cdc, reddit_df=reddit, twitter_df=twitter
    )

    # Regression guard: the anonymous merge key must never leak into output.
    assert not any(c.startswith("key_") for c in merged.columns)
    assert "_join_date" not in merged.columns
    assert len(merged) == len(nhamcs)
    assert "news_mentions" in merged.columns
    assert "reddit_sentiment" in merged.columns
    assert "tweet_count" in merged.columns


def test_merge_datasets_without_scraped_sources():
    cleaner = DataCleaner()
    nhamcs = cleaner.clean_nhamcs_data(_raw()["er_visits"])
    merged = cleaner.merge_datasets(nhamcs)
    assert len(merged) == len(nhamcs)
    assert "_join_date" not in merged.columns


def test_feature_engineering_creates_features():
    cleaner = DataCleaner()
    nhamcs = cleaner.clean_nhamcs_data(_raw()["er_visits"])
    feats = FeatureEngineer().create_all_features(nhamcs, date_col="visit_date")
    for col in ["day_of_week_sin", "month_cos", "is_flu_season", "quarter"]:
        assert col in feats.columns
    assert feats.shape[1] > nhamcs.shape[1]
