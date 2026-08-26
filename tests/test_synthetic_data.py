"""Tests for the synthetic data generator."""

import pandas as pd

from src.data.generate_synthetic_data import NHAMCS_COLUMNS, SyntheticDataGenerator


def test_er_visits_schema_and_size():
    gen = SyntheticDataGenerator(seed=1)
    df = gen.generate_er_visits(n_records=500)
    assert len(df) == 500
    for col in NHAMCS_COLUMNS:
        assert col in df.columns


def test_er_visits_value_ranges():
    gen = SyntheticDataGenerator(seed=1)
    df = gen.generate_er_visits(n_records=1000)
    assert df["AGE"].between(0, 100).all()
    assert df["IMMEDR"].between(1, 5).all()
    assert df["SEX"].isin([1, 2]).all()
    assert (df["ARRTIME"] < 2400).all()


def test_generator_is_deterministic():
    a = SyntheticDataGenerator(seed=7).generate_er_visits(n_records=300)
    b = SyntheticDataGenerator(seed=7).generate_er_visits(n_records=300)
    pd.testing.assert_frame_equal(a, b)


def test_different_seeds_differ():
    a = SyntheticDataGenerator(seed=1).generate_er_visits(n_records=300)
    b = SyntheticDataGenerator(seed=2).generate_er_visits(n_records=300)
    assert not a.equals(b)


def test_scraped_sources_have_expected_columns():
    gen = SyntheticDataGenerator(seed=1)
    cdc = gen.generate_cdc_news(n_records=50)
    reddit = gen.generate_reddit_posts(n_records=50)
    twitter = gen.generate_twitter_posts(n_records=50)

    assert {"date", "title", "keywords"}.issubset(cdc.columns)
    assert {"date", "post_id", "sentiment_polarity"}.issubset(reddit.columns)
    assert {"date", "tweet_id", "likes", "retweets"}.issubset(twitter.columns)
    assert reddit["sentiment_polarity"].between(-1, 1).all()


def test_generate_all_keys():
    datasets = SyntheticDataGenerator(seed=1).generate_all(n_visits=200)
    assert set(datasets) == {"er_visits", "cdc_news", "reddit_posts", "twitter_posts"}
    assert all(isinstance(v, pd.DataFrame) for v in datasets.values())


def test_monday_evening_surge_signal():
    """The generator should bake in a Monday-evening high-acuity signal."""
    gen = SyntheticDataGenerator(seed=42)
    df = gen.generate_er_visits(n_records=8000)
    df["visit_date"] = pd.to_datetime(df["VDATE"])
    df["hour"] = df["ARRTIME"] // 100
    df["high_acuity"] = df["IMMEDR"].isin([1, 2]).astype(int)

    monday_eve = df[(df["visit_date"].dt.dayofweek == 0) & df["hour"].between(18, 21)]
    other = df[~((df["visit_date"].dt.dayofweek == 0) & df["hour"].between(18, 21))]
    # Monday-evening high-acuity rate should exceed the baseline.
    assert monday_eve["high_acuity"].mean() > other["high_acuity"].mean()
