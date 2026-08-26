"""
Synthetic Data Generator
=========================

Generates realistic, NHAMCS-like emergency-room visit records plus matching
CDC / Reddit / Twitter "scraped" samples so the whole pipeline can be run
end-to-end without any private data or API credentials.

The generator is fully deterministic for a given seed and intentionally bakes
in the patterns the project analyses (Monday-evening surge, weekend effect,
flu-season seasonality, and an insurance/acuity relationship) so downstream
statistics and models have real signal to find.

NOTE: This is *simulated* data for demonstration and testing only. It does not
represent any real patients, and headline metrics derived from it are
illustrative, not clinical findings.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Column layout mirrors the NHAMCS fields consumed by DataCleaner.clean_nhamcs_data
NHAMCS_COLUMNS = [
    "VDATE",  # visit date
    "AGE",  # patient age in years
    "SEX",  # 1 = female, 2 = male
    "ARRTIME",  # arrival time as military integer (e.g. 1830)
    "IMMEDR",  # triage acuity 1 (immediate) .. 5 (non-urgent); 1-2 = high acuity
    "PAYTYPER",  # expected payment source; 5 = self-pay, 6 = no charge (uninsured)
    "DIAG1",  # primary diagnosis category
]

HEALTH_KEYWORDS = [
    "flu",
    "covid",
    "rsv",
    "outbreak",
    "fever",
    "cough",
    "shortness of breath",
    "chest pain",
    "vaccination",
]

DIAGNOSES = [
    "respiratory",
    "cardiac",
    "injury",
    "abdominal",
    "neurological",
    "infection",
    "psychiatric",
    "other",
]


class SyntheticDataGenerator:
    """Generate reproducible synthetic healthcare datasets."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.logger = logger

    # ------------------------------------------------------------------ #
    # Emergency-room visits (NHAMCS-like)                                 #
    # ------------------------------------------------------------------ #
    def generate_er_visits(
        self,
        n_records: int = 20000,
        start_date: str = "2023-01-01",
        end_date: str = "2023-12-31",
        sites: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Generate patient-level ER visit records.

        Args:
            n_records: Number of visits to generate.
            start_date: First possible visit date (inclusive).
            end_date: Last possible visit date (inclusive).
            sites: Optional hospital-site identifiers. When given, each record
                gets a ``SITE`` column, with uneven per-site volume and a small
                per-site shift in high-acuity rate (multi-hospital simulation).

        Returns:
            DataFrame with NHAMCS-style columns (plus ``SITE`` when ``sites`` set).
        """
        self.logger.info("Generating %d synthetic ER visits...", n_records)

        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)
        n_days = (end - start).days + 1
        days = np.arange(n_days)
        dates = start + pd.to_timedelta(days, unit="D")

        # Daily arrival intensity: flu-season seasonality + weekend dip.
        month = dates.month.to_numpy()
        dow = dates.dayofweek.to_numpy()
        flu_season = np.isin(month, [10, 11, 12, 1, 2, 3]).astype(float)
        weekend = (dow >= 5).astype(float)
        daily_weight = 1.0 + 0.35 * flu_season - 0.15 * weekend
        daily_weight = np.clip(daily_weight, 0.1, None)
        daily_prob = daily_weight / daily_weight.sum()

        # Assign each visit to a day, then a within-day arrival hour.
        day_idx = self.rng.choice(n_days, size=n_records, p=daily_prob)
        visit_dates = dates[day_idx]
        visit_dow = visit_dates.dayofweek.to_numpy()

        arrival_hour = self._sample_arrival_hours(visit_dow)
        arrtime = arrival_hour * 100 + self.rng.integers(0, 60, size=n_records)

        age = np.clip(
            self.rng.gamma(shape=4.0, scale=11.0, size=n_records), 0, 100
        ).astype(int)
        sex = self.rng.integers(1, 3, size=n_records)

        # Insurance: ~18% uninsured (self-pay / no charge).
        uninsured = self.rng.random(n_records) < 0.18
        paytyper = np.where(
            uninsured,
            self.rng.choice([5, 6], size=n_records),
            self.rng.integers(1, 5, size=n_records),
        )

        # Acuity: Monday-evening surge and flu season raise high-acuity odds;
        # uninsured patients skew toward non-urgent (higher IMMEDR).
        immedr = self._sample_acuity(visit_dates, arrival_hour, uninsured)

        diag1 = self.rng.choice(DIAGNOSES, size=n_records)

        columns = {
            "VDATE": visit_dates.strftime("%Y-%m-%d"),
            "AGE": age,
            "SEX": sex,
            "ARRTIME": arrtime,
            "IMMEDR": immedr,
            "PAYTYPER": paytyper,
            "DIAG1": diag1,
        }

        if sites:
            # Uneven site volumes; a couple of higher-acuity (tertiary) centers.
            weights = np.linspace(1.0, 2.0, len(sites))
            site_probs = weights / weights.sum()
            site = self.rng.choice(sites, size=n_records, p=site_probs)
            # Larger-index sites bump some low-acuity visits up to high-acuity.
            site_rank = {s: i / max(len(sites) - 1, 1) for i, s in enumerate(sites)}
            bump = np.array([site_rank[s] for s in site]) * 0.15
            flip = (columns["IMMEDR"] >= 3) & (self.rng.random(n_records) < bump)
            columns["IMMEDR"] = np.where(flip, 2, columns["IMMEDR"])
            columns["SITE"] = site

        df = pd.DataFrame(columns)
        df = df.sort_values("VDATE").reset_index(drop=True)
        self.logger.info("Generated %d ER visit records", len(df))
        return df

    def _sample_arrival_hours(self, dow: np.ndarray) -> np.ndarray:
        """Sample arrival hours with an evening peak that is stronger on Mondays."""
        base = np.array(
            [1, 1, 1, 1, 1, 2, 3, 4, 5, 6, 6, 6, 6, 6, 6, 6, 7, 8, 9, 9, 8, 6, 4, 2],
            dtype=float,
        )
        hours = np.empty(dow.shape[0], dtype=int)
        for i, d in enumerate(dow):
            weights = base.copy()
            if d == 0:  # Monday: amplify the 18-21h window (the README's finding)
                weights[18:22] *= 1.8
            weights /= weights.sum()
            hours[i] = self.rng.choice(24, p=weights)
        return hours

    def _sample_acuity(
        self,
        visit_dates: pd.DatetimeIndex,
        arrival_hour: np.ndarray,
        uninsured: np.ndarray,
    ) -> np.ndarray:
        """Sample triage acuity (1=emergent .. 5=non-urgent)."""
        n = len(visit_dates)
        dow = visit_dates.dayofweek.to_numpy()
        month = visit_dates.month.to_numpy()

        # Probability of being HIGH acuity (IMMEDR in {1,2}).
        p_high = np.full(n, 0.30)
        monday_evening = (dow == 0) & (arrival_hour >= 18) & (arrival_hour <= 21)
        p_high = np.where(monday_evening, p_high + 0.12, p_high)
        p_high = np.where(np.isin(month, [12, 1, 2]), p_high + 0.05, p_high)
        p_high = np.where(uninsured, p_high - 0.10, p_high)  # uninsured skew non-urgent
        p_high = np.clip(p_high, 0.05, 0.95)

        is_high = self.rng.random(n) < p_high
        high_vals = self.rng.choice([1, 2], size=n)
        low_vals = self.rng.choice([3, 4, 5], size=n, p=[0.35, 0.4, 0.25])
        return np.where(is_high, high_vals, low_vals)

    # ------------------------------------------------------------------ #
    # Scraped-source samples                                             #
    # ------------------------------------------------------------------ #
    def generate_cdc_news(
        self, n_records: int = 500, start_date: str = "2023-01-01"
    ) -> pd.DataFrame:
        """Generate CDC/WHO-style news article samples."""
        dates = self._random_dates(n_records, start_date)
        categories = self.rng.choice(
            ["Outbreak", "Vaccination", "Advisory", "Research"], size=n_records
        )
        keywords = [
            ", ".join(self.rng.choice(HEALTH_KEYWORDS, size=2, replace=False))
            for _ in range(n_records)
        ]
        df = pd.DataFrame(
            {
                "date": dates,
                "title": [f"{c}: {k}" for c, k in zip(categories, keywords)],
                "content": [
                    f"Health officials report developments related to {k}."
                    for k in keywords
                ],
                "keywords": keywords,
                "category": categories,
            }
        )
        return df.sort_values("date").reset_index(drop=True)

    def generate_reddit_posts(
        self, n_records: int = 1200, start_date: str = "2023-01-01"
    ) -> pd.DataFrame:
        """Generate Reddit health-discussion samples."""
        dates = self._random_dates(n_records, start_date)
        df = pd.DataFrame(
            {
                "date": dates,
                "post_id": [f"r_{i:06d}" for i in range(n_records)],
                "text": [
                    f"Discussion about {self.rng.choice(HEALTH_KEYWORDS)} symptoms."
                    for _ in range(n_records)
                ],
                "sentiment_polarity": np.clip(
                    self.rng.normal(-0.1, 0.4, n_records), -1, 1
                ),
                "symptoms_mentioned": self.rng.choice(HEALTH_KEYWORDS, size=n_records),
            }
        )
        return df.sort_values("date").reset_index(drop=True)

    def generate_twitter_posts(
        self, n_records: int = 1800, start_date: str = "2023-01-01"
    ) -> pd.DataFrame:
        """Generate Twitter/X health-related samples."""
        dates = self._random_dates(n_records, start_date)
        df = pd.DataFrame(
            {
                "date": dates,
                "tweet_id": [f"t_{i:06d}" for i in range(n_records)],
                "text": [
                    f"Feeling {self.rng.choice(HEALTH_KEYWORDS)} today."
                    for _ in range(n_records)
                ],
                "sentiment_polarity": np.clip(
                    self.rng.normal(-0.05, 0.45, n_records), -1, 1
                ),
                "likes": self.rng.integers(0, 500, size=n_records),
                "retweets": self.rng.integers(0, 100, size=n_records),
            }
        )
        return df.sort_values("date").reset_index(drop=True)

    def _random_dates(self, n: int, start_date: str, span_days: int = 364) -> pd.Series:
        start = pd.Timestamp(start_date)
        offsets = self.rng.integers(0, span_days + 1, size=n)
        return pd.Series(start + pd.to_timedelta(offsets, unit="D"))

    # ------------------------------------------------------------------ #
    # Convenience                                                        #
    # ------------------------------------------------------------------ #
    def generate_all(self, n_visits: int = 20000) -> Dict[str, pd.DataFrame]:
        """Generate every dataset and return them keyed by name."""
        return {
            "er_visits": self.generate_er_visits(n_records=n_visits),
            "cdc_news": self.generate_cdc_news(),
            "reddit_posts": self.generate_reddit_posts(),
            "twitter_posts": self.generate_twitter_posts(),
        }

    def save_all(
        self, output_dir: str = "data/raw", n_visits: int = 20000
    ) -> Dict[str, str]:
        """Generate all datasets and write them to CSV files."""
        os.makedirs(output_dir, exist_ok=True)
        datasets = self.generate_all(n_visits=n_visits)
        paths: Dict[str, str] = {}
        for name, df in datasets.items():
            path = os.path.join(output_dir, f"{name}.csv")
            df.to_csv(path, index=False)
            paths[name] = path
            self.logger.info("Wrote %s (%d rows)", path, len(df))
        return paths


def main() -> None:
    """CLI: generate the full synthetic dataset into data/raw/."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate synthetic healthcare datasets."
    )
    parser.add_argument(
        "--output-dir", default="data/raw", help="Directory to write CSVs into."
    )
    parser.add_argument(
        "--visits", type=int, default=20000, help="Number of ER visit records."
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility."
    )
    args = parser.parse_args()

    generator = SyntheticDataGenerator(seed=args.seed)
    paths = generator.save_all(output_dir=args.output_dir, n_visits=args.visits)
    print("Synthetic data written:")
    for name, path in paths.items():
        print(f"  {name:14s} -> {path}")


if __name__ == "__main__":
    main()
