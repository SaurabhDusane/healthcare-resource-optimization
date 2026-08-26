"""
Dashboard Data Preparation
=========================

Transforms the processed feature table into small, BI-tool-ready aggregate CSVs
(Tableau / Power BI). Each output is a tidy table keyed by the dimensions a
dashboard tile needs, so no in-tool reshaping is required.

Outputs (written to ``data/processed/dashboard/`` by default):
  * ``hourly_heatmap.csv``      visits by day-of-week x arrival-hour
  * ``daily_visits.csv``        daily visit counts (+ 7-day rolling mean)
  * ``acuity_by_insurance.csv`` high-acuity rate by insurance status
  * ``web_signals.csv``         daily news mentions + social sentiment
"""

from __future__ import annotations

import logging
import os
from typing import Dict

import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DOW_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


class DashboardPrep:
    """Build BI-ready aggregate tables from the processed feature set."""

    def __init__(self, output_dir: str = "data/processed/dashboard"):
        self.output_dir = output_dir
        self.logger = logger

    # ------------------------------------------------------------------ #
    def hourly_heatmap(self, df: pd.DataFrame) -> pd.DataFrame:
        """Visit counts by day-of-week and arrival hour (long format)."""
        needed = {"day_of_week", "arrival_hour"}
        if not needed.issubset(df.columns):
            return pd.DataFrame(
                columns=["day_of_week", "day_name", "arrival_hour", "visits"]
            )
        grouped = (
            df.groupby(["day_of_week", "arrival_hour"])
            .size()
            .rename("visits")
            .reset_index()
        )
        grouped["day_name"] = grouped["day_of_week"].map(
            lambda d: DOW_NAMES[int(d)] if 0 <= int(d) < 7 else str(d)
        )
        return grouped[["day_of_week", "day_name", "arrival_hour", "visits"]]

    def daily_visits(self, df: pd.DataFrame) -> pd.DataFrame:
        """Daily visit totals with a 7-day rolling mean."""
        if "visit_date" not in df.columns:
            return pd.DataFrame(columns=["date", "visits", "rolling_mean_7d"])
        daily = (
            df.assign(visit_date=pd.to_datetime(df["visit_date"]))
            .groupby(pd.Grouper(key="visit_date", freq="D"))
            .size()
            .rename("visits")
            .reset_index()
            .rename(columns={"visit_date": "date"})
        )
        daily["rolling_mean_7d"] = (
            daily["visits"].rolling(7, min_periods=1).mean().round(2)
        )
        return daily

    def acuity_by_insurance(self, df: pd.DataFrame) -> pd.DataFrame:
        """High-acuity rate and visit counts split by insurance status."""
        if not {"has_insurance", "high_acuity"}.issubset(df.columns):
            return pd.DataFrame(
                columns=["insurance_status", "visits", "high_acuity_rate"]
            )
        grouped = (
            df.groupby("has_insurance")
            .agg(
                visits=("high_acuity", "size"), high_acuity_rate=("high_acuity", "mean")
            )
            .reset_index()
        )
        grouped["insurance_status"] = grouped["has_insurance"].map(
            {1: "insured", 0: "uninsured"}
        )
        grouped["high_acuity_rate"] = grouped["high_acuity_rate"].round(4)
        return grouped[["insurance_status", "visits", "high_acuity_rate"]]

    def web_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Daily web-scraped signals aligned to the visit date."""
        if "visit_date" not in df.columns:
            return pd.DataFrame(columns=["date"])
        cols = [
            c
            for c in ["news_mentions", "reddit_sentiment", "twitter_sentiment"]
            if c in df.columns
        ]
        if not cols:
            return pd.DataFrame(columns=["date"])
        agg = {c: "mean" for c in cols}
        signals = (
            df.assign(visit_date=pd.to_datetime(df["visit_date"]))
            .groupby(pd.Grouper(key="visit_date", freq="D"))
            .agg(agg)
            .round(4)
            .reset_index()
            .rename(columns={"visit_date": "date"})
        )
        return signals

    # ------------------------------------------------------------------ #
    def build_all(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Compute every dashboard table."""
        return {
            "hourly_heatmap": self.hourly_heatmap(df),
            "daily_visits": self.daily_visits(df),
            "acuity_by_insurance": self.acuity_by_insurance(df),
            "web_signals": self.web_signals(df),
        }

    def export(self, df: pd.DataFrame) -> Dict[str, str]:
        """Compute and write all dashboard tables; return their paths."""
        os.makedirs(self.output_dir, exist_ok=True)
        tables = self.build_all(df)
        paths: Dict[str, str] = {}
        for name, table in tables.items():
            path = os.path.join(self.output_dir, f"{name}.csv")
            table.to_csv(path, index=False)
            paths[name] = path
            self.logger.info("Wrote %s (%d rows)", path, len(table))
        return paths


def main() -> None:
    """CLI: read data/processed/features.csv and export dashboard tables."""
    import argparse

    parser = argparse.ArgumentParser(description="Export BI-ready dashboard tables.")
    parser.add_argument(
        "--features",
        default="data/processed/features.csv",
        help="Path to the processed features CSV (run main.py first).",
    )
    parser.add_argument("--output-dir", default="data/processed/dashboard")
    args = parser.parse_args()

    if not os.path.exists(args.features):
        raise SystemExit(
            f"{args.features} not found. Run `python main.py` first to generate it."
        )

    df = pd.read_csv(args.features)
    paths = DashboardPrep(output_dir=args.output_dir).export(df)
    print("Dashboard tables written:")
    for name, path in paths.items():
        print(f"  {name:20s} -> {path}")


if __name__ == "__main__":
    main()
