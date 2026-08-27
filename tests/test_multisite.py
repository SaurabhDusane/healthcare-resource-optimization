"""Tests for multi-hospital (multi-site) data support."""

import pandas as pd

from src.data.generate_synthetic_data import SyntheticDataGenerator
from src.data_processing.dashboard_prep import DashboardPrep

SITES = ["Site_A", "Site_B", "Site_C"]


def _multisite_df(n=3000, seed=1):
    return SyntheticDataGenerator(seed=seed).generate_er_visits(
        n_records=n, sites=SITES
    )


def test_site_column_present_and_covers_all_sites():
    df = _multisite_df()
    assert "SITE" in df.columns
    assert set(df["SITE"].unique()) == set(SITES)


def test_no_site_column_by_default():
    df = SyntheticDataGenerator(seed=1).generate_er_visits(n_records=500)
    assert "SITE" not in df.columns


def test_site_acuity_gradient():
    df = _multisite_df(n=6000)
    df["high"] = df["IMMEDR"].isin([1, 2]).astype(int)
    rates = {s: df[df["SITE"] == s]["high"].mean() for s in SITES}
    # Higher-tier sites carry a higher high-acuity rate by construction.
    assert rates["Site_A"] < rates["Site_C"]


def test_daily_visits_by_site_table():
    df = _multisite_df()
    df["visit_date"] = pd.to_datetime(df["VDATE"])
    table = DashboardPrep().daily_visits_by_site(df)
    assert {"date", "SITE", "visits"}.issubset(table.columns)
    assert set(table["SITE"].unique()) == set(SITES)
    assert table["visits"].sum() == len(df)


def test_build_all_includes_site_table_only_when_present():
    df = _multisite_df()
    df["visit_date"] = pd.to_datetime(df["VDATE"])
    assert "daily_visits_by_site" in DashboardPrep().build_all(df)

    df_no_site = SyntheticDataGenerator(seed=2).generate_er_visits(n_records=500)
    df_no_site["visit_date"] = pd.to_datetime(df_no_site["VDATE"])
    assert "daily_visits_by_site" not in DashboardPrep().build_all(df_no_site)
