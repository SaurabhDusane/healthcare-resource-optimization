"""Tests for the dashboard data exporter."""

import pandas as pd

from src.data.generate_synthetic_data import SyntheticDataGenerator
from src.data_processing.cleaning import DataCleaner
from src.data_processing.dashboard_prep import DashboardPrep
from src.data_processing.feature_engineering import FeatureEngineer


def _features():
    raw = SyntheticDataGenerator(seed=5).generate_er_visits(n_records=1500)
    cleaned = DataCleaner().clean_nhamcs_data(raw)
    return FeatureEngineer().create_all_features(cleaned, date_col="visit_date")


def test_build_all_tables_present():
    tables = DashboardPrep().build_all(_features())
    assert set(tables) == {
        "hourly_heatmap",
        "daily_visits",
        "acuity_by_insurance",
        "web_signals",
    }


def test_hourly_heatmap_shape():
    heat = DashboardPrep().hourly_heatmap(_features())
    assert {"day_of_week", "day_name", "arrival_hour", "visits"}.issubset(heat.columns)
    assert heat["visits"].sum() > 0
    assert heat["day_name"].isin(["Mon", "Sun"]).any()


def test_acuity_by_insurance_rate_bounds():
    tbl = DashboardPrep().acuity_by_insurance(_features())
    assert set(tbl["insurance_status"]).issubset({"insured", "uninsured"})
    assert tbl["high_acuity_rate"].between(0, 1).all()


def test_export_writes_files(tmp_path):
    prep = DashboardPrep(output_dir=str(tmp_path))
    paths = prep.export(_features())
    for path in paths.values():
        assert pd.read_csv(path) is not None


def test_missing_columns_return_empty_frames():
    empty = pd.DataFrame({"unrelated": [1, 2, 3]})
    prep = DashboardPrep()
    assert prep.hourly_heatmap(empty).empty
    assert prep.acuity_by_insurance(empty).empty
