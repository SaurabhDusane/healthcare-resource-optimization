"""Tests for the real-data CSV loader."""

import pandas as pd
import pytest

from src.data.data_loader import DataValidationError, load_er_visits_csv
from src.data.generate_synthetic_data import SyntheticDataGenerator


def _write_csv(df, tmp_path, name="visits.csv"):
    path = tmp_path / name
    df.to_csv(path, index=False)
    return str(path)


def test_loads_valid_nhamcs_csv(tmp_path):
    df = SyntheticDataGenerator(seed=3).generate_er_visits(500)
    path = _write_csv(df, tmp_path)
    loaded = load_er_visits_csv(path)
    assert len(loaded) == 500
    for col in ["VDATE", "AGE", "ARRTIME", "IMMEDR", "PAYTYPER"]:
        assert col in loaded.columns


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_er_visits_csv("/no/such/file.csv")


def test_missing_columns_raises(tmp_path):
    bad = pd.DataFrame(
        {"AGE": [30, 40], "IMMEDR": [1, 3]}
    )  # missing VDATE, ARRTIME, PAYTYPER
    path = _write_csv(bad, tmp_path)
    with pytest.raises(DataValidationError):
        load_er_visits_csv(path)


def test_empty_csv_raises(tmp_path):
    path = _write_csv(
        pd.DataFrame(columns=["VDATE", "AGE", "ARRTIME", "IMMEDR", "PAYTYPER"]),
        tmp_path,
    )
    with pytest.raises(DataValidationError):
        load_er_visits_csv(path)


def test_non_numeric_coerced_to_nan(tmp_path):
    df = pd.DataFrame(
        {
            "VDATE": ["2024-01-01", "2024-01-02"],
            "AGE": ["forty", 55],
            "ARRTIME": [1830, 900],
            "IMMEDR": [2, 4],
            "PAYTYPER": [1, 5],
        }
    )
    path = _write_csv(df, tmp_path)
    loaded = load_er_visits_csv(path)
    assert pd.isna(loaded["AGE"].iloc[0])
    assert loaded["AGE"].iloc[1] == 55
