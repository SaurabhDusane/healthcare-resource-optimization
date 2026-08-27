"""Tests for simulation scenario presets."""

import pandas as pd
import pytest

from src.data.generate_synthetic_data import SyntheticDataGenerator
from src.data.scenarios import Scenario, get_scenario, list_scenarios


def _gen(scenario_name=None, n=8000, seed=7):
    scenario = get_scenario(scenario_name) if scenario_name else None
    return SyntheticDataGenerator(seed=seed).generate_er_visits(n, scenario=scenario)


def _winter_share(df):
    d = pd.to_datetime(df["VDATE"])
    return d.dt.month.isin([12, 1, 2]).mean()


def _high_acuity_rate(df):
    return df["IMMEDR"].isin([1, 2]).mean()


def _uninsured_rate(df):
    return df["PAYTYPER"].isin([5, 6]).mean()


def test_registry_contents():
    names = list_scenarios()
    assert "baseline" in names
    assert {"flu_surge", "outbreak_spike", "mild_winter", "high_uninsured"} <= set(
        names
    )


def test_get_scenario_case_insensitive_and_unknown():
    assert get_scenario("FLU_SURGE").name == "flu_surge"
    with pytest.raises(KeyError):
        get_scenario("does_not_exist")


def test_baseline_matches_default_generation():
    """The baseline preset must not change deterministic output."""
    a = SyntheticDataGenerator(seed=42).generate_er_visits(3000)
    b = SyntheticDataGenerator(seed=42).generate_er_visits(
        3000, scenario=get_scenario("baseline")
    )
    pd.testing.assert_frame_equal(a, b)


def test_flu_surge_increases_winter_and_acuity():
    base, flu = _gen(), _gen("flu_surge")
    assert _winter_share(flu) > _winter_share(base)
    assert _high_acuity_rate(flu) > _high_acuity_rate(base)


def test_mild_winter_reduces_seasonality():
    assert _winter_share(_gen("mild_winter")) < _winter_share(_gen())


def test_high_uninsured_raises_uninsured_share():
    base, unins = _gen(), _gen("high_uninsured")
    assert _uninsured_rate(unins) > _uninsured_rate(base) + 0.1


def test_outbreak_spike_concentrates_visits_in_window():
    def window_share(df):
        doy = (pd.to_datetime(df["VDATE"]) - pd.Timestamp("2023-01-01")).dt.days
        return ((doy >= 180) & (doy <= 210)).mean()

    assert window_share(_gen("outbreak_spike")) > window_share(_gen()) * 1.3


def test_scenario_is_frozen_dataclass():
    sc = get_scenario("baseline")
    assert isinstance(sc, Scenario)
    with pytest.raises(Exception):
        sc.flu_intensity = 1.0  # frozen


def test_pipeline_records_scenario(tmp_path):
    from src.pipeline import Pipeline, PipelineConfig

    cfg = PipelineConfig(
        n_visits=1200,
        model_type="random_forest",
        scenario="flu_surge",
        raw_dir=str(tmp_path / "raw"),
        processed_dir=str(tmp_path / "processed"),
        models_dir=str(tmp_path / "models"),
        reports_dir=str(tmp_path / "reports"),
        visualizations_dir=str(tmp_path / "viz"),
        dirs_to_create=[
            str(tmp_path / p) for p in ("raw", "processed", "models", "reports", "viz")
        ],
    )
    metrics = Pipeline(cfg).run()
    assert metrics["scenario"] == "flu_surge"
