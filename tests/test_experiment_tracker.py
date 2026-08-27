"""Tests for the lightweight experiment tracker."""

import json

from src.utils.experiment_tracker import ExperimentTracker


def test_log_writes_record(tmp_path):
    tracker = ExperimentTracker(experiments_dir=str(tmp_path / "exp"))
    path = tracker.log(
        run_name="unit_test",
        params={"lr": 0.1, "n": 10},
        metrics={"accuracy": 0.9},
        tags={"task": "classification"},
    )
    with open(path, encoding="utf-8") as fh:
        record = json.load(fh)

    assert record["run_name"] == "unit_test"
    assert record["params"]["lr"] == 0.1
    assert record["metrics"]["accuracy"] == 0.9
    assert record["tags"]["task"] == "classification"
    assert "timestamp" in record


def test_load_history_returns_all_runs(tmp_path):
    tracker = ExperimentTracker(experiments_dir=str(tmp_path / "exp"))
    tracker.log("run_a", {}, {"metric": 1})
    tracker.log("run_b", {}, {"metric": 2})
    history = tracker.load_history()
    assert len(history) == 2
    names = {r["run_name"] for r in history}
    assert names == {"run_a", "run_b"}


def test_load_history_empty_when_missing(tmp_path):
    tracker = ExperimentTracker(experiments_dir=str(tmp_path / "does_not_exist"))
    assert tracker.load_history() == []
