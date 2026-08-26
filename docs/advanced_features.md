# Advanced Features (Phase 4)

Phase 4 adds the differentiating, scale-oriented capabilities: a neural
forecaster benchmarked against ARIMA, data-drift monitoring, multi-hospital
data support, and optional workflow orchestration.

> Metrics are illustrative (synthetic data) — see the README reproducibility note.

---

## 1. Neural forecaster (MLP)

`src/modeling/neural_forecaster.py` — `MLPForecaster` frames forecasting as
supervised regression on a sliding window of lagged values (scikit-learn
`MLPRegressor`), then forecasts multiple steps ahead **recursively**. It needs no
deep-learning runtime, so it trains and is benchmarked in CI.

It participates automatically in the forecast backtest as the `mlp` model,
ranked by MAE alongside `seasonal_naive` and `arima`:

```python
from src.modeling.time_series_forecaster import TimeSeriesForecaster
result = TimeSeriesForecaster().backtest(daily_df, test_size=30)
# result["metrics_by_model"] -> {"seasonal_naive": ..., "arima": ..., "mlp": ...}
```

On the synthetic daily series ARIMA typically still wins — a short, univariate
series doesn't give an MLP much to exploit — which the backtest reports honestly
rather than hiding.

### Extension point: LSTM / Temporal Fusion Transformer

For a true recurrent/attention model, add a `torch`-based forecaster with the
same `fit(series)` / `forecast(periods)` interface and register it as a new
model key in `TimeSeriesForecaster.backtest`. `torch` is intentionally **not**
in `requirements-ci.txt`, keeping CI light; install it separately
(`pip install torch`) to develop the LSTM path. The MLP is the always-available
default so the pipeline never depends on a heavy runtime.

---

## 2. Data-drift monitoring

`src/monitoring/drift.py` — `DriftMonitor` compares a **reference** dataset
against **current** data per numeric feature using two signals:

- **PSI** (Population Stability Index): `<0.1` stable, `0.1–0.2` moderate,
  `>=0.2` significant.
- **KS** two-sample test: `p < 0.05` flags a shift.

A feature drifts if `PSI >= threshold` **or** the KS p-value is below `alpha`.

The pipeline uses it automatically:

- **First run** establishes a compact reference profile
  (`reports/drift_reference.json`).
- **Later runs** compare current features to that profile and write
  `reports/drift_report.json`, with a summary under `metrics["drift"]`.

```python
from src.monitoring.drift import DriftMonitor
report = DriftMonitor().compare(reference_df, current_df)
report.drifted_features   # -> ["news_mentions", ...]
```

This is the signal to trigger retraining before silent performance decay.

---

## 3. Multi-hospital (multi-site) data

The synthetic generator accepts a `sites` list, adding a `SITE` column with
uneven per-site volumes and a per-tier high-acuity gradient (tertiary centers
see more high-acuity visits):

```python
from src.data.generate_synthetic_data import SyntheticDataGenerator
df = SyntheticDataGenerator().generate_er_visits(
    n_records=20000, sites=["Site_A", "Site_B", "Site_C"]
)
```

When `SITE` is present, `DashboardPrep` adds a `daily_visits_by_site` table for
per-hospital demand views. This is the data-model foundation for per-site models
and geographic generalization.

---

## 4. Orchestration (optional Prefect)

`src/orchestration/flow.py` provides a dependency-free `PipelineFlow` that runs
the pipeline as retried, logged steps with exponential backoff:

```python
from src.orchestration.flow import run_flow
summary = run_flow()      # {"ok": True, "steps": [...], "metrics": {...}}
```

If `prefect` is installed, `build_prefect_flow()` adapts the same steps into a
real Prefect flow (tasks, retries, scheduling) without changing the core logic:

```python
from src.orchestration.flow import build_prefect_flow
flow = build_prefect_flow()   # requires: pip install prefect
flow()                        # or deploy on a schedule
```

Prefect is kept out of the default and CI dependency sets so the core stays
lightweight.

---

## Summary of Phase 4 modules

| capability            | module                                   | heavy dep? |
|-----------------------|------------------------------------------|------------|
| Neural forecaster     | `src/modeling/neural_forecaster.py`      | no (sklearn) |
| Drift monitoring      | `src/monitoring/drift.py`                | no (scipy)   |
| Multi-site data       | `src/data/generate_synthetic_data.py`    | no           |
| Orchestration         | `src/orchestration/flow.py`              | optional (prefect) |
| LSTM / TFT forecaster | _extension point_ (see §1)               | optional (torch)   |
