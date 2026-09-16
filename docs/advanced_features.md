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

## 5. Model registry (versioning & rollback)

`src/modeling/registry.py` — `ModelRegistry` versions trained models instead of
overwriting them. Each pipeline run registers a new version under
`models/registry/acuity/v<N>/` with the serialized estimator and a `meta.json`
(metrics, params, timestamp, git SHA), and updates a `production` pointer.

```python
from src.modeling.registry import ModelRegistry
reg = ModelRegistry()
reg.list_versions("acuity")     # -> [1, 2, 3]
reg.promote("acuity", 2)        # roll back production to v2
model, meta = reg.load("acuity")  # load the production version
```

## 6. A/B testing (intervention evaluation)

`src/analysis/ab_testing.py` — `ABTest` evaluates a two-arm experiment:

- `proportion_test` — binary outcomes (e.g. non-urgent revisit rate) via a
  two-proportion z-test, with Cohen's h.
- `mean_test` — continuous outcomes (e.g. wait time) via Welch's t-test, with
  Cohen's d.

Each returns the effect, p-value, significance at `alpha`, and a plain-language
decision — the statistical backbone for testing staffing or outreach changes.

## 7. Enriched dashboard

The API root (`GET /`) renders inline-SVG charts (recent daily-visit trend and
top acuity predictors) directly from the run artifacts — no JS framework or
extra dependency. A full Streamlit/React frontend is a natural next step; the
same `/metrics`, `/forecast`, and `/alerts` endpoints back it.

## 8. Simulation scenario presets

`src/data/scenarios.py` turns synthetic data's key advantage — controllable
ground truth — into one-flag experiments. Each preset is a small set of
deterministic overrides on the generator, so a whole "what-if" run is a single
argument rather than manual parameter edits:

```bash
python main.py --scenario flu_surge        # severe flu season (more winter demand + acuity)
python main.py --scenario outbreak_spike   # a sharp ~month-long mid-year spike
python main.py --scenario mild_winter      # weak seasonality
python main.py --scenario high_uninsured   # higher uninsured share
python -m src.data.generate_synthetic_data --scenario outbreak_spike
```

| preset           | what it models                                             |
|------------------|-----------------------------------------------------------|
| `baseline`       | default behavior (byte-identical to the generator default)|
| `flu_surge`      | strong winter seasonality + higher acuity                 |
| `outbreak_spike` | a demand-multiplier window (fires the early-warning alerts)|
| `mild_winter`    | reduced flu-season intensity                              |
| `high_uninsured` | larger uninsured/self-pay share                           |

The run records its scenario under `metrics["scenario"]`, and `baseline`
reproduces the original output exactly — so existing runs are unchanged. These
are precisely the situations real, fixed history won't hand you on demand: a rare
event, a counterfactual season, or a shifted population for stress-testing the
drift monitor and alerting.

## Summary of advanced modules

| capability            | module                                   | heavy dep? |
|-----------------------|------------------------------------------|------------|
| Neural forecaster     | `src/modeling/neural_forecaster.py`      | no (sklearn) |
| Drift monitoring      | `src/monitoring/drift.py`                | no (scipy)   |
| Multi-site data       | `src/data/generate_synthetic_data.py`    | no           |
| Model registry        | `src/modeling/registry.py`               | no (joblib)  |
| A/B testing           | `src/analysis/ab_testing.py`             | no (scipy)   |
| Orchestration         | `src/orchestration/flow.py`              | optional (prefect) |
| LSTM / TFT forecaster | _extension point_ (see §1)               | optional (torch)   |
