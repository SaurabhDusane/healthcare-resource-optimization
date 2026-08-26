# Model Documentation

This document describes the models in the platform, how they are trained and
evaluated, and how results are made reproducible.

> **Data note.** Unless stated otherwise, the metrics discussed here are produced
> by the **synthetic** data generator (`src/data/generate_synthetic_data.py`), so
> they are *illustrative and reproducible*, not clinical findings. Every run
> writes its own numbers to `reports/pipeline_metrics.json` and
> `reports/experiments/`.

---

## 1. Acuity Classifier

**Task.** Binary classification — predict whether an ER visit is *high acuity*
(`IMMEDR ∈ {1, 2}`) from non-leaking visit attributes.

**Module.** `src/modeling/classification_model.py` (`ClassificationModel`).

**Algorithms.** Selectable via `--model`:

| key              | estimator                       |
|------------------|---------------------------------|
| `xgboost`        | `XGBClassifier` (default)       |
| `random_forest`  | `RandomForestClassifier`        |
| `gradient_boost` | `GradientBoostingClassifier`    |
| `logistic`       | `LogisticRegression`            |

**Features.** An explicit allow-list (`CLASSIFIER_FEATURES` in
`src/pipeline.py`) is used so the target column and other leakage-prone fields
can never enter the model: age, sex, arrival hour, insurance status, calendar
fields, the `weekend_evening` interaction, and cyclical (sin/cos) encodings of
day-of-week and month.

**Split & evaluation.** A stratified `train_test_split` (default 80/20). The
`ModelEvaluator` reports accuracy, weighted precision/recall/F1, ROC-AUC, and a
confusion matrix, and saves:

- `reports/acuity_classification.json` — metrics
- `visualizations/confusion_matrix.png`, `roc_curve.png`, `feature_importance.png`

**Interpreting the numbers.** On synthetic data the signal is deliberately
modest, so ROC-AUC sits near 0.5–0.55. This is expected and honest: the point of
the demo is a working, leakage-free training/eval loop, not an inflated score.
Feature importance typically surfaces `has_insurance` and calendar features,
consistent with how the generator builds acuity.

---

## 2. Demand Forecaster

**Task.** Forecast daily ER-visit counts (univariate time series).

**Module.** `src/modeling/time_series_forecaster.py` (`TimeSeriesForecaster`).

**Models.**

| key              | method                                            | dependency        |
|------------------|---------------------------------------------------|-------------------|
| `seasonal_naive` | repeat last weekly cycle (baseline)               | core              |
| `arima`          | `statsmodels` ARIMA, default order `(5, 1, 1)`    | statsmodels       |
| `prophet`        | additive model w/ weekly + yearly seasonality     | prophet (optional)|

Prophet is imported lazily; if it isn't installed the backtest simply skips it.

**Backtest.** `backtest()` performs a **chronological** holdout (default: last 30
days), fits each model on the earlier data only, forecasts the holdout horizon,
and ranks models by MAE. There is no shuffling and no future leakage. Metrics:
MAE, RMSE, MAPE, and a `100 − MAPE` "accuracy" figure (MAPE is divide-by-zero
guarded).

**Typical result (synthetic, full year).** ARIMA beats the seasonal-naive
baseline — e.g. MAE ≈ 3.2 vs 4.3, ≈ 82% accuracy on a 30-day holdout. The best
model and its metrics are written to `reports/pipeline_metrics.json` under
`forecast`.

---

## 3. Reproducibility & Experiment Tracking

- **Seeded.** The generator and splits are seeded (`--seed`, default 42), so a
  run reproduces exactly.
- **Experiment log.** `src/utils/experiment_tracker.py` writes one JSON record
  per model run to `reports/experiments/`, capturing parameters, metrics, a UTC
  timestamp, and the current git SHA. Collate them with
  `ExperimentTracker.load_history()`.
- **Artifacts.** The trained classifier is persisted to
  `models/acuity_model.joblib` via `joblib`.

Run everything with:

```bash
python main.py                 # or: make pipeline
python main.py --model xgboost --visits 20000
```

---

## 4. Roadmap (beyond current state)

- Hyperparameter search (grid/Bayesian) with cross-validation for the classifier.
- SARIMA / auto-order selection and multivariate forecasting that incorporates
  the scraped news/sentiment lag features already present in the merged dataset.
- Deep learning forecasters (LSTM, Temporal Fusion Transformer) benchmarked
  against ARIMA (Phase 4).
