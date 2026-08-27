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

**Interpreting the numbers.** The synthetic generator builds acuity from
observable drivers (age is dominant, plus arrival hour, calendar, and insurance),
so a leakage-free model recovers **real, learnable structure**: ROC-AUC ≈
0.66–0.69 and accuracy ≈ 0.76 on a held-out split. Feature importance surfaces
`AGE` and `has_insurance` first, consistent with how acuity is generated. These
are illustrative synthetic numbers, but they reflect genuine signal, not noise.

**Hyperparameter tuning.** `ClassificationModel.tune()` runs a
`RandomizedSearchCV` (stratified CV, scored by ROC-AUC) over a small per-model
search space and refits to the best estimator. Enable it with `python main.py
--tune`; the best params and CV score are recorded under
`metrics["acuity_model"]["tuning"]` and in the experiment log. Tuning gives a
measurable lift (e.g. test ROC-AUC 0.67 → 0.69).

---

## 2. Demand Forecaster

**Task.** Forecast daily ER-visit counts (univariate time series).

**Module.** `src/modeling/time_series_forecaster.py` (`TimeSeriesForecaster`).

**Models.**

| key              | method                                            | dependency        |
|------------------|---------------------------------------------------|-------------------|
| `seasonal_naive` | repeat last weekly cycle (baseline)               | core              |
| `arima`          | `statsmodels` ARIMA, default order `(5, 1, 1)`    | statsmodels       |
| `mlp`            | neural MLP on lagged windows, recursive forecast  | scikit-learn      |
| `prophet`        | additive model w/ weekly + yearly seasonality     | prophet (optional)|

Prophet is imported lazily; if it isn't installed the backtest simply skips it.

**Backtest.** `backtest()` performs a **chronological** holdout (default: last 30
days), fits each model on the earlier data only, forecasts the holdout horizon,
and ranks models by MAE. There is no shuffling and no future leakage. Metrics:
MAE, RMSE, MAPE, and a `100 − MAPE` "accuracy" figure (MAPE is divide-by-zero
guarded).

**Rolling-origin cross-validation.** `backtest_cv()` runs several successive
holdouts (default 4 folds × 14-day horizon), each training only on the data
before it, and averages MAE per model — a far more robust ranking than one
holdout. Results land under `forecast["cross_validation"]`.

**Exogenous signals (the early-warning thesis).** `backtest_exog()` compares
univariate ARIMA against **SARIMAX** with the scraped daily signals
(`news_mentions`, `reddit_sentiment`, `twitter_sentiment`) as exogenous
regressors, on the same holdout. It reports both MAEs and an `exog_helps` flag —
an honest, measured test of whether the web signals actually improve the
forecast (`forecast["exogenous"]`).

**Typical result (synthetic, full year).** ARIMA beats the seasonal-naive and
MLP models (e.g. MAE ≈ 3.2 vs 4.3 vs 5.3), and SARIMAX-with-exog edges out
univariate ARIMA by a small margin — the scraped signals help modestly, as the
generator links them only weakly. Everything is written to
`reports/pipeline_metrics.json` under `forecast`.

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

## 4. Running on real data

The pipeline is data-source agnostic. To run on a real (de-identified)
NHAMCS-format ER-visits CSV instead of synthetic data:

```bash
python main.py --data-csv /path/to/nhamcs_visits.csv
```

`src/data/data_loader.py` validates the required NHAMCS columns
(`VDATE, AGE, ARRTIME, IMMEDR, PAYTYPER`), coerces numeric types, and hands the
frame to the exact same cleaning → feature-engineering → modeling path — no code
changes. Scraped-signal samples remain synthetic unless you supply real feeds.

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
