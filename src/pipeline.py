"""
End-to-End Analytics Pipeline
=============================

Runs the full workflow the project describes, using the synthetic data
generator so it works with zero credentials and no external network access:

    generate -> clean -> feature-engineer -> train -> evaluate -> persist

Two models are produced:
  * an acuity classifier (predicts high-acuity ER visits), and
  * a daily-visit time-series summary (for forecasting downstream).

Artifacts written:
  * data/raw/*.csv          raw synthetic datasets
  * data/processed/*.csv    cleaned + feature-engineered tables
  * models/acuity_model.joblib
  * reports/pipeline_metrics.json
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.data_loader import load_er_visits_csv
from src.data.generate_synthetic_data import SyntheticDataGenerator
from src.data_processing.cleaning import DataCleaner
from src.data_processing.dashboard_prep import DashboardPrep
from src.data_processing.feature_engineering import FeatureEngineer
from src.modeling.classification_model import ClassificationModel
from src.modeling.model_evaluator import ModelEvaluator
from src.modeling.registry import ModelRegistry
from src.modeling.time_series_forecaster import TimeSeriesForecaster
from src.monitoring.drift import DriftMonitor
from src.utils.experiment_tracker import ExperimentTracker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("pipeline")

# Numeric features fed to the acuity classifier. Kept explicit so the model
# never accidentally trains on a leakage column (e.g. IMMEDR itself).
CLASSIFIER_FEATURES = [
    "AGE",
    "SEX",
    "arrival_hour",
    "has_insurance",
    "day_of_week",
    "is_weekend",
    "is_monday",
    "is_flu_season",
    "month",
    "weekend_evening",
    "day_of_week_sin",
    "day_of_week_cos",
    "month_sin",
    "month_cos",
]


@dataclass
class PipelineConfig:
    """Configuration for a pipeline run."""

    n_visits: int = 20000
    seed: int = 42
    test_size: float = 0.2
    model_type: str = "xgboost"
    tune: bool = False
    forecast_test_size: int = 30
    # Data source: "synthetic" (default) or "csv" to load real NHAMCS-format data.
    data_source: str = "synthetic"
    er_visits_csv: Optional[str] = None
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    models_dir: str = "models"
    reports_dir: str = "reports"
    visualizations_dir: str = "visualizations"
    dirs_to_create: List[str] = field(
        default_factory=lambda: [
            "data/raw",
            "data/processed",
            "models",
            "reports",
            "visualizations",
        ]
    )


class Pipeline:
    """Orchestrates the full analytics workflow."""

    def __init__(self, config: PipelineConfig | None = None):
        self.config = config or PipelineConfig()
        self.logger = logger
        self.metrics: Dict[str, object] = {}
        self.tracker = ExperimentTracker(
            experiments_dir=os.path.join(self.config.reports_dir, "experiments")
        )

    # ------------------------------------------------------------------ #
    def run(self) -> Dict[str, object]:
        """Execute the pipeline and return a metrics dictionary."""
        self.logger.info("=== Pipeline start ===")
        self._ensure_dirs()

        raw = self._generate()
        cleaned = self._clean(raw)
        features = self._engineer(cleaned)
        self._train_acuity_model(features)
        daily = self._summarize_timeseries(features)
        self._forecast(daily, features)
        self._export_dashboard(features)
        self._monitor_drift(features)
        self._write_metrics()

        self.logger.info("=== Pipeline complete ===")
        return self.metrics

    # ------------------------------------------------------------------ #
    def _ensure_dirs(self) -> None:
        for d in self.config.dirs_to_create:
            os.makedirs(d, exist_ok=True)

    def _generate(self) -> Dict[str, pd.DataFrame]:
        gen = SyntheticDataGenerator(seed=self.config.seed)
        # Scraped-source samples are always synthetic here; swap in real feeds
        # the same way the ER visits are swapped below.
        datasets = gen.generate_all(n_visits=self.config.n_visits)

        if self.config.data_source == "csv":
            if not self.config.er_visits_csv:
                raise ValueError("data_source='csv' requires er_visits_csv to be set")
            self.logger.info(
                "Step 1/6: loading real ER visits from %s", self.config.er_visits_csv
            )
            datasets["er_visits"] = load_er_visits_csv(self.config.er_visits_csv)
        else:
            self.logger.info("Step 1/6: generating synthetic data")

        for name, df in datasets.items():
            df.to_csv(os.path.join(self.config.raw_dir, f"{name}.csv"), index=False)
        self.metrics["data_source"] = self.config.data_source
        self.metrics["record_counts"] = {k: int(len(v)) for k, v in datasets.items()}
        return datasets

    def _clean(self, raw: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        self.logger.info("Step 2/6: cleaning")
        cleaner = DataCleaner()
        nhamcs = cleaner.clean_nhamcs_data(raw["er_visits"])
        cdc = cleaner.clean_scraped_data(raw["cdc_news"], "cdc")
        reddit = cleaner.clean_scraped_data(raw["reddit_posts"], "reddit")
        twitter = cleaner.clean_scraped_data(raw["twitter_posts"], "twitter")
        merged = cleaner.merge_datasets(
            nhamcs, cdc_df=cdc, reddit_df=reddit, twitter_df=twitter
        )
        merged.to_csv(
            os.path.join(self.config.processed_dir, "merged.csv"), index=False
        )
        return merged

    def _engineer(self, cleaned: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("Step 3/6: feature engineering")
        engineer = FeatureEngineer()
        features = engineer.create_all_features(cleaned, date_col="visit_date")
        features.to_csv(
            os.path.join(self.config.processed_dir, "features.csv"), index=False
        )
        return features

    def _train_acuity_model(self, features: pd.DataFrame) -> None:
        self.logger.info("Step 4/6: training acuity classifier")
        available = [c for c in CLASSIFIER_FEATURES if c in features.columns]
        missing = sorted(set(CLASSIFIER_FEATURES) - set(available))
        if missing:
            self.logger.warning("Missing expected features (skipped): %s", missing)

        model_df = features.dropna(subset=available + ["high_acuity"])
        X = model_df[available].astype(float)
        y = model_df["high_acuity"].astype(int)

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.config.test_size,
            random_state=self.config.seed,
            stratify=y,
        )

        model = ClassificationModel(model_type=self.config.model_type)
        tuning: Dict[str, object] = {"tuned": False}
        if self.config.tune:
            tuning = model.tune(X_train, y_train)
        else:
            model.train(X_train, y_train)

        # Detailed, headless-safe evaluation + report artifacts.
        evaluator = ModelEvaluator()
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        clf_metrics = evaluator.evaluate_classification(y_test, y_pred, y_proba)
        evaluator.save_report(
            clf_metrics, self.config.reports_dir, "acuity_classification"
        )
        evaluator.plot_confusion_matrix(
            y_test,
            y_pred,
            labels=["low", "high"],
            save_path=os.path.join(
                self.config.visualizations_dir, "confusion_matrix.png"
            ),
        )
        evaluator.plot_roc_curve(
            y_test,
            y_proba,
            save_path=os.path.join(self.config.visualizations_dir, "roc_curve.png"),
        )

        model_path = os.path.join(self.config.models_dir, "acuity_model.joblib")
        model.save_model(model_path)
        # Persist the feature order so the API can build inference rows correctly.
        features_path = os.path.join(self.config.models_dir, "acuity_features.json")
        with open(features_path, "w", encoding="utf-8") as fh:
            json.dump(available, fh)

        importance = model.get_feature_importance()
        top_features = (
            importance.head(5).to_dict(orient="records")
            if importance is not None
            else []
        )
        if importance is not None:
            evaluator.plot_feature_importance(
                importance,
                save_path=os.path.join(
                    self.config.visualizations_dir, "feature_importance.png"
                ),
            )

        headline = {k: v for k, v in clf_metrics.items() if k != "confusion_matrix"}

        # Register a versioned copy in the model registry (promote to production).
        registry = ModelRegistry(
            base_dir=os.path.join(self.config.models_dir, "registry")
        )
        registry_version = registry.register(
            model.model,
            name="acuity",
            metrics=headline,
            params={"model_type": self.config.model_type, "features": available},
        )

        self.metrics["acuity_model"] = {
            "model_type": self.config.model_type,
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
            "features_used": available,
            "metrics": headline,
            "confusion_matrix": clf_metrics["confusion_matrix"],
            "top_features": top_features,
            "tuning": tuning,
            "model_path": model_path,
            "registry_version": registry_version,
        }
        self.logger.info("Acuity model metrics: %s", headline)

        self.tracker.log(
            run_name=f"acuity_{self.config.model_type}",
            params={
                "model_type": self.config.model_type,
                "n_visits": self.config.n_visits,
                "test_size": self.config.test_size,
                "seed": self.config.seed,
                "n_features": len(available),
                "tuned": tuning.get("tuned", False),
                "best_params": tuning.get("best_params", {}),
            },
            metrics=headline,
            tags={"task": "classification"},
        )

    def _summarize_timeseries(self, features: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("Step 5/6: building daily-visits time series")
        daily = (
            features.groupby(features["visit_date"].dt.date)
            .size()
            .rename("visits")
            .reset_index()
            .rename(columns={"index": "date"})
        )
        daily.columns = ["date", "visits"]
        ts_path = os.path.join(self.config.processed_dir, "daily_visits.csv")
        daily.to_csv(ts_path, index=False)

        self.metrics["timeseries"] = {
            "n_days": int(len(daily)),
            "mean_daily_visits": round(float(daily["visits"].mean()), 2),
            "max_daily_visits": int(daily["visits"].max()),
            "path": ts_path,
        }
        return daily

    def _build_daily_exog(self, features: pd.DataFrame) -> pd.DataFrame:
        """Aggregate scraped signals to a daily exogenous-regressor frame."""
        signal_cols = [
            c
            for c in ["news_mentions", "reddit_sentiment", "twitter_sentiment"]
            if c in features.columns
        ]
        if not signal_cols or "visit_date" not in features.columns:
            return pd.DataFrame()
        exog = (
            features.assign(visit_date=pd.to_datetime(features["visit_date"]))
            .groupby(pd.Grouper(key="visit_date", freq="D"))[signal_cols]
            .mean()
        )
        return exog

    def _forecast(self, daily: pd.DataFrame, features: pd.DataFrame) -> None:
        self.logger.info("Step 6/6: forecasting daily demand (backtest)")
        test_size = self.config.forecast_test_size
        if len(daily) <= test_size + self.config.forecast_test_size:
            self.logger.warning(
                "Not enough days (%d) for a %d-day forecast backtest; skipping",
                len(daily),
                test_size,
            )
            self.metrics["forecast"] = {
                "skipped": True,
                "reason": "insufficient_history",
            }
            return

        forecaster = TimeSeriesForecaster()
        result = forecaster.backtest(daily, test_size=test_size)
        # Rolling-origin cross-validation for a more robust model ranking.
        try:
            result["cross_validation"] = forecaster.backtest_cv(
                daily, horizon=min(14, test_size), n_folds=4
            )
        except ValueError as exc:
            self.logger.warning("Skipping forecast CV: %s", exc)
            result["cross_validation"] = {"skipped": True, "reason": str(exc)}

        # Test whether scraped signals (news / sentiment) improve the forecast.
        exog = self._build_daily_exog(features)
        if not exog.empty:
            try:
                result["exogenous"] = TimeSeriesForecaster().backtest_exog(
                    daily, exog, test_size=test_size
                )
                self.logger.info(
                    "Exogenous signals help forecast: %s",
                    result["exogenous"]["exog_helps"],
                )
            except Exception as exc:  # pragma: no cover - numerical edge cases
                self.logger.warning("Exogenous backtest failed: %s", exc)
                result["exogenous"] = {"skipped": True, "reason": str(exc)}

        self.metrics["forecast"] = result
        self.logger.info(
            "Best forecaster: %s (%s)", result["best_model"], result["best_metrics"]
        )

        self.tracker.log(
            run_name=f"forecast_{result['best_model']}",
            params={
                "test_size": test_size,
                "arima_order": result["arima_order"],
                "n_train": result["n_train"],
            },
            metrics=result["best_metrics"],
            tags={"task": "forecasting", "best_model": result["best_model"]},
        )

    def _export_dashboard(self, features: pd.DataFrame) -> None:
        self.logger.info("Exporting BI-ready dashboard tables")
        prep = DashboardPrep(
            output_dir=os.path.join(self.config.processed_dir, "dashboard")
        )
        paths = prep.export(features)
        self.metrics["dashboard_tables"] = paths

    def _monitor_drift(self, features: pd.DataFrame) -> None:
        """Establish a drift reference on first run; compare against it after."""
        self.logger.info("Monitoring data drift")
        monitor = DriftMonitor()
        cols = [c for c in CLASSIFIER_FEATURES if c in features.columns]
        current = features[cols]
        ref_path = os.path.join(self.config.reports_dir, "drift_reference.json")

        if not os.path.exists(ref_path):
            monitor.save_profile(current, ref_path)
            self.metrics["drift"] = {
                "status": "baseline_established",
                "reference": ref_path,
            }
            return

        report = monitor.compare_to_profile(current, ref_path)
        report_path = os.path.join(self.config.reports_dir, "drift_report.json")
        with open(report_path, "w", encoding="utf-8") as fh:
            json.dump(report.to_dict(), fh, indent=2, default=str)
        self.metrics["drift"] = {
            "status": "compared",
            "n_features": report.n_features,
            "n_drifted": report.n_drifted,
            "drift_share": report.drift_share,
            "drifted_features": report.drifted_features,
            "report": report_path,
        }
        self.logger.info(
            "Drift: %d/%d features drifted", report.n_drifted, report.n_features
        )

    def _write_metrics(self) -> None:
        self.metrics["generated_at"] = datetime.utcnow().isoformat() + "Z"
        self.metrics["seed"] = self.config.seed
        path = os.path.join(self.config.reports_dir, "pipeline_metrics.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.metrics, fh, indent=2, default=str)
        self.logger.info("Wrote metrics to %s", path)


def run_pipeline(config: PipelineConfig | None = None) -> Dict[str, object]:
    """Convenience wrapper used by main.py and tests."""
    return Pipeline(config).run()


if __name__ == "__main__":
    run_pipeline()
