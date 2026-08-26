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
from typing import Dict, List

import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.generate_synthetic_data import SyntheticDataGenerator
from src.data_processing.cleaning import DataCleaner
from src.data_processing.feature_engineering import FeatureEngineer
from src.modeling.classification_model import ClassificationModel

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
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    models_dir: str = "models"
    reports_dir: str = "reports"
    dirs_to_create: List[str] = field(
        default_factory=lambda: ["data/raw", "data/processed", "models", "reports"]
    )


class Pipeline:
    """Orchestrates the full analytics workflow."""

    def __init__(self, config: PipelineConfig | None = None):
        self.config = config or PipelineConfig()
        self.logger = logger
        self.metrics: Dict[str, object] = {}

    # ------------------------------------------------------------------ #
    def run(self) -> Dict[str, object]:
        """Execute the pipeline and return a metrics dictionary."""
        self.logger.info("=== Pipeline start ===")
        self._ensure_dirs()

        raw = self._generate()
        cleaned = self._clean(raw)
        features = self._engineer(cleaned)
        self._train_acuity_model(features)
        self._summarize_timeseries(features)
        self._write_metrics()

        self.logger.info("=== Pipeline complete ===")
        return self.metrics

    # ------------------------------------------------------------------ #
    def _ensure_dirs(self) -> None:
        for d in self.config.dirs_to_create:
            os.makedirs(d, exist_ok=True)

    def _generate(self) -> Dict[str, pd.DataFrame]:
        self.logger.info("Step 1/5: generating synthetic data")
        gen = SyntheticDataGenerator(seed=self.config.seed)
        datasets = gen.generate_all(n_visits=self.config.n_visits)
        for name, df in datasets.items():
            df.to_csv(os.path.join(self.config.raw_dir, f"{name}.csv"), index=False)
        self.metrics["record_counts"] = {k: int(len(v)) for k, v in datasets.items()}
        return datasets

    def _clean(self, raw: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        self.logger.info("Step 2/5: cleaning")
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
        self.logger.info("Step 3/5: feature engineering")
        engineer = FeatureEngineer()
        features = engineer.create_all_features(cleaned, date_col="visit_date")
        features.to_csv(
            os.path.join(self.config.processed_dir, "features.csv"), index=False
        )
        return features

    def _train_acuity_model(self, features: pd.DataFrame) -> None:
        self.logger.info("Step 4/5: training acuity classifier")
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
        model.train(X_train, y_train)
        metrics = model.evaluate(X_test, y_test)

        model_path = os.path.join(self.config.models_dir, "acuity_model.joblib")
        model.save_model(model_path)

        importance = model.get_feature_importance()
        top_features = (
            importance.head(5).to_dict(orient="records")
            if importance is not None
            else []
        )

        self.metrics["acuity_model"] = {
            "model_type": self.config.model_type,
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
            "features_used": available,
            "metrics": {k: round(float(v), 4) for k, v in metrics.items()},
            "top_features": top_features,
            "model_path": model_path,
        }
        self.logger.info(
            "Acuity model metrics: %s", self.metrics["acuity_model"]["metrics"]
        )

    def _summarize_timeseries(self, features: pd.DataFrame) -> None:
        self.logger.info("Step 5/5: building daily-visits time series")
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
