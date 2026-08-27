"""
Command-line entry point for the Healthcare Resource Optimization pipeline.

Examples
--------
    python main.py                      # full run with defaults
    python main.py --visits 5000        # smaller, faster run
    python main.py --model random_forest
"""

from __future__ import annotations

import argparse
import json

from src.pipeline import Pipeline, PipelineConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the end-to-end healthcare analytics pipeline on synthetic data."
    )
    parser.add_argument(
        "--visits",
        type=int,
        default=20000,
        help="Number of ER visit records to generate.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--model",
        default="xgboost",
        choices=["xgboost", "random_forest", "logistic", "gradient_boost"],
        help="Classifier used for acuity prediction.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data held out for evaluation.",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Run randomized hyperparameter search for the classifier.",
    )
    parser.add_argument(
        "--data-csv",
        default=None,
        help="Path to a real NHAMCS-format ER-visits CSV (uses it instead of synthetic).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PipelineConfig(
        n_visits=args.visits,
        seed=args.seed,
        model_type=args.model,
        test_size=args.test_size,
        tune=args.tune,
        data_source="csv" if args.data_csv else "synthetic",
        er_visits_csv=args.data_csv,
    )
    metrics = Pipeline(config).run()

    print("\n=== Pipeline summary ===")
    print(json.dumps(metrics, indent=2, default=str))


if __name__ == "__main__":
    main()
