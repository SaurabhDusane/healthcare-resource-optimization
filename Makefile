.PHONY: help install install-dev data pipeline test lint format check clean

help:
	@echo "Healthcare Resource Optimization - common tasks"
	@echo ""
	@echo "  make install       Install the lightweight (CI/pipeline) dependencies"
	@echo "  make install-dev   Install the full feature-set dependencies"
	@echo "  make data          Generate synthetic datasets into data/raw/"
	@echo "  make pipeline      Run the end-to-end pipeline on synthetic data"
	@echo "  make test          Run the test suite"
	@echo "  make lint          Run pylint (errors only)"
	@echo "  make format        Auto-format with black"
	@echo "  make check         format-check + lint + test (what CI runs)"
	@echo "  make clean         Remove generated data, models and caches"

install:
	pip install -r requirements-ci.txt

install-dev:
	pip install -r requirements.txt

data:
	python -m src.data.generate_synthetic_data

pipeline:
	python main.py

test:
	pytest -q

lint:
	pylint src main.py --errors-only

format:
	black src tests main.py conftest.py

check:
	black --check src tests main.py conftest.py
	pylint src main.py --errors-only
	pytest -q

clean:
	rm -rf data/raw/*.csv data/processed/*.csv models/*.joblib reports/pipeline_metrics.json
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type d -name .pytest_cache -prune -exec rm -rf {} +
