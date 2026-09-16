# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/), and this project's versioning is
informal (pre-1.0).

## [Unreleased]

### Added
- **Simulation scenario presets** (`src/data/scenarios.py`): named what-if
  presets (`flu_surge`, `outbreak_spike`, `mild_winter`, `high_uninsured`) that
  reshape the synthetic generator via `--scenario`. `baseline` reproduces the
  default output exactly; the run records its scenario in `metrics["scenario"]`.
- **Runtime configuration** (`src/config.py`): a single, env-driven `Settings`
  object for artifact paths, API key, rate limit, and server bind.
- **API hardening**: optional `X-API-Key` authentication (enabled by setting
  `API_KEY`), in-process sliding-window rate limiting (`RATE_LIMIT_PER_MINUTE`,
  returns HTTP 429), and bounded input validation on prediction payloads.
- **Real-data path**: `python main.py --data-csv <nhamcs.csv>` runs the full
  pipeline on a real NHAMCS-format CSV via `src/data/data_loader.py`.
- **Modeling depth**: neural MLP forecaster, rolling-origin cross-validation,
  SARIMAX with exogenous scraped signals, and `RandomizedSearchCV` tuning
  (`--tune`).
- **CI**: Docker-build job, coverage reporting (`pytest-cov`), and a Prefect
  optional-deps job. Python matrix moved to 3.11 / 3.12.
- **Repo hygiene**: `CONTRIBUTING.md`, this changelog, PR and issue templates,
  and coverage configuration.

### Changed
- Acuity signal in the synthetic generator strengthened so the classifier learns
  real structure (ROC-AUC ≈ 0.66–0.69, up from ~0.53).
- Dependency pins relaxed to compatible lower bounds (`>=`) for a current,
  installable stack. Minimum Python raised to 3.10.

### Fixed
- pandas 3 `select_dtypes` deprecation in the cleaner.
- `setup.py` now strips inline comments when parsing requirements.
- `merge_datasets` dtype mismatch and leaked `key_0` columns (earlier).

## [1.0.0] — Phased platform build

- **Phase 1**: reproducible synthetic pipeline, CLI, CI, tests, honest README.
- **Phase 2**: forecasting depth, evaluation reports, experiment tracking.
- **Phase 3**: FastAPI service, early-warning alerts, BI exports, Docker.
- **Phase 4**: neural forecaster, drift monitoring, multi-site data, optional
  Prefect orchestration.
