# Contributing

Thanks for your interest in improving this project. This guide covers the local
workflow, quality gates, and conventions.

## Development setup

```bash
git clone https://github.com/saurabhdusane/healthcare-resource-optimization.git
cd healthcare-resource-optimization
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements-ci.txt                # lightweight; runs tests + pipeline
# For the full feature set (live scraping, Prophet, spaCy): pip install -r requirements.txt
pre-commit install                                # optional: format on commit
```

Python **3.10+** is required (the scientific stack the project targets has
dropped 3.9).

## Everyday commands

```bash
make pipeline     # run the end-to-end pipeline on synthetic data
make test         # run the test suite
make check        # format-check + lint + tests (mirrors CI)
make format       # auto-format with black
```

## Quality gates (must pass before a PR is merged)

CI runs these; run them locally first:

1. **Format** — `black --check src tests main.py conftest.py`
2. **Lint** — `pylint src main.py --errors-only`
3. **Tests** — `pytest -q` (add `--cov=src` for coverage)
4. **Pipeline smoke** — `python main.py --visits 2000 --model random_forest`

A Docker build and a Prefect optional-deps job also run in CI.

## Conventions

- **Formatting**: `black` (line length 88). Don't hand-format.
- **Tests**: every new module or behavior ships with tests. Keep tests offline
  and deterministic (seed RNGs; mock any network). No test may hit the network.
- **Optional dependencies**: heavy/optional packages (`prophet`, `snscrape`,
  `torch`, `prefect`) must be imported lazily and degrade gracefully so the core
  never depends on them.
- **Data honesty**: metrics from synthetic data are illustrative — label them as
  such; never present them as clinical findings.
- **No secrets** in code or tests. API keys and SMTP/Slack config come from the
  environment (see `src/config.py`).

## Commit & PR

- Write clear, imperative commit messages describing the *what* and *why*.
- Keep PRs focused; update docs and `CHANGELOG.md` alongside code.
- Fill in the PR template checklist.

## Reporting issues

Use the issue templates (bug report / feature request). Include repro steps,
expected vs actual behavior, and environment details for bugs.
