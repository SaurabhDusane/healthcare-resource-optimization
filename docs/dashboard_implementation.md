# Dashboard & API Guide

Phase 3 adds a **product layer**: a FastAPI service, an early-warning system, and
BI-ready data exports. Everything runs on the synthetic pipeline output, with no
credentials required.

> Metrics shown are illustrative (synthetic data). See the reproducibility note
> in the README.

---

## 1. Generate the data

```bash
pip install -r requirements-ci.txt
python main.py            # writes models/, data/processed/, reports/, and
                          # data/processed/dashboard/*.csv
```

The pipeline now also exports dashboard tables automatically. To (re)build just
the BI tables from an existing run:

```bash
python -m src.data_processing.dashboard_prep
```

### BI export tables (`data/processed/dashboard/`)

| file                      | grain                         | use in a dashboard                 |
|---------------------------|-------------------------------|------------------------------------|
| `hourly_heatmap.csv`      | day-of-week × arrival hour    | demand heatmap (peak staffing)     |
| `daily_visits.csv`        | day (+ 7-day rolling mean)    | trend line / forecast overlay      |
| `acuity_by_insurance.csv` | insurance status              | high-acuity rate comparison        |
| `web_signals.csv`         | day                           | news/sentiment early-signal series |

These are tidy CSVs — connect Tableau or Power BI directly to the folder; no
in-tool reshaping needed.

---

## 2. Run the API

```bash
uvicorn src.api.app:app --reload      # http://127.0.0.1:8000
```

Open `http://127.0.0.1:8000/` for a minimal live dashboard (KPIs + links).

### Endpoints

| method & path          | description                                        |
|------------------------|----------------------------------------------------|
| `GET /`                | HTML dashboard (KPIs, links)                       |
| `GET /health`          | service status + which artifacts are loaded        |
| `GET /metrics`         | the full `pipeline_metrics.json`                   |
| `POST /predict/acuity` | probability a visit is high-acuity                 |
| `GET /forecast?days=N` | N-day demand forecast (ARIMA), clamped to 1–90     |
| `GET /alerts?days=N`   | early-warning surge alerts over the forecast       |

If the model/data haven't been generated yet, artifact-backed endpoints return
`503` with a clear message instead of crashing.

### Example: acuity prediction

```bash
curl -s -X POST http://127.0.0.1:8000/predict/acuity \
  -H 'Content-Type: application/json' \
  -d '{"features": {"AGE": 72, "has_insurance": 0, "arrival_hour": 19, "is_monday": 1}}'
```

Any features not supplied default to 0; the server aligns the payload to the
model's persisted feature order (`models/acuity_features.json`).

### Example: forecast + alerts

```bash
curl -s "http://127.0.0.1:8000/forecast?days=14"
curl -s "http://127.0.0.1:8000/alerts?days=14"
```

---

## 3. Early-warning system

`src/alerts/early_warning.py` compares each forecast day to a historical baseline
(`mean + k·std`) and assigns a severity: `none → watch → warning → critical`
(thresholds at 1σ / 2σ / 3σ by default).

Delivery is pluggable (`src/alerts/notifier.py`):

- `LoggingChannel` — default, always on, no configuration.
- `SlackWebhookChannel` — set `SLACK_WEBHOOK_URL`.
- `EmailChannel` — set `SMTP_HOST`, `ALERT_EMAIL_FROM`, `ALERT_EMAIL_TO`
  (and `SMTP_USERNAME` / `SMTP_PASSWORD` if your relay needs auth).

Optional channels fail soft: if they aren't configured, they log a warning and
return `False` rather than raising.

---

## 4. Run with Docker

```bash
docker compose up --build      # serves on http://localhost:8000
```

The image generates artifacts at build time and the entrypoint regenerates them
if the mounted `models/` volume is empty. Tune the synthetic size with the
`PIPELINE_VISITS` environment variable (see `docker-compose.yml`).
