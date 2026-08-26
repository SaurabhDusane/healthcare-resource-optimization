#!/usr/bin/env bash
set -euo pipefail

# Ensure model + data artifacts exist (e.g. when models/ is a fresh volume).
if [ ! -f "models/acuity_model.joblib" ] || [ ! -f "data/processed/daily_visits.csv" ]; then
  echo "Artifacts missing - running the pipeline to generate them..."
  python main.py --visits "${PIPELINE_VISITS:-8000}"
fi

exec uvicorn src.api.app:app --host 0.0.0.0 --port "${PORT:-8000}"
