# Healthcare Resource Optimization - API image
FROM python:3.11-slim

# libgomp1 is required by xgboost's shared library.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

WORKDIR /app

# Install dependencies first for better layer caching.
COPY requirements-ci.txt ./
RUN pip install --no-cache-dir -r requirements-ci.txt

# Copy the project.
COPY . .

# Generate initial artifacts at build time so the image serves immediately.
RUN python main.py --visits 8000

EXPOSE 8000

# Regenerate artifacts if the mounted volume is empty, then serve.
ENTRYPOINT ["./docker/entrypoint.sh"]
