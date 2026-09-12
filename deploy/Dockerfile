# Base image for every Essentia service. Build from the repo root:
#   docker build -f deploy/Dockerfile -t essentia:latest .
FROM node:22-alpine AS web
WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN npm ci
COPY web/ ./
RUN npm run build

FROM python:3.11-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY contract ./contract
COPY scripts ./scripts
COPY --from=web /web/dist ./web/dist
RUN pip install --no-cache-dir -e ".[analysis]"

# Bake the EffNet graph so the container needs no network at start.
RUN python3 scripts/fetch_models.py

ENV TF_CPP_MIN_LOG_LEVEL=2
EXPOSE 8000
CMD ["uvicorn", "music_recommendations.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
