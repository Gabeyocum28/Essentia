# Base image for every Essentia service. Build from the repo root:
#   docker build -f deploy/Dockerfile -t essentia:latest .
FROM python:3.11-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY contract ./contract
COPY scripts ./scripts
RUN pip install --no-cache-dir -e ".[analysis]"

# Bake the EffNet graph so the container needs no network at start.
RUN python3 scripts/fetch_models.py

ENV TF_CPP_MIN_LOG_LEVEL=2
CMD ["python3", "-c", "import music_recommendations.analysis as a; print('analysis ok, v', a.FEATURES_VERSION)"]
