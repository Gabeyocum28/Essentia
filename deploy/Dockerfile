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

# Torch from the CPU-only index first: the default PyPI wheel drags in the
# CUDA runtime (~2 GB of it) that a CPU VM will never load. aarch64 wheels
# for cp311 exist on this index. Installed as its own layer so the rest of
# the extra resolves against the torch that is already there.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
      torch torchaudio
RUN pip install --no-cache-dir -e ".[analysis]"

# Bake every model file (CLAP, Beat This!, and the v1 EffNet graph) so the
# container needs no network at start.
RUN python3 scripts/fetch_models.py

ENV TF_CPP_MIN_LOG_LEVEL=2
EXPOSE 8000
CMD ["uvicorn", "music_recommendations.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
