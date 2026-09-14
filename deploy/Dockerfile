# Base image for every Essentia service. Build from the repo root:
#   docker build -f deploy/Dockerfile -t essentia:latest .
FROM node:22-alpine AS web
WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN npm ci
COPY web/ ./
RUN npm run build

FROM python:3.11-slim

# ffmpeg  decodes previews behind librosa/audioread (a separate process, see
#         docs/THIRD_PARTY.md on the GPL build)
# git     the analysis extra installs beat_this from a git URL; pip needs a
#         git binary to clone it and the build fails without one
# libsndfile1  soundfile's shared library: librosa reads wav/flac through it
# curl    the compose healthcheck and manual poking
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      ffmpeg git libsndfile1 curl \
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

# Bake the model files (CLAP weights, the Beat This! checkpoint, the GPT-2
# tokenizer) into /app/models/v2 so the container needs no network at start.
# Nothing else: the non-commercial model this project existed to remove, and
# the ~600 MB framework that ran it, both went at the cutover.
RUN python3 scripts/fetch_models.py

EXPOSE 8000
CMD ["uvicorn", "music_recommendations.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
