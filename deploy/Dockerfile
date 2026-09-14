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
# build-essential  arm64 is the platform most likely to have no wheel for
#         some transitive dependency; without a compiler that turns into a
#         build failure at the bottom of a long log instead of a slow sdist
# curl    the compose healthcheck and manual poking
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      ffmpeg git libsndfile1 build-essential curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY contract ./contract
COPY scripts ./scripts
COPY --from=web /web/dist ./web/dist

# Torch from the CPU-only index first: the default PyPI wheel drags in the
# CUDA runtime (~2 GB of it) that a CPU VM will never load.
#
# --extra-index-url, NOT --index-url: the CPU index carries torch and a
# handful of siblings and NOTHING else, so making it the only index means a
# dependency it does not mirror fails to resolve (and, worse, a name it does
# mirror partially resolves to an unexpected build).
#
# Pinned exactly, and to a +cpu local version that exists only on that index,
# because "whatever torch is newest on build day" is how an image that worked
# last month stops working -- and because torchaudio must match torch or
# msclap fails at import. Both have cp311 manylinux_2_28_aarch64 wheels at
# this version, which is what the ARM VM needs.
ARG TORCH_VERSION=2.10.0
RUN pip install --no-cache-dir \
      --extra-index-url https://download.pytorch.org/whl/cpu \
      "torch==${TORCH_VERSION}+cpu" "torchaudio==${TORCH_VERSION}+cpu"
RUN pip install --no-cache-dir -e ".[analysis]"

# Bake the model files (CLAP weights, the Beat This! checkpoint, the GPT-2
# tokenizer) into /app/models/v2 so the container needs no network at start.
# Nothing else: the non-commercial model this project existed to remove, and
# the ~600 MB framework that ran it, both went at the cutover.
RUN python3 scripts/fetch_models.py

EXPOSE 8000
CMD ["uvicorn", "music_recommendations.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
