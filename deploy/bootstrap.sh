#!/usr/bin/env bash
# One-time (and re-runnable) setup on the VM, as ubuntu:
#   curl -fsSL https://raw.githubusercontent.com/Gabeyocum28/Essentia/main/deploy/bootstrap.sh | bash
# or, from a checkout: bash deploy/bootstrap.sh
set -euo pipefail

STACK=/home/ubuntu/stacks/essentia
CADDY=/home/ubuntu/stacks/caddy
REPO=https://github.com/Gabeyocum28/Essentia.git
BRANCH="${ESSENTIA_BRANCH:-main}"

# 1. Checkout (or update) the repo as the stack directory.
if [ ! -d "$STACK/.git" ]; then
  git clone -q "$REPO" "$STACK"
fi
cd "$STACK"
git fetch -q origin && git checkout -q "$BRANCH" && git pull -q --ff-only origin "$BRANCH"

# 2. Secrets file: created empty once; never overwritten.
if [ ! -f deploy/.env ]; then
  cp deploy/.env.example deploy/.env
  chmod 600 deploy/.env
  echo ">> Created $STACK/deploy/.env — put your MONGODB_URI in it, then re-run this script."
fi

# 3. Caddy site block, appended once, with a backup like the existing ones.
if ! grep -q "^essentia.gabeyocum.com" "$CADDY/Caddyfile"; then
  cp "$CADDY/Caddyfile" "$CADDY/Caddyfile.bak.$(date +%s)"
  printf '\n' >> "$CADDY/Caddyfile"
  cat deploy/Caddyfile.essentia >> "$CADDY/Caddyfile"
  echo ">> Caddy: added essentia.gabeyocum.com"
fi

# Validate before reloading: the Caddyfile is shared with other sites, and a
# syntax error anywhere in it must not take Caddy down for all of them.
if ! docker compose -f "$CADDY/docker-compose.yml" exec -T caddy caddy validate --config /etc/caddy/Caddyfile; then
  echo ">> Caddyfile is invalid (shared config?) — not reloading; fix it and re-run"
else
  # Reload Caddy on every run (idempotent); -T avoids "input device is not a
  # TTY" when this script runs non-interactively (e.g. piped from curl).
  if docker compose -f "$CADDY/docker-compose.yml" exec -T caddy caddy reload --config /etc/caddy/Caddyfile; then
    echo ">> Caddy: reloaded"
  else
    echo ">> Caddy reload failed; stack still deploying"
  fi
fi

# 4. Build and (re)start the stack.
docker compose -f deploy/docker-compose.yml up -d --build
docker compose -f deploy/docker-compose.yml ps
echo ">> https://essentia.gabeyocum.com/api/axes should answer within a minute (certificate issuance)."
