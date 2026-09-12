#!/bin/bash
# Essencia embed worker, supervised by launchd.
#
# The Oracle VM is ARM and cannot run essentia, so the embed/attribution work
# happens on a Mac, run directly against MongoDB Atlas -- reachable straight
# over the internet, no tunnel needed. If the worker dies, this script exits
# and launchd restarts it.
#
# Installed copy (what launchd actually runs) lives at
#   ~/Library/Application Support/essencia/embed_worker_run.sh
# loaded by ~/Library/LaunchAgents/com.essencia.embed-worker.plist, which
# also carries MONGODB_URI in its EnvironmentVariables.
# launchd agents cannot read ~/Desktop, so the installed copy runs against its
# own clone rather than a Desktop checkout. After editing this file, reinstall:
#   cp scripts/embed_worker_run.sh "$HOME/Library/Application Support/essencia/"
#   launchctl kickstart -k gui/$(id -u)/com.essencia.embed-worker
set -u

REPO="${ESSENCIA_REPO:-$HOME/Library/Application Support/essencia/repo}"
PYTHON="${ESSENCIA_PYTHON:-$HOME/.pyenv/versions/3.11.9/bin/python3}"

cd "$REPO"
# Stay current with main; tolerate being offline.
git pull --ff-only -q 2>/dev/null || true
PYTHONPATH=src exec "$PYTHON" scripts/embed_worker.py
