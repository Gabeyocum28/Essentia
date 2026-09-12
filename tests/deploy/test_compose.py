"""Static checks on the deploy files: the things a typo would silently break."""
import re
from pathlib import Path

DEPLOY = Path(__file__).resolve().parents[2] / "deploy"


def test_compose_services_follow_the_vm_conventions():
    text = (DEPLOY / "docker-compose.yml").read_text()
    for needle in ("container_name: essentia-api", "container_name: essentia-worker",
                   "networks: [web]", "env_file: .env", "expose:", 'mem_limit', "cpus"):
        assert needle in text, needle
    assert "ports:" not in text                      # only Caddy binds host ports
    assert "python -m music_recommendations.worker" in text
    assert "external: true" in text


def test_env_example_has_placeholders_only():
    text = (DEPLOY / ".env.example").read_text()
    assert "MONGODB_URI=" in text and "mongodb+srv://" not in text.split("MONGODB_URI=")[1].splitlines()[0]
    assert "PUBLIC_BASE_URL=https://essentia.gabeyocum.com/api" in text


def test_caddy_block_routes_api_and_strips_prefix():
    text = (DEPLOY / "Caddyfile.essentia").read_text()
    assert text.lstrip().startswith("essentia.gabeyocum.com {")
    assert re.search(r"handle_path /api/\* \{\s*reverse_proxy essentia-api:8000", text)


def test_bootstrap_backs_up_caddyfile_before_editing():
    text = (DEPLOY / "bootstrap.sh").read_text()
    assert "Caddyfile.bak." in text
    assert "caddy reload" in text
    assert "exec -T caddy caddy reload" in text
    assert "set -euo pipefail" in text
