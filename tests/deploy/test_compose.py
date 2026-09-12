"""Static checks on the deploy files: the things a typo would silently break."""
import re
from pathlib import Path

DEPLOY = Path(__file__).resolve().parents[2] / "deploy"


def test_compose_services_follow_the_vm_conventions():
    text = (DEPLOY / "docker-compose.yml").read_text()
    for needle in ("container_name: essentia-api", "container_name: essentia-worker",
                   "networks: [web]", "env_file: .env", "expose:", 'mem_limit'):
        assert needle in text, needle
    assert re.search(r"cpus: [0-9.]+", text)
    assert "ports:" not in text                      # only Caddy binds host ports
    assert "python -m music_recommendations.worker" in text
    assert "external: true" in text
    assert "name: essentia" in text
    # PUBLIC_BASE_URL already sets the public origin; uvicorn's own
    # forwarded-header trust is unnecessary and --forwarded-allow-ips=*
    # would trust it from anyone.
    assert "--proxy-headers" not in text
    assert "--forwarded-allow-ips" not in text


def test_env_example_has_placeholders_only():
    text = (DEPLOY / ".env.example").read_text()
    assert "MONGODB_URI=" in text and "mongodb+srv://" not in text.split("MONGODB_URI=")[1].splitlines()[0]
    assert "PUBLIC_BASE_URL=https://essentia.gabeyocum.com/api" in text


def test_caddy_block_routes_api_and_strips_prefix():
    text = (DEPLOY / "Caddyfile.essentia").read_text()
    assert text.lstrip().startswith("essentia.gabeyocum.com {")
    assert re.search(r"handle_path /api/\* \{\s*reverse_proxy essentia-api:8000", text)
    assert "redir /api /api/ 308" in text                # bare /api falls through handle_path otherwise


def test_bootstrap_backs_up_caddyfile_before_editing():
    text = (DEPLOY / "bootstrap.sh").read_text()
    assert "Caddyfile.bak." in text
    assert "caddy reload" in text
    assert "exec -T caddy caddy reload" in text
    assert "set -euo pipefail" in text
    backup_idx = text.index("Caddyfile.bak.")
    append_idx = text.index("cat deploy/Caddyfile.essentia")
    assert backup_idx < append_idx


def test_bootstrap_validates_before_reloading():
    text = (DEPLOY / "bootstrap.sh").read_text()
    assert "caddy validate --config /etc/caddy/Caddyfile" in text
    validate_idx = text.index("caddy validate")
    reload_idx = text.index("caddy reload")
    assert validate_idx < reload_idx
    assert "not reloading; fix it and re-run" in text
    assert "Caddy reload failed; stack still deploying" in text
