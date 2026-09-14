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
    # No real credential of any kind, ever, in a committed file.
    assert "mongodb+srv://" not in text


def test_env_example_documents_the_source_switches():
    """SOURCES is what separates the development catalogue from the one we
    could ship, so the file a VM is configured from has to name both."""
    text = (DEPLOY / ".env.example").read_text()
    assert "SOURCES=deezer" in text
    assert "JAMENDO_CLIENT_ID=" in text
    assert text.split("JAMENDO_CLIENT_ID=")[1].splitlines()[0] == ""


def test_the_containers_are_sized_for_the_models_they_can_load():
    """The worker always loads CLAP's audio tower (~2.5 GB resident). The API
    loads nothing by default but pulls the WHOLE model on the first
    /search/text when TEXT_SEARCH=1 -- msclap has no text-tower-only load --
    so its limit has to cover that or the feature OOM-kills the API the first
    time anyone uses it. Both still fit the 12 GB box beside Caddy."""
    text = (DEPLOY / "docker-compose.yml").read_text()
    limits = re.findall(r"mem_limit: (\d+)g", text)
    assert limits == ["6", "4"]


def test_the_image_installs_git_for_the_beat_tracker():
    """beat_this is installed from a git URL; pip cannot clone without it,
    and the failure is a build-time one nobody sees until a rebuild."""
    text = (DEPLOY / "Dockerfile").read_text()
    apt = next(line for line in text.splitlines()
               if "ffmpeg" in line and not line.lstrip().startswith("#"))
    assert "git" in apt.split()


def test_the_image_carries_no_tensorflow():
    text = (DEPLOY / "Dockerfile").read_text()
    code = "\n".join(line for line in text.splitlines()
                     if not line.lstrip().startswith("#"))
    assert "tensorflow" not in code.lower()
    assert "TF_CPP" not in code
    assert "scripts/fetch_models.py" in code


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
