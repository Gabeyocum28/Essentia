"""Download the Discogs-EffNet graph into models/.

One command from checkout to working analysis. Skips files already present.
Usage: python3 scripts/fetch_models.py
"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from music_recommendations.analysis import registry


def fetch(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"  have  {dest.name}")
        return
    print(f"  fetch {dest.name}")
    urllib.request.urlretrieve(url, dest)


def main() -> None:
    registry.MODELS_DIR.mkdir(exist_ok=True)
    fetch(registry.EFFNET_URL, registry.MODELS_DIR / registry.EFFNET_FILE)
    print("models ready")


if __name__ == "__main__":
    main()
