"""Download every model file the analysis needs into models/.

One command from checkout to working analysis. Skips files already present.
Usage: python3 scripts/fetch_models.py

Two generations are fetched during the transition:

  models/v2/  CLAP weights + the Beat This! checkpoint -- what analysis/v2.py
              uses, and the only thing the image will need after the cutover.
  models/     the Discogs-EffNet graph and the eleven feel heads -- v1, kept
              until the parity test is deleted.

Both msclap and beat_this can fetch their own weights on first use. We do
not let them: a container that has to reach huggingface.co the first time a
job arrives is a container that fails on a bad network at the worst moment,
and "whatever the library downloads today" is not a version we have tested.
"""
from __future__ import annotations

import shutil
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from music_recommendations.analysis import registry


def fetch(url: str, dest: Path) -> None:
    """Download `url` to `dest`, atomically and only if it is complete.

    Downloading straight to `dest` is how you end up with a truncated model
    file that exists, so every later run skips it and the failure surfaces
    much later as "corrupted checkpoint" (this happened: a 58 MB prefix of an
    81 MB Beat This! checkpoint). Write to .part, verify the length the
    server promised, then rename.
    """
    if dest.exists():
        print(f"  have  {dest.name}")
        return
    print(f"  fetch {dest.name}")
    part = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url) as response, part.open("wb") as out:
        expected = response.headers.get("Content-Length")
        shutil.copyfileobj(response, out)
    size = part.stat().st_size
    if expected is not None and size != int(expected):
        part.unlink()
        raise IOError(
            f"{dest.name}: got {size} bytes, server promised {expected}"
        )
    part.rename(dest)


def fetch_v2() -> None:
    """CLAP (audio + text embedding) and Beat This! (beat tracker)."""
    registry.V2_DIR.mkdir(parents=True, exist_ok=True)
    fetch(registry.CLAP_URL, registry.CLAP_WEIGHTS)
    fetch(registry.BEAT_THIS_URL, registry.BEAT_THIS_CKPT)

    # CLAP's text tower is GPT-2's tokenizer, which msclap pulls from the
    # hub at load time rather than from the weights file. Warm the cache so
    # that first analysis in a fresh container needs no network either.
    try:
        from transformers import AutoTokenizer

        AutoTokenizer.from_pretrained("gpt2")
        print("  have  gpt2 tokenizer")
    except Exception as exc:  # noqa: BLE001 - a nice-to-have, not a failure
        print(f"  warn  could not pre-cache the gpt2 tokenizer ({exc})")


def fetch_v1() -> None:
    """The Discogs-EffNet graph and the eleven feel heads (~0.5 MB each).

    Non-commercial licence; deleted at the cutover. The heads' JSON sits
    beside each graph because it carries the class order every positive
    index in registry.HEADS is checked against.
    """
    registry.MODELS_DIR.mkdir(exist_ok=True)
    fetch(registry.EFFNET_URL, registry.MODELS_DIR / registry.EFFNET_FILE)
    registry.HEADS_DIR.mkdir(parents=True, exist_ok=True)
    for head in registry.HEADS.values():
        graph_url, meta_url = head.urls
        fetch(graph_url, head.graph)
        fetch(meta_url, head.metadata)


def main() -> None:
    print("v2 (CLAP + Beat This!):")
    fetch_v2()
    print("v1 (Discogs-EffNet, removed at the cutover):")
    fetch_v1()
    print("models ready")


if __name__ == "__main__":
    main()
