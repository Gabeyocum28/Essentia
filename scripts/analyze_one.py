"""Analyze one audio file and print the embedding's shape, norm, and timing.
Usage: python3 scripts/analyze_one.py path/to/preview.mp3"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from music_recommendations.analysis import analyze_track  # noqa: E402


def main() -> None:
    path = Path(sys.argv[1])
    t0 = time.perf_counter()
    vec = analyze_track(path)["embedding"]
    dt = time.perf_counter() - t0
    print(f"{path.name}: shape={vec.shape} norm={float((vec ** 2).sum() ** 0.5):.3f} "
          f"seconds={dt:.2f}")


if __name__ == "__main__":
    main()
