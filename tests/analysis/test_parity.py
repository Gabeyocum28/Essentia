"""Old pipeline vs new on real audio. Runs only where essentia is installed
(the Mac); the VM and CI skip it. This is the merge gate for the ARM port."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from music_recommendations.analysis import analyze_track_v1
from tests.analysis.conftest import needs_effnet, needs_essentia, run_essentia

FIXTURE = Path(__file__).resolve().parents[2] / "contract" / "fixture.json"
BASELINE = Path(__file__).resolve().parent / "parity_baseline.json"
N_TRACKS = int(os.environ.get("PARITY_TRACKS", "8"))
pytestmark = [
    needs_essentia,
    needs_effnet,
    pytest.mark.skipif(
        os.environ.get("PARITY") != "1",
        reason="set PARITY=1 to run the Essentia parity gate (network + ~1 min)",
    ),
]


def _essentia_embedding(mp3: Path, out: Path) -> np.ndarray:
    from music_recommendations.analysis import registry

    script = (
        "import numpy as np\n"
        "import essentia\n"
        "essentia.log.infoActive = False\n"
        "essentia.log.warningActive = False\n"
        "from essentia.standard import MonoLoader, TensorflowPredictEffnetDiscogs\n"
        "model = TensorflowPredictEffnetDiscogs(\n"
        f"    graphFilename={str(registry.MODELS_DIR / registry.EFFNET_FILE)!r},\n"
        f"    output={registry.EFFNET_OUTPUT!r},\n"
        ")\n"
        f"frames = model(MonoLoader(filename={str(mp3)!r}, sampleRate=16000)())\n"
        f"np.save({str(out)!r}, np.asarray(frames).mean(axis=0))\n"
    )
    return run_essentia(script, out)


def _download(track: dict, dest: Path) -> Path | None:
    from music_recommendations.corpus import deezer, download

    try:
        return download.download_preview(track, dest)
    except Exception:  # noqa: BLE001 - signed URL expired; refresh once
        try:
            url = deezer.fresh_preview_url(track["track_id"])
            if not url:
                return None
            return download.download_preview({**track, "preview_url": url}, dest)
        except Exception:  # noqa: BLE001 - network flake; skip this track
            return None


def _cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def test_fixture_tracks_cosine_above_099(tmp_path):
    tracks = json.loads(FIXTURE.read_text())["tracks"][:N_TRACKS]
    scores = []
    for track in tracks:
        mp3 = _download(track, tmp_path)
        if mp3 is None:
            continue
        new = analyze_track_v1(mp3)["embedding"]
        old = _essentia_embedding(mp3, tmp_path / f"{track['track_id']}.npy")
        track_id = track["track_id"]
        title = track["title"]
        scores.append((track_id, _cos(new, old)))
        print(f"{scores[-1][1]:.4f}  {track_id}  {title}")
    assert len(scores) >= 3, "network: could not fetch enough previews"

    if BASELINE.exists():
        baseline = json.loads(BASELINE.read_text())
        for track_id, c in scores:
            if track_id in baseline:
                print(f"  delta vs baseline {track_id}: {c - baseline[track_id]:+.4f}")

    worst = min(scores, key=lambda s: s[1])
    assert worst[1] > 0.99, f"worst parity {worst}"
    assert float(np.mean([c for _, c in scores])) > 0.985
