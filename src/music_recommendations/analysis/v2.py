"""analyze_tracks v2: decode once, then CLAP + feel + rhythm off that decode.

The shape of the entry point the worker was already built on is kept exactly
-- a list of paths in, one slot per path holding either a feature dict or the
exception that path raised -- because the worker's group loop, its retry
accounting and its tests are all built on it. What changed at the cutover is
inside each slot:

    before: embedding (1280, a non-commercial model)  feel (11, trained heads)
    now:    embedding (1024, CLAP)   feel (8, zero-shot)   rhythm (7 numbers)

Decoding is the one expensive thing every stage needs, so it happens once
per path and the waveform is handed to all three. CLAP runs on the whole
group in a single forward pass; feel is eight dot products on the result;
rhythm is per track and CPU-bound but small.

`_features_version` rides along in the returned dict so a caller that
persists features cannot forget to stamp it -- mixing version-3 and
version-4 vectors in one corpus ranks two unrelated embedding spaces against
each other and silently produces nonsense.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .schema import FEATURES_VERSION

__all__ = ["DecodeError", "analyze_tracks", "analyze_track", "decode"]

SAMPLE_RATE = 44100

# Below this there is not enough audio for a meaningful embedding (CLAP's
# window is 7 s) or a tempo (three beats at 60 BPM is 2 s).
MIN_SECONDS = 1.0


class DecodeError(Exception):
    """The file could not be turned into audio: not audio, truncated, empty."""


def decode(path: Path | str) -> np.ndarray:
    """Mono float32 at 44.1 kHz, via librosa (audioread/soundfile + ffmpeg).

    Raises DecodeError rather than whichever of librosa's five backends
    failed first, so the worker sees one exception type in the slot and can
    tell "this preview is broken" from "this host is broken".
    """
    import librosa

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    try:
        y, _sr = librosa.load(str(p), sr=SAMPLE_RATE, mono=True)
    except Exception as exc:  # noqa: BLE001 - normalized into DecodeError
        raise DecodeError(f"{p}: {type(exc).__name__}: {exc}") from exc
    y = np.asarray(y, dtype=np.float32)
    if y.size < int(MIN_SECONDS * SAMPLE_RATE):
        raise DecodeError(f"{p}: {y.size / SAMPLE_RATE:.2f}s is too short")
    return y


def analyze_tracks(paths: list[Path | str]) -> list[dict | Exception]:
    """One slot per input path: the feature dict, or that path's exception.

    A file that fails to decode costs itself and nothing else -- previews
    404, get truncated mid-download, or turn out to be HTML error pages, and
    one of those must not cost a group of eight its analysis.
    """
    from . import clap, feel, rhythm

    results: list[dict | Exception | None] = [None] * len(paths)
    ready: list[int] = []
    waves: list[np.ndarray] = []
    for i, raw in enumerate(paths):
        try:
            waves.append(decode(raw))
            ready.append(i)
        except Exception as exc:  # noqa: BLE001 - reported in this path's slot
            results[i] = exc

    if ready:
        embeddings = clap.embed_audio(waves, sr=SAMPLE_RATE)
        feels = feel.feel_scores(embeddings)
        for slot, i in enumerate(ready):
            features = {
                "embedding": embeddings[slot].astype(np.float32),
                "feel": feels[slot].astype(np.float32),
                "_features_version": FEATURES_VERSION,
            }
            try:
                features["rhythm"] = rhythm.rhythm_features(waves[slot],
                                                            SAMPLE_RATE)
            except Exception as exc:  # noqa: BLE001
                # The embedding is the part recommendations cannot work
                # without; losing the tempo of one track costs that track a
                # ranking term (store.put_track simply stores no rhythm, and
                # the tempo distance treats a missing BPM as no penalty), not
                # its place in the corpus. Said once per process, because a
                # broken Beat This! checkpoint would otherwise print per track.
                _warn_rhythm_failed(exc)
            results[i] = features
    return results  # type: ignore[return-value]


_rhythm_warned = False


def _warn_rhythm_failed(exc: Exception) -> None:
    """Say once per process that rhythm extraction is failing."""
    global _rhythm_warned
    if _rhythm_warned:
        return
    _rhythm_warned = True
    print(f"analysis: rhythm unavailable ({type(exc).__name__}: {exc}); "
          f"tracks will be stored without tempo, key or loudness", flush=True)


def analyze_track(path: Path | str) -> dict:
    """The one-path case of analyze_tracks; raises instead of returning."""
    result = analyze_tracks([path])[0]
    if isinstance(result, Exception):
        raise result
    return result
