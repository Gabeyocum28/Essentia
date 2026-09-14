"""The v2 stack: CLAP embedding, zero-shot feel, rhythm/loudness/key.

Run these with the interpreter that has the analysis extra installed:

    PYTHONPATH=$PWD/src <v2-python> -m pytest tests/analysis/test_v2.py

`python3 -m pytest` skips the whole file (needs_v2), because during the
transition the TensorFlow/Essentia interpreter and the torch one are not the
same interpreter.

Everything here runs on audio this file synthesizes, except one smoke test
on a real preview that skips when the file is not there. Synthetic audio is
not a weaker test for the DSP: a 120 BPM click track has exactly one right
answer for tempo, and silence has exactly one right answer for loudness,
which no real recording does.
"""
from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import pytest

from tests.analysis.conftest import needs_clap_weights, needs_v2

pytestmark = needs_v2

SR = 44100
PREVIEW = Path(
    "/private/tmp/claude-501/-Users-gabrielyocum-Projects-Essentia"
    "/a19167c7-7d14-4a2a-9bcb-b87a72c2843c/scratchpad/parity/2711781.mp3"
)


# ---- synthetic audio --------------------------------------------------------

def click_track(bpm: float = 120.0, seconds: float = 12.0,
                sr: int = SR) -> np.ndarray:
    """Short noise bursts at `bpm`. Unambiguous tempo, no pitch content."""
    y = np.zeros(int(seconds * sr), dtype=np.float32)
    period = int(round(60.0 / bpm * sr))
    rng = np.random.default_rng(0)
    burst = (rng.standard_normal(int(0.01 * sr)).astype(np.float32)
             * np.linspace(1.0, 0.0, int(0.01 * sr), dtype=np.float32) * 0.9)
    for start in range(0, len(y) - len(burst), period):
        y[start:start + len(burst)] += burst
    return y


def c_major_loop(seconds: float = 12.0, sr: int = SR) -> np.ndarray:
    """A C-E-G triad restruck every half second, plus its octaves.

    Chroma is pitch-class energy, so the answer must be C major regardless of
    which octave the partials land in; including them keeps the test honest.
    """
    freqs = [261.63, 329.63, 392.00, 523.25, 659.26, 784.00]
    t = np.arange(int(seconds * sr)) / sr
    y = np.zeros_like(t, dtype=np.float32)
    for f in freqs:
        y += np.sin(2 * np.pi * f * t).astype(np.float32)
    env = 0.5 * (1 + np.cos(2 * np.pi * 2.0 * t)).astype(np.float32)
    return (y / len(freqs) * env * 0.8).astype(np.float32)


def write_wav(path: Path, y: np.ndarray, sr: int = SR) -> Path:
    pcm = (np.clip(y, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    return path


# ---- rhythm -----------------------------------------------------------------

def test_click_track_tempo_is_recovered():
    from music_recommendations.analysis import rhythm

    feats = rhythm.rhythm_features(click_track(120.0), SR)
    assert set(feats) == set(rhythm.RHYTHM_KEYS)
    assert abs(feats["tempo_bpm"] - 120.0) <= 3.0
    assert 0.0 <= feats["beat_strength"] <= 1.0


def test_a_slower_click_track_is_not_reported_as_the_same_tempo():
    """Guards against a tracker that has quietly stopped tracking: a constant
    output would pass the 120 BPM test on its own."""
    from music_recommendations.analysis import rhythm

    feats = rhythm.rhythm_features(click_track(90.0), SR)
    assert abs(feats["tempo_bpm"] - 90.0) <= 3.0


def test_silence_is_floored_not_infinite_and_claims_no_key():
    from music_recommendations.analysis import rhythm

    feats = rhythm.rhythm_features(np.zeros(SR * 10, dtype=np.float32), SR)
    assert feats["loudness_lufs"] == -70.0
    assert feats["loudness_range"] == 0.0
    assert feats["key_strength"] == 0.0
    assert np.isfinite(feats["tempo_bpm"])


def test_c_major_loop_is_c_major():
    from music_recommendations.analysis import rhythm

    feats = rhythm.rhythm_features(c_major_loop(), SR)
    assert feats["key"] == 0
    assert feats["mode"] == "major"
    assert feats["key_strength"] > 0.5


def test_loudness_tracks_level():
    """A signal attenuated by 20 dB must read ~20 LU quieter -- the units are
    not arbitrary, and a wrong scale would still pass a shape assertion."""
    from music_recommendations.analysis import rhythm

    loud = c_major_loop()
    quiet = loud * 0.1
    a = rhythm.rhythm_features(loud, SR)["loudness_lufs"]
    b = rhythm.rhythm_features(quiet, SR)["loudness_lufs"]
    assert -70.0 < b < a
    assert abs((a - b) - 20.0) < 1.0


# ---- feel -------------------------------------------------------------------

def test_feel_scores_saturate_at_the_poles():
    """An "audio" vector that IS the positive prompt must score ~1 on that
    axis, and one that is the negative prompt ~0. Pure arithmetic on a fake
    text matrix -- no model, so this runs anywhere the extra is installed."""
    from music_recommendations.analysis import feel_v2

    n = len(feel_v2.FEEL_KEYS)
    rng = np.random.default_rng(0)
    text = rng.standard_normal((n, 2, 1024)).astype(np.float32)
    text /= np.linalg.norm(text, axis=2, keepdims=True)
    feel_v2._text = text
    try:
        pos = np.stack([text[i, 0] for i in range(n)])
        neg = np.stack([text[i, 1] for i in range(n)])
        scores_pos = feel_v2.feel_scores(pos)
        scores_neg = feel_v2.feel_scores(neg)
        assert scores_pos.shape == (n, n)
        for i in range(n):
            assert scores_pos[i, i] > 0.99
            assert scores_neg[i, i] < 0.01
        assert scores_pos.min() >= 0.0 and scores_pos.max() <= 1.0
    finally:
        feel_v2._text = None


def test_feel_keys_and_prompt_bank_agree():
    from music_recommendations.analysis import feel_v2

    assert len(feel_v2.FEEL_KEYS) == 8
    assert feel_v2.FEEL_KEYS == [n for n, _p, _q in feel_v2.PROMPT_BANK]


# ---- clap -------------------------------------------------------------------

def test_windows_tile_a_short_clip_and_span_a_long_one():
    from music_recommendations.analysis import clap

    short = np.ones(SR * 2, dtype=np.float32)
    wins = clap._windows(short, SR)
    assert len(wins) == 1 and wins[0].shape == (clap.WINDOW_SECONDS * SR,)

    long = np.arange(SR * 30, dtype=np.float32)
    wins = clap._windows(long, SR)
    assert len(wins) == clap.MAX_WINDOWS
    assert all(w.shape == (clap.WINDOW_SECONDS * SR,) for w in wins)
    # Last window must reach the end of the clip: a tracker that only ever
    # embedded the first 7 s would make every long track look like its intro.
    assert wins[-1][-1] == long[-1]


@needs_clap_weights
def test_embed_text_is_unit_length_and_discriminates():
    from music_recommendations.analysis import clap

    emb = clap.embed_text(["a drum solo", "a solo violin", "a drum solo"])
    assert emb.shape == (3, clap.DIM)
    assert np.allclose(np.linalg.norm(emb, axis=1), 1.0, atol=1e-5)
    assert float(emb[0] @ emb[2]) > float(emb[0] @ emb[1])


@needs_clap_weights
def test_embed_audio_is_unit_length_and_batch_matches_single(tmp_path):
    from music_recommendations.analysis import clap, v2

    y = v2.decode(write_wav(tmp_path / "loop.wav", c_major_loop(seconds=20)))
    batch = clap.embed_audio([y, click_track()])
    assert batch.shape == (2, clap.DIM)
    assert np.allclose(np.linalg.norm(batch, axis=1), 1.0, atol=1e-5)
    # Batching is an optimization, not a different computation.
    alone = clap.embed_audio([y])
    assert float(alone[0] @ batch[0]) > 0.999


# ---- analyze_tracks ---------------------------------------------------------

@needs_clap_weights
def test_analyze_tracks_shapes_and_one_bad_path(tmp_path):
    """One unreadable file must cost itself and nothing else."""
    from music_recommendations.analysis import FEATURES_VERSION, analyze_tracks
    from music_recommendations.analysis import as_json, feel_v2, rhythm

    good = write_wav(tmp_path / "good.wav", c_major_loop(seconds=15))
    out = analyze_tracks([good, tmp_path / "nope.mp3", good])

    assert isinstance(out[1], Exception)
    for feats in (out[0], out[2]):
        assert set(feats) == {"embedding", "feel", "rhythm", "_features_version"}
        assert feats["embedding"].shape == (1024,)
        assert feats["embedding"].dtype == np.float32
        assert abs(float(np.linalg.norm(feats["embedding"])) - 1.0) < 1e-4
        assert feats["feel"].shape == (len(feel_v2.FEEL_KEYS),)
        assert 0.0 <= float(feats["feel"].min())
        assert float(feats["feel"].max()) <= 1.0
        assert set(feats["rhythm"]) == set(rhythm.RHYTHM_KEYS)
        assert feats["_features_version"] == FEATURES_VERSION == 4

    js = as_json(out[0])
    assert len(js["embedding"]) == 1024 and isinstance(js["embedding"][0], float)
    assert len(js["feel"]) == 8
    assert js["rhythm"]["mode"] in ("major", "minor")
    assert isinstance(js["rhythm"]["key"], int)


def test_decode_rejects_a_file_that_is_not_audio(tmp_path):
    from music_recommendations.analysis import v2

    junk = tmp_path / "preview.mp3"
    junk.write_bytes(b"<html>404</html>")
    with pytest.raises(v2.DecodeError):
        v2.decode(junk)


def test_decode_rejects_a_clip_too_short_to_analyze(tmp_path):
    from music_recommendations.analysis import v2

    tiny = write_wav(tmp_path / "tiny.wav", np.zeros(1000, dtype=np.float32))
    with pytest.raises(v2.DecodeError):
        v2.decode(tiny)


def test_decode_reports_a_missing_file_as_missing(tmp_path):
    from music_recommendations.analysis import v2

    with pytest.raises(FileNotFoundError):
        v2.decode(tmp_path / "gone.mp3")


@needs_clap_weights
@pytest.mark.skipif(not PREVIEW.exists(), reason="no real preview on this host")
def test_real_preview_gets_plausible_numbers():
    """A jazz preview ("All Blues"): acoustic, not very dance-y, a real tempo
    and a loudness in the band a mastered recording actually occupies."""
    from music_recommendations.analysis import analyze_track, feel_v2

    feats = analyze_track(PREVIEW)
    feel = dict(zip(feel_v2.FEEL_KEYS, feats["feel"].tolist()))
    rhythm = feats["rhythm"]

    assert feel["acoustic"] > 0.5
    assert feel["energy"] < 0.5
    assert 40.0 < rhythm["tempo_bpm"] < 220.0
    assert -30.0 < rhythm["loudness_lufs"] < -3.0
    assert rhythm["key_strength"] > 0.0
