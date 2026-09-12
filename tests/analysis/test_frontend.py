from __future__ import annotations

import numpy as np
import pytest

from music_recommendations.analysis import frontend
from tests.analysis.conftest import SR, needs_essentia


def test_decode_returns_mono_16k_float32(tone_wav):
    audio = frontend.decode(tone_wav)
    assert audio.dtype == np.float32
    assert audio.ndim == 1
    assert abs(len(audio) - 3 * SR) < SR // 100  # within 10 ms of 3 s
    assert 0.4 < np.abs(audio).max() <= 1.0


def test_decode_missing_file_raises(tmp_path):
    with pytest.raises(frontend.DecodeError):
        frontend.decode(tmp_path / "nope.mp3")


@needs_essentia
def test_decode_matches_essentia_monoloader(tone_wav):
    from essentia.standard import MonoLoader

    ours = frontend.decode(tone_wav)
    ref = MonoLoader(filename=str(tone_wav), sampleRate=SR)()
    n = min(len(ours), len(ref))
    # Different resamplers; agree on the waveform to well under 1%.
    assert np.abs(ours[:n] - ref[:n]).max() < 0.01
