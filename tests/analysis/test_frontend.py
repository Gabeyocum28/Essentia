from __future__ import annotations

import subprocess
import sys

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
def test_decode_matches_essentia_monoloader(tone_wav, tmp_path):
    # Essentia and TensorFlow cannot both be loaded in one process (they
    # deadlock/abort — verified locally), and this test suite also exercises
    # embedding.py's TensorFlow graph. Run the essentia reference decode in
    # a subprocess so the two never share a process.
    ref_path = tmp_path / "ref.npy"
    script = (
        "import numpy as np\n"
        "from essentia.standard import MonoLoader\n"
        f"ref = MonoLoader(filename={str(tone_wav)!r}, sampleRate={SR})()\n"
        f"np.save({str(ref_path)!r}, ref)\n"
    )
    subprocess.run([sys.executable, "-c", script], check=True)
    ref = np.load(ref_path)

    ours = frontend.decode(tone_wav)
    n = min(len(ours), len(ref))
    # Different resamplers; agree on the waveform to well under 1%.
    assert np.abs(ours[:n] - ref[:n]).max() < 0.01


def _tone(seconds=2.0, hz=440.0):
    t = np.arange(int(seconds * SR)) / SR
    return (0.5 * np.sin(2 * np.pi * hz * t)).astype(np.float32)


def test_mel_frames_shape_and_count():
    audio = _tone(2.0)
    mel = frontend.mel_frames(audio)
    n = len(audio)
    expected = -(-(n - frontend.FRAME_SIZE) // frontend.HOP_SIZE) + 1  # ceil + 1
    assert mel.shape == (expected, frontend.N_MELS)
    assert mel.dtype == np.float32


def test_mel_frames_tone_peaks_in_expected_band():
    mel = frontend.mel_frames(_tone(1.0, hz=440.0))
    fb = frontend.mel_filterbank()
    centre_hz = np.arange(fb.shape[1]) * SR / frontend.FRAME_SIZE
    band_centres = (fb * centre_hz).sum(axis=1) / fb.sum(axis=1)
    peak_band = mel[5:-5].mean(axis=0).argmax()
    assert abs(band_centres[peak_band] - 440.0) < 60.0


def test_mel_frames_silence_is_zero():
    mel = frontend.mel_frames(np.zeros(SR, dtype=np.float32))
    assert np.allclose(mel, 0.0)


def test_patches_shape_and_hop():
    mel = np.random.default_rng(0).random((400, 96), dtype=np.float32)
    p = frontend.patches(mel)
    assert p.shape == (1 + (400 - 128) // 62, 128, 96)
    assert np.array_equal(p[1], mel[62:62 + 128])


def test_patches_too_short_is_empty():
    assert frontend.patches(np.zeros((100, 96), np.float32)).shape == (0, 128, 96)


@needs_essentia
def test_mel_frames_match_essentia_input_musicnn(tmp_path):
    # See test_decode_matches_essentia_monoloader: essentia and TensorFlow
    # cannot share a process, so the reference values are computed out of
    # process and only numpy arrays cross back over.
    rng = np.random.default_rng(1)
    audio = (_tone(1.0) + 0.1 * rng.standard_normal(SR)).astype(np.float32)
    ours = frontend.mel_frames(audio)

    audio_path = tmp_path / "audio.npy"
    ref_path = tmp_path / "ref.npy"
    np.save(audio_path, audio)
    indices = list(range(0, 40, 7))
    script = (
        "import numpy as np\n"
        "from essentia.standard import TensorflowInputMusiCNN\n"
        f"audio = np.load({str(audio_path)!r})\n"
        f"frame_size = {frontend.FRAME_SIZE}\n"
        f"hop_size = {frontend.HOP_SIZE}\n"
        "ref_fn = TensorflowInputMusiCNN()\n"
        "refs = []\n"
        f"for i in {indices!r}:\n"
        "    frame = audio[i * hop_size: i * hop_size + frame_size]\n"
        "    refs.append(ref_fn(frame))\n"
        f"np.save({str(ref_path)!r}, np.array(refs))\n"
    )
    subprocess.run([sys.executable, "-c", script], check=True)
    refs = np.load(ref_path)

    for row, i in enumerate(indices):  # a spread of full frames
        ref = refs[row]
        assert np.abs(ours[i] - ref).max() / np.abs(ref).max() < 1e-4
