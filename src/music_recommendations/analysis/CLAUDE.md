# analysis/

Pure function: audio file path in, feature dict out (spec §2.1). Knows
nothing about HTTP, Deezer, the store, or the phone.

## v2 (current)

`analyze_tracks(paths)` -> per path either an exception or

    {"embedding": (1024,) float32,   # CLAP, L2-normalized
     "feel":      (8,)    float32,   # zero-shot contrastive axes, [0, 1]
     "rhythm":    {...},             # the seven contract RHYTHM_KEYS
     "_features_version": 4}

- `v2.py` decodes once per path with librosa (mono, 44.1 kHz) and feeds all
  three stages from that one waveform; CLAP runs the whole group in one
  forward pass. Decode failures become `DecodeError` in that path's slot.
- `clap.py` loads the Microsoft CLAP 2023 weights from `models/v2/` (never
  the library's own hub download) and calls the wrapper's tensor entry
  point, not its file loader. Three 7 s windows per track, mean, normalize.
- `feel_v2.py` is the feel vector as eight *pairs* of prompts: the score is
  a softmax over (positive, negative) cosine. Changing a prompt changes
  every number, so it is a FEATURES_VERSION bump.
- `rhythm.py` is exact DSP, no model except Beat This!: tempo from the
  median inter-beat interval (librosa fallback when the checkpoint is
  missing), loudness from pyloudnorm (BS.1770, floored at -70 LUFS), key
  from chroma against the Krumhansl profiles with `key_strength` as the
  honest "this number means nothing" signal.
- Nothing under v2 imports TensorFlow or Essentia. Keep every heavy import
  inside a function: importing this package must stay cheap for callers
  that only want `FEATURES_VERSION`.
- Weights come from `scripts/fetch_models.py` into `models/v2/`, baked into
  the image. New dependency => a row in `docs/THIRD_PARTY.md`, or
  `tests/test_licences.py` fails.
- v2 tests (`tests/analysis/test_v2.py`) need the analysis extra and are
  skipped by the default interpreter; run them with the torch interpreter.

## v1 (being removed)

`analyze_tracks_v1` / `analyze_track_v1` plus `frontend.py`, `embedding.py`
and `feel.py`: ffmpeg decode + numpy mel + Discogs-EffNet in TensorFlow, and
eleven classifier heads. The model is CC BY-NC-SA, which is why v2 exists.
Kept only so `tests/analysis/test_parity.py` can still run until the cutover
deletes all of it. Do not call it from anything.

Bump FEATURES_VERSION in schema.py whenever the numbers change.
