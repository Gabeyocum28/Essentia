# analysis/ — Person 4

Pure function: MP3 path in, `{"embedding": (1280,) float32}` out (spec §2.1).
Knows nothing about HTTP, Deezer, Redis, or the phone.

- Decoding is the ffmpeg CLI (`frontend.decode`), 16 kHz mono via an
  explicit `pan=mono|c0<c0+c1` downmix — NOT `-ac 1`, which applies a
  sqrt(2) gain and breaks parity with Essentia's plain (L+R)/2 average.
- `frontend.py` reproduces Essentia's TensorflowInputMusiCNN mel front-end
  in numpy; the parameters were measured against Essentia and are recorded
  in docs/superpowers/plans/2026-09-11-analysis-on-arm.md.
- `embedding.py` runs Discogs-EffNet through TensorFlow directly (the graph
  was frozen with a fixed batch of 64 patches). No Essentia import anywhere
  under src/ — Linux aarch64 has no Essentia wheels.
- Parity tests against Essentia live in tests/analysis (test_parity.py) and
  run only where Essentia is installed, in a subprocess.
- Bump FEATURES_VERSION in schema.py whenever the numbers change.
