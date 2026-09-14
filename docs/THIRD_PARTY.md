# Third-party components

Everything this service ships or loads at runtime, with the licence it is
under. The point of the register is one question: **can this be sold?** A
component that cannot is a component that has to go, and the only way to
know that before a customer asks is to write them all down.

`tests/test_licences.py` keeps this file honest. It reads every dependency
out of `pyproject.toml` (including the optional extras), demands a row here
for each one, and fails on any licence containing `NC` or `NonCommercial`.
Adding a dependency without a row is a test failure, not a code review
catch.

## Python packages

| Package | Licence | Used for | Link |
|---|---|---|---|
| fastapi | MIT | the HTTP API | https://github.com/fastapi/fastapi |
| uvicorn | BSD-3-Clause | ASGI server | https://github.com/encode/uvicorn |
| pymongo | Apache-2.0 | the store | https://github.com/mongodb/mongo-python-driver |
| numpy | BSD-3-Clause | every vector in the system | https://github.com/numpy/numpy |
| umap-learn | BSD-3-Clause | the 2D galaxy layout behind /viz/map and /viz/walk | https://github.com/lmcinnes/umap |
| torch | BSD-3-Clause | runs CLAP and Beat This! on CPU | https://github.com/pytorch/pytorch |
| msclap | MIT | Microsoft CLAP: the 1024-d audio embedding and the text tower behind the feel axes and text search | https://github.com/microsoft/CLAP |
| librosa | ISC | decoding, chroma/key, the fallback beat tracker | https://github.com/librosa/librosa |
| pyloudnorm | MIT | ITU-R BS.1770 integrated loudness and loudness range | https://github.com/csteinmetz1/pyloudnorm |
| soxr | LGPL-2.1-or-later | resampling inside Beat This! (dynamically linked, unmodified) | https://github.com/dofuuz/python-soxr |
| beat_this | MIT | transformer beat tracker → tempo and beat strength | https://github.com/CPJKU/beat_this |
| pytest | MIT | tests (dev only) | https://github.com/pytest-dev/pytest |
| httpx | BSD-3-Clause | test client (dev only) | https://github.com/encode/httpx |
| mongomock | BSD-3-Clause | store tests (dev only) | https://github.com/mongomock/mongomock |

Pulled in transitively by the above and worth naming, because they ship in
the image too:

| Package | Licence | Comes with | Link |
|---|---|---|---|
| torchaudio | BSD-2-Clause | msclap (imported at module load) | https://github.com/pytorch/audio |
| transformers | Apache-2.0 | msclap (GPT-2 tokenizer for the text tower) | https://github.com/huggingface/transformers |
| huggingface-hub | Apache-2.0 | transformers | https://github.com/huggingface/huggingface_hub |
| torchlibrosa | MIT | msclap (mel front end inside the audio tower) | https://github.com/qiuqiangkong/torchlibrosa |
| scikit-learn | BSD-3-Clause | msclap, umap-learn | https://github.com/scikit-learn/scikit-learn |
| pandas | BSD-3-Clause | msclap | https://github.com/pandas-dev/pandas |
| scipy | BSD-3-Clause | librosa | https://github.com/scipy/scipy |
| numba | BSD-2-Clause | librosa, umap-learn | https://github.com/numba/numba |
| llvmlite | BSD-2-Clause | numba | https://github.com/numba/llvmlite |
| pynndescent | BSD-2-Clause | umap-learn (the approximate k-NN graph) | https://github.com/lmcinnes/pynndescent |
| soundfile | BSD-3-Clause | librosa (libsndfile, LGPL-2.1, dynamically linked) | https://github.com/bastibe/python-soundfile |
| einops | MIT | beat_this | https://github.com/arogozhnikov/einops |
| PyYAML | MIT | msclap (model config) | https://github.com/yaml/pyyaml |

## Model weights

| Model | Licence | Used for | Link |
|---|---|---|---|
| CLAP 2023 (`CLAP_weights_2023.pth`) | MIT | the audio embedding, the eight feel axes, text search | https://huggingface.co/microsoft/msclap |
| Beat This! `final0` checkpoint | code MIT; weights CC BY 4.0 (upstream statement: https://github.com/CPJKU/beat_this#license — attribute Foscarin, Schlüter & Widmer, ISMIR 2024) | tempo and beat strength | https://github.com/CPJKU/beat_this |
| GPT-2 tokenizer | MIT | CLAP's text tower | https://huggingface.co/gpt2 |

Nothing under a non-commercial licence is shipped any more. The two
`models/msd-musicnn-1.*` files (MSD-MusiCNN, **CC BY-NC-SA 4.0**) were
tracked in git until this commit even though only the frozen `legacy/` MVP
ever loaded them; they are now out of the repo. Running anything in
`legacy/` therefore means downloading those weights yourself from
https://essentia.upf.edu/models.html into `models/` — and doing so under a
licence this project does not sell against.

## System

| Component | Licence | Used for |
|---|---|---|
| ffmpeg | **GPL-2.0+/GPL-3.0+** as built in Debian (`ffmpeg` from bookworm, configured `--enable-gpl`), not the LGPL build | decoding previews behind librosa/audioread |
| Deezer preview URLs | Deezer API terms — 30 s previews, attribution required; **development only**, never in a shipped product | `SOURCES=deezer` |
| Jamendo API + audio | Jamendo API terms; each track under its own Creative Commons licence, attribution and backlink required (served as `attribution_url`) | `SOURCES=jamendo` — the shippable catalogue |

### ffmpeg and the GPL

The image installs Debian's `ffmpeg` binary, which is GPL-licensed, not the
LGPL build. Nothing links it: librosa/audioread **exec it as a separate
process** and talk to it over a pipe, so its licence does not propagate to
this code. What that obliges us to do if we distribute the image is offer
ffmpeg's source, which Debian publishes; we neither modify nor patch it. If
we ever wanted to link ffmpeg's libraries directly, this row would have to be
revisited first.

### LGPL decision (recorded)

Three components are LGPL: **soxr** (python-soxr, LGPL-2.1+), **libsndfile**
behind `soundfile` (LGPL-2.1), and — in an LGPL build — ffmpeg. For all
three, and as a standing rule:

- they are **dynamically linked** (loaded as shared objects via their Python
  bindings), never statically linked into anything we build;
- they are shipped **unmodified**, from upstream wheels or Debian packages;
- we **do not patch** them, and any future need to would end this
  arrangement and require the patched source to be published;
- any artifact we distribute ships their **licence notices and an offer of
  source** (the upstream URLs in this file are that offer).

Under LGPL-2.1 §6 that is the clause-compliant arrangement: a user can
replace the library with their own build without touching our code. Nothing
here obliges us to publish this project's source.
