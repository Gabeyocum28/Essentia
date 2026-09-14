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
| torch | BSD-3-Clause | runs CLAP and Beat This! on CPU | https://github.com/pytorch/pytorch |
| msclap | MIT | Microsoft CLAP: the 1024-d audio embedding and the text tower behind the feel axes and text search | https://github.com/microsoft/CLAP |
| librosa | ISC | decoding, chroma/key, the fallback beat tracker | https://github.com/librosa/librosa |
| pyloudnorm | MIT | ITU-R BS.1770 integrated loudness and loudness range | https://github.com/csteinmetz1/pyloudnorm |
| soxr | LGPL-2.1-or-later | resampling inside Beat This! (dynamically linked, unmodified) | https://github.com/dofuuz/python-soxr |
| beat_this | MIT | transformer beat tracker → tempo and beat strength | https://github.com/CPJKU/beat_this |
| tensorflow | Apache-2.0 | **v1 only**, runs the Discogs-EffNet graph; removed at the cutover | https://github.com/tensorflow/tensorflow |
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
| scikit-learn | BSD-3-Clause | msclap | https://github.com/scikit-learn/scikit-learn |
| pandas | BSD-3-Clause | msclap | https://github.com/pandas-dev/pandas |
| scipy | BSD-3-Clause | librosa | https://github.com/scipy/scipy |
| numba | BSD-2-Clause | librosa | https://github.com/numba/numba |
| soundfile | BSD-3-Clause | librosa (libsndfile, LGPL-2.1, dynamically linked) | https://github.com/bastibe/python-soundfile |
| einops | MIT | beat_this | https://github.com/arogozhnikov/einops |
| PyYAML | MIT | msclap (model config) | https://github.com/yaml/pyyaml |

## Model weights

| Model | Licence | Used for | Link |
|---|---|---|---|
| CLAP 2023 (`CLAP_weights_2023.pth`) | MIT | the audio embedding, the eight feel axes, text search | https://huggingface.co/microsoft/msclap |
| Beat This! `final0` checkpoint | MIT (CC BY 4.0 for the weights per the authors) | tempo and beat strength | https://github.com/CPJKU/beat_this |
| GPT-2 tokenizer | MIT | CLAP's text tower | https://huggingface.co/gpt2 |
| Discogs-EffNet + eleven classification heads | **CC BY-NC-SA 4.0 — non-commercial** | v1 embedding and feel heads. This is the reason for the clean-room stack. Not loaded by v2; both the files and the TensorFlow dependency are deleted at the cutover. | https://essentia.upf.edu/models.html |

## System

| Component | Licence | Used for |
|---|---|---|
| ffmpeg | LGPL-2.1+ (build in the image) | decoding previews behind librosa/audioread |
| Deezer preview URLs | Deezer API terms — 30 s previews, attribution required | the current catalogue source |
