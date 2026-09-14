# Jazz Recommender

Audio-based jazz recommendations for iPhone. Pick a track, pick what
"similar" means ("More sounds like this" / "Nothing like this"), get 5–10
tracks with 30 s previews. Design: `Essencia_design_spec.md`.

## Setup

    python3 -m pip install -e ".[dev]"             # server + tests, no models
    python3 -m pip install -e ".[analysis]"        # + torch/CLAP: only the worker needs this
    brew install ffmpeg                            # or apt install ffmpeg
    python3 scripts/fetch_models.py                # CLAP + Beat This! into models/v2
    export MONGODB_URI="mongodb+srv://..."   # Atlas connection string, never committed
    uvicorn music_recommendations.server.app:app --reload

## Analysis

Each 30-second preview is decoded once (librosa, mono 44.1 kHz) and three
things are computed off that one waveform (`src/music_recommendations/analysis/`):

- **embedding** — Microsoft CLAP 2023, 1024-d, L2-normalized. Three 7 s
  windows spread across the clip, meaned. This is what "sounds like" is a
  cosine of.
- **feel** — eight axes (energy, valence, tension, acoustic, danceable,
  vocal, brightness, density), each a softmax over a *pair* of sentences
  scored against the same embedding by CLAP's text tower. Zero-shot: no
  trained classifier, so an axis is added by writing a sentence.
- **rhythm** — tempo and beat strength from Beat This!, integrated loudness
  and loudness range from pyloudnorm (BS.1770), key/mode/key_strength from
  chroma against the Krumhansl profiles.

`FEATURES_VERSION` (`analysis/schema.py`) stamps every stored row, and the
ranking reads only rows at the current version — two vectors from different
stacks have no meaningful cosine, and nothing in the numbers would say so.
When the version goes up, the visible corpus shrinks to what the new stack
has analyzed and the **worker's re-analysis arm** refills it: each tick it
takes the oldest `GROUP_SIZE` superseded rows, re-fetches their audio from
the source that owns them, re-analyzes, and logs `reanalyzed N, remaining M`
— before it crawls for anything new. A row whose audio can no longer be
fetched or decoded is marked on its own document and skipped, so one dead
preview cannot stall the queue behind it.

Only the worker loads the audio model. `POST /seed` for an unknown track
queues it for the worker and waits; the API process never analyzes anything,
because a second copy of CLAP alongside the worker's is an OOM kill. A seed
whose row was analyzed by an older version is not "ready" either — it is
pushed to the front of the re-analysis queue and waited on like a cold one,
and `/recommend` answers 409 `unanalyzed` rather than ranking a vector from
another feature space.

`GET /search/text` (search by description) is the exception and is **off by
default**: msclap has no text-tower-only load, so the first query pulls the
whole model into the API process (~2.5 GB resident, permanently). Set
`TEXT_SEARCH=1` to switch it on — `GET /axes` reports `text_search`, and the
web app shows the toggle only when it is true.

## Sources and licensing

Where tracks come from is `SOURCES` (`corpus/sources/`):

- `SOURCES=deezer` — 30-second previews. **Development only.** Deezer's API
  terms do not permit shipping it in a product.
- `SOURCES=jamendo` — Creative Commons music, full audio, and the catalogue
  this can actually be sold with. Needs a free `JAMENDO_CLIENT_ID`. Every
  track carries an `attribution_url`; the web app and the iPhone app both
  render a "via Jamendo · CC BY-SA" credit that links to it, because the
  licence obliges it. (The iOS credit line ships in this branch **unbuilt** —
  there is no Xcode toolchain in the environment it was written in, so it has
  not been compiled or seen on a device.)

Both may be listed at once (`SOURCES=deezer,jamendo`); the crawler
round-robins them, and a track id resolves back to its own source whether or
not that source is still switched on.

Every dependency and model file is MIT, BSD, ISC or Apache — see
`docs/THIRD_PARTY.md`, which `tests/test_licences.py` keeps honest: a new
dependency without a row fails the suite, and so does anything
non-commercial. That register is the whole point of the current stack. The
previous one embedded audio with Discogs-EffNet (CC BY-NC-SA) under
TensorFlow; both are gone, along with ~600 MB of image.

## Web app

    cd web && npm install && npm run dev   # proxies /api to the live server
    npm test
    npm run build

Production is built into the API image by `deploy/Dockerfile` and served
at `/` by the FastAPI app (`web/dist`).

### The feel slider

Embedding cosine ranks by *style*: it puts a hushed solo take and a full-band
blast of the same idiom in the same corner of the space, because idiomatically
they are the same thing. The eight feel axes (`analysis/feel.py`) score what
the cosine drops — energy, mood, texture — and "More sounds like this" ranks on
`cos(embedding) − w · mean|z(feel_rec) − z(feel_seed)|`, where the "Match the
feel" slider on the recommendations screen is `w` and each axis is z-scored
over the corpus first so no axis dominates by being wider. At `w = 0` the order
is exactly the embedding-only order, which is what makes the slider safe to
drag either way; the default is 0.3 and lives on the server (`FEEL_DEFAULT` in
`server/app.py`) — the web client omits the `feel` parameter at that value so
the number can be retuned without a redeploy of the bundle. A second slider,
"Match the tempo", subtracts an octave-folded tempo distance the same way.
Slider positions are remembered in `localStorage` and carried into Insights,
where the math panel shows the seed's and the rec's eight dimensions side by
side, the `feel_dist` between them, and both tracks' BPM, key and loudness. A
track with no feel vector is never penalized, and `surprise` is left alone
entirely, since "nothing like this" is already a request to leave the seed's
neighbourhood.

### SOUND mode

SOUND mode shows what a track actually sounds like rather than where it sits
in feature space: a mel spectrogram (96 bands, FFT 2048 / hop 1024, 70 dB
window — the same recipe as the iOS view), a self-similarity matrix of the
pooled mel frames (bright off-diagonal blocks are restated material), and a
band-solo strip you drag to hear only one range of frequencies. All of it is
computed in the browser, in a worker, from the decoded preview.

Ordinary playback still uses `GET /preview/{id}`, a 302 to the source's CDN,
so the mp3 bytes never touch our host. Only SOUND mode needs the raw samples —
a cross-origin stream decodes to silence through a `MediaElementSource` —
so the audio is proxied through `GET /preview/{id}/audio` only once a solo
is asked for.

## Deploy

The API and worker run as a Docker Compose stack on the Oracle VM, behind
the shared Caddy instance, at `https://essentia.gabeyocum.com`. See
`deploy/README.md` for the full picture; the three commands you need are:

    bash deploy/bootstrap.sh                                    # first-time setup / pick up new commits
    docker compose -f deploy/docker-compose.yml logs -f worker   # tail the worker's logs
    docker compose -f deploy/docker-compose.yml up -d --build    # redeploy after pulling new commits

The worker idles until `MONGODB_URI` is set in `~/stacks/essentia/deploy/.env`
on the VM.

## Layout

    contract/                     HTTP contract + fixture (cross-cutting: change
                                   server, app, and tests/test_contract.py together)
    src/music_recommendations/    analysis / server (FastAPI backend on MongoDB
                                   Atlas) / corpus components
    ios/                          SwiftUI app
    scripts/                      operator entry points
    legacy/                       pre-spec MVP, frozen reference

Tests: `python3 -m pytest` (and `cd web && npm test`). The analysis tests
skip unless the `analysis` extra is installed and `scripts/fetch_models.py`
has run — both runs must be green.
