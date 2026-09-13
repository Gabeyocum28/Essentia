# Jazz Recommender

Audio-based jazz recommendations for iPhone. Pick a track, pick what
"similar" means ("More sounds like this" / "Nothing like this"), get 5–10
tracks with 30 s previews. Design: `Essencia_design_spec.md`.

## Setup

    python3 -m pip install -e ".[dev,analysis]"   # analysis extra = tensorflow
    brew install ffmpeg                            # or apt install ffmpeg
    python3 scripts/fetch_models.py                # downloads EffNet into models/
    export MONGODB_URI="mongodb+srv://..."   # Atlas connection string, never committed
    uvicorn music_recommendations.server.app:app --reload

## Web app

    cd web && npm install && npm run dev   # proxies /api to the live server
    npm test
    npm run build

Production is built into the API image by `deploy/Dockerfile` and served
at `/` by the FastAPI app (`web/dist`).

### The feel slider

Embedding cosine ranks by *style*: it puts a hushed solo take and a full-band
blast of the same idiom in the same corner of the space, because idiomatically
they are the same thing. Eleven small classifier heads (`analysis/feel.py`,
graphs fetched by `scripts/fetch_models.py`) read the same EffNet embedding and
score what the cosine drops — energy, mood, texture. "More sounds like this"
then ranks on `cos(embedding) − w · mean|feel_rec − feel_seed|`, and the "Match
the feel" slider on the recommendations screen is `w`. At `w = 0` the order is
exactly the embedding-only order the app served before the heads shipped, which
is what makes the slider safe to drag either way; the default is 0.3 and lives
on the server (`FEEL_DEFAULT` in `server/app.py`) — the web client omits the
`feel` parameter at that value so the number can be retuned without a redeploy
of the bundle. The slider position is remembered in `localStorage` and carried
into Insights, where the math panel shows the seed's and the rec's eleven
dimensions side by side and the `feel_dist` between them. A track with no feel
vector yet (a row `scripts/feel_backfill.py` has not reached) is never
penalized — a partial backfill must not hide tracks — and `surprise` is left
alone entirely, since "nothing like this" is already a request to leave the
seed's neighbourhood.

### SOUND mode

SOUND mode shows what a track actually sounds like rather than where it sits
in feature space: a mel spectrogram (96 bands, FFT 2048 / hop 1024, 70 dB
window — the same recipe as the iOS view), a self-similarity matrix of the
pooled mel frames (bright off-diagonal blocks are restated material), and a
band-solo strip you drag to hear only one range of frequencies. All of it is
computed in the browser, in a worker, from the decoded preview.

Ordinary playback still uses `GET /preview/{id}`, a 302 to Deezer's CDN, so
the mp3 bytes never touch our host. Only SOUND mode needs the raw samples —
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

Tests: `python3 -m pytest`
Before merging analysis changes, also run
`PARITY=1 python3 -m pytest tests/analysis/test_parity.py -s` (needs
Essentia, network).
