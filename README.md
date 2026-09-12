# Jazz Recommender

Audio-based jazz recommendations for iPhone. Pick a track, pick what
"similar" means (sound / feeling / style / groove / surprise), get 5–10
tracks with 30 s previews. Design: `Essencia_design_spec.md`.

## Setup

    python3 -m pip install -e ".[dev,analysis]"   # analysis extra = tensorflow
    brew install ffmpeg                            # or apt install ffmpeg
    python3 scripts/fetch_models.py                # downloads EffNet into models/
    redis-server &                          # storage
    uvicorn music_recommendations.server.app:app --reload

## Layout

    contract/                     frozen HTTP contract + fixture (read-only)
    src/music_recommendations/    analysis / server / corpus lanes
    ios/                          SwiftUI app
    scripts/                      operator entry points
    legacy/                       pre-spec MVP, frozen reference

Tests: `python3 -m pytest`
