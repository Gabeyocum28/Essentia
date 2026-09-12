# Working agreement

Essentia is a jazz recommender: Deezer previews, Essentia audio analysis,
and axis-based ranking, served to an iPhone app. It began as a 4-person,
24-hour hackathon and is now a solo project. One person owns everything,
so there are no ownership lanes; edit whatever the task needs.

The design is in `Essencia_design_spec.md`. Read the relevant section
before changing behavior.

## Layout

  ios/                                  — the iPhone app (SwiftUI)
  src/music_recommendations/server/     — the FastAPI backend
  src/music_recommendations/corpus/     — the Deezer crawler
  src/music_recommendations/analysis/   — the Essentia pipeline
  contract/                             — HTTP contract, feature axes, fixture
  scripts/                              — corpus build, seed, push, export tools
  tests/                                — pytest, mirrors src/ by folder
  legacy/                               — the pre-spec MVP, frozen

## contract/

contract/ defines the HTTP shapes and feature axes that the server and the
iOS app both depend on. It is no longer frozen, but a change there is a
cross-cutting change: update the server, the app, and tests/test_contract.py
in the same PR so nothing drifts.

## Test data

contract/fixture.json holds 30 real jazz tracks with real Deezer preview
URLs. Use it for testing. Do not invent test tracks.

## Scope

Build what is asked, nothing more. If something extra seems needed, say so
and ask before adding it. Dependencies go in pyproject.toml via `uv`.

## legacy/

legacy/ is frozen. Copy from it if useful; never import it, never edit it.

## Git

Branch as <topic> off main and open a PR; do not commit directly to main.
Run `python3 -m pytest` before committing. Commit small changes often.
