# Jazz Recommender — Design Spec

**Date:** 2026-08-31
**Constraint:** 24-hour sprint, 4 people, all working with AI coding agents.

---

## 1. What we are building

An iPhone app that recommends tracks similar to a track you pick, where *similar* is something you choose the meaning of.

The loop:

1. User searches for a seed track.
2. User taps one of five intentions — sounds like this, keep the feeling, keep the style, keep the groove, different songs.
3. App returns 5–10 tracks.
4. User plays the 30-second previews.

Audio comes from Deezer's public preview URLs. Analysis is Essentia, running server-side only.

### Explicit non-goals for v1

These were considered and deliberately cut. Do not build them.

- **No musical profile screen.** The app shows the track and the five buttons. No tags, no metrics, no visualization.
- **No explanation text.** Recommendations come back as a plain list.
- **No harmony axis.** Key and chord analysis on a 30-second lossy preview is hard to do well and would eat the sprint.
- **No instrumentation axis.** It was proposed to make explanations legible; with no explanations, it earns nothing.
- **No accounts, no saving, no history, no playlists.**
- **No Android, no web client.**
- **No hosting.** Backend runs on a laptop behind a tunnel — superseded by `docs/superpowers/specs/2026-09-11-oracle-atlas-migration-design.md`.

---

## 2. Architecture

Four components. Each is independently buildable and testable.

```
 iOS app  ──HTTP──>  server  ──calls──>  analysis  <──calls──  corpus crawler
                        │                                            │
                        └──────────────>  Redis  <──────────────────┘
```

### 2.1 analysis — audio to numbers

A pure function: MP3 path in, a dictionary of feature vectors out. Knows nothing about HTTP, Deezer, Redis, or the phone.

The important property is that **one EffNet pass feeds everything.** Decoding the audio and running Discogs-EffNet is the only slow part; every classifier head then runs on the resulting embedding for almost nothing.

```
audio ─→ MonoLoader(16 kHz) ─→ EffNet ─→ embedding[1280] ─┬─→ genre_discogs400 head → genre[400]
                                                          ├─→ moodtheme head       → mood[56]
                                                          └─→ (any future head)

audio ─→ MonoLoader(44.1 kHz) ─→ rhythm DSP ────────────────→ groove[?]
```

Consequence: adding an axis that rides on the embedding later costs a small model download and no extra compute. Adding a DSP axis costs a second decode and a second pass over the audio.

**Genre is a head, not a second model output.** An earlier draft of this spec had EffNet fanning out to both `PartitionedCall:1` (embedding) and `PartitionedCall:0` (400 style activations) in one pass. That is not possible: `TensorflowPredictEffnetDiscogs` takes a *single* `output` string, so reading both means instantiating it twice and paying for two full forward passes. Use the `genre_discogs400` classification head on the embedding instead. Verified: it reproduces the direct `PartitionedCall:0` output to within float error (cosine 1.0000001, identical argmax) at **0.014 s instead of 0.35 s**. Genre behaves exactly like every other head.

Those 400 activations include **25 jazz substyles**: Afro-Cuban, Afrobeat, Avant-garde, Big Band, Bop, Bossa Nova, Contemporary, Cool Jazz, Dixieland, Easy Listening, Free Improvisation, Free Jazz, Fusion, Gypsy Jazz, Hard Bop, Jazz-Funk, Jazz-Rock, Latin Jazz, Modal, Post Bop, Ragtime, Smooth Jazz, Soul-Jazz, Space-Age, Swing.

**Two decodes, not one.** `RhythmExtractor2013` and `OnsetRate` both require 44100 Hz and throw otherwise, while EffNet requires 16 kHz. The pipeline calls `MonoLoader` twice. Each decode is ~0.03 s, so this costs nothing.

**Verified available** (all take the EffNet embedding as input, all confirmed downloadable from `essentia.upf.edu/models/classification-heads/`):

| Model | Size | Output |
|---|---|---|
| `genre_discogs400` | 2.06 MB | 400 Discogs style activations |
| `mtg_jamendo_moodtheme` | 2.74 MB | 56 mood/theme tags |
| `mtg_jamendo_instrument` | 2.71 MB | 40 instrument tags |
| `mood_happy` / `sad` / `relaxed` / `aggressive` / `acoustic` | 0.51 MB each | binary probability |
| `timbre` | 0.51 MB | bright ↔ dark |
| `voice_instrumental` | 0.51 MB | vocal presence |
| `danceability`, `approachability`, `engagement` | 0.51 MB each | scalar |

The base model `discogs-effnet-bs64-1.pb` (18 MB) is confirmed working.

**Head node names are not uniform. This will bite.** `TensorflowPredict2D` defaults to `input="model/Placeholder"`, and not every head uses it. Verified on two heads:

| Head | `input` | `output` |
|---|---|---|
| `genre_discogs400` | `serving_default_model_Placeholder` | `PartitionedCall:0` |
| `mtg_jamendo_moodtheme` | `model/Placeholder` | `model/Sigmoid` |

Every head added must have its node names checked, not assumed. This is why the head table is a data structure in the code (`analysis/registry.py`) rather than arguments scattered across call sites.

**Measured cost**, one 30 s Deezer preview, laptop CPU:

| Stage | Time |
|---|---|
| `MonoLoader` @ 16 kHz | 0.04 s |
| EffNet → embedding | 0.48 s |
| each classification head | ~0.01 s |
| `MonoLoader` @ 44.1 kHz | 0.03 s |
| `RhythmExtractor2013` | 0.22 s |
| `OnsetRate` + `Danceability` | 0.05 s |
| **total** | **~0.8 s** |

So 300 tracks is roughly four minutes of analysis, not twenty. Analysis throughput is not a constraint on the corpus target; crawling and downloading previews is.

Groove is the only axis not derived from the embedding — it needs rhythm DSP (tempo, onset rate, beat confidence, and some measure of swing). **Exact extractor choices are still open.** Essentia has no swing algorithm, so any swing feature is hand-built from `RhythmExtractor2013` ticks and `OnsetRate` onsets. `BpmHistogramDescriptors` is worth considering because it consumes `bpmIntervals` and surfaces a second tempo peak, which turns the octave-error risk below into a feature rather than a wrong number. These choices live entirely inside this component and changing them breaks nothing else.

### 2.2 server — the thing the phone talks to

FastAPI. Proxies Deezer search, orchestrates seed analysis, ranks candidates, returns results.

Ranking is a normalized matrix multiply plus an argsort, in numpy, in-process. At a few hundred to a few thousand tracks this is sub-millisecond. **Do not add FAISS, pgvector, or any ANN index** — at this scale it is pure overhead.

The server owns an **axis registry**: a mapping from axis id to (feature matrix, distance function, direction). Adding, removing, or reweighting an axis is a change to this one table.

### 2.3 corpus — collecting the jazz

Deezer has **no usable genre browse**. Verified: `GET /genre/129/artists` returns Dolly Parton, Bad Bunny, and Drake. It is not genre-filtered. `GET /chart/129/tracks` caps at 100 results. There is no endpoint that enumerates a genre.

What does work is the related-artists graph. Verified: `GET /artist/1910/related` (Miles Davis) returns Coltrane, Monk, Chet Baker, Parker, Rollins, Mingus, Gillespie, Bill Evans, Blakey, Ellington, Wes Montgomery, Cannonball Adderley, Horace Silver, Kenny Burrell, Oscar Peterson, Wynton Marsalis, Dexter Gordon, Brubeck, Ahmad Jamal, Clifford Brown — twenty artists, all jazz, no contamination.

**Strategy:** snowball. Start from ~8 roots chosen to span the space rather than cluster:

```
Miles Davis · Duke Ellington · Django Reinhardt · Ornette Coleman
Bill Evans · Stan Getz · Jimmy Smith · Weather Report
```

Two hops of `/related` (20 per artist) yields 100–200 artists. Take `/artist/{id}/top?limit=20` from each, dedupe, require a preview URL. That lands at roughly 2,000–4,000 candidates.

**Sprint target is ~300 analyzed tracks,** not 2,000. Three hundred is enough for recommendations to feel real and takes ~20 minutes to analyze instead of hours. Grow only if there is time left after integration.

### 2.4 iOS — the app

SwiftUI. Search field → results list with artwork → tap to select → five buttons → recommendation list → AVPlayer preview playback.

---

## 3. The contract

**This is the only truly shared thing in the project. It must be frozen in hour 0 and changed only by agreement of all four people.**

```
GET /search?q=miles+davis+so+what
→ { "results": [ Track, ... ] }

POST /seed  { "track_id": "3135556" }
→ { "track_id": "3135556", "status": "ready" }
   Blocks until analysis completes. Cold ~1-2s (0.8s analysis plus
   preview download), warm instant.

GET /axes
→ { "axes": [ { "id": "sounds_like", "label": "Sounds like this" },
              { "id": "mood",        "label": "Keep the feeling"  },
              { "id": "genre",       "label": "Keep the style"    },
              { "id": "groove",      "label": "Keep the groove"   },
              { "id": "surprise",    "label": "Surprise me"       } ] }

GET /recommend?track_id=3135556&axis=groove&limit=10
→ { "seed_track_id": "3135556", "axis": "groove", "results": [ Track, ... ] }
```

Every `Track` object is the same shape at every endpoint:

```json
{
  "track_id": "3135556",
  "title": "So What",
  "artist": "Miles Davis",
  "album": "Kind of Blue",
  "artwork_url": "https://...",
  "preview_url": "https://...",
  "score": 0.91
}
```

`score` is present on recommendation results only, and exists for debugging. The v1 UI ignores it.

### Three deliberate choices

**`/axes` is an endpoint, not hardcoded in Swift.** The axis list is not settled and may shrink. The client renders whatever buttons the server sends, so changing the axis list never requires touching iOS.

**`POST /seed` is synchronous.** An async status/polling design costs the server a state machine and the client another one. A five-second blocking HTTP request is fine and removes real work from both sides of the biggest seam.

**Uniform `Track` shape.** One Swift struct, decoded identically everywhere.

---

## 4. Axes

| Axis id | Source | Independent of embedding? |
|---|---|---|
| `sounds_like` | EffNet embedding, cosine | — (is the embedding) |
| `mood` | `mtg_jamendo_moodtheme` + mood binaries | No |
| `genre` | Discogs-400, jazz substyles | No |
| `groove` | rhythm DSP | **Yes** |
| `surprise` | EffNet embedding, most distant | No |

`surprise` returns the 10 **most distant** tracks by embedding cosine.

**Known risk, accepted:** `sounds_like`, `mood`, `genre`, and `surprise` all derive from the same 1280-d vector, so they are correlated and may return overlapping results. If two buttons go to the same place, the premise of the app is weakened. Measuring this was scoped as a spike and then cut for time. If the demo shows the buttons behaving identically, the fallback is to ship three axes: `sounds_like`, `groove`, `surprise` — `groove` is the only genuinely independent one.

**Known risk, accepted:** the most-distant track in an embedding space tends to be a degenerate outlier — near-silent intros, applause, bad encodes, mis-crawled non-jazz. `surprise` may return the same small set of junk for every seed. Mitigation if it shows up: drop tracks flagged as outliers from the candidate pool, or sample from the far tail instead of taking a strict argmax.

**Known risk, accepted:** tempo estimation commonly halves or doubles (68 vs 136 BPM), which would make `groove` cluster wrongly. Using log-tempo or folding tempo into a single octave helps if it appears.

---

## 5. Storage

**Redis**, with ranking done in Python.

Redis holds track metadata and feature vectors. The server loads the corpus matrices into numpy at startup and ranks in-process; newly analyzed seeds get written back so the second request for a given track is instant.

Redis Stack's vector search was considered and rejected for the sprint: it requires a specific Redis build, and it makes the ranking a black box at exactly the scale where a three-line numpy matmul is both exact and instant. It is the upgrade path if the index ever gets large.

---

## 6. Repo layout and agent boundaries

All four people are working with AI agents, which introduces two failure modes that will otherwise destroy the integration:

1. **Agents invent interfaces.** If the contract lives in chat, one agent produces `artworkURL` and another `cover_image`, and nothing catches it until integration.
2. **Agents fix things outside their lane.** An agent debugging a failing call will happily edit the server, the contract, or someone else's folder to make its own code pass — silently reshaping the seam everyone agreed on.

Both are prevented by putting the contract in the repo as a file and giving every agent written boundaries it reads automatically.

```
music-recommendations/
├── CLAUDE.md                     ← working agreement, every agent reads it
├── pyproject.toml                ← one package, one venv, one `uv sync`
├── uv.lock  .python-version  .gitignore  README.md
│
├── contract/                     ← READ-ONLY. all four people.
│   ├── CLAUDE.md                 ← "changing this needs all four"
│   ├── contract.md               ← the frozen HTTP contract (Section 3)
│   ├── features.py               ← the dict shape analysis returns
│   └── fixture.json              ← 30 real jazz tracks; everyone's test data
│
├── src/music_recommendations/
│   ├── analysis/                 ← Person 4
│   │   ├── CLAUDE.md
│   │   ├── __init__.py           ← analyze_track(mp3_path) -> dict
│   │   ├── registry.py           ← head name -> (file, input node, output node)
│   │   ├── embedding.py          ← MonoLoader(16k) + EffNet -> (n, 1280)
│   │   ├── heads.py              ← TensorflowPredict2D over the registry
│   │   └── groove.py             ← MonoLoader(44.1k) + rhythm DSP
│   ├── server/                   ← Person 2
│   │   ├── CLAUDE.md
│   │   ├── app.py                ← FastAPI; routes mirror contract.md exactly
│   │   ├── deezer.py             ← search proxy
│   │   ├── store.py              ← Redis read/write
│   │   ├── axes.py               ← axis registry: id -> (matrix, metric, direction)
│   │   └── rank.py               ← normalize, matmul, argsort
│   └── corpus/                   ← Person 3
│       ├── CLAUDE.md
│       ├── crawl.py              ← snowball over /artist/{id}/related
│       ├── download.py           ← previews -> audio_cache/
│       └── ingest.py             ← calls analysis in a loop, writes Redis
│
├── ios/                          ← Person 1. no Python.
│   ├── CLAUDE.md
│   └── JazzRec.xcodeproj/ + JazzRec/
│
├── scripts/                      ← operator entry points, not libraries
│   ├── fetch_models.py           ← downloads EffNet + every head in registry.py
│   ├── build_corpus.py
│   └── analyze_corpus.py
│
├── tests/
│   └── analysis/  server/  corpus/   ← one dir per lane, same ownership rules
│
├── notebooks/
│   └── discogs_style_embedding_comparison.ipynb
│
├── models/                       ← gitignored, populated by fetch_models.py
└── audio_cache/                  ← gitignored
```

### Four decisions in that tree

**One Python package, not four top-level folders.** Flat `analysis/ server/ corpus/` directories look cleaner but break on the first real import — `server` must call `analysis`, and so must `corpus`. Flat dirs mean `sys.path` hacks, four editable installs, or a uv workspace with four `pyproject.toml` files. Under one package it is just `from music_recommendations.analysis import analyze_track`. Ownership does not come from depth; it comes from the per-directory `CLAUDE.md`, which agents load when they touch files in that directory at any nesting level. The boundary survives, the packaging tax does not.

**`contract/` sits outside `src/`.** It is less code the app imports than the thing four people agreed on, and its position at the top of the tree is part of how it stays visible.

**`registry.py` is load-bearing, not a detail.** Head node names vary per head and do not match `TensorflowPredict2D`'s defaults (Section 2.1). One table, read by both `fetch_models.py` and `heads.py`, makes adding an axis a one-row diff:

```python
HEADS = {
    "genre":     Head("genre_discogs400",      400, "serving_default_model_Placeholder", "PartitionedCall:0"),
    "moodtheme": Head("mtg_jamendo_moodtheme",  56, "model/Placeholder",                 "model/Sigmoid"),
}
```

**`models/` and `audio_cache/` are gitignored, with a fetch script.** Nobody commits 18 MB of protobuf, and nobody hand-downloads it either — `scripts/fetch_models.py` iterating `registry.py` puts a new machine one command from a working checkout. `audio_cache/` is derived data.

```gitignore
models/
audio_cache/
.venv/
__pycache__/
*.pyc
.DS_Store
ios/**/xcuserdata/
```

### Root `CLAUDE.md`

```markdown
# Working agreement

This is a 4-person, 24-hour sprint. Four people are running coding agents
against this repo at the same time.

## Ownership

You own exactly ONE of these folders. It was named in your task.

  ios/                                  — the iPhone app
  src/music_recommendations/server/     — the FastAPI backend
  src/music_recommendations/corpus/     — the Deezer crawler
  src/music_recommendations/analysis/   — the Essentia pipeline

You also own the matching folder under tests/ and any script in scripts/
that drives your lane.

Do not create, edit, or delete files outside the folder you own. This
includes pyproject.toml: if you need a dependency added, ask.

## contract/ is read-only

Everything in contract/ is shared by four people. Do not edit it.

If the contract appears wrong, incomplete, or blocking, STOP and tell the
human. Do not work around it. Do not add a field. Do not rename anything.
A contract change requires all four people to agree, and routing around a
mismatch silently breaks three other people's work.

## Test data

contract/fixture.json holds 30 real jazz tracks with real Deezer preview
URLs. Use it for all testing. Do not invent your own test tracks.

## Scope

This is a 24-hour sprint. Build what is asked, nothing more. No profile
screens, no explanation text, no accounts, no persistence beyond Redis,
no deployment config — superseded by `docs/superpowers/specs/2026-09-11-oracle-atlas-migration-design.md`. If you think something extra is needed, ask.

## Merging

Commit and push small changes often. Do not sit on large diffs.
```

### Why `fixture.json` carries weight

Thirty real jazz tracks — real Deezer IDs, real artwork, real preview URLs — used by all four folders. The iOS app displays them, the mock server serves them, the crawler must produce that shape, and the analyzer runs on those exact clips. When something breaks, all four people are looking at the same Miles Davis track.

---

## 7. Who does what

**Person 1 — iOS.** The whole client. Search, results list, seed selection, buttons rendered from `/axes`, recommendation list, preview playback. Touches no Python. The largest single track.

**Person 2 — server.** Ships the **mock server first** — a small FastAPI app that ignores the question and returns tracks from `fixture.json`. That unblocks Person 1 permanently and is the highest-priority item in the project. Then the real backend: Deezer search proxy, seed orchestration, Redis, numpy ranking, axis registry. Touches no Essentia and no crawling.

**Person 3 — corpus.** Writes `fixture.json` first (the other unblocker). Then the snowball crawler, metadata ingest, preview downloads, dedup, and the Redis schema. Runs the batch analysis over the corpus by calling Person 4's function. Touches no models and no HTTP.

**Person 4 — analysis.** The Essentia pipeline: decode, EffNet pass, classifier heads, groove DSP, per-axis normalization. All undecided extractor choices live here. Touches no HTTP and no crawler.

### The boundary between Person 2 and Person 4

**Person 4 owns *how* a song is analyzed. Person 2 owns *when* a song gets analyzed.**

Person 4 writes one function: MP3 in, feature dictionary out. It does not know the phone, the server, Deezer, or Redis exist.

Person 2 decides when to call it — cached lookup for known tracks, download-and-analyze for new ones, write the result back so nobody waits twice.

The split matters because Person 4's work is slow, model-heavy, and full of open decisions, while Person 2's is fast plumbing that must be nailed down early so Person 1 can build against it. One person owning both would block the phone app on Essentia decisions.

Person 3 calls the same function in a batch loop. Same function, different caller.

**Person 4's function is the only thing three other people depend on**, so its *shape* should be agreed in hour 0 even though its insides stay open. Roughly: MP3 path in, a dict of named float arrays out. Person 4 can change what is inside that dict for the whole sprint without breaking anyone, as long as the shape holds.

---

## 8. Timeline

The way 24-hour sprints fail is four people building excellent components and integrating at hour 20. Everything works, nothing connects, no time left. **Integrating early with garbage data is worth more than any component you could build in the same hours.**

| Hour | What |
|---|---|
| **0–1** | All four together. Repo skeleton, `CLAUDE.md`, freeze the contract, write `fixture.json`. Nobody codes alone yet. |
| **1–2** | P2 ships the mock server. P1 starts the app. P3 starts the crawler. P4 downloads the models and analyzes one MP3. |
| **2–6** | P1 has search → pick → buttons → list → playback working against the mock. **Vertical slice done.** |
| **6–12** | P4's analyzer produces all features for one file. P3 has ~300 tracks downloaded. P2 has Redis + ranking working on the 30 fixture tracks. |
| **12–16** | P3 runs batch analysis over all ~300. P2 swaps mock responses for real ranking. |
| **16–18** | **Integration.** P1 points the app at the real server. |
| **18–20** | Grow the corpus if time allows. Fix what is obviously broken. |
| **20** | **Feature freeze.** No new anything. |
| **20–24** | Bugs, demo rehearsal, sleep in shifts. |

Two hard commitments: **vertical slice by hour 6**, **feature freeze at hour 20**. Everything else can slip.

---

## 9. Cut list

If behind, cut in this order. Each is safe on its own.

1. **Growing the corpus past 300 tracks** — skip it.
2. **Five buttons → three** (`sounds_like`, `groove`, `surprise`).
3. **Open Deezer search → closed catalog.** Seed only from the ~300 analyzed tracks. This deletes on-demand analysis entirely, makes every response instant, and removes a large chunk of Person 2's work. The demo barely suffers. This is the highest-value cut available.
4. **Three buttons → one.** The app becomes a plain "more like this."

---

## 10. Known risks

| Risk | Impact | Response |
|---|---|---|
| Axes are correlated; buttons return similar results | Undermines the core premise | Fall back to 3 axes; `groove` is the only independent one |
| `surprise` returns degenerate outliers | Bad demo moment | Filter outliers or sample the far tail |
| Tempo octave errors (68 vs 136 BPM) | `groove` clusters wrongly | Log-tempo or fold into one octave |
| Deezer search returns the wrong recording | User seeds the wrong "Autumn Leaves" | Show several hits with artwork; user picks |
| **No evaluation of recommendation quality** | No way to tell if any change helps or hurts | Accepted for the sprint. Nothing distinguishes a good recommender from a broken one without listening tests, and there is no time. |
| ~~Analysis slower than expected on laptop CPU~~ | — | **Retired.** Measured at 0.8 s/track; 300 tracks is ~4 min. The corpus bottleneck is crawling and downloading previews, not Essentia. |
| A classification head's node names differ from the defaults | Silent `TensorflowPredict2D` failure or wrong tensor | Node names are checked per head and recorded in `analysis/registry.py`; never assume the defaults |
| Four agents colliding in the repo | Lost work, broken seams | Folder ownership + read-only `contract/` in `CLAUDE.md` |

---

## 11. Decisions made

| Decision | Choice |
|---|---|
| Client | SwiftUI native iPhone app |
| Backend hosting | Laptop + tunnel, dev only |
| Seed universe | Any Deezer track, analyzed on demand |
| Candidate pool | ~300 jazz tracks (target; 2,000 if time) |
| Corpus source | `/artist/{id}/related` snowball from 8 roots |
| Axes | sounds_like, mood, genre, groove, surprise |
| Harmony axis | Out |
| Instrumentation axis | Out |
| Surprise semantics | Most distant tracks |
| Results per query | 5–10 |
| Playback | Yes, 30s preview |
| Profile screen | None |
| Explanation text | None |
| Storage | Redis; ranking in numpy in-process |
| Seed endpoint | Synchronous |
| Evaluation | None (accepted risk) |
| Repo layout | One Python package under `src/`, one lane per subpackage |
| Genre extraction | `genre_discogs400` head on the embedding, not a second EffNet pass |
| Head configuration | Per-head node names in `analysis/registry.py` |
| Groove extractors | **Still open** — see Section 2.1 |
