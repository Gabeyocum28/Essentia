# server/ — Person 2

FastAPI backend. Routes mirror contract/contract.md EXACTLY — same paths,
same query params, same JSON keys. The mock (fixture-serving) behavior
ships first and stays as the fallback until the corpus lands.

Rules:
- No Essentia imports; call music_recommendations.analysis.analyze_track.
  You own WHEN a track is analyzed (cache lookup, download, write-back);
  analysis owns HOW.
- Ranking is normalize + matmul + argsort in numpy, in-process. Never add
  FAISS/pgvector/ANN — pure overhead at this scale (spec §2.2).
- The axis registry in axes.py is the one table for adding/removing/
  reweighting axes.
- POST /seed is one blocking HTTP request -- no polling state machine in the
  contract. Internally, a cold seed on a host without Essentia enqueues the
  track and polls the store for the Mac embed worker to finish before
  responding.
- Storage is MongoDB Atlas; see store.py's docstring for the three
  collections. Set MONGODB_URI (and optionally MONGODB_DB) to run.
- Analysis failures on POST /seed return 502 with body
  `{"detail": "analysis failed"}` -- the spec's `{"error": ...}` shape was
  not adopted, to match every other error response in app.py.
- While Atlas is unreachable, the API does not fail closed: `_safe()`
  swallows the store error and the endpoint falls back to serving the
  fixture with dummy scores, same as before the corpus landed. That silent
  fallback is a known gap for this sprint, to be replaced by proper 503s in
  the deployment sub-project.
