# server/ — Person 2

FastAPI backend. Routes mirror contract/contract.md EXACTLY — same paths,
same query params, same JSON keys. The mock (fixture-serving) behavior
ships first and stays as the fallback until the corpus lands.

Rules:
- The API process NEVER analyzes. A cold seed is handed to the worker, which
  is the one process that loads the audio model. The single exception is
  GET /search/text, which is OFF unless TEXT_SEARCH=1 because msclap has no
  text-tower-only load and the first query would pull the whole ~2.5 GB
  model into this process permanently.
- A seed is "ready" only at the CURRENT features_version. A row analyzed by
  an older stack is prioritized for re-analysis (store.prioritize_reanalysis)
  and waited on like a cold one; /recommend and the viz endpoints answer 409
  `{"status": "unanalyzed"}` for it rather than letting numpy's shape error
  become a 500.
- Ranking is normalize + matmul + argsort in numpy, in-process. Never add
  FAISS/pgvector/ANN — pure overhead at this scale (spec §2.2).
- The axis registry in axes.py is the one table for adding/removing/
  reweighting axes.
- POST /seed is one blocking HTTP request -- no polling state machine in the
  contract. Internally, a cold seed stores the metadata, enqueues an embed
  job and polls the store for the worker to finish, answering "ready" or
  "unanalyzed" within _EMBED_WAIT_S.
- Storage is MongoDB Atlas; see store.py's docstring for the three
  collections. Set MONGODB_URI (and optionally MONGODB_DB) to run.
- POST /seed no longer 502s: nothing about the audio is known here any more,
  so a broken preview is the worker's problem and the response is
  `{"status": "unanalyzed"}`. Other error responses keep app.py's `detail`
  shape rather than the spec's `{"error": ...}`.
- While Atlas is unreachable, the API does not fail closed: `_safe()`
  swallows the store error and the endpoint falls back to serving the
  fixture with dummy scores, same as before the corpus landed. That silent
  fallback is a known gap for this sprint, to be replaced by proper 503s in
  the deployment sub-project.
