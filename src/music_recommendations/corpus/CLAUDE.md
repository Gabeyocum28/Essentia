# corpus/ — Person 3

Snowball crawler over Deezer /artist/{id}/related from 8 jazz roots
(spec §2.3), preview downloads into audio_cache/, batch ingest that calls
music_recommendations.analysis.analyze_track and writes the store.

Rules:
- No Essentia imports; call analysis.analyze_track only (from ingest.py).
- No HTTP serving; this lane only *consumes* the Deezer API.
- Sleep ≥0.2 s between API calls; dedupe by track_id; require preview_url.
- Sprint target ~300 tracks. Do not gold-plate the crawler.
- Every track dict you produce is the contract Track shape — see
  contract/features.py TRACK_FIELDS.
