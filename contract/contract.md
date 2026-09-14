# HTTP contract — frozen

```
GET /search?q=miles+davis+so+what
→ { "results": [ Track, ... ] }

POST /seed  { "track_id": "3135556" }
→ { "track_id": "3135556", "status": "ready" }
   Blocks until analysis completes. Cold ~1-2s (0.8s analysis plus
   preview download), warm instant.

GET /axes
→ { "axes": [ { "id": "sounds_like", "label": "More sounds like this" },
              { "id": "surprise",    "label": "Nothing like this"      } ],
    "text_search": false }

   `text_search` says whether THIS HOST can answer GET /search/text (the
   CLAP text tower costs the API process ~2.5 GB resident, so it is off
   unless TEXT_SEARCH=1). A client that does not know the key ignores it;
   a client that offers a "by description" search hides it when false.

GET /recommend?track_id=3135556&axis=surprise&limit=10
→ { "seed_track_id": "3135556", "axis": "surprise", "results": [ Track, ... ] }
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
