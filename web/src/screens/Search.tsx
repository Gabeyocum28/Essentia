import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { api } from "../api/client";
import { TrackRow } from "../components/TrackRow";
import type { Track } from "../api/types";

type Status = "idle" | "loading" | "error" | "ready";

export function Search() {
  const [query, setQuery] = useState("");
  const [status, setStatus] = useState<Status>("idle");
  const [results, setResults] = useState<Track[]>([]);
  const navigate = useNavigate();

  const runSearch = async (q: string) => {
    if (!q.trim()) return;
    setStatus("loading");
    try {
      const { results } = await api.search(q);
      setResults(results);
      setStatus("ready");
    } catch {
      setStatus("error");
    }
  };

  const onSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    void runSearch(query);
  };

  return (
    <div className="screen search-screen">
      <form onSubmit={onSubmit} className="search-form">
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Search for a track or artist"
          aria-label="Search"
        />
        <button type="submit">Search</button>
      </form>

      {status === "idle" && <p className="hint">Search for a track to get started.</p>}
      {status === "loading" && <p className="hint">Searching…</p>}
      {status === "error" && (
        <div className="error-box">
          <p>Something went wrong</p>
          <button type="button" onClick={() => void runSearch(query)}>
            Try again
          </button>
        </div>
      )}
      {status === "ready" && results.length === 0 && <p className="hint">No tracks found.</p>}
      {status === "ready" && results.length > 0 && (
        <div className="track-list">
          {results.map((t) => (
            <TrackRow
              key={t.track_id}
              track={t}
              onSelect={(track) => navigate(`/seed/${track.track_id}`, { state: { track } })}
            />
          ))}
        </div>
      )}
    </div>
  );
}
