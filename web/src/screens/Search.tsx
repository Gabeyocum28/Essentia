import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { api, ApiError } from "../api/client";
import { TrackRow } from "../components/TrackRow";
import { SkeletonTrackList } from "../components/Skeleton";
import type { Track } from "../api/types";

type Status = "idle" | "loading" | "error" | "ready";

const DEFAULT_ERROR = "Something went wrong";

export function Search() {
  const [query, setQuery] = useState("");
  const [status, setStatus] = useState<Status>("idle");
  const [byDescription, setByDescription] = useState(false);
  const [error, setError] = useState(DEFAULT_ERROR);
  const [results, setResults] = useState<Track[]>([]);
  const navigate = useNavigate();

  // Two different questions, deliberately one box: by NAME asks the
  // catalogue ("Kind of Blue"), by DESCRIPTION asks the analyzed corpus what
  // it sounds like ("hazy late-night trumpet"). Only the second one needs
  // CLAP, and the server answers 503 when it cannot load it -- so that
  // detail is shown rather than a generic failure.
  const runSearch = async (q: string, description: boolean) => {
    if (!q.trim()) return;
    setStatus("loading");
    try {
      const { results } = description ? await api.searchText(q) : await api.search(q);
      setResults(results);
      setStatus("ready");
    } catch (e) {
      setError(e instanceof ApiError ? e.detail : DEFAULT_ERROR);
      setStatus("error");
    }
  };

  const onSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    void runSearch(query, byDescription);
  };

  const onToggle = (next: boolean) => {
    setByDescription(next);
    // Re-answer the question they already asked, rather than leaving the
    // previous mode's results under the new label.
    if (status !== "idle") void runSearch(query, next);
  };

  return (
    <div className="screen search-screen">
      <div className="search-hero">
        <h1 className="search-hero-title">Find a track to start from</h1>
        <p className="search-hero-sub">
          Pick a seed and Essentia maps the corpus around it.
        </p>
      </div>

      <form onSubmit={onSubmit} className="search-form">
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder={byDescription
            ? "Describe what you want to hear"
            : "Search for a track or artist"}
          aria-label="Search"
        />
        <button type="submit" className="btn btn-primary">
          Search
        </button>
      </form>

      <label className="search-mode">
        <input
          type="checkbox"
          checked={byDescription}
          onChange={(e) => onToggle(e.target.checked)}
          aria-label="Search by description"
        />
        <span>by description</span>
      </label>

      {status === "idle" && (
        <p className="hint">
          {byDescription
            ? "Describe a sound and Essentia finds the closest tracks it has analyzed."
            : "Search for a track to get started."}
        </p>
      )}
      {status === "loading" && (
        <>
          <p className="hint">Searching…</p>
          <SkeletonTrackList rows={6} />
        </>
      )}
      {status === "error" && (
        <div className="error-box">
          <p>{error}</p>
          <button type="button" onClick={() => void runSearch(query, byDescription)}>
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
