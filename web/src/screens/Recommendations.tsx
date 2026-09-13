import { useCallback, useEffect, useRef, useState } from "react";
import { Link, useLocation, useParams } from "react-router-dom";
import { api } from "../api/client";
import { Artwork } from "../components/Artwork";
import { TrackRow } from "../components/TrackRow";
import { Card } from "../components/Card";
import { SkeletonTrackList } from "../components/Skeleton";
import type { Track } from "../api/types";

type Status = "loading" | "ready" | "error";

const FEEL_KEY = "essentia.feel";
const FEEL_DEBOUNCE_MS = 250;
const DEFAULT_FEEL = 0.5;

function loadFeel(): number {
  try {
    const raw = localStorage.getItem(FEEL_KEY);
    const n = raw === null ? NaN : Number(raw);
    return Number.isFinite(n) ? n : DEFAULT_FEEL;
  } catch {
    return DEFAULT_FEEL;
  }
}

export function Recommendations() {
  const { id = "", axis = "" } = useParams();
  const location = useLocation();
  const seedTrack = (location.state as { track?: Track } | null)?.track;

  const [status, setStatus] = useState<Status>("loading");
  const [results, setResults] = useState<Track[]>([]);
  const [feel, setFeel] = useState(loadFeel);
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const run = useCallback(async (feelValue: number) => {
    setStatus("loading");
    try {
      const res = await api.recommend(id, axis, 10, feelValue);
      setResults(res.results);
      setStatus("ready");
    } catch {
      setStatus("error");
    }
  }, [id, axis]);

  useEffect(() => {
    void run(feel);
    // Only re-run immediately on id/axis change; feel changes are debounced separately.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [id, axis]);

  const handleFeelChange = (value: number) => {
    setFeel(value);
    try {
      localStorage.setItem(FEEL_KEY, String(value));
    } catch {
      /* localStorage unavailable */
    }
    if (debounceRef.current) clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(() => {
      void run(value);
    }, FEEL_DEBOUNCE_MS);
  };

  useEffect(() => {
    return () => {
      if (debounceRef.current) clearTimeout(debounceRef.current);
    };
  }, []);

  return (
    <div className="screen recs-screen">
      <Card className="seed-header seed-header-compact">
        <Artwork url={seedTrack?.artwork_url} size={72} />
        <div className="seed-header-info">
          <div className="seed-eyebrow">{axis || "Seed"}</div>
          <div className="seed-title">{seedTrack?.title ?? id}</div>
          {seedTrack && <div className="seed-artist">{seedTrack.artist}</div>}
        </div>
      </Card>

      <label className="feel-slider">
        <span className="feel-slider-label">Match the feel</span>
        <input
          type="range"
          min={0}
          max={2}
          step={0.1}
          value={feel}
          onChange={(e) => handleFeelChange(Number(e.target.value))}
          className="feel-slider-input"
          aria-label="Match the feel"
        />
        <span className="mono feel-slider-value">{feel.toFixed(1)}</span>
      </label>

      {status === "loading" && (
        <>
          <p className="hint">Loading recommendations…</p>
          <SkeletonTrackList rows={6} />
        </>
      )}

      {status === "error" && (
        <div className="error-box">
          <p>Something went wrong</p>
          <button type="button" onClick={() => void run(feel)}>
            Try again
          </button>
        </div>
      )}

      {status === "ready" && (
        <>
          <div className="recs-header-row">
            <span className="recs-count">
              {results.length} {results.length === 1 ? "track" : "tracks"}
            </span>
            <Link className="insights-link" to={`/insights/${id}/${axis}?feel=${feel}`}>
              See the math ✦
            </Link>
          </div>
          <div className="track-list">
            {results.map((t) => (
              <TrackRow key={t.track_id} track={t} showScore />
            ))}
          </div>
        </>
      )}
    </div>
  );
}
