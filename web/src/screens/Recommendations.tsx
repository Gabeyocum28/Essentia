import { useCallback, useEffect, useState } from "react";
import { Link, useLocation, useParams } from "react-router-dom";
import { api } from "../api/client";
import { Artwork } from "../components/Artwork";
import { TrackRow } from "../components/TrackRow";
import { Card } from "../components/Card";
import { SkeletonTrackList } from "../components/Skeleton";
import type { Track } from "../api/types";

type Status = "loading" | "ready" | "error";

export function Recommendations() {
  const { id = "", axis = "" } = useParams();
  const location = useLocation();
  const seedTrack = (location.state as { track?: Track } | null)?.track;

  const [status, setStatus] = useState<Status>("loading");
  const [results, setResults] = useState<Track[]>([]);

  const run = useCallback(async () => {
    setStatus("loading");
    try {
      const res = await api.recommend(id, axis);
      setResults(res.results);
      setStatus("ready");
    } catch {
      setStatus("error");
    }
  }, [id, axis]);

  useEffect(() => {
    void run();
  }, [run]);

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

      {status === "loading" && (
        <>
          <p className="hint">Loading recommendations…</p>
          <SkeletonTrackList rows={6} />
        </>
      )}

      {status === "error" && (
        <div className="error-box">
          <p>Something went wrong</p>
          <button type="button" onClick={() => void run()}>
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
            <Link className="insights-link" to={`/insights/${id}/${axis}`}>
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
