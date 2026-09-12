import { useCallback, useEffect, useState } from "react";
import { useLocation, useNavigate, useParams } from "react-router-dom";
import { api, ApiError } from "../api/client";
import { Artwork } from "../components/Artwork";
import type { Axis, Track } from "../api/types";

type Status = "loading" | "ready" | "unanalyzed" | "error";

export function Seed() {
  const { id = "" } = useParams();
  const location = useLocation();
  const navigate = useNavigate();
  const track = (location.state as { track?: Track } | null)?.track;

  const [status, setStatus] = useState<Status>("loading");
  const [errorText, setErrorText] = useState("");
  const [axes, setAxes] = useState<Axis[]>([]);

  const run = useCallback(async () => {
    setStatus("loading");
    try {
      const [seedResult, axesResult] = await Promise.all([api.seed(id), api.axes()]);
      setAxes(axesResult.axes);
      if (seedResult.status === "ready") {
        setStatus("ready");
      } else {
        setStatus("unanalyzed");
      }
    } catch (err) {
      if (err instanceof ApiError && err.status === 502) {
        setErrorText("Couldn't prepare this track (undecodable preview).");
      } else {
        setErrorText("Couldn't prepare this track.");
      }
      setStatus("error");
    }
  }, [id]);

  useEffect(() => {
    void run();
  }, [run]);

  return (
    <div className="screen seed-screen">
      <div className="seed-header">
        <Artwork url={track?.artwork_url} size={160} />
        <div className="seed-header-info">
          <div className="seed-title">{track?.title ?? id}</div>
          {track && <div className="seed-artist">{track.artist}</div>}
          {track && <div className="seed-album">{track.album}</div>}
        </div>
      </div>

      {status === "loading" && <p className="hint">Analyzing track…</p>}

      {status === "unanalyzed" && (
        <div className="error-box">
          <p>This track hasn&apos;t been analyzed yet.</p>
          <button type="button" onClick={() => void run()}>
            Try again
          </button>
        </div>
      )}

      {status === "error" && (
        <div className="error-box">
          <p>{errorText}</p>
          <button type="button" onClick={() => void run()}>
            Try again
          </button>
        </div>
      )}

      {status === "ready" && (
        <div className="axis-list">
          {axes.map((axis) => (
            <button
              key={axis.id}
              type="button"
              className="axis-button"
              onClick={() => navigate(`/recs/${id}/${axis.id}`, { state: { track } })}
            >
              {axis.label}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
