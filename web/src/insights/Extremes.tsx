import { useCallback, useEffect, useState } from "react";
import { api } from "../api/client";
import { Artwork } from "../components/Artwork";
import { usePlayer } from "../player/usePlayer";
import type { Track, VizExtremes as VizExtremesData } from "../api/types";

type Status = "loading" | "ready" | "error";

const PCS = [1, 2, 3, 4];

function Rail({ label, tracks }: { label: string; tracks: Track[] }) {
  const { play } = usePlayer();
  return (
    <div className="extremes-rail">
      <div className="extremes-rail-label">{label}</div>
      {tracks.map((t) => (
        <button
          key={t.track_id}
          type="button"
          className="extremes-item"
          onClick={() => play(t)}
          aria-label={`Play ${t.title}`}
        >
          <Artwork url={t.artwork_url} size={52} />
          <div className="extremes-item-info">
            <div className="extremes-item-title">{t.title}</div>
            <div className="extremes-item-artist">{t.artist}</div>
          </div>
        </button>
      ))}
    </div>
  );
}

export function Extremes() {
  const [status, setStatus] = useState<Status>("loading");
  const [pcs, setPcs] = useState<VizExtremesData[]>([]);

  const run = useCallback(async () => {
    setStatus("loading");
    try {
      const results = await Promise.all(PCS.map((pc) => api.vizExtremes(pc, 4)));
      setPcs(results);
      setStatus("ready");
    } catch {
      setStatus("error");
    }
  }, []);

  useEffect(() => {
    void run();
  }, [run]);

  if (status === "loading") {
    return <p className="hint">Loading eigen listening…</p>;
  }
  if (status === "error") {
    return (
      <div className="error-box">
        <p>Couldn&apos;t load the extremes.</p>
        <button type="button" onClick={() => void run()}>
          Try again
        </button>
      </div>
    );
  }

  return (
    <div className="extremes">
      {pcs.map((pc) => (
        <div key={pc.pc} className="extremes-row">
          <p className="mono extremes-caption">
            PC {pc.pc} · {pc.variance_pct.toFixed(1)}% of variance
          </p>
          <div className="extremes-rails">
            <Rail label="Low" tracks={pc.low} />
            <Rail label="High" tracks={pc.high} />
          </div>
        </div>
      ))}
    </div>
  );
}
