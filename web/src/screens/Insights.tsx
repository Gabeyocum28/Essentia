import { useCallback, useEffect, useState } from "react";
import { useParams } from "react-router-dom";
import { api } from "../api/client";
import type { VizMap } from "../api/types";
import { RecStrip } from "../insights/RecStrip";
import { Galaxy } from "../insights/Galaxy";
import { Topology } from "../insights/Topology";

type Status = "loading" | "ready" | "unanalyzed" | "error";
type Mode = "GALAXY" | "SOUND" | "PROOF";
type GalaxyChip = "Explore" | "Walk" | "Tour" | "Topo";

export function Insights() {
  const { id = "", axis = "" } = useParams();

  const [status, setStatus] = useState<Status>("loading");
  const [map, setMap] = useState<VizMap | null>(null);
  const [mode, setMode] = useState<Mode>("GALAXY");
  const [chip, setChip] = useState<GalaxyChip>("Explore");
  const [selectedId, setSelectedId] = useState<string | null>(null);

  const run = useCallback(async () => {
    setStatus("loading");
    try {
      const seedResult = await api.seed(id);
      if (seedResult.status !== "ready") {
        setStatus("unanalyzed");
        return;
      }
      const mapResult = await api.vizMap(id, axis);
      setMap(mapResult);
      setStatus("ready");
    } catch {
      setStatus("error");
    }
  }, [id, axis]);

  useEffect(() => {
    void run();
  }, [run]);

  return (
    <div className="screen insights-screen">
      {status === "loading" && <p className="hint">Loading insights…</p>}

      {status === "unanalyzed" && (
        <div className="error-box">
          <p>This track is not analyzed on the server yet.</p>
        </div>
      )}

      {status === "error" && (
        <div className="error-box">
          <p>Something went wrong</p>
          <button type="button" onClick={() => void run()}>
            Try again
          </button>
        </div>
      )}

      {status === "ready" && map && (
        <>
          <RecStrip seed={map.seed} recs={map.recs} selectedId={selectedId} onSelect={setSelectedId} />

          <div className="segmented-control">
            {(["GALAXY", "SOUND", "PROOF"] as Mode[]).map((m) => (
              <button
                key={m}
                type="button"
                className={`segmented-control-item${mode === m ? " segmented-control-item-active" : ""}`}
                onClick={() => setMode(m)}
              >
                {m}
              </button>
            ))}
          </div>

          {mode === "GALAXY" && (
            <>
              <div className="chip-row">
                {(["Explore", "Walk", "Tour", "Topo"] as GalaxyChip[]).map((c) => (
                  <button
                    key={c}
                    type="button"
                    className={`chip${chip === c ? " chip-active" : ""}`}
                    onClick={() => setChip(c)}
                  >
                    {c}
                  </button>
                ))}
              </div>

              {chip === "Explore" && <Galaxy map={map} selectedId={selectedId} onSelect={setSelectedId} />}
              {chip === "Walk" && <p className="hint">Walk is coming in the next task.</p>}
              {chip === "Tour" && <p className="hint">Tour is coming in the next task.</p>}
              {chip === "Topo" && <Topology map={map} selectedId={selectedId} onSelect={setSelectedId} />}
            </>
          )}

          {mode === "SOUND" && (
            <p className="hint">Spectrogram, self-similarity and band solo are not in the web app yet.</p>
          )}

          {mode === "PROOF" && <p className="hint">Proof mode arrives in Task 5.</p>}
        </>
      )}
    </div>
  );
}
