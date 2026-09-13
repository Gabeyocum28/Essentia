import { useCallback, useEffect, useMemo, useState } from "react";
import { useParams } from "react-router-dom";
import { api, ApiError } from "../api/client";
import type { VizMap } from "../api/types";
import { RecStrip } from "../insights/RecStrip";
import { Galaxy } from "../insights/Galaxy";
import { Topology } from "../insights/Topology";
import { Walk } from "../insights/Walk";
import { Tour } from "../insights/Tour";
import { Proof } from "../insights/Proof";
import { Extremes } from "../insights/Extremes";
import { MathPanel } from "../insights/MathPanel";
import { WhySimilar } from "../insights/WhySimilar";

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

  // The global viz endpoints (tour/mst/hubs/extremes) build their own
  // seed-anchored subset, which need not contain a far `surprise` rec. Hand
  // them the ids this map drew so the highlighting lines up across chips.
  const recIds = useMemo(() => map?.recs.map((r) => r.track_id) ?? [], [map]);

  const selectedRec = useMemo(
    () => map?.recs.find((r) => r.track_id === selectedId) ?? null,
    [map, selectedId],
  );

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
    } catch (err) {
      if (err instanceof ApiError && err.status === 404) {
        setStatus("unanalyzed");
      } else {
        setStatus("error");
      }
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
          <button type="button" onClick={() => void run()}>
            Try again
          </button>
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

          {selectedRec && (
            <>
              <MathPanel rec={selectedRec} />
              {mode === "PROOF" && (
                <WhySimilar seedId={map.seed.track_id} recId={selectedRec.track_id} />
              )}
            </>
          )}

          <div className="segmented-control">
            {(["GALAXY", "SOUND", "PROOF"] as Mode[]).map((m) => (
              <button
                key={m}
                type="button"
                className={`segmented-control-item${mode === m ? " segmented-control-item-active" : ""}`}
                aria-pressed={mode === m}
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
                    aria-pressed={chip === c}
                    onClick={() => setChip(c)}
                  >
                    {c}
                  </button>
                ))}
              </div>

              {chip === "Explore" && <Galaxy map={map} selectedId={selectedId} onSelect={setSelectedId} />}
              {chip === "Walk" && <Walk map={map} selectedId={selectedId} onSelect={setSelectedId} />}
              {chip === "Tour" && (
                <Tour map={map} seedId={id} recIds={recIds} selectedId={selectedId} onSelect={setSelectedId} />
              )}
              {chip === "Topo" && (
                <Topology map={map} seedId={id} recIds={recIds} selectedId={selectedId} onSelect={setSelectedId} />
              )}
            </>
          )}

          {mode === "SOUND" && (
            <p className="hint">Spectrogram, self-similarity and band solo are not in the web app yet.</p>
          )}

          {mode === "PROOF" && (
            <>
              <Proof seedId={id} recIds={recIds} />
              <Extremes seedId={id} recIds={recIds} />
            </>
          )}
        </>
      )}
    </div>
  );
}
