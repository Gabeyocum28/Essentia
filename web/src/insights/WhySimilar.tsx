import { useEffect, useState } from "react";
import { api } from "../api/client";
import { formatHz } from "../audio/mel";
import { soloBand, soloOff, useSoloBand, useSoloError } from "../audio/soloStore";
import type { VizAttribution } from "../api/types";

interface Props {
  seedId: string;
  recId: string;
}

type Phase = "pending" | "ready" | "failed" | "timeout";

const POLL_MS = 1500;
const TIMEOUT_MS = 60_000;

/** Formats a frequency in Hz as e.g. "240" or "1.2k". */
export const hz = formatHz;

export function WhySimilar({ seedId, recId }: Props) {
  const [phase, setPhase] = useState<Phase>("pending");
  const [data, setData] = useState<VizAttribution | null>(null);
  const solo = useSoloBand();
  const soloFailure = useSoloError();

  useEffect(() => {
    setPhase("pending");
    setData(null);

    let cancelled = false;
    let interval: ReturnType<typeof setInterval> | undefined;
    let timeout: ReturnType<typeof setTimeout> | undefined;

    const stop = () => {
      if (interval !== undefined) clearInterval(interval);
      if (timeout !== undefined) clearTimeout(timeout);
    };

    const poll = async () => {
      try {
        const result = await api.vizAttribute(seedId, recId);
        if (cancelled) return;
        if (result.status === "pending") {
          setPhase("pending");
          return;
        }
        setData(result);
        setPhase(result.status);
        stop();
      } catch {
        if (!cancelled) {
          setPhase("failed");
          stop();
        }
      }
    };

    void poll();
    interval = setInterval(() => void poll(), POLL_MS);
    timeout = setTimeout(() => {
      stop();
      if (!cancelled) setPhase("timeout");
    }, TIMEOUT_MS);

    return () => {
      cancelled = true;
      stop();
    };
  }, [seedId, recId]);

  if (phase === "timeout") {
    return <p className="hint why-similar">Still waiting on the analysis worker.</p>;
  }

  if (phase === "failed") {
    return <p className="error-box why-similar">{data?.error ?? "Something went wrong"}</p>;
  }

  if (phase === "pending" || !data) {
    return <p className="hint why-similar">Working out why these sound alike…</p>;
  }

  if (!data.bands) {
    return <p className="error-box why-similar">No attribution data.</p>;
  }

  const bands = data.bands;
  const peak = bands.reduce((m, b) => Math.max(m, b.delta), 0) || 1;
  const topIdx = bands.reduce((best, b, i) => (b.delta > bands[best].delta ? i : best), 0);

  return (
    <div className="why-similar">
      {data.base !== undefined && <p className="mono why-similar-base">base cos {data.base.toFixed(4)}</p>}
      <div className="why-similar-bars">
        {bands.map((band, i) => {
          const soloed = solo !== null && solo[0] === band.lo_hz && solo[1] === band.hi_hz;
          return (
            <button
              type="button"
              key={`${band.lo_hz}-${band.hi_hz}`}
              className={`why-similar-bar-col${soloed ? " why-similar-bar-col-solo" : ""}`}
              aria-pressed={soloed}
              aria-label={`Solo ${hz(band.lo_hz)} to ${hz(band.hi_hz)} Hz`}
              onClick={() => (soloed ? soloOff() : soloBand(band.lo_hz, band.hi_hz))}
            >
              <div
                className="why-similar-bar"
                style={{
                  height: `${Math.max(0, band.delta) / peak * 100}%`,
                  opacity: soloed || i === topIdx ? 1 : 0.45,
                }}
              />
              <div className="mono why-similar-bar-label">
                {hz(band.lo_hz)} / {hz(band.hi_hz)}
              </div>
            </button>
          );
        })}
      </div>
      {soloFailure && <p className="error-box why-similar-solo-error">{soloFailure}</p>}
      <div className="why-similar-note">
        <span className="hint">Tap a band to hear only that band.</span>
        <button type="button" className="why-similar-solo-off" onClick={soloOff} disabled={solo === null}>
          Solo off
        </button>
      </div>
    </div>
  );
}
