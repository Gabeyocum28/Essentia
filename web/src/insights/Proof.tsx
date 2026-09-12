import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "../api/client";
import { gaussianCurve, scaleBars } from "./histogram";
import { Artwork } from "../components/Artwork";
import { usePlayer } from "../player/usePlayer";
import type { Track, VizHistogram, VizHubs, VizMap } from "../api/types";

interface Props {
  trackId: string;
  selectedId: string | null;
  onSelect: (id: string | null) => void;
}

type Status = "loading" | "ready" | "error";
type Correction = "off" | "on";

const PAD_X = 24;
const PAD_Y = 16;

function HubRail({
  title,
  items,
  badge,
}: {
  title: string;
  items: (Track & { count?: number; centrality?: number })[];
  badge: (item: Track & { count?: number; centrality?: number }) => string;
}) {
  const { play } = usePlayer();
  return (
    <div className="proof-rail">
      <div className="proof-rail-title">{title}</div>
      <div className="proof-rail-items">
        {items.map((item) => (
          <button
            key={item.track_id}
            type="button"
            className="proof-rail-item"
            onClick={() => play(item)}
            aria-label={`Play ${item.title}`}
          >
            <Artwork url={item.artwork_url} size={52} />
            <div className="proof-rail-info">
              <div className="proof-rail-item-title">{item.title}</div>
              <div className="proof-rail-item-artist">{item.artist}</div>
            </div>
            <span className="mono proof-rail-badge">{badge(item)}</span>
          </button>
        ))}
      </div>
    </div>
  );
}

export function Proof({ trackId, selectedId, onSelect }: Props) {
  const [status, setStatus] = useState<Status>("loading");
  const [hist, setHist] = useState<VizHistogram | null>(null);
  const [hubs, setHubs] = useState<VizHubs | null>(null);

  const [correction, setCorrection] = useState<Correction>("off");
  const [correctedMap, setCorrectedMap] = useState<VizMap | null>(null);
  const [correctedStatus, setCorrectedStatus] = useState<Status>("loading");

  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [size, setSize] = useState({ width: 0, height: 0 });
  const { play } = usePlayer();

  const run = useCallback(async () => {
    setStatus("loading");
    try {
      const [h, hb] = await Promise.all([api.vizHistogram(trackId), api.vizHubs()]);
      setHist(h);
      setHubs(hb);
      setStatus("ready");
    } catch {
      setStatus("error");
    }
  }, [trackId]);

  useEffect(() => {
    void run();
  }, [run]);

  const runCorrected = useCallback(
    async (c: Correction) => {
      setCorrectedStatus("loading");
      try {
        const result = await api.vizMap(trackId, "surprise", 10, c);
        setCorrectedMap(result);
        setCorrectedStatus("ready");
      } catch {
        setCorrectedStatus("error");
      }
    },
    [trackId],
  );

  useEffect(() => {
    void runCorrected(correction);
  }, [correction, runCorrected]);

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const observer = new ResizeObserver((entries) => {
      const entry = entries[0];
      if (!entry) return;
      setSize({ width: entry.contentRect.width, height: entry.contentRect.height });
    });
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    if (!hist || size.width === 0 || size.height === 0) return;
    const canvas = canvasRef.current;
    if (!canvas) return;

    const dpr = window.devicePixelRatio || 1;
    canvas.width = size.width * dpr;
    canvas.height = size.height * dpr;
    canvas.style.width = `${size.width}px`;
    canvas.style.height = `${size.height}px`;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, size.width, size.height);

    const innerW = size.width - PAD_X * 2;
    const innerH = size.height - PAD_Y * 2;
    if (hist.bins.length === 0) return;

    const binWidth = hist.bins.length > 1 ? hist.bins[1] - hist.bins[0] : 0.1;
    const total = hist.counts.reduce((a, b) => a + b, 0);
    const curve = gaussianCurve(hist.bins, total, binWidth, hist.null.sd, hist.null.mean);
    const maxCount = hist.counts.reduce((m, c) => Math.max(m, c), 0) || 1;
    const barHeights = scaleBars(hist.counts, innerH);
    const curveHeights = curve.map((c) => (c / maxCount) * innerH);

    const xToScreen = (x: number) => PAD_X + ((x + 1) / 2) * innerW;
    const barW = innerW / hist.bins.length;

    // Cyan bars: the actual score histogram.
    ctx.fillStyle = "#64D2FF";
    hist.bins.forEach((x, i) => {
      const h = barHeights[i];
      const sx = xToScreen(x) - (barW * 0.8) / 2;
      ctx.fillRect(sx, PAD_Y + innerH - h, barW * 0.8, h);
    });

    // White Gaussian null curve.
    ctx.strokeStyle = "rgba(255,255,255,.85)";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    hist.bins.forEach((x, i) => {
      const sx = xToScreen(x);
      const sy = PAD_Y + innerH - curveHeights[i];
      if (i === 0) ctx.moveTo(sx, sy);
      else ctx.lineTo(sx, sy);
    });
    ctx.stroke();

    // Yellow ticks at each rec score.
    ctx.strokeStyle = "#FFD60A";
    ctx.lineWidth = 2;
    for (const score of hist.rec_scores) {
      const sx = xToScreen(score);
      ctx.beginPath();
      ctx.moveTo(sx, PAD_Y + innerH);
      ctx.lineTo(sx, PAD_Y + innerH - 12);
      ctx.stroke();
    }
  }, [hist, size]);

  if (status === "loading") {
    return <p className="hint">Loading proof…</p>;
  }
  if (status === "error" || !hist || !hubs) {
    return (
      <div className="error-box">
        <p>Couldn&apos;t load the proof data.</p>
        <button type="button" onClick={() => void run()}>
          Try again
        </button>
      </div>
    );
  }

  const percentile = Math.round(hist.percentile);

  return (
    <div className="proof">
      <div className="proof-histogram" ref={containerRef}>
        <canvas ref={canvasRef} />
      </div>
      <div className="proof-axis-labels mono">
        <span>−1</span>
        <span>0</span>
        <span>1</span>
      </div>
      <p className="mono proof-caption">seed&apos;s recs sit at the {percentile}th percentile</p>

      <div className="proof-correction">
        <span className="proof-correction-label">Correction</span>
        <div className="segmented-control proof-correction-toggle">
          {(["off", "on"] as Correction[]).map((c) => (
            <button
              key={c}
              type="button"
              className={`segmented-control-item${correction === c ? " segmented-control-item-active" : ""}`}
              onClick={() => setCorrection(c)}
            >
              {c}
            </button>
          ))}
        </div>
      </div>

      {correctedStatus === "loading" && <p className="hint">Loading recs…</p>}
      {correctedStatus === "error" && <p className="error-box">Couldn&apos;t load the corrected recs.</p>}
      {correctedStatus === "ready" && correctedMap && (
        <div className="proof-corrected-list">
          {correctedMap.recs.map((rec) => (
            <button
              key={rec.track_id}
              type="button"
              className={`proof-corrected-item${selectedId === rec.track_id ? " proof-corrected-item-selected" : ""}`}
              onClick={() => {
                play(rec);
                onSelect(rec.track_id);
              }}
            >
              <span className="proof-corrected-title">{rec.title}</span>
              <span className="proof-corrected-artist">{rec.artist}</span>
              <span className="mono proof-corrected-score">{rec.score.toFixed(4)}</span>
            </button>
          ))}
        </div>
      )}

      <HubRail title="Hubs" items={hubs.hubs} badge={(item) => `×${item.count ?? 0}`} />
      <HubRail title="Central" items={hubs.central} badge={(item) => (item.centrality ?? 0).toFixed(3)} />
      <HubRail title="Isolated" items={hubs.isolated} badge={(item) => (item.centrality ?? 0).toFixed(3)} />
    </div>
  );
}
