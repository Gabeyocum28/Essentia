import { useEffect, useRef, useState } from "react";
import { canvasPalette } from "../theme";
import { api } from "../api/client";
import { gaussianCurve, scaleBars } from "./histogram";
import { Artwork } from "../components/Artwork";
import { usePlayer } from "../player/usePlayer";
import type { Track, VizHistogram, VizHubs, VizMap } from "../api/types";

interface Props {
  seedId: string;
  recIds: string[];
}

type Status = "loading" | "ready" | "error";

interface ComparisonRow {
  track: Track;
  rawScore?: number;
  correctedScore?: number;
  rawRank?: number;
  correctedRank?: number;
}

const PAD_X = 24;
const PAD_Y = 16;

// Ordered by the corrected list's order, then any extra raw-only recs.
function buildComparison(raw: VizMap, corrected: VizMap): ComparisonRow[] {
  const rawRankById = new Map(raw.recs.map((r, i) => [r.track_id, i + 1]));
  const rawScoreById = new Map(raw.recs.map((r) => [r.track_id, r.score]));

  const rows: ComparisonRow[] = corrected.recs.map((r, i) => ({
    track: r,
    correctedScore: r.score,
    correctedRank: i + 1,
    rawScore: rawScoreById.get(r.track_id),
    rawRank: rawRankById.get(r.track_id),
  }));

  const seen = new Set(corrected.recs.map((r) => r.track_id));
  for (const [i, r] of raw.recs.entries()) {
    if (seen.has(r.track_id)) continue;
    rows.push({ track: r, rawScore: r.score, rawRank: i + 1 });
  }
  return rows;
}

// "↑2" moved up (better) two ranks by correction, "↓1" moved down one, "·" unchanged or
// not comparable (missing from one side).
function rankBadge(rawRank: number | undefined, correctedRank: number | undefined): string {
  if (rawRank === undefined || correctedRank === undefined) return "·";
  const diff = rawRank - correctedRank;
  if (diff > 0) return `↑${diff}`;
  if (diff < 0) return `↓${-diff}`;
  return "·";
}

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

export function Proof({ seedId, recIds }: Props) {
  const [status, setStatus] = useState<Status>("loading");
  const [hist, setHist] = useState<VizHistogram | null>(null);
  const [hubs, setHubs] = useState<VizHubs | null>(null);
  const [rawMap, setRawMap] = useState<VizMap | null>(null);
  const [correctedMap, setCorrectedMap] = useState<VizMap | null>(null);
  const [reloadKey, setReloadKey] = useState(0);

  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [size, setSize] = useState({ width: 0, height: 0 });
  const { play } = usePlayer();

  useEffect(() => {
    let cancelled = false;
    setStatus("loading");
    (async () => {
      try {
        const [h, hb, raw, corrected] = await Promise.all([
          api.vizHistogram(seedId),
          api.vizHubs(seedId, recIds),
          api.vizMap(seedId, "surprise", 10, "off"),
          api.vizMap(seedId, "surprise", 10, "on"),
        ]);
        if (cancelled) return;
        setHist(h);
        setHubs(hb);
        setRawMap(raw);
        setCorrectedMap(corrected);
        setStatus("ready");
      } catch {
        if (!cancelled) setStatus("error");
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [seedId, recIds, reloadKey]);

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
    const c = canvasPalette();
    if (!canvas) return;

    const dpr = window.devicePixelRatio || 1;
    canvas.width = size.width * dpr;
    canvas.height = size.height * dpr;
    canvas.style.width = `${size.width}px`;
    canvas.style.height = `${size.height}px`;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = c.bg;
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
    ctx.fillStyle = c.cyan;
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
    ctx.strokeStyle = c.seed;
    ctx.lineWidth = 2;
    for (const rawScore of hist.rec_scores) {
      const score = Math.min(1, Math.max(-1, rawScore));
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
  if (status === "error" || !hist || !hubs || !rawMap || !correctedMap) {
    return (
      <div className="error-box">
        <p>Couldn&apos;t load the proof data.</p>
        <button type="button" onClick={() => setReloadKey((k) => k + 1)}>
          Try again
        </button>
      </div>
    );
  }

  const percentile = Math.round(hist.percentile);
  const comparisonRows = buildComparison(rawMap, correctedMap);

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

      <p className="mono proof-section-title">Nothing like this · raw vs corrected scores</p>
      <div className="proof-corrected-list">
        <div className="proof-corrected-header mono">
          <span className="proof-corrected-header-title" />
          <span className="proof-corrected-score-heading">raw</span>
          <span className="proof-corrected-score-heading">corrected</span>
          <span className="proof-corrected-rank-heading" />
        </div>
        {comparisonRows.map((row) => (
          <button
            key={row.track.track_id}
            type="button"
            className="proof-corrected-item"
            onClick={() => play(row.track)}
            aria-label={`Play ${row.track.title}`}
          >
            <span className="proof-corrected-title">{row.track.title}</span>
            <span className="proof-corrected-artist">{row.track.artist}</span>
            <span className="mono proof-corrected-score">
              {row.rawScore !== undefined ? row.rawScore.toFixed(4) : "—"}
            </span>
            <span className="mono proof-corrected-score">
              {row.correctedScore !== undefined ? row.correctedScore.toFixed(4) : "—"}
            </span>
            <span className="mono proof-corrected-rank">{rankBadge(row.rawRank, row.correctedRank)}</span>
          </button>
        ))}
      </div>

      <HubRail title="Hubs" items={hubs.hubs} badge={(item) => `×${item.count ?? 0}`} />
      <HubRail title="Central" items={hubs.central} badge={(item) => (item.centrality ?? 0).toFixed(3)} />
      <HubRail title="Isolated" items={hubs.isolated} badge={(item) => (item.centrality ?? 0).toFixed(3)} />
    </div>
  );
}
