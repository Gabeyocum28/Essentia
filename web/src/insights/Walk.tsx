import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api } from "../api/client";
import { fitTransform, nearestIndex, toScreen } from "./geometry";
import { Artwork } from "../components/Artwork";
import { usePlayer } from "../player/usePlayer";
import { WalkStrip } from "./WalkStrip";
import type { Track, VizMap, VizWalk } from "../api/types";

interface Props {
  map: VizMap;
  selectedId: string | null;
  onSelect: (id: string | null) => void;
}

const PAD = 24;
const CLICK_MAX_DIST = 36;
const STEP_MS = 250;

export function Walk({ map, selectedId, onSelect }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const { play } = usePlayer();

  const [size, setSize] = useState({ width: 0, height: 0 });
  const [fromId, setFromId] = useState(map.seed.track_id);
  const [toId, setToId] = useState<string | null>(null);
  const [walk, setWalk] = useState<VizWalk | null>(null);
  const [walkError, setWalkError] = useState(false);
  const [revealCount, setRevealCount] = useState(0);
  const [callout, setCallout] = useState<Track | null>(null);

  const allPoints = useMemo(() => {
    const points: { x: number; y: number; track: Track; kind: "corpus" | "rec" | "seed" }[] = [];
    for (let i = 0; i < map.points.ids.length; i++) {
      points.push({ x: map.points.x[i], y: map.points.y[i], track: map.points.tracks[i], kind: "corpus" });
    }
    for (const rec of map.recs) {
      points.push({ x: rec.x, y: rec.y, track: rec, kind: "rec" });
    }
    points.push({ x: map.seed.x, y: map.seed.y, track: map.seed, kind: "seed" });
    return points;
  }, [map]);

  const byId = useMemo(() => new Map(allPoints.map((p) => [p.track.track_id, p])), [allPoints]);

  const transform = useMemo(() => {
    const xs = allPoints.map((p) => p.x);
    const ys = allPoints.map((p) => p.y);
    return fitTransform(xs, ys, size.width || 1, size.height || 1, PAD);
  }, [allPoints, size]);

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

  // Reset picking to a fresh "from" whenever the seed changes underneath us.
  useEffect(() => {
    setFromId(map.seed.track_id);
    setToId(null);
    setWalk(null);
  }, [map.seed.track_id]);

  const runWalk = useCallback(async (from: string, to: string) => {
    setWalkError(false);
    setWalk(null);
    setRevealCount(0);
    try {
      const result = await api.vizWalk(from, to, 8);
      setWalk(result);
    } catch {
      setWalkError(true);
    }
  }, []);

  useEffect(() => {
    if (toId) void runWalk(fromId, toId);
  }, [fromId, toId, runWalk]);

  // Reveal one more step of the path every 250ms, driven by rAF so it survives tab throttling
  // gracefully and is trivially cancellable on unmount or a new walk.
  useEffect(() => {
    if (!walk) return;
    let raf = 0;
    let cancelled = false;
    const start = performance.now();
    const total = walk.path.length;

    const tick = (now: number) => {
      if (cancelled) return;
      const elapsed = now - start;
      const next = Math.min(total, Math.floor(elapsed / STEP_MS) + 1);
      setRevealCount((prev) => (next > prev ? next : prev));
      if (next < total) raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => {
      cancelled = true;
      cancelAnimationFrame(raf);
    };
  }, [walk]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || size.width === 0 || size.height === 0) return;

    let raf = 0;
    const draw = () => {
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

      ctx.fillStyle = "rgba(255,255,255,.55)";
      for (let i = 0; i < map.points.ids.length; i++) {
        const [sx, sy] = toScreen(transform, map.points.x[i], map.points.y[i]);
        ctx.beginPath();
        ctx.arc(sx, sy, 1.5, 0, Math.PI * 2);
        ctx.fill();
      }

      for (const rec of map.recs) {
        const [sx, sy] = toScreen(transform, rec.x, rec.y);
        if (selectedId === rec.track_id) {
          ctx.fillStyle = "rgba(10,132,255,.35)";
          ctx.beginPath();
          ctx.arc(sx, sy, 11, 0, Math.PI * 2);
          ctx.fill();
        }
        ctx.fillStyle = "#0A84FF";
        ctx.beginPath();
        ctx.arc(sx, sy, 4.5, 0, Math.PI * 2);
        ctx.fill();
      }

      const [seedSx, seedSy] = toScreen(transform, map.seed.x, map.seed.y);
      ctx.fillStyle = "rgba(255,214,10,.6)";
      ctx.beginPath();
      ctx.arc(seedSx, seedSy, 9, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = "#FFD60A";
      ctx.beginPath();
      ctx.arc(seedSx, seedSy, 4, 0, Math.PI * 2);
      ctx.fill();

      const from = byId.get(fromId);
      const to = toId ? byId.get(toId) : undefined;

      if (from) {
        const [sx, sy] = toScreen(transform, from.x, from.y);
        ctx.strokeStyle = "#FFD60A";
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(sx, sy, 8, 0, Math.PI * 2);
        ctx.stroke();
      }

      if (from && to) {
        const [fsx, fsy] = toScreen(transform, from.x, from.y);
        const [tsx, tsy] = toScreen(transform, to.x, to.y);
        ctx.save();
        ctx.strokeStyle = "rgba(255,255,255,.5)";
        ctx.lineWidth = 1;
        ctx.setLineDash([4, 4]);
        ctx.beginPath();
        ctx.moveTo(fsx, fsy);
        ctx.lineTo(tsx, tsy);
        ctx.stroke();
        ctx.restore();

        ctx.strokeStyle = "#0A84FF";
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(tsx, tsy, 8, 0, Math.PI * 2);
        ctx.stroke();
      }

      if (walk && walk.path.length > 0) {
        ctx.strokeStyle = "#FFD60A";
        ctx.lineWidth = 2;
        ctx.beginPath();
        const shown = walk.path.slice(0, revealCount);
        shown.forEach((step, i) => {
          const [sx, sy] = toScreen(transform, step.x, step.y);
          if (i === 0) ctx.moveTo(sx, sy);
          else ctx.lineTo(sx, sy);
        });
        ctx.stroke();
        for (const step of shown) {
          const [sx, sy] = toScreen(transform, step.x, step.y);
          ctx.fillStyle = "#FFD60A";
          ctx.beginPath();
          ctx.arc(sx, sy, 3.5, 0, Math.PI * 2);
          ctx.fill();
        }
      }
    };

    raf = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(raf);
  }, [map, size, transform, selectedId, byId, fromId, toId, walk, revealCount]);

  const handleClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const rect = canvasRef.current?.getBoundingClientRect();
    if (!rect) return;
    const px = e.clientX - rect.left;
    const py = e.clientY - rect.top;
    const xs = allPoints.map((p) => p.x);
    const ys = allPoints.map((p) => p.y);
    const idx = nearestIndex(xs, ys, transform, px, py, CLICK_MAX_DIST);
    if (idx === -1) return;
    const point = allPoints[idx];
    setCallout(point.track);

    if (toId === null) {
      if (point.track.track_id === fromId) return;
      setToId(point.track.track_id);
      const rec = map.recs.find((r) => r.track_id === point.track.track_id);
      onSelect(rec ? point.track.track_id : selectedId);
    } else {
      setFromId(point.track.track_id);
      setToId(null);
      setWalk(null);
    }
  };

  return (
    <div className="walk">
      <div className="galaxy" ref={containerRef}>
        <canvas ref={canvasRef} onClick={handleClick} />
        {callout && (
          <div className="galaxy-callout">
            <Artwork url={callout.artwork_url} size={56} />
            <div className="galaxy-callout-info">
              <div className="galaxy-callout-title">{callout.title}</div>
              <div className="galaxy-callout-artist">{callout.artist}</div>
            </div>
            <button
              type="button"
              className="galaxy-callout-play"
              aria-label={`Play ${callout.title}`}
              onClick={() => play(callout)}
            >
              ▶
            </button>
          </div>
        )}
      </div>
      <p className="hint walk-hint">
        {toId === null
          ? "Click a point to walk to it."
          : "Click another point to start a new walk."}
      </p>
      {walkError && <p className="error-box">Couldn&apos;t compute that walk.</p>}
      {walk && <WalkStrip walk={walk} revealCount={revealCount} />}
    </div>
  );
}
