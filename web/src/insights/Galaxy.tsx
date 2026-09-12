import { useEffect, useMemo, useRef, useState } from "react";
import { applyZoomPan, fitTransform, nearestIndex, toScreen } from "./geometry";
import { Artwork } from "../components/Artwork";
import { usePlayer } from "../player/usePlayer";
import type { Track, VizMap } from "../api/types";

interface Props {
  map: VizMap;
  selectedId: string | null;
  onSelect: (id: string | null) => void;
}

interface CalloutPoint {
  track: Track;
  score?: number;
  isRec: boolean;
}

const PAD = 24;
const CLICK_MAX_DIST = 36;

export function Galaxy({ map, selectedId, onSelect }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const { play } = usePlayer();

  const [size, setSize] = useState({ width: 0, height: 0 });
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const dragRef = useRef<{ startX: number; startY: number; panX: number; panY: number } | null>(null);
  const [callout, setCallout] = useState<CalloutPoint | null>(null);

  const recById = useMemo(() => new Map(map.recs.map((r) => [r.track_id, r])), [map.recs]);

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

  const baseTransform = useMemo(() => {
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
      const { width, height } = entry.contentRect;
      setSize({ width, height });
    });
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

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
      ctx.clearRect(0, 0, size.width, size.height);
      ctx.fillStyle = "#000";
      ctx.fillRect(0, 0, size.width, size.height);

      const cx = size.width / 2;
      const cy = size.height / 2;
      const t = applyZoomPan(baseTransform, zoom, pan.x, pan.y, cx, cy);

      // Corpus dots
      ctx.fillStyle = "rgba(255,255,255,.55)";
      for (let i = 0; i < map.points.ids.length; i++) {
        const [sx, sy] = toScreen(t, map.points.x[i], map.points.y[i]);
        ctx.beginPath();
        ctx.arc(sx, sy, 1.5, 0, Math.PI * 2);
        ctx.fill();
      }

      const [seedSx, seedSy] = toScreen(t, map.seed.x, map.seed.y);

      // Spokes seed -> rec
      ctx.strokeStyle = "rgba(10,132,255,.6)";
      ctx.lineWidth = 1;
      for (const rec of map.recs) {
        const [rsx, rsy] = toScreen(t, rec.x, rec.y);
        ctx.beginPath();
        ctx.moveTo(seedSx, seedSy);
        ctx.lineTo(rsx, rsy);
        ctx.stroke();
      }

      // Recs
      for (const rec of map.recs) {
        const [rsx, rsy] = toScreen(t, rec.x, rec.y);
        if (selectedId === rec.track_id) {
          ctx.fillStyle = "rgba(10,132,255,.35)";
          ctx.beginPath();
          ctx.arc(rsx, rsy, 11, 0, Math.PI * 2);
          ctx.fill();
        }
        ctx.fillStyle = "#0A84FF";
        ctx.beginPath();
        ctx.arc(rsx, rsy, 4.5, 0, Math.PI * 2);
        ctx.fill();
      }

      // Seed glow + dot
      ctx.fillStyle = "rgba(255,214,10,.25)";
      ctx.beginPath();
      ctx.arc(seedSx, seedSy, 18, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = "rgba(255,214,10,.6)";
      ctx.beginPath();
      ctx.arc(seedSx, seedSy, 9, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = "#FFD60A";
      ctx.beginPath();
      ctx.arc(seedSx, seedSy, 4, 0, Math.PI * 2);
      ctx.fill();
    };

    raf = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(raf);
  }, [map, size, zoom, pan, selectedId, baseTransform]);

  const handleWheel = (e: React.WheelEvent<HTMLCanvasElement>) => {
    e.preventDefault();
    const delta = -e.deltaY * 0.002;
    setZoom((z) => Math.min(8, Math.max(1, z * (1 + delta))));
  };

  const handleMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    dragRef.current = { startX: e.clientX, startY: e.clientY, panX: pan.x, panY: pan.y };
  };

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!dragRef.current) return;
    const dx = e.clientX - dragRef.current.startX;
    const dy = e.clientY - dragRef.current.startY;
    setPan({ x: dragRef.current.panX + dx, y: dragRef.current.panY + dy });
  };

  const handleMouseUp = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const drag = dragRef.current;
    dragRef.current = null;
    if (!drag) return;
    const moved = Math.hypot(e.clientX - drag.startX, e.clientY - drag.startY);
    if (moved > 4) return; // was a drag, not a click

    const rect = canvasRef.current?.getBoundingClientRect();
    if (!rect) return;
    const px = e.clientX - rect.left;
    const py = e.clientY - rect.top;
    const cx = size.width / 2;
    const cy = size.height / 2;
    const t = applyZoomPan(baseTransform, zoom, pan.x, pan.y, cx, cy);

    const xs = allPoints.map((p) => p.x);
    const ys = allPoints.map((p) => p.y);
    const idx = nearestIndex(xs, ys, t, px, py, CLICK_MAX_DIST);
    if (idx === -1) {
      setCallout(null);
      return;
    }
    const point = allPoints[idx];
    const rec = recById.get(point.track.track_id);
    setCallout({ track: point.track, score: rec?.score, isRec: point.kind === "rec" });
    if (point.kind === "rec" || point.kind === "seed") {
      onSelect(point.track.track_id === map.seed.track_id ? null : point.track.track_id);
    }
  };

  return (
    <div className="galaxy" ref={containerRef}>
      <canvas
        ref={canvasRef}
        onWheel={handleWheel}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={() => (dragRef.current = null)}
      />
      {callout && (
        <div className="galaxy-callout">
          <Artwork url={callout.track.artwork_url} size={56} />
          <div className="galaxy-callout-info">
            <div className="galaxy-callout-title">{callout.track.title}</div>
            <div className="galaxy-callout-artist">{callout.track.artist}</div>
            {callout.isRec && callout.score !== undefined && (
              <div className="mono galaxy-callout-score">{callout.score.toFixed(4)}</div>
            )}
          </div>
          <button
            type="button"
            className="galaxy-callout-play"
            aria-label={`Play ${callout.track.title}`}
            onClick={() => play(callout.track)}
          >
            ▶
          </button>
        </div>
      )}
    </div>
  );
}
