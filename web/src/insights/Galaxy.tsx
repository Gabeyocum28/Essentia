import { useEffect, useMemo, useRef, useState } from "react";
import { canvasPalette } from "../theme";
import { applyZoomPan, fitTransform, nearestIndex, toScreen, zoomAbout } from "./geometry";
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

  // Kept in sync with state so the native (non-passive) wheel listener, which is attached
  // once, can always read the latest values without re-subscribing.
  const zoomRef = useRef(zoom);
  const panRef = useRef(pan);
  const sizeRef = useRef(size);
  useEffect(() => {
    zoomRef.current = zoom;
    panRef.current = pan;
    sizeRef.current = size;
  }, [zoom, pan, size]);

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
  const baseTransformRef = useRef(baseTransform);
  useEffect(() => {
    baseTransformRef.current = baseTransform;
  }, [baseTransform]);

  // Tracks the canvas backing-store size actually applied, so the draw loop only touches
  // canvas.width/height (which clears the canvas and is comparatively expensive) when the
  // container size or DPR genuinely changed, not on every frame.
  const appliedSizeRef = useRef({ width: 0, height: 0, dpr: 0 });

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
      const c = canvasPalette();
      const dpr = window.devicePixelRatio || 1;
      const applied = appliedSizeRef.current;
      if (applied.width !== size.width || applied.height !== size.height || applied.dpr !== dpr) {
        canvas.width = size.width * dpr;
        canvas.height = size.height * dpr;
        canvas.style.width = `${size.width}px`;
        canvas.style.height = `${size.height}px`;
        appliedSizeRef.current = { width: size.width, height: size.height, dpr };
      }
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, size.width, size.height);
      ctx.fillStyle = c.bg;
      ctx.fillRect(0, 0, size.width, size.height);

      const cx = size.width / 2;
      const cy = size.height / 2;
      const t = applyZoomPan(baseTransform, zoom, pan.x, pan.y, cx, cy);

      // Corpus dots
      ctx.fillStyle = c.dot;
      for (let i = 0; i < map.points.ids.length; i++) {
        const [sx, sy] = toScreen(t, map.points.x[i], map.points.y[i]);
        ctx.beginPath();
        ctx.arc(sx, sy, 1.5, 0, Math.PI * 2);
        ctx.fill();
      }

      const [seedSx, seedSy] = toScreen(t, map.seed.x, map.seed.y);

      // Spokes seed -> rec
      ctx.strokeStyle = c.accentLine;
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
          ctx.fillStyle = c.accentSoft;
          ctx.beginPath();
          ctx.arc(rsx, rsy, 11, 0, Math.PI * 2);
          ctx.fill();
        }
        ctx.fillStyle = c.accent;
        ctx.beginPath();
        ctx.arc(rsx, rsy, 4.5, 0, Math.PI * 2);
        ctx.fill();
      }

      // Seed glow + dot
      ctx.fillStyle = c.seedSofter;
      ctx.beginPath();
      ctx.arc(seedSx, seedSy, 18, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = c.seedSoft;
      ctx.beginPath();
      ctx.arc(seedSx, seedSy, 9, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = c.seed;
      ctx.beginPath();
      ctx.arc(seedSx, seedSy, 4, 0, Math.PI * 2);
      ctx.fill();
    };

    raf = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(raf);
  }, [map, size, zoom, pan, selectedId, baseTransform]);

  // React's onWheel is a passive listener, so preventDefault() there is silently ignored and
  // the page scrolls instead of zooming. Attach a native listener with { passive: false } so
  // preventDefault actually stops the scroll, and anchor the zoom to the cursor position.
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const rect = canvas.getBoundingClientRect();
      const mx = e.clientX - rect.left;
      const my = e.clientY - rect.top;
      const cx = sizeRef.current.width / 2;
      const cy = sizeRef.current.height / 2;
      const factor = 1 - e.deltaY * 0.002;
      const { zoom: nextZoom, pan: nextPan } = zoomAbout(
        baseTransformRef.current,
        zoomRef.current,
        panRef.current,
        factor,
        mx,
        my,
        cx,
        cy,
      );
      setZoom(nextZoom);
      setPan(nextPan);
    };

    canvas.addEventListener("wheel", onWheel, { passive: false });
    return () => canvas.removeEventListener("wheel", onWheel);
  }, []);

  const handlePointerDown = (e: React.PointerEvent<HTMLCanvasElement>) => {
    dragRef.current = { startX: e.clientX, startY: e.clientY, panX: pan.x, panY: pan.y };
    canvasRef.current?.setPointerCapture(e.pointerId);
  };

  const handlePointerMove = (e: React.PointerEvent<HTMLCanvasElement>) => {
    if (!dragRef.current) return;
    const dx = e.clientX - dragRef.current.startX;
    const dy = e.clientY - dragRef.current.startY;
    setPan({ x: dragRef.current.panX + dx, y: dragRef.current.panY + dy });
  };

  const handlePointerUp = (e: React.PointerEvent<HTMLCanvasElement>) => {
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
        onPointerDown={handlePointerDown}
        onPointerMove={handlePointerMove}
        onPointerUp={handlePointerUp}
        onPointerCancel={() => (dragRef.current = null)}
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
