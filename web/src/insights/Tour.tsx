import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api } from "../api/client";
import { fitTransform, toScreen } from "./geometry";
import { givensFrame, project } from "./tourMath";
import type { VizMap, VizTour } from "../api/types";

interface Props {
  map: VizMap;
  seedId: string;
  selectedId: string | null;
  onSelect: (id: string | null) => void;
}

type Status = "loading" | "ready" | "error";

const PAD = 24;

export function Tour({ map, seedId, selectedId }: Props) {
  const [status, setStatus] = useState<Status>("loading");
  const [tour, setTour] = useState<VizTour | null>(null);
  const [playing, setPlaying] = useState(true);

  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [size, setSize] = useState({ width: 0, height: 0 });

  const tRef = useRef(0);
  const playingRef = useRef(playing);
  useEffect(() => {
    playingRef.current = playing;
  }, [playing]);

  // Tracks the canvas backing-store size actually applied, so the draw loop only touches
  // canvas.width/height (which clears the canvas and is comparatively expensive) when the
  // container size or DPR genuinely changed, not on every animation frame.
  const appliedSizeRef = useRef({ width: 0, height: 0, dpr: 0 });

  const run = useCallback(async () => {
    setStatus("loading");
    try {
      const result = await api.vizTour(seedId);
      setTour(result);
      setStatus("ready");
    } catch {
      setStatus("error");
    }
  }, [seedId]);

  useEffect(() => {
    void run();
  }, [run]);

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

  const radius = useMemo(() => {
    if (!tour) return 1;
    let max = 0;
    for (const row of tour.coords) {
      let s = 0;
      for (let d = 0; d < row.length; d++) s += row[d] * row[d];
      max = Math.max(max, Math.sqrt(s));
    }
    return max || 1;
  }, [tour]);

  const idIndex = useMemo(() => {
    if (!tour) return new Map<string, number>();
    return new Map(tour.ids.map((id, i) => [id, i]));
  }, [tour]);

  const recIds = useMemo(() => new Set(map.recs.map((r) => r.track_id)), [map.recs]);
  const seedIdx = tour ? idIndex.get(map.seed.track_id) : undefined;

  useEffect(() => {
    if (!tour || size.width === 0 || size.height === 0) return;
    const canvas = canvasRef.current;
    if (!canvas) return;

    let raf = 0;
    let lastTime = performance.now();
    let cancelled = false;

    const draw = (now: number) => {
      if (cancelled) return;
      // Clamp dt so a tab that was backgrounded and resumes doesn't jump the tour forward.
      const dt = Math.min((now - lastTime) / 1000, 0.1);
      lastTime = now;
      if (playingRef.current) {
        tRef.current += dt;
      }

      const dpr = window.devicePixelRatio || 1;
      const applied = appliedSizeRef.current;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      if (applied.width !== size.width || applied.height !== size.height || applied.dpr !== dpr) {
        canvas.width = size.width * dpr;
        canvas.height = size.height * dpr;
        canvas.style.width = `${size.width}px`;
        canvas.style.height = `${size.height}px`;
        appliedSizeRef.current = { width: size.width, height: size.height, dpr };
      }
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.fillStyle = "#000";
      ctx.fillRect(0, 0, size.width, size.height);

      const frame = givensFrame(tRef.current);
      const { x, y } = project(tour.coords, frame);
      const t = fitTransform([-radius, radius], [-radius, radius], size.width, size.height, PAD);

      ctx.fillStyle = "rgba(255,255,255,.7)";
      for (let i = 0; i < x.length; i++) {
        if (i === seedIdx || recIds.has(tour.ids[i])) continue;
        const [sx, sy] = toScreen(t, x[i], y[i]);
        ctx.beginPath();
        ctx.arc(sx, sy, 2.4, 0, Math.PI * 2);
        ctx.fill();
      }

      for (let i = 0; i < x.length; i++) {
        if (!recIds.has(tour.ids[i]) || i === seedIdx) continue;
        const [sx, sy] = toScreen(t, x[i], y[i]);
        if (selectedId === tour.ids[i]) {
          ctx.fillStyle = "rgba(10,132,255,.35)";
          ctx.beginPath();
          ctx.arc(sx, sy, 9, 0, Math.PI * 2);
          ctx.fill();
        }
        ctx.fillStyle = "#0A84FF";
        ctx.beginPath();
        ctx.arc(sx, sy, 4, 0, Math.PI * 2);
        ctx.fill();
      }

      if (seedIdx !== undefined) {
        const [sx, sy] = toScreen(t, x[seedIdx], y[seedIdx]);
        ctx.fillStyle = "rgba(255,214,10,.6)";
        ctx.beginPath();
        ctx.arc(sx, sy, 7, 0, Math.PI * 2);
        ctx.fill();
        ctx.fillStyle = "#FFD60A";
        ctx.beginPath();
        ctx.arc(sx, sy, 3.5, 0, Math.PI * 2);
        ctx.fill();
      }

      raf = requestAnimationFrame(draw);
    };

    raf = requestAnimationFrame(draw);
    return () => {
      cancelled = true;
      cancelAnimationFrame(raf);
    };
  }, [tour, size, radius, seedIdx, recIds, selectedId]);

  const variancePct = useMemo(() => {
    if (!tour) return 0;
    return Math.round(tour.variance.reduce((a, b) => a + b, 0) * 100);
  }, [tour]);

  if (status === "loading") {
    return <p className="hint">Loading tour…</p>;
  }
  if (status === "error" || !tour) {
    return (
      <div className="error-box">
        <p>Couldn&apos;t load the tour data.</p>
        <button type="button" onClick={() => void run()}>
          Try again
        </button>
      </div>
    );
  }

  return (
    <div className="tour">
      <div className="galaxy" ref={containerRef}>
        <canvas ref={canvasRef} />
      </div>
      <div className="tour-controls">
        <button type="button" className="tour-play" onClick={() => setPlaying((p) => !p)}>
          {playing ? "Pause" : "Play"}
        </button>
        <p className="mono tour-caption">top-8 PCs hold {variancePct}% of variance</p>
      </div>
    </div>
  );
}
