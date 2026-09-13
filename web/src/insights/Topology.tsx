import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { canvasPalette } from "../theme";
import { api } from "../api/client";
import { UnionFind } from "./unionfind";
import { fitTransform, nearestIndex, toScreen } from "./geometry";
import type { VizMap, VizMst, VizTour } from "../api/types";

interface Props {
  map: VizMap;
  seedId: string;
  recIds: string[];
  selectedId: string | null;
  onSelect: (id: string | null) => void;
}

type Status = "loading" | "ready" | "error";

const PAD = 24;
const CLICK_MAX_DIST = 36;

function percentile(sorted: number[], p: number): number {
  if (sorted.length === 0) return 0;
  const idx = Math.min(sorted.length - 1, Math.max(0, Math.floor(p * (sorted.length - 1))));
  return sorted[idx];
}

export function Topology({ map, seedId, recIds, selectedId, onSelect }: Props) {
  const [status, setStatus] = useState<Status>("loading");
  const [mst, setMst] = useState<VizMst | null>(null);
  const [tour, setTour] = useState<VizTour | null>(null);
  const [threshold, setThreshold] = useState(0);

  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const barcodeRef = useRef<HTMLCanvasElement>(null);
  const [size, setSize] = useState({ width: 0, height: 0 });

  const run = useCallback(async () => {
    setStatus("loading");
    try {
      const [mstResult, tourResult] = await Promise.all([api.vizMst(seedId, recIds), api.vizTour(seedId, recIds)]);
      if (mstResult.ids.length !== tourResult.ids.length) {
        setStatus("error");
        return;
      }
      setMst(mstResult);
      setTour(tourResult);
      const distances = mstResult.edges.map((e) => e[2]).sort((a, b) => a - b);
      setThreshold(percentile(distances, 0.3));
      setStatus("ready");
    } catch {
      setStatus("error");
    }
  }, [seedId, recIds]);

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

  const xs = useMemo(() => (tour ? tour.coords.map((c) => c[0]) : []), [tour]);
  const ys = useMemo(() => (tour ? tour.coords.map((c) => c[1]) : []), [tour]);

  const maxD = useMemo(() => {
    if (!mst || mst.edges.length === 0) return 1;
    return Math.max(...mst.edges.map((e) => e[2]));
  }, [mst]);

  const { ranks, activeEdges } = useMemo(() => {
    if (!mst) return { ranks: [] as number[], activeEdges: [] as [number, number, number][] };
    const n = mst.ids.length;
    const uf = new UnionFind(n);
    const active: [number, number, number][] = [];
    for (const [a, b, d] of mst.edges) {
      if (d <= threshold) {
        uf.union(a, b);
        active.push([a, b, d]);
      }
    }
    const ids = Array.from({ length: n }, (_, i) => i);
    return { ranks: uf.componentRank(ids), activeEdges: active };
  }, [mst, threshold]);

  const componentCount = useMemo(() => {
    if (!mst) return 0;
    const nonSingleton = new Set(ranks.filter((r) => r >= 0));
    const singletons = ranks.filter((r) => r === -1).length;
    return nonSingleton.size + singletons;
  }, [ranks, mst]);

  const idIndex = useMemo(() => {
    if (!mst) return new Map<string, number>();
    return new Map(mst.ids.map((id, i) => [id, i]));
  }, [mst]);

  const seedIdx = mst ? idIndex.get(map.seed.track_id) : undefined;
  // The highlight set for drawing, distinct from the `recIds` prop, which is
  // the list sent to the server so those tracks are in the subset at all.
  const recSet = useMemo(() => new Set(map.recs.map((r) => r.track_id)), [map.recs]);

  useEffect(() => {
    if (!mst || !tour || size.width === 0 || size.height === 0) return;
    const canvas = canvasRef.current;
    if (!canvas) return;

    let raf = 0;
    const draw = () => {
      const c = canvasPalette();
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

      const t = fitTransform(xs, ys, size.width, size.height, PAD);

      // Edges under threshold
      ctx.strokeStyle = c.edge;
      ctx.lineWidth = 1;
      for (const [a, b] of activeEdges) {
        const [sxA, syA] = toScreen(t, xs[a], ys[a]);
        const [sxB, syB] = toScreen(t, xs[b], ys[b]);
        ctx.beginPath();
        ctx.moveTo(sxA, syA);
        ctx.lineTo(sxB, syB);
        ctx.stroke();
      }

      // Dots
      for (let i = 0; i < mst.ids.length; i++) {
        const rank = ranks[i];
        const [sx, sy] = toScreen(t, xs[i], ys[i]);
        ctx.fillStyle = rank === -1 ? "rgba(255,255,255,.25)" : `hsl(${(rank % 12) * 30}, 72%, 60%)`;
        ctx.beginPath();
        ctx.arc(sx, sy, 2.5, 0, Math.PI * 2);
        ctx.fill();
      }

      // Recs and seed on top
      for (let i = 0; i < mst.ids.length; i++) {
        if (!recSet.has(mst.ids[i])) continue;
        const [sx, sy] = toScreen(t, xs[i], ys[i]);
        if (selectedId === mst.ids[i]) {
          ctx.fillStyle = c.accentSoft;
          ctx.beginPath();
          ctx.arc(sx, sy, 11, 0, Math.PI * 2);
          ctx.fill();
        }
        ctx.fillStyle = c.accent;
        ctx.beginPath();
        ctx.arc(sx, sy, 4.5, 0, Math.PI * 2);
        ctx.fill();
      }

      if (seedIdx !== undefined) {
        const [sx, sy] = toScreen(t, xs[seedIdx], ys[seedIdx]);
        ctx.fillStyle = c.seedSofter;
        ctx.beginPath();
        ctx.arc(sx, sy, 18, 0, Math.PI * 2);
        ctx.fill();
        ctx.fillStyle = c.seedSoft;
        ctx.beginPath();
        ctx.arc(sx, sy, 9, 0, Math.PI * 2);
        ctx.fill();
        ctx.fillStyle = c.seed;
        ctx.beginPath();
        ctx.arc(sx, sy, 4, 0, Math.PI * 2);
        ctx.fill();
      }
    };

    raf = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(raf);
  }, [mst, tour, size, xs, ys, activeEdges, ranks, seedIdx, recSet, selectedId]);

  useEffect(() => {
    if (!mst) return;
    const canvas = barcodeRef.current;
    if (!canvas || size.width === 0) return;
    const c = canvasPalette();
    const height = 28;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = size.width * dpr;
    canvas.height = height * dpr;
    canvas.style.width = `${size.width}px`;
    canvas.style.height = `${height}px`;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = c.bg;
    ctx.fillRect(0, 0, size.width, height);

    ctx.strokeStyle = c.cyan;
    ctx.lineWidth = 1;
    for (const [, , d] of mst.edges) {
      const x = (d / maxD) * size.width;
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, height);
      ctx.stroke();
    }

    const cursorX = (threshold / maxD) * size.width;
    ctx.strokeStyle = c.seed;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(cursorX, 0);
    ctx.lineTo(cursorX, height);
    ctx.stroke();
  }, [mst, size, maxD, threshold]);

  if (status === "loading") {
    return <p className="hint">Loading topology…</p>;
  }
  if (status === "error" || !mst || !tour) {
    return (
      <div className="error-box">
        <p>Couldn&apos;t load the topology data.</p>
        <button type="button" onClick={() => void run()}>
          Try again
        </button>
      </div>
    );
  }

  return (
    <div className="topology">
      <div className="galaxy" ref={containerRef}>
        <canvas
          ref={canvasRef}
          onClick={(e) => {
            const rect = canvasRef.current?.getBoundingClientRect();
            if (!rect || size.width === 0 || size.height === 0) return;
            const px = e.clientX - rect.left;
            const py = e.clientY - rect.top;
            const t = fitTransform(xs, ys, size.width, size.height, PAD);
            const idx = nearestIndex(xs, ys, t, px, py, CLICK_MAX_DIST);
            onSelect(idx === -1 ? null : mst.ids[idx]);
          }}
        />
      </div>
      <canvas ref={barcodeRef} className="topology-barcode" />
      <input
        type="range"
        min={0}
        max={maxD}
        step={maxD / 1000}
        value={threshold}
        onChange={(e) => setThreshold(Number(e.target.value))}
        className="topology-slider"
        aria-label="MST distance threshold"
      />
      <p className="mono topology-caption">
        {componentCount} components at threshold {threshold.toFixed(3)}
      </p>
    </div>
  );
}
