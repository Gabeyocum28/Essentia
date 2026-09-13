import type * as T from "./types";

export class ApiError extends Error {
  status: number;
  detail: string;
  constructor(status: number, detail: string) {
    super(`${status}: ${detail}`);
    this.status = status;
    this.detail = detail;
  }
}

const BASE = "/api";

async function request<R>(path: string, init?: RequestInit): Promise<R> {
  const res = await fetch(BASE + path, { headers: { "content-type": "application/json" }, ...init });
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail ?? detail; } catch { /* non-JSON error body */ }
    throw new ApiError(res.status, detail);
  }
  return res.json() as Promise<R>;
}

const q = (params: Record<string, string | number | undefined>) =>
  "?" + new URLSearchParams(Object.entries(params).filter(([, v]) => v !== undefined)
    .map(([k, v]) => [k, String(v)])).toString();

// The subset the global viz endpoints build is seed-anchored, so a `surprise`
// rec can fall outside it and have no row to highlight. Passing the rec ids
// guarantees each one a row. undefined (not "") when there are none, so `q`
// drops the param entirely rather than sending an empty list.
const recList = (recs?: string[]) => (recs && recs.length ? recs.join(",") : undefined);

export const previewUrl = (trackId: string) => `${BASE}/preview/${trackId}`;

export function decodeCoords8(b64: string, n: number): Float32Array[] {
  const bytes = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
  const all = new Float32Array(bytes.buffer, bytes.byteOffset, n * 8);
  return Array.from({ length: n }, (_, i) => all.subarray(i * 8, i * 8 + 8));
}

export const api = {
  search: (query: string) => request<{ results: T.Track[] }>(`/search${q({ q: query })}`),
  seed: (track_id: string) => request<T.SeedResponse>("/seed", { method: "POST", body: JSON.stringify({ track_id }) }),
  axes: () => request<{ axes: T.Axis[] }>("/axes"),
  recommend: (track_id: string, axis: string, limit = 10) =>
    request<T.RecommendResponse>(`/recommend${q({ track_id, axis, limit })}`),
  vizMap: (track_id: string, axis: string, limit = 10, correction?: "on" | "off") =>
    request<T.VizMap>(`/viz/map${q({ track_id, axis, limit, correction })}`),
  vizWalk: (from: string, to: string, k = 8) => request<T.VizWalk>(`/viz/walk${q({ from, to, k })}`),
  vizHistogram: (track_id: string) => request<T.VizHistogram>(`/viz/histogram${q({ track_id })}`),
  vizHubs: (track_id?: string, recs?: string[]) =>
    request<T.VizHubs>(`/viz/hubs${q({ track_id, recs: recList(recs) })}`),
  vizTour: async (track_id?: string, recs?: string[]): Promise<T.VizTour> => {
    const raw = await request<{ ids: string[]; coords8: string; variance: number[] }>(
      `/viz/tour${q({ track_id, recs: recList(recs) })}`,
    );
    return { ids: raw.ids, coords: decodeCoords8(raw.coords8, raw.ids.length), variance: raw.variance };
  },
  vizMst: (track_id?: string, recs?: string[]) =>
    request<T.VizMst>(`/viz/mst${q({ track_id, recs: recList(recs) })}`),
  vizExtremes: (pc: number, limit = 4, track_id?: string, recs?: string[]) =>
    request<T.VizExtremes>(`/viz/extremes${q({ pc, limit, track_id, recs: recList(recs) })}`),
  vizAttribute: (seed: string, rec: string) => request<T.VizAttribution>(`/viz/attribute${q({ seed, rec })}`),
};
