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

// ---- the feel weight ----
//
// The DEFAULT lives on the SERVER (app.py's FEEL_DEFAULT). This constant is
// only what the slider starts at and what a missing stored value falls back
// to; `feelParam` below OMITS the query parameter when the value equals it,
// so a change on the server takes effect without shipping a new bundle.
export const DEFAULT_FEEL = 0.3;
export const FEEL_MIN = 0;
export const FEEL_MAX = 2;
export const FEEL_STORAGE_KEY = "essentia.feel";

/** A user-supplied weight, clamped to the slider's range; DEFAULT_FEEL if unusable. */
export function clampFeel(value: unknown): number {
  // Number(null) and Number("") are both 0, which would read a missing value
  // as "turn the blend off" rather than as "use the default".
  if (value === null || value === undefined || value === "") return DEFAULT_FEEL;
  const n = Number(value);
  if (!Number.isFinite(n)) return DEFAULT_FEEL;
  return Math.min(FEEL_MAX, Math.max(FEEL_MIN, n));
}

/** The weight the user last chose on the Recommendations slider. */
export function loadStoredFeel(): number {
  try {
    const raw = localStorage.getItem(FEEL_STORAGE_KEY);
    return raw === null ? DEFAULT_FEEL : clampFeel(raw);
  } catch {
    return DEFAULT_FEEL;
  }
}

export function storeFeel(value: number): void {
  try {
    localStorage.setItem(FEEL_STORAGE_KEY, String(value));
  } catch {
    /* localStorage unavailable (private mode, blocked site data) */
  }
}

// undefined at the default, so `q` drops the param and the server's own
// FEEL_DEFAULT is what answers.
const feelParam = (feel?: number) =>
  feel === undefined || feel === DEFAULT_FEEL ? undefined : feel;

// ---- the tempo weight ----
//
// The same arrangement as the feel weight above, for the same reason: the
// DEFAULT lives on the server (app.py's TEMPO_DEFAULT) and `tempoParam`
// omits the parameter at that value, so it can be retuned without shipping
// a bundle. Separate storage key, because the two sliders are separate
// decisions -- "same mood" and "same speed" are not the same request.
export const DEFAULT_TEMPO = 0.2;
export const TEMPO_MIN = 0;
export const TEMPO_MAX = 2;
export const TEMPO_STORAGE_KEY = "essentia.tempo";

/** A user-supplied weight, clamped to the slider's range; DEFAULT_TEMPO if unusable. */
export function clampTempo(value: unknown): number {
  if (value === null || value === undefined || value === "") return DEFAULT_TEMPO;
  const n = Number(value);
  if (!Number.isFinite(n)) return DEFAULT_TEMPO;
  return Math.min(TEMPO_MAX, Math.max(TEMPO_MIN, n));
}

/** The weight the user last chose on the Recommendations slider. */
export function loadStoredTempo(): number {
  try {
    const raw = localStorage.getItem(TEMPO_STORAGE_KEY);
    return raw === null ? DEFAULT_TEMPO : clampTempo(raw);
  } catch {
    return DEFAULT_TEMPO;
  }
}

export function storeTempo(value: number): void {
  try {
    localStorage.setItem(TEMPO_STORAGE_KEY, String(value));
  } catch {
    /* localStorage unavailable (private mode, blocked site data) */
  }
}

const tempoParam = (tempo?: number) =>
  tempo === undefined || tempo === DEFAULT_TEMPO ? undefined : tempo;

export const previewUrl = (trackId: string) => `${BASE}/preview/${trackId}`;

// Same-origin mp3 bytes, for SOUND mode only: decodeAudioData needs the
// bytes, and the plain preview URL 302s to a CDN with no CORS header.
export const previewAudioUrl = (trackId: string) => `${BASE}/preview/${trackId}/audio`;

export function decodeCoords8(b64: string, n: number): Float32Array[] {
  const bytes = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
  const all = new Float32Array(bytes.buffer, bytes.byteOffset, n * 8);
  return Array.from({ length: n }, (_, i) => all.subarray(i * 8, i * 8 + 8));
}

export const api = {
  search: (query: string) => request<{ results: T.Track[] }>(`/search${q({ q: query })}`),
  seed: (track_id: string) => request<T.SeedResponse>("/seed", { method: "POST", body: JSON.stringify({ track_id }) }),
  axes: () => request<{ axes: T.Axis[] }>("/axes"),
  // Search the CORPUS by description rather than the catalogue by name: the
  // phrase is embedded by CLAP's text tower and cosined against every
  // analyzed track. 503 (an ApiError, with the server's detail) when CLAP is
  // not loadable on the API host.
  searchText: (query: string, limit?: number) =>
    request<{ results: T.Track[] }>(`/search/text${q({ q: query, limit })}`),
  recommend: (track_id: string, axis: string, limit = 10, feel?: number, tempo?: number) =>
    request<T.RecommendResponse>(
      `/recommend${q({ track_id, axis, limit, feel: feelParam(feel), tempo: tempoParam(tempo) })}`),
  vizMap: (track_id: string, axis: string, limit = 10, correction?: "on" | "off",
           feel?: number, tempo?: number) =>
    request<T.VizMap>(
      `/viz/map${q({ track_id, axis, limit, correction, feel: feelParam(feel), tempo: tempoParam(tempo) })}`),
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
