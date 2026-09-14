// `source` and `attribution_url` are the contract's optional Track fields
// (contract/features.py TRACK_OPTIONAL_FIELDS): a Creative Commons source
// requires a credit and a backlink, Deezer sends neither.
export interface Track { track_id: string; title: string; artist: string; album: string;
  artwork_url: string | null; preview_url: string | null; score?: number;
  source?: string; attribution_url?: string | null; }
export interface Axis { id: string; label: string; }
export interface SeedResponse { track_id: string; status: "ready" | "unanalyzed"; }
export interface RecommendResponse { seed_track_id: string; axis: string; results: Track[]; }
// contract/features.py RHYTHM_KEYS: the named, human-readable numbers CLAP
// cannot give. `key` is 0-11 with 0 = C; `key_strength` 0 means the key field
// means nothing.
export interface Rhythm { tempo_bpm: number; beat_strength: number;
  loudness_lufs: number; loudness_range: number;
  key: number; mode: string; key_strength: number; }
export interface ScoreMath { metric: string; dot: number; seed_norm: number; rec_norm: number;
  distance?: number | null; centrality?: number | null;
  feel_dist?: number | null; feel?: { seed: number[]; rec: number[] } | null;
  tempo_dist?: number | null; rhythm?: { seed: Rhythm; rec: Rhythm } | null; }
export interface VizPoint extends Track { x: number; y: number; }
export interface VizRec extends VizPoint { score: number; math: ScoreMath; }
export interface VizMap { points: { ids: string[]; x: number[]; y: number[]; tracks: Track[] };
  seed: VizPoint; recs: VizRec[]; axis: { id: string; metric: string; direction: number };
  feel_keys?: string[]; }
export interface VizWalk { path: VizPoint[]; geodesic: number; ambient: number; detour: number; k: number; }
export interface VizHistogram { bins: number[]; counts: number[]; rec_scores: number[]; percentile: number;
  null: { mean: number; sd: number }; corpus: { mean: number; sd: number }; }
export interface VizHubs { hubs: (Track & { count: number })[]; central: (Track & { centrality: number })[];
  isolated: (Track & { centrality: number })[]; expected_k: number; }
export interface VizTour { ids: string[]; coords: Float32Array[]; variance: number[]; }
export interface VizMst { ids: string[]; edges: [number, number, number][]; }
export interface VizExtremes { pc: number; variance_pct: number; low: Track[]; high: Track[]; }
export interface VizAttribution { status: "pending" | "ready" | "failed"; base?: number;
  bands?: { lo_hz: number; hi_hz: number; delta: number }[]; error?: string; }
