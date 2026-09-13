export interface Track { track_id: string; title: string; artist: string; album: string;
  artwork_url: string | null; preview_url: string | null; score?: number; }
export interface Axis { id: string; label: string; }
export interface SeedResponse { track_id: string; status: "ready" | "unanalyzed"; }
export interface RecommendResponse { seed_track_id: string; axis: string; results: Track[]; }
export interface ScoreMath { metric: string; dot: number; seed_norm: number; rec_norm: number;
  distance?: number | null; centrality?: number | null;
  feel_dist?: number | null; feel?: { seed: number[]; rec: number[] } | null; }
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
