// Self-similarity ("recurrence") matrix, same recipe as iOS
// SelfSimilarityMatrix.swift: average-pool the mel frames down to <=256 time
// columns, L2-normalize each column across bands, take the cosine Gram
// matrix. Repeating structure shows up as bright off-diagonal blocks.

export const MAX_COLUMNS = 256;

export interface Ssm {
  columns: number;
  /** Row-major `columns × columns` cosine similarities in [−1, 1]. */
  values: Float32Array;
}

/** Average-pool frames in time to at most `target` columns. Never upsamples. */
export function pool(frames: Float32Array[], bands: number, target: number): Float32Array[] {
  const total = frames.length;
  if (total <= target) return frames;
  const pooled: Float32Array[] = [];
  for (let col = 0; col < target; col++) {
    const start = Math.floor((total * col) / target);
    const end = Math.max(start + 1, Math.floor((total * (col + 1)) / target));
    const sum = new Float32Array(bands);
    for (let f = start; f < end; f++) {
      const frame = frames[f];
      for (let b = 0; b < bands; b++) sum[b] += frame[b];
    }
    const n = end - start;
    for (let b = 0; b < bands; b++) sum[b] /= n;
    pooled.push(sum);
  }
  return pooled;
}

export function selfSimilarity(frames: Float32Array[], maxColumns = MAX_COLUMNS): Ssm {
  const bands = frames[0]?.length ?? 0;
  if (!bands || frames.length === 0) return { columns: 0, values: new Float32Array(0) };

  const pooled = pool(frames, bands, maxColumns).map((frame) => {
    const out = new Float32Array(frame);
    let sq = 0;
    for (let b = 0; b < bands; b++) sq += out[b] * out[b];
    const mag = Math.sqrt(sq);
    if (mag > 0) for (let b = 0; b < bands; b++) out[b] /= mag;
    return out;
  });

  const columns = pooled.length;
  const values = new Float32Array(columns * columns);
  for (let r = 0; r < columns; r++) {
    for (let c = r; c < columns; c++) {
      let dot = 0;
      const a = pooled[r];
      const b = pooled[c];
      for (let i = 0; i < bands; i++) dot += a[i] * b[i];
      values[r * columns + c] = dot;
      values[c * columns + r] = dot;
    }
  }
  return { columns, values };
}
