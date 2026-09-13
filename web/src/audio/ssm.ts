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

/** Percentile window the off-diagonal values are stretched across. */
const LOW_PCT = 0.02;
const HIGH_PCT = 0.98;

/**
 * Contrast-stretch an SSM for display: off-diagonal values map from their
 * 2nd–98th percentile onto 0–1, the diagonal stays 1.
 *
 * Pooled mel frames of one track are all fairly alike, so their cosines sit
 * in a narrow band near the top of [−1, 1] — the naive `(s + 1) / 2` sent
 * essentially every cell to the bright end of the colormap and the picture
 * was a uniform slab. Stretching across the percentile window (rather than
 * min–max) spends the whole ramp on the differences that are actually
 * there, without letting one outlier cell set the scale.
 *
 * Pure and total: a constant matrix has no spread, so every off-diagonal
 * cell becomes a mid-grey 0.5 rather than a NaN.
 */
export function stretch(values: Float32Array): Float32Array {
  const out = new Float32Array(values.length);
  if (values.length === 0) return out;
  const columns = Math.round(Math.sqrt(values.length));

  const offDiagonal: number[] = [];
  for (let r = 0; r < columns; r++) {
    for (let c = 0; c < columns; c++) {
      if (r !== c) offDiagonal.push(values[r * columns + c]);
    }
  }
  if (offDiagonal.length === 0) {
    out.fill(1);
    return out;
  }

  offDiagonal.sort((a, b) => a - b);
  const at = (p: number) =>
    offDiagonal[Math.max(0, Math.min(offDiagonal.length - 1, Math.round(p * (offDiagonal.length - 1))))];
  const lo = at(LOW_PCT);
  const hi = at(HIGH_PCT);
  const span = hi - lo;

  for (let r = 0; r < columns; r++) {
    for (let c = 0; c < columns; c++) {
      const i = r * columns + c;
      if (r === c) {
        out[i] = 1;
      } else if (span <= 0) {
        out[i] = 0.5; // no spread to show; a flat mid-tone, never a NaN
      } else {
        out[i] = Math.max(0, Math.min(1, (values[i] - lo) / span));
      }
    }
  }
  return out;
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
