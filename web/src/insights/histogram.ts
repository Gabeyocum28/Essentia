// Gaussian null curve and bar-scaling helpers for the Proof-mode histogram.

/** total·binWidth·pdf(x; mean, sd), evaluated at each bin centre in `bins`. */
export function gaussianCurve(bins: number[], total: number, binWidth: number, sd: number, mean = 0): number[] {
  const coeff = 1 / (sd * Math.sqrt(2 * Math.PI));
  return bins.map((x) => {
    const z = (x - mean) / sd;
    const pdf = coeff * Math.exp(-0.5 * z * z);
    return total * binWidth * pdf;
  });
}

/** Scales `counts` so the largest count maps to `height`; all-zero (or empty) input stays zero. */
export function scaleBars(counts: number[], height: number): number[] {
  const max = counts.reduce((m, c) => Math.max(m, c), 0);
  if (max <= 0) return counts.map(() => 0);
  return counts.map((c) => (c / max) * height);
}
