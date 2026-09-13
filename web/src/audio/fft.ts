// Radix-2 FFT, magnitudes only. Small enough to keep in the app rather than
// pull a dependency in: one 2048-point transform per 1024-sample hop over a
// 30 s preview is ~1300 transforms, which the worker chews through in well
// under a second.

/** Hann window of length `n` (periodic-free "denormalized" form, matching iOS). */
export function hann(n: number): Float32Array {
  const w = new Float32Array(n);
  for (let i = 0; i < n; i++) w[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (n - 1)));
  return w;
}

function bitReverse(re: Float64Array, im: Float64Array, n: number): void {
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      let t = re[i]; re[i] = re[j]; re[j] = t;
      t = im[i]; im[i] = im[j]; im[j] = t;
    }
  }
}

/**
 * In-place complex FFT of `re`/`im` (length must be a power of two).
 * Exported for tests; callers usually want `magnitudes`.
 */
export function fftInPlace(re: Float64Array, im: Float64Array): void {
  const n = re.length;
  if (n <= 1) return;
  if ((n & (n - 1)) !== 0) throw new Error(`fft size ${n} is not a power of two`);
  bitReverse(re, im, n);
  for (let len = 2; len <= n; len <<= 1) {
    const ang = (-2 * Math.PI) / len;
    const wRe = Math.cos(ang);
    const wIm = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let curRe = 1;
      let curIm = 0;
      for (let k = 0; k < len / 2; k++) {
        const a = i + k;
        const b = a + len / 2;
        const tRe = re[b] * curRe - im[b] * curIm;
        const tIm = re[b] * curIm + im[b] * curRe;
        re[b] = re[a] - tRe;
        im[b] = im[a] - tIm;
        re[a] += tRe;
        im[a] += tIm;
        const nextRe = curRe * wRe - curIm * wIm;
        curIm = curRe * wIm + curIm * wRe;
        curRe = nextRe;
      }
    }
  }
}

/**
 * Magnitude spectrum of a real signal: `size / 2` bins, bin b centered at
 * `b * sampleRate / size` Hz. `samples` must be at least `size` long from
 * `offset`; a `window` of length `size` is applied when given.
 */
export function magnitudes(
  samples: Float32Array,
  offset: number,
  size: number,
  window?: Float32Array,
): Float64Array {
  const re = new Float64Array(size);
  const im = new Float64Array(size);
  for (let i = 0; i < size; i++) {
    const s = samples[offset + i] ?? 0;
    re[i] = window ? s * window[i] : s;
  }
  fftInPlace(re, im);
  const half = size / 2;
  const out = new Float64Array(half);
  for (let b = 0; b < half; b++) out[b] = Math.hypot(re[b], im[b]);
  return out;
}
