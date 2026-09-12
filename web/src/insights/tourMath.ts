// Grand Tour: a smooth rotation of an 8-D point cloud onto a 2-D viewing plane, driven by a
// sequence of Givens rotations (one per pair of dimensions) whose angles advance at rates set
// by the first 28 primes. Starting the frame from the standard basis vectors e0/e1 and applying
// every (i, j) pair, i < j, in a fixed order keeps the two frame vectors orthonormal for all t
// (each Givens rotation is itself an orthogonal transform, and composing orthogonal transforms
// stays orthogonal).

const FIRST_28_PRIMES = [
  2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97,
  101, 103, 107,
];

/** All (i, j) pairs, i < j, for dimension `dim`, in row-major order — the fixed rotation order. */
function pairs(dim: number): [number, number][] {
  const out: [number, number][] = [];
  for (let i = 0; i < dim; i++) {
    for (let j = i + 1; j < dim; j++) {
      out.push([i, j]);
    }
  }
  return out;
}

/** Rotate both vectors in-place within the (i, j) plane by angle theta. */
function applyGivens(v0: Float32Array, v1: Float32Array, i: number, j: number, theta: number) {
  const cos = Math.cos(theta);
  const sin = Math.sin(theta);
  for (const v of [v0, v1]) {
    const vi = v[i];
    const vj = v[j];
    v[i] = cos * vi - sin * vj;
    v[j] = sin * vi + cos * vj;
  }
}

/**
 * Compute the Grand Tour frame at time t: two orthonormal `dim`-dimensional vectors, starting
 * from the standard basis e0/e1, after applying all C(dim, 2) Givens rotations in a fixed
 * (i, j) order, with angle theta_k = t * 0.05 * sqrt(prime_k) for the k-th pair.
 */
export function givensFrame(t: number, dim = 8): [Float32Array, Float32Array] {
  const v0 = new Float32Array(dim);
  const v1 = new Float32Array(dim);
  v0[0] = 1;
  v1[1] = 1;

  const ps = pairs(dim);
  for (let k = 0; k < ps.length; k++) {
    const [i, j] = ps[k];
    const prime = FIRST_28_PRIMES[k % FIRST_28_PRIMES.length];
    const theta = t * 0.05 * Math.sqrt(prime);
    applyGivens(v0, v1, i, j, theta);
  }

  return [v0, v1];
}

/** Project each `dim`-dimensional row in `coords` onto the 2-D frame via dot products. */
export function project(
  coords: Float32Array[],
  frame: [Float32Array, Float32Array],
): { x: Float32Array; y: Float32Array } {
  const [fx, fy] = frame;
  const n = coords.length;
  const x = new Float32Array(n);
  const y = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const row = coords[i];
    let dx = 0;
    let dy = 0;
    for (let d = 0; d < row.length; d++) {
      dx += row[d] * fx[d];
      dy += row[d] * fy[d];
    }
    x[i] = dx;
    y[i] = dy;
  }
  return { x, y };
}
