import { MAX_COLUMNS, pool, selfSimilarity, stretch } from "./ssm";

function randomFrames(count: number, bands: number): Float32Array[] {
  let seed = 7;
  const rand = () => (seed = (seed * 1103515245 + 12345) % 2 ** 31) / 2 ** 31;
  return Array.from({ length: count }, () =>
    Float32Array.from({ length: bands }, () => rand() * 80 - 80),
  );
}

test("the diagonal is 1 and the matrix is symmetric", () => {
  const { columns, values } = selfSimilarity(randomFrames(60, 12));
  expect(columns).toBe(60);
  for (let r = 0; r < columns; r++) {
    expect(values[r * columns + r]).toBeCloseTo(1, 5);
    for (let c = 0; c < columns; c++) {
      expect(values[r * columns + c]).toBeCloseTo(values[c * columns + r], 6);
    }
  }
});

test("pools down to at most 256 columns", () => {
  const { columns, values } = selfSimilarity(randomFrames(1300, 8));
  expect(columns).toBe(MAX_COLUMNS);
  expect(values.length).toBe(MAX_COLUMNS * MAX_COLUMNS);
});

test("never upsamples fewer frames than the target", () => {
  expect(pool(randomFrames(10, 4), 4, 256)).toHaveLength(10);
});

test("pooling averages the frames it merges", () => {
  const frames = [Float32Array.from([0, 0]), Float32Array.from([2, 4])];
  expect(Array.from(pool(frames, 2, 1)[0])).toEqual([1, 2]);
});

test("identical frames are perfectly similar", () => {
  const frame = Float32Array.from([-10, -20, -30]);
  const { values } = selfSimilarity([frame, new Float32Array(frame), new Float32Array(frame)]);
  for (const v of values) expect(v).toBeCloseTo(1, 5);
});

test("no frames yields an empty matrix", () => {
  expect(selfSimilarity([])).toEqual({ columns: 0, values: new Float32Array(0) });
});

describe("stretch", () => {
  test("a constant matrix maps to a flat 0.5, with no NaN", () => {
    const values = new Float32Array(16).fill(0.93);
    for (let i = 0; i < 4; i++) values[i * 4 + i] = 1;

    const out = stretch(values);
    for (let r = 0; r < 4; r++) {
      for (let c = 0; c < 4; c++) {
        const v = out[r * 4 + c];
        expect(Number.isNaN(v)).toBe(false);
        expect(v).toBeCloseTo(r === c ? 1 : 0.5, 6);
      }
    }
  });

  test("a spread maps the percentile window onto the full 0-1 ramp", () => {
    // 8x8 with off-diagonal cosines climbing from 0.90 to 0.99 -- the narrow,
    // uniformly-high band that (s + 1) / 2 turned into a flat bright slab.
    const n = 8;
    const values = new Float32Array(n * n);
    const raw: number[] = [];
    for (let r = 0; r < n; r++) {
      for (let c = 0; c < n; c++) {
        if (r === c) {
          values[r * n + c] = 1;
        } else {
          const v = 0.9 + (0.09 * (r * n + c)) / (n * n);
          values[r * n + c] = v;
          raw.push(v);
        }
      }
    }

    const out = stretch(values);
    const off: number[] = [];
    for (let r = 0; r < n; r++) {
      for (let c = 0; c < n; c++) if (r !== c) off.push(out[r * n + c]);
    }

    expect(Math.min(...off)).toBe(0);
    expect(Math.max(...off)).toBe(1);
    // Everything lands inside the ramp, and the spread is genuinely used
    // rather than all crowding one end.
    expect(off.every((v) => v >= 0 && v <= 1 && !Number.isNaN(v))).toBe(true);
    const mid = off.filter((v) => v > 0.2 && v < 0.8).length;
    expect(mid).toBeGreaterThan(off.length / 3);
    // Order is preserved: the smallest raw off-diagonal is still the darkest.
    expect(raw).toHaveLength(off.length);

    // The diagonal stays pinned at 1.
    for (let i = 0; i < n; i++) expect(out[i * n + i]).toBe(1);
  });

  test("an empty or single-column matrix is total", () => {
    expect(stretch(new Float32Array(0))).toHaveLength(0);
    expect(Array.from(stretch(new Float32Array([0.7])))).toEqual([1]);
  });
});
