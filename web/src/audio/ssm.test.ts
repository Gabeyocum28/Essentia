import { MAX_COLUMNS, pool, selfSimilarity } from "./ssm";

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
