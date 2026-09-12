import { describe, expect, test } from "vitest";
import { givensFrame, project } from "./tourMath";

function dot(a: Float32Array, b: Float32Array): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

function norm(a: Float32Array): number {
  return Math.sqrt(dot(a, a));
}

describe("givensFrame", () => {
  test.each([0, 1, 7.3])("produces two unit, orthogonal vectors at t = %s", (t) => {
    const [v0, v1] = givensFrame(t);
    expect(norm(v0)).toBeCloseTo(1, 5);
    expect(norm(v1)).toBeCloseTo(1, 5);
    expect(Math.abs(dot(v0, v1))).toBeLessThan(1e-5);
  });

  test("returns vectors of the requested dimension", () => {
    const [v0, v1] = givensFrame(1, 4);
    expect(v0.length).toBe(4);
    expect(v1.length).toBe(4);
  });
});

describe("project", () => {
  test("projection of a unit-norm point has norm <= 1", () => {
    const frame = givensFrame(2.5);
    const point = new Float32Array(8);
    point[3] = 1; // unit norm along one axis
    const { x, y } = project([point], frame);
    const projNorm = Math.hypot(x[0], y[0]);
    expect(projNorm).toBeLessThanOrEqual(1 + 1e-6);
  });

  test("projects multiple rows", () => {
    const frame = givensFrame(0);
    const a = new Float32Array(8);
    a[0] = 1;
    const b = new Float32Array(8);
    b[1] = 1;
    const { x, y } = project([a, b], frame);
    // At t = 0 the frame is exactly e0, e1, so projection is the identity on those axes.
    expect(x[0]).toBeCloseTo(1, 5);
    expect(y[0]).toBeCloseTo(0, 5);
    expect(x[1]).toBeCloseTo(0, 5);
    expect(y[1]).toBeCloseTo(1, 5);
  });
});
