import { describe, expect, test } from "vitest";
import { gaussianCurve, scaleBars } from "./histogram";

// Bin centres deliberately include 0 (and 0.4 for the mean test) so "nearest bin" is unambiguous.
function bins(lo: number, hi: number, width: number): number[] {
  const out: number[] = [];
  for (let x = lo; x <= hi + 1e-9; x += width) out.push(Math.round(x * 1e6) / 1e6);
  return out;
}

describe("gaussianCurve", () => {
  test("peaks at the bin nearest 0", () => {
    const b = bins(-1, 1, 0.1);
    const curve = gaussianCurve(b, 300, 0.1, 0.3);
    let peakIdx = 0;
    for (let i = 1; i < curve.length; i++) if (curve[i] > curve[peakIdx]) peakIdx = i;
    const nearestZero = b.reduce(
      (best, x, i) => (Math.abs(x) < Math.abs(b[best]) ? i : best),
      0,
    );
    expect(peakIdx).toBe(nearestZero);
    expect(b[peakIdx]).toBeCloseTo(0, 9);
  });

  test("integrates to approximately total", () => {
    // Each bin already holds total·binWidth·pdf (an expected count for that bin), so summing
    // the bins directly approximates `total` — the Riemann sum total·Σ(pdf·binWidth) ≈ total·1.
    const binWidth = 0.05;
    const b = bins(-1, 1, binWidth);
    const total = 500;
    const curve = gaussianCurve(b, total, binWidth, 0.3);
    const integral = curve.reduce((a, c) => a + c, 0);
    expect(Math.abs(integral - total) / total).toBeLessThan(0.02);
  });

  test("respects a non-zero mean", () => {
    const b = bins(-1, 1, 0.1);
    const curve = gaussianCurve(b, 300, 0.1, 0.3, 0.4);
    let peakIdx = 0;
    for (let i = 1; i < curve.length; i++) if (curve[i] > curve[peakIdx]) peakIdx = i;
    const nearestMean = b.reduce(
      (best, x, i) => (Math.abs(x - 0.4) < Math.abs(b[best] - 0.4) ? i : best),
      0,
    );
    expect(peakIdx).toBe(nearestMean);
    expect(b[peakIdx]).toBe(0.4);
  });
});

describe("scaleBars", () => {
  test("maps the max count to height", () => {
    const scaled = scaleBars([1, 5, 3, 5, 2], 40);
    expect(Math.max(...scaled)).toBeCloseTo(40, 6);
  });

  test("scales proportionally", () => {
    const scaled = scaleBars([2, 4], 10);
    expect(scaled[0]).toBeCloseTo(5, 6);
    expect(scaled[1]).toBeCloseTo(10, 6);
  });

  test("empty counts produce all zeros", () => {
    expect(scaleBars([], 40)).toEqual([]);
    expect(scaleBars([0, 0, 0], 40)).toEqual([0, 0, 0]);
  });
});
