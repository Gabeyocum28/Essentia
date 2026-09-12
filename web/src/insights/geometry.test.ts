import { describe, expect, test } from "vitest";
import { applyZoomPan, fitTransform, nearestIndex, toScreen, zoomAbout } from "./geometry";

describe("fitTransform", () => {
  test("maps min -> pad and max -> width - pad on the limiting axis", () => {
    const xs = [0, 10];
    const ys = [0, 10];
    const width = 100;
    const height = 100;
    const pad = 10;
    const t = fitTransform(xs, ys, width, height, pad);

    const [minSx] = toScreen(t, 0, 0);
    const [maxSx] = toScreen(t, 10, 10);
    expect(minSx).toBeCloseTo(pad);
    expect(maxSx).toBeCloseTo(width - pad);

    const [, minSy] = toScreen(t, 0, 0);
    const [, maxSy] = toScreen(t, 10, 10);
    expect(minSy).toBeCloseTo(pad);
    expect(maxSy).toBeCloseTo(height - pad);
  });

  test("handles non-square data ranges by fitting the limiting axis", () => {
    const xs = [0, 100];
    const ys = [0, 10];
    const width = 200;
    const height = 200;
    const pad = 0;
    const t = fitTransform(xs, ys, width, height, pad);

    const [minSx] = toScreen(t, 0, 0);
    const [maxSx] = toScreen(t, 100, 0);
    expect(minSx).toBeCloseTo(0);
    expect(maxSx).toBeCloseTo(200);
  });
});

describe("applyZoomPan", () => {
  test("zooming about a centre keeps that centre fixed", () => {
    const t = fitTransform([0, 10], [0, 10], 100, 100, 10);
    const cx = 50;
    const cy = 50;
    const zoomed = applyZoomPan(t, 3, 0, 0, cx, cy);

    // A point that maps to the centre under t should still map to the centre after zoom.
    // Solve for data point mapping to (cx, cy) under t, then check it still maps there.
    const dataX = (cx - t.tx) / t.sx;
    const dataY = (cy - t.ty) / t.sy;
    const [sx, sy] = toScreen(zoomed, dataX, dataY);
    expect(sx).toBeCloseTo(cx);
    expect(sy).toBeCloseTo(cy);
  });

  test("pan shifts the transform by the given screen offset", () => {
    const t = fitTransform([0, 10], [0, 10], 100, 100, 10);
    const panned = applyZoomPan(t, 1, 20, -5, 50, 50);
    const [sx0, sy0] = toScreen(t, 5, 5);
    const [sx1, sy1] = toScreen(panned, 5, 5);
    expect(sx1 - sx0).toBeCloseTo(20);
    expect(sy1 - sy0).toBeCloseTo(-5);
  });
});

describe("zoomAbout", () => {
  test("keeps the data point under the cursor fixed on the screen after a 2x zoom", () => {
    const t = fitTransform([0, 10], [0, 10], 100, 100, 10);
    const cx = 50;
    const cy = 50;
    const zoom = 1;
    const pan = { x: 3, y: -4 };

    // Pick an arbitrary cursor point and compute the screen position under it before zooming.
    const mx = 30;
    const my = 65;
    const before = applyZoomPan(t, zoom, pan.x, pan.y, cx, cy);
    // The data point currently under the cursor, inverted from the pre-zoom transform.
    const dataX = (mx - before.tx) / before.sx;
    const dataY = (my - before.ty) / before.sy;

    const { zoom: zoom2, pan: pan2 } = zoomAbout(t, zoom, pan, 2, mx, my, cx, cy);
    expect(zoom2).toBeCloseTo(2);
    const after = applyZoomPan(t, zoom2, pan2.x, pan2.y, cx, cy);
    const [sx, sy] = toScreen(after, dataX, dataY);
    expect(sx).toBeCloseTo(mx);
    expect(sy).toBeCloseTo(my);
  });

  test("keeps the data point under the cursor fixed on the screen after a 0.5x zoom", () => {
    const t = fitTransform([0, 10], [0, 10], 100, 100, 10);
    const cx = 50;
    const cy = 50;
    const zoom = 4;
    const pan = { x: -10, y: 6 };

    const mx = 70;
    const my = 20;
    const before = applyZoomPan(t, zoom, pan.x, pan.y, cx, cy);
    const dataX = (mx - before.tx) / before.sx;
    const dataY = (my - before.ty) / before.sy;

    const { zoom: zoomHalf, pan: panHalf } = zoomAbout(t, zoom, pan, 0.5, mx, my, cx, cy);
    expect(zoomHalf).toBeCloseTo(2);
    const after = applyZoomPan(t, zoomHalf, panHalf.x, panHalf.y, cx, cy);
    const [sx, sy] = toScreen(after, dataX, dataY);
    expect(sx).toBeCloseTo(mx);
    expect(sy).toBeCloseTo(my);
  });

  test("zoom stays clamped to [1, 8]", () => {
    const t = fitTransform([0, 10], [0, 10], 100, 100, 10);
    const low = zoomAbout(t, 1, { x: 0, y: 0 }, 0.1, 50, 50, 50, 50);
    expect(low.zoom).toBe(1);
    const high = zoomAbout(t, 8, { x: 0, y: 0 }, 10, 50, 50, 50, 50);
    expect(high.zoom).toBe(8);
  });
});

describe("nearestIndex", () => {
  test("returns the closest point index within maxDist", () => {
    const t = fitTransform([0, 10], [0, 10], 100, 100, 10);
    const xs = [0, 5, 10];
    const ys = [0, 5, 10];
    const [px, py] = toScreen(t, 5, 5);
    expect(nearestIndex(xs, ys, t, px, py, 36)).toBe(1);
  });

  test("returns -1 beyond maxDist", () => {
    const t = fitTransform([0, 10], [0, 10], 100, 100, 10);
    const xs = [0, 10];
    const ys = [0, 10];
    expect(nearestIndex(xs, ys, t, 500, 500, 36)).toBe(-1);
  });

  test("returns -1 for an empty point set", () => {
    const t = fitTransform([0, 1], [0, 1], 100, 100, 10);
    expect(nearestIndex([], [], t, 50, 50, 36)).toBe(-1);
  });
});
