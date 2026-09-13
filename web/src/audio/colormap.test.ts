import { STOPS, heat, heatBytes, luminance } from "./colormap";

test("t = 0 is nearly black and t = 1 is nearly white", () => {
  expect(luminance(heat(0))).toBeLessThan(0.05);
  expect(luminance(heat(1))).toBeGreaterThan(0.9);
});

test("luminance rises monotonically across the ramp", () => {
  let previous = -1;
  for (let i = 0; i <= 100; i++) {
    const l = luminance(heat(i / 100));
    expect(l).toBeGreaterThan(previous);
    previous = l;
  }
});

test("hits each declared stop exactly", () => {
  STOPS.forEach((stop, i) => {
    const got = heat(i / (STOPS.length - 1));
    got.forEach((v, c) => expect(v).toBeCloseTo(stop[c], 6));
  });
});

test("clamps outside [0, 1]", () => {
  expect(heat(-5)).toEqual(heat(0));
  expect(heat(5)).toEqual(heat(1));
});

test("heatBytes are 0–255 integers", () => {
  for (const v of heatBytes(0.37)) {
    expect(Number.isInteger(v)).toBe(true);
    expect(v).toBeGreaterThanOrEqual(0);
    expect(v).toBeLessThanOrEqual(255);
  }
});
