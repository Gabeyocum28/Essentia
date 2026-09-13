// The inferno-like heat ramp the iOS SOUND mode uses, so a spectrogram looks
// the same on both clients. Six stops, linearly interpolated in RGB.

export const STOPS: readonly (readonly [number, number, number])[] = [
  [0.0, 0.0, 0.02],
  [0.2, 0.03, 0.35],
  [0.55, 0.1, 0.42],
  [0.9, 0.35, 0.15],
  [0.99, 0.75, 0.2],
  [0.99, 0.98, 0.8],
];

/** `t` in [0, 1] (clamped) to r/g/b in [0, 1]. */
export function heat(t: number): [number, number, number] {
  const clamped = Math.max(0, Math.min(1, t));
  const scaled = clamped * (STOPS.length - 1);
  const i = Math.min(Math.floor(scaled), STOPS.length - 2);
  const f = scaled - i;
  const a = STOPS[i];
  const b = STOPS[i + 1];
  return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f];
}

/** The same ramp as 0–255 bytes, for writing straight into ImageData. */
export function heatBytes(t: number): [number, number, number] {
  const [r, g, b] = heat(t);
  return [Math.round(r * 255), Math.round(g * 255), Math.round(b * 255)];
}

/** Rec. 709 relative luminance of a stop, used to check the ramp is monotone. */
export const luminance = ([r, g, b]: [number, number, number]) =>
  0.2126 * r + 0.7152 * g + 0.0722 * b;
