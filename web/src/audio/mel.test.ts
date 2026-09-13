import { BANDS, DB_FLOOR, FFT_SIZE, HOP, bandEdges, formatHz, melSpectrogram } from "./mel";

const SR = 44100;

function tone(hz: number, n: number): Float32Array {
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = Math.sin((2 * Math.PI * hz * i) / SR);
  return out;
}

test("silence floors every band at −80 dB", () => {
  const { frames } = melSpectrogram(new Float32Array(SR), SR);
  expect(frames.length).toBeGreaterThan(0);
  for (const frame of frames) for (const v of frame) expect(v).toBe(DB_FLOOR);
});

test("frame count is floor((n − 2048) / 1024) + 1", () => {
  const n = SR; // 44100 samples
  const { frames } = melSpectrogram(new Float32Array(n), SR);
  expect(frames.length).toBe(Math.floor((n - FFT_SIZE) / HOP) + 1);
});

test("a 440 Hz tone's loudest band spans 440 Hz", () => {
  const { frames, bands, bandEdgesHz } = melSpectrogram(tone(440, SR), SR);
  const frame = frames[Math.floor(frames.length / 2)];
  let argmax = 0;
  for (let b = 1; b < bands; b++) if (frame[b] > frame[argmax]) argmax = b;
  expect(bandEdgesHz[argmax]).toBeLessThanOrEqual(440);
  expect(bandEdgesHz[argmax + 2]).toBeGreaterThanOrEqual(440);
});

test("peak is the loudest dB anywhere, and silence peaks at the floor", () => {
  const loud = melSpectrogram(tone(440, SR), SR);
  expect(loud.peak).toBeGreaterThan(DB_FLOOR);
  const maxSeen = Math.max(...loud.frames.map((f) => Math.max(...f)));
  expect(loud.peak).toBeCloseTo(maxSeen, 4); // peak is kept in double, frames in float32
  expect(melSpectrogram(new Float32Array(SR), SR).peak).toBe(DB_FLOOR);
});

test("too-short input yields no frames rather than throwing", () => {
  const { frames, peak } = melSpectrogram(new Float32Array(100), SR);
  expect(frames).toEqual([]);
  expect(peak).toBe(DB_FLOOR);
});

test("band edges are ascending and cover up to Nyquist", () => {
  const edges = bandEdges(BANDS, SR);
  expect(edges.length).toBe(BANDS + 2);
  for (let i = 1; i < edges.length; i++) expect(edges[i]).toBeGreaterThan(edges[i - 1]);
  expect(edges[edges.length - 1]).toBeCloseTo(SR / 2, 3);
});

test("formatHz reads as Hz below 1k and kHz above", () => {
  expect(formatHz(240)).toBe("240");
  expect(formatHz(1200)).toBe("1.2k");
});
