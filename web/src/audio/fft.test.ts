import { magnitudes, hann } from "./fft";

function tone(hz: number, sampleRate: number, n: number): Float32Array {
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = Math.sin((2 * Math.PI * hz * i) / sampleRate);
  return out;
}

test("a 440 Hz tone at 44.1 kHz peaks at bin 20 for N=2048", () => {
  // round(440 * 2048 / 44100) = 20
  const mag = magnitudes(tone(440, 44100, 2048), 0, 2048, hann(2048));
  let argmax = 0;
  for (let b = 1; b < mag.length; b++) if (mag[b] > mag[argmax]) argmax = b;
  expect(argmax).toBe(20);
});

test("returns size/2 bins", () => {
  expect(magnitudes(new Float32Array(2048), 0, 2048).length).toBe(1024);
});

test("silence has no energy", () => {
  const mag = magnitudes(new Float32Array(2048), 0, 2048, hann(2048));
  expect(Math.max(...mag)).toBe(0);
});

test("reads from the requested offset", () => {
  const samples = new Float32Array(4096);
  samples.set(tone(440, 44100, 2048), 2048);
  const mag = magnitudes(samples, 2048, 2048, hann(2048));
  let argmax = 0;
  for (let b = 1; b < mag.length; b++) if (mag[b] > mag[argmax]) argmax = b;
  expect(argmax).toBe(20);
});

test("rejects a non-power-of-two size", () => {
  expect(() => magnitudes(new Float32Array(100), 0, 100)).toThrow(/power of two/);
});
