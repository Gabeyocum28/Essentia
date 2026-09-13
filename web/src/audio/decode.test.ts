import { clearAudioCache, loadTrackAudio, rememberAnalysis, toMono } from "./decode";
import type { SoundAnalysis } from "./sound";

function stubBuffer(channels: number[][], sampleRate = 44100): AudioBuffer {
  return {
    numberOfChannels: channels.length,
    length: channels[0]?.length ?? 0,
    sampleRate,
    duration: (channels[0]?.length ?? 0) / sampleRate,
    getChannelData: (c: number) => Float32Array.from(channels[c]),
  } as unknown as AudioBuffer;
}

function stubContext(buffer: AudioBuffer) {
  return { decodeAudioData: vi.fn(async () => buffer) } as unknown as BaseAudioContext & {
    decodeAudioData: ReturnType<typeof vi.fn>;
  };
}

beforeEach(() => {
  clearAudioCache();
  globalThis.fetch = vi.fn(async () => ({
    ok: true,
    status: 200,
    arrayBuffer: async () => new ArrayBuffer(8),
  })) as unknown as typeof fetch;
});

test("fetches the same-origin audio proxy and decodes it to mono", async () => {
  const context = stubContext(stubBuffer([[1, 1, 1], [0, 0, 0]]));
  const audio = await loadTrackAudio("721063", { context });

  expect(globalThis.fetch).toHaveBeenCalledWith("/api/preview/721063/audio", { signal: undefined });
  expect(context.decodeAudioData).toHaveBeenCalledTimes(1);
  expect(Array.from(audio.samples)).toEqual([0.5, 0.5, 0.5]);
  expect(audio.sampleRate).toBe(44100);
});

test("caches per track id: a second load neither refetches nor re-decodes", async () => {
  const context = stubContext(stubBuffer([[1, 0]]));
  await loadTrackAudio("721063", { context });
  await loadTrackAudio("721063", { context });
  expect(globalThis.fetch).toHaveBeenCalledTimes(1);
  expect(context.decodeAudioData).toHaveBeenCalledTimes(1);
});

test("the cache holds 5 tracks; the 6th evicts the oldest", async () => {
  const context = stubContext(stubBuffer([[1, 0]]));
  for (const id of ["a", "b", "c", "d", "e", "f"]) await loadTrackAudio(id, { context });
  expect(globalThis.fetch).toHaveBeenCalledTimes(6);

  await loadTrackAudio("f", { context });
  expect(globalThis.fetch).toHaveBeenCalledTimes(6); // still cached
  await loadTrackAudio("a", { context });
  expect(globalThis.fetch).toHaveBeenCalledTimes(7); // evicted, refetched
});

test("a non-ok response throws", async () => {
  globalThis.fetch = vi.fn(async () => ({ ok: false, status: 404 })) as unknown as typeof fetch;
  await expect(loadTrackAudio("nope", { context: stubContext(stubBuffer([[0]])) })).rejects.toThrow(
    /404/,
  );
});

test("a remembered analysis comes back with the cached samples", async () => {
  const context = stubContext(stubBuffer([[1, 0]]));
  const analysis = { frameCount: 3 } as unknown as SoundAnalysis;

  await loadTrackAudio("721063", { context });
  rememberAnalysis("721063", analysis);

  const again = await loadTrackAudio("721063", { context });
  expect(again.analysis).toBe(analysis);
  expect(globalThis.fetch).toHaveBeenCalledTimes(1);
});

test("an evicted track loses its analysis with its samples", async () => {
  const context = stubContext(stubBuffer([[1, 0]]));
  await loadTrackAudio("a", { context });
  rememberAnalysis("a", { frameCount: 3 } as unknown as SoundAnalysis);
  for (const id of ["b", "c", "d", "e", "f"]) await loadTrackAudio(id, { context });

  expect((await loadTrackAudio("a", { context })).analysis).toBeUndefined();
});

test("remembering an analysis for an uncached track is a no-op", () => {
  expect(() => rememberAnalysis("never-loaded", {} as SoundAnalysis)).not.toThrow();
});

test("toMono averages channels", () => {
  expect(Array.from(toMono(stubBuffer([[1, 1], [-1, 3]])))).toEqual([0, 2]);
});
