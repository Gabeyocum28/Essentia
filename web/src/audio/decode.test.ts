import { clearAudioCache, loadTrackAudio, toMono } from "./decode";

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

test("toMono averages channels", () => {
  expect(Array.from(toMono(stubBuffer([[1, 1], [-1, 3]])))).toEqual([0, 2]);
});
