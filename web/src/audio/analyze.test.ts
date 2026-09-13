import { AnalysisAborted, analyzeSound } from "./analyze";

// jsdom has no Workers, so stand one in: it records that it was terminated,
// which is the whole point of cancelling on a track switch.
class StubWorker {
  static instances: StubWorker[] = [];
  onmessage: ((event: MessageEvent<unknown>) => void) | null = null;
  onerror: ((event: { message: string }) => void) | null = null;
  posted: unknown[] = [];
  terminated = false;

  constructor() {
    StubWorker.instances.push(this);
  }

  postMessage(message: unknown) {
    this.posted.push(message);
  }

  terminate() {
    this.terminated = true;
  }

  reply(data: unknown) {
    this.onmessage?.({ data } as MessageEvent<unknown>);
  }
}

const ANALYSIS = { frameCount: 2, bands: 96 };

beforeEach(() => {
  StubWorker.instances = [];
  globalThis.Worker = StubWorker as unknown as typeof Worker;
});

afterEach(() => {
  Reflect.deleteProperty(globalThis, "Worker");
});

test("hands the samples to a worker and resolves with its result", async () => {
  const promise = analyzeSound(new Float32Array(4096), 44100);
  const worker = StubWorker.instances[0];
  expect(worker.posted).toHaveLength(1);
  worker.reply(ANALYSIS);
  await expect(promise).resolves.toEqual(ANALYSIS);
  expect(worker.terminated).toBe(true);
});

test("aborting terminates the in-flight worker", async () => {
  const controller = new AbortController();
  const promise = analyzeSound(new Float32Array(4096), 44100, { signal: controller.signal });
  const worker = StubWorker.instances[0];

  controller.abort();
  await expect(promise).rejects.toBeInstanceOf(AnalysisAborted);
  expect(worker.terminated).toBe(true);
});

test("an already-aborted signal never starts a worker", async () => {
  const controller = new AbortController();
  controller.abort();
  await expect(
    analyzeSound(new Float32Array(4096), 44100, { signal: controller.signal }),
  ).rejects.toBeInstanceOf(AnalysisAborted);
  expect(StubWorker.instances).toHaveLength(0);
});

test("a worker error rejects", async () => {
  const promise = analyzeSound(new Float32Array(4096), 44100);
  StubWorker.instances[0].onerror?.({ message: "boom" });
  await expect(promise).rejects.toThrow("boom");
});

test("without Workers it computes inline", async () => {
  Reflect.deleteProperty(globalThis, "Worker");
  const samples = new Float32Array(4096);
  const result = await analyzeSound(samples, 44100);
  expect(result.bands).toBe(96);
  expect(result.frameCount).toBe(3);
});
