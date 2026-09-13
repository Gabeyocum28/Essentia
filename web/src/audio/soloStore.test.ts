import { attachGraph } from "../player/usePlayer";
import { currentBand, resetSolo, soloBand, soloError, soloOff } from "./soloStore";

vi.mock("../player/usePlayer", () => ({ attachGraph: vi.fn() }));

function fakeGraph() {
  const param = () => ({ value: 0, setTargetAtTime(v: number) { this.value = v; } });
  const made = { filters: 0, gains: 0, disconnects: 0, connects: 0 };
  const context = {
    destination: {},
    currentTime: 0,
    createBiquadFilter() {
      made.filters++;
      return { type: "", frequency: param(), Q: param(), connect: () => {} };
    },
    createGain() {
      made.gains++;
      return { gain: param(), connect: () => {} };
    },
  };
  const source = {
    connect: () => {
      made.connects++;
    },
    disconnect: () => {
      made.disconnects++;
    },
  };
  return { made, graph: { context, source } };
}

function mockAttach(graph: unknown) {
  vi.mocked(attachGraph).mockResolvedValue(graph as Awaited<ReturnType<typeof attachGraph>>);
}

beforeEach(() => {
  resetSolo();
  vi.mocked(attachGraph).mockReset();
});

test("builds the chain on the first solo and reuses it after", async () => {
  const { made, graph } = fakeGraph();
  mockAttach(graph);

  await soloBand(240, 1200);
  expect(currentBand()).toEqual([240, 1200]);
  await soloBand(2000, 6000);
  expect(currentBand()).toEqual([2000, 6000]);

  expect(made.filters).toBe(4); // one graph, not two
  expect(made.gains).toBe(1);
  expect(attachGraph).toHaveBeenCalledTimes(1);
});

test("a burst of solos while attaching builds one chain and applies the last band", async () => {
  const { made, graph } = fakeGraph();
  mockAttach(graph);

  await Promise.all([soloBand(100, 200), soloBand(300, 400), soloBand(500, 600)]);

  expect(made.filters).toBe(4);
  expect(attachGraph).toHaveBeenCalledTimes(1);
  expect(currentBand()).toEqual([500, 600]);
});

test("soloOff clears the band", async () => {
  const { graph } = fakeGraph();
  mockAttach(graph);
  await soloBand(240, 1200);
  soloOff();
  expect(currentBand()).toBeNull();
});

test("soloOff during the attach wins over the in-flight solo", async () => {
  const { graph } = fakeGraph();
  mockAttach(graph);
  const pending = soloBand(240, 1200);
  soloOff();
  await pending;
  expect(currentBand()).toBeNull();
});

test("no Web Audio means no solo, not a crash", async () => {
  mockAttach(null);
  await soloBand(240, 1200);
  expect(currentBand()).toBeNull();
});

test("no Web Audio is not an error message, just no solo", async () => {
  mockAttach(null);
  await soloBand(240, 1200);
  expect(soloError()).toBeNull();
});

test("a failed attach becomes a message, not an unhandled rejection", async () => {
  vi.mocked(attachGraph).mockRejectedValue(new Error("timed out loading the audio proxy"));

  await expect(soloBand(240, 1200)).resolves.toBeUndefined();
  expect(currentBand()).toBeNull();
  expect(soloError()).toBe("Couldn't enable band solo.");

  // The next attempt gets to try again, and a success clears the message.
  const { graph } = fakeGraph();
  mockAttach(graph);
  await soloBand(240, 1200);
  expect(soloError()).toBeNull();
  expect(currentBand()).toEqual([240, 1200]);
});

test("soloOff clears a failure message too", async () => {
  vi.mocked(attachGraph).mockRejectedValue(new Error("nope"));
  await soloBand(240, 1200);
  expect(soloError()).not.toBeNull();
  soloOff();
  expect(soloError()).toBeNull();
});
