import { SOLO_Q, createBandSolo, type AudioGraphContext } from "./bandSolo";

interface StubParam {
  value: number;
  setTargetAtTime(value: number, when: number, tc: number): void;
}

function param(initial = 0): StubParam {
  return {
    value: initial,
    setTargetAtTime(value: number) {
      this.value = value;
    },
  };
}

function stubContext() {
  const filters: { type: string; frequency: StubParam; Q: StubParam; connect: unknown }[] = [];
  const gains: { gain: StubParam; connect: unknown }[] = [];
  /** Every connect/disconnect in order, e.g. ["source", "destination"]. */
  const connections: [string, string][] = [];
  const disconnections: [string, string][] = [];
  const names = new Map<object, string>();
  const name = (node: object) => names.get(node) ?? "unknown";
  const label = (to: object) => (to === context.destination ? "destination" : name(to));

  const connect = (from: object) => (to: object) => {
    connections.push([name(from), label(to)]);
  };

  const context = {
    destination: { id: "destination" },
    currentTime: 0,
    createBiquadFilter() {
      const node = { type: "", frequency: param(1), Q: param(1), connect: (_to: object) => {} };
      node.connect = connect(node);
      names.set(node, `filter${filters.length}`);
      filters.push(node);
      return node as unknown as BiquadFilterNode;
    },
    createGain() {
      const node = { gain: param(1), connect: (_to: object) => {} };
      node.connect = connect(node);
      names.set(node, `gain${gains.length}`);
      gains.push(node);
      return node as unknown as GainNode;
    },
  };

  const source = {
    connect: (_to: object) => {},
    disconnect: (to: object) => {
      disconnections.push([name(source), label(to)]);
    },
  };
  source.connect = connect(source);
  names.set(source, "source");

  return {
    context: context as unknown as AudioGraphContext,
    source,
    filters,
    gains,
    connections,
    disconnections,
  };
}

test("builds the filter chain once, alongside the graph's direct connection", () => {
  const { context, source, filters, gains, connections } = stubContext();
  createBandSolo(context, source);

  expect(filters).toHaveLength(4);
  expect(gains).toHaveLength(1); // no bypass gain: the direct source -> destination is the bypass
  expect(filters.map((f) => f.type)).toEqual(["highpass", "highpass", "lowpass", "lowpass"]);
  for (const f of filters) expect(f.Q.value).toBeCloseTo(SOLO_Q, 6);

  expect(connections).toEqual([
    ["source", "filter0"],
    ["filter0", "filter1"],
    ["filter1", "filter2"],
    ["filter2", "filter3"],
    ["filter3", "gain0"],
    ["gain0", "destination"],
  ]);
});

test("starts silent on the filter path, leaving the direct connection audible", () => {
  const { context, source, gains, disconnections } = stubContext();
  const solo = createBandSolo(context, source);
  expect(gains[0].gain.value).toBe(0);
  expect(disconnections).toEqual([]);
  expect(solo.band()).toBeNull();
});

test("setBand updates all four filter frequencies and takes over from the direct path", () => {
  const { context, source, filters, gains, disconnections } = stubContext();
  const solo = createBandSolo(context, source);
  solo.setBand(240, 1200);

  expect(filters.map((f) => f.frequency.value)).toEqual([240, 240, 1200, 1200]);
  expect(gains[0].gain.value).toBe(1);
  expect(disconnections).toEqual([["source", "destination"]]);
  expect(solo.band()).toEqual([240, 1200]);
});

test("setBand re-tunes the same nodes rather than building more, and disconnects once", () => {
  const { context, source, filters, gains, disconnections } = stubContext();
  const solo = createBandSolo(context, source);
  solo.setBand(240, 1200);
  solo.setBand(2000, 6000);
  expect(filters).toHaveLength(4);
  expect(gains).toHaveLength(1);
  expect(filters.map((f) => f.frequency.value)).toEqual([2000, 2000, 6000, 6000]);
  expect(disconnections).toHaveLength(1);
});

test("a reversed or degenerate range is normalized", () => {
  const { context, source, filters } = stubContext();
  const solo = createBandSolo(context, source);
  solo.setBand(1200, 240);
  expect(filters.map((f) => f.frequency.value)).toEqual([240, 240, 1200, 1200]);
  solo.setBand(500, 500);
  expect(solo.band()).toEqual([500, 501]);
});

test("clear() mutes the filters and restores the direct connection", () => {
  const { context, source, gains, connections } = stubContext();
  const solo = createBandSolo(context, source);
  solo.setBand(240, 1200);
  solo.clear();
  expect(gains[0].gain.value).toBe(0);
  expect(connections.at(-1)).toEqual(["source", "destination"]);
  expect(solo.band()).toBeNull();
});

test("clear() on an already-bypassed chain doesn't double-connect", () => {
  const { context, source, connections } = stubContext();
  const solo = createBandSolo(context, source);
  const before = connections.length;
  solo.clear();
  expect(connections).toHaveLength(before);
});
