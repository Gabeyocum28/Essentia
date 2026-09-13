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
  const connections: [string, string][] = [];
  const names = new Map<object, string>();
  const name = (node: object) => names.get(node) ?? "unknown";

  const connect = (from: object) => (to: object) => {
    connections.push([name(from), to === context.destination ? "destination" : name(to)]);
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

  const source = { connect: (_to: object) => {} };
  source.connect = connect(source);
  names.set(source, "source");

  return { context: context as unknown as AudioGraphContext, source, filters, gains, connections };
}

test("builds the filter chain once, wired to destination twice", () => {
  const { context, source, filters, gains, connections } = stubContext();
  createBandSolo(context, source);

  expect(filters).toHaveLength(4);
  expect(gains).toHaveLength(2);
  expect(filters.map((f) => f.type)).toEqual(["highpass", "highpass", "lowpass", "lowpass"]);
  for (const f of filters) expect(f.Q.value).toBeCloseTo(SOLO_Q, 6);

  // source -> hp -> hp -> lp -> lp -> bandGain -> destination, plus the bypass.
  expect(connections).toEqual([
    ["source", "filter0"],
    ["filter0", "filter1"],
    ["filter1", "filter2"],
    ["filter2", "filter3"],
    ["filter3", "gain0"],
    ["gain0", "destination"],
    ["source", "gain1"],
    ["gain1", "destination"],
  ]);
});

test("starts bypassed: band gain 0, bypass gain 1", () => {
  const { context, source, gains } = stubContext();
  const solo = createBandSolo(context, source);
  expect(gains[0].gain.value).toBe(0);
  expect(gains[1].gain.value).toBe(1);
  expect(solo.band()).toBeNull();
});

test("setBand updates all four filter frequencies and opens the band path", () => {
  const { context, source, filters, gains } = stubContext();
  const solo = createBandSolo(context, source);
  solo.setBand(240, 1200);

  expect(filters.map((f) => f.frequency.value)).toEqual([240, 240, 1200, 1200]);
  expect(gains[0].gain.value).toBe(1);
  expect(gains[1].gain.value).toBe(0);
  expect(solo.band()).toEqual([240, 1200]);
});

test("setBand re-tunes the same nodes rather than building more", () => {
  const { context, source, filters, gains } = stubContext();
  const solo = createBandSolo(context, source);
  solo.setBand(240, 1200);
  solo.setBand(2000, 6000);
  expect(filters).toHaveLength(4);
  expect(gains).toHaveLength(2);
  expect(filters.map((f) => f.frequency.value)).toEqual([2000, 2000, 6000, 6000]);
});

test("a reversed or degenerate range is normalized", () => {
  const { context, source, filters } = stubContext();
  const solo = createBandSolo(context, source);
  solo.setBand(1200, 240);
  expect(filters.map((f) => f.frequency.value)).toEqual([240, 240, 1200, 1200]);
  solo.setBand(500, 500);
  expect(solo.band()).toEqual([500, 501]);
});

test("clear() bypasses the filters again", () => {
  const { context, source, gains } = stubContext();
  const solo = createBandSolo(context, source);
  solo.setBand(240, 1200);
  solo.clear();
  expect(gains[0].gain.value).toBe(0);
  expect(gains[1].gain.value).toBe(1);
  expect(solo.band()).toBeNull();
});
