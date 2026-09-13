import { attachGraph } from "../player/usePlayer";
import { currentBand, resetSolo, soloBand, soloOff } from "./soloStore";

vi.mock("../player/usePlayer", () => ({ attachGraph: vi.fn() }));

function fakeGraph() {
  const param = () => ({ value: 0, setTargetAtTime(v: number) { this.value = v; } });
  const made = { filters: 0, gains: 0 };
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
  return { made, graph: { context, source: { connect: () => {} } } };
}

beforeEach(() => {
  resetSolo();
  vi.mocked(attachGraph).mockReset();
});

test("builds the chain on the first solo and reuses it after", () => {
  const { made, graph } = fakeGraph();
  vi.mocked(attachGraph).mockReturnValue(graph as unknown as ReturnType<typeof attachGraph>);

  soloBand(240, 1200);
  expect(currentBand()).toEqual([240, 1200]);
  soloBand(2000, 6000);
  expect(currentBand()).toEqual([2000, 6000]);

  expect(made.filters).toBe(4); // one graph, not two
  expect(made.gains).toBe(2);
});

test("soloOff clears the band", () => {
  const { graph } = fakeGraph();
  vi.mocked(attachGraph).mockReturnValue(graph as unknown as ReturnType<typeof attachGraph>);
  soloBand(240, 1200);
  soloOff();
  expect(currentBand()).toBeNull();
});

test("no Web Audio means no solo, not a crash", () => {
  vi.mocked(attachGraph).mockReturnValue(null);
  soloBand(240, 1200);
  expect(currentBand()).toBeNull();
});
