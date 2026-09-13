import { render, screen, waitFor } from "@testing-library/react";
import { Proof } from "./Proof";
import { api } from "../api/client";
import { PlayerProvider } from "../player/usePlayer";
import type { VizHistogram, VizHubs, VizMap } from "../api/types";

vi.mock("../api/client", async () => {
  const actual = await vi.importActual<typeof import("../api/client")>("../api/client");
  return {
    ...actual,
    api: {
      ...actual.api,
      vizHistogram: vi.fn(),
      vizHubs: vi.fn(),
      vizMap: vi.fn(),
    },
  };
});

function track(id: string, title: string) {
  return { track_id: id, title, artist: "Artist", album: "Album", artwork_url: "", preview_url: "" };
}

function rec(id: string, title: string, score: number) {
  return {
    ...track(id, title),
    x: 0,
    y: 0,
    score,
    math: { metric: "cosine", dot: score, seed_norm: 1, rec_norm: 1, centrality: null, distance: null },
  };
}

function makeHistogram(): VizHistogram {
  return {
    bins: [-0.5, 0, 0.5],
    counts: [1, 5, 2],
    rec_scores: [0.2, 0.4],
    percentile: 82,
    null: { mean: 0, sd: 0.3 },
    corpus: { mean: 0, sd: 0.3 },
  };
}

function makeHubs(): VizHubs {
  return { hubs: [], central: [], isolated: [], expected_k: 8 };
}

function makeMap(recs: ReturnType<typeof rec>[]): VizMap {
  return {
    points: { ids: [], x: [], y: [], tracks: [] },
    seed: { ...track("seed", "Seed Track"), x: 0, y: 0 },
    recs,
    axis: { id: "surprise", metric: "cosine", direction: 1 },
  };
}

function renderProof() {
  return render(
    <PlayerProvider>
      <Proof seedId="seed" recIds={["rec1"]} />
    </PlayerProvider>,
  );
}

beforeEach(() => {
  vi.spyOn(HTMLMediaElement.prototype, "play").mockResolvedValue(undefined);
  vi.spyOn(HTMLMediaElement.prototype, "pause").mockImplementation(() => {});
  vi.mocked(api.vizHistogram).mockReset();
  vi.mocked(api.vizHubs).mockReset();
  vi.mocked(api.vizMap).mockReset();
});

test("fetches raw and corrected maps in parallel, exactly once each", async () => {
  vi.mocked(api.vizHistogram).mockResolvedValue(makeHistogram());
  vi.mocked(api.vizHubs).mockResolvedValue(makeHubs());
  vi.mocked(api.vizMap).mockImplementation(async (_id, _axis, _limit, correction) =>
    correction === "off"
      ? makeMap([rec("a", "Track A", 0.9), rec("b", "Track B", 0.7)])
      : makeMap([rec("b", "Track B", 0.85), rec("c", "Track C", 0.6)]),
  );

  renderProof();

  await waitFor(() => expect(screen.getByText("Track A")).toBeInTheDocument());

  expect(api.vizMap).toHaveBeenCalledTimes(2);
  expect(api.vizMap).toHaveBeenCalledWith("seed", "surprise", 10, "off", undefined);
  expect(api.vizMap).toHaveBeenCalledWith("seed", "surprise", 10, "on", undefined);
  expect(api.vizHubs).toHaveBeenCalledWith("seed", ["rec1"]);
});

test("renders both score columns for a shared track and a dash for a one-sided track", async () => {
  vi.mocked(api.vizHistogram).mockResolvedValue(makeHistogram());
  vi.mocked(api.vizHubs).mockResolvedValue(makeHubs());
  vi.mocked(api.vizMap).mockImplementation(async (_id, _axis, _limit, correction) =>
    correction === "off"
      ? makeMap([rec("a", "Track A", 0.9), rec("b", "Track B", 0.7)])
      : makeMap([rec("b", "Track B", 0.85), rec("c", "Track C", 0.6)]),
  );

  renderProof();

  await waitFor(() => expect(screen.getByText("Track B")).toBeInTheDocument());

  // "Track B" is present on both sides: both scores should render.
  expect(screen.getByText("0.7000")).toBeInTheDocument();
  expect(screen.getByText("0.8500")).toBeInTheDocument();

  // "Track A" is raw-only, "Track C" is corrected-only: each should show a dash on the
  // side it's missing from.
  expect(screen.getByText("Track A")).toBeInTheDocument();
  expect(screen.getByText("Track C")).toBeInTheDocument();
  const dashes = screen.getAllByText("—");
  expect(dashes.length).toBe(2);
});
