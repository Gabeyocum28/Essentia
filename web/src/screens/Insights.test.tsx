import { render, screen, waitFor } from "@testing-library/react";
import { fireEvent } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { Insights } from "./Insights";
import { api, ApiError } from "../api/client";
import { PlayerProvider } from "../player/usePlayer";
import type { VizMap } from "../api/types";

vi.mock("../api/client", async () => {
  const actual = await vi.importActual<typeof import("../api/client")>("../api/client");
  return {
    ...actual,
    api: {
      ...actual.api,
      seed: vi.fn(),
      vizMap: vi.fn(),
    },
  };
});

function track(id: string, title: string) {
  return { track_id: id, title, artist: "Artist", album: "Album", artwork_url: "", preview_url: "" };
}

function makeMap(): VizMap {
  return {
    points: {
      ids: ["a", "b", "c"],
      x: [0, 1, 2],
      y: [0, 1, 2],
      tracks: [track("a", "A"), track("b", "B"), track("c", "C")],
    },
    seed: { ...track("seed", "Seed Track"), x: 0, y: 0 },
    recs: [
      { ...track("rec1", "Rec One"), x: 1, y: 1, score: 0.9, math: { metric: "cosine", dot: 0.9, seed_norm: 1, rec_norm: 1, centrality: null, distance: null } },
      { ...track("rec2", "Rec Two"), x: 2, y: 2, score: 0.8, math: { metric: "cosine", dot: 0.8, seed_norm: 1, rec_norm: 1, centrality: null, distance: null } },
    ],
    axis: { id: "energy", metric: "cosine", direction: 1 },
  };
}

function renderInsights(id = "seed", axis = "energy") {
  return render(
    <PlayerProvider>
      <MemoryRouter initialEntries={[`/insights/${id}/${axis}`]}>
        <Routes>
          <Route path="/insights/:id/:axis" element={<Insights />} />
        </Routes>
      </MemoryRouter>
    </PlayerProvider>,
  );
}

beforeEach(() => {
  vi.spyOn(HTMLMediaElement.prototype, "play").mockResolvedValue(undefined);
  vi.spyOn(HTMLMediaElement.prototype, "pause").mockImplementation(() => {});
});

test("shows the unanalyzed message when seed is not ready", async () => {
  vi.mocked(api.seed).mockResolvedValue({ track_id: "seed", status: "unanalyzed" });
  renderInsights();

  await waitFor(() =>
    expect(screen.getByText("This track is not analyzed on the server yet.")).toBeInTheDocument(),
  );
});

test("shows the not-analyzed message when vizMap 404s", async () => {
  vi.mocked(api.seed).mockResolvedValue({ track_id: "seed", status: "ready" });
  vi.mocked(api.vizMap).mockRejectedValue(new ApiError(404, "not found"));
  renderInsights();

  await waitFor(() =>
    expect(screen.getByText("This track is not analyzed on the server yet.")).toBeInTheDocument(),
  );
});

test("renders the rec strip with seed and recs, and clicking a rec plays it", async () => {
  vi.mocked(api.seed).mockResolvedValue({ track_id: "seed", status: "ready" });
  vi.mocked(api.vizMap).mockResolvedValue(makeMap());
  renderInsights();

  await waitFor(() => expect(screen.getByText("Seed Track")).toBeInTheDocument());
  expect(screen.getByText("Rec One")).toBeInTheDocument();
  expect(screen.getByText("Rec Two")).toBeInTheDocument();

  const playSpy = vi.spyOn(HTMLMediaElement.prototype, "play");
  fireEvent.click(screen.getByRole("button", { name: "Play Rec One" }));

  await waitFor(() => expect(playSpy).toHaveBeenCalled());
});
