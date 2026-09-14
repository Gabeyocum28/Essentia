import { render, screen, waitFor } from "@testing-library/react";
import { fireEvent } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { Insights } from "./Insights";
import { api, ApiError, DEFAULT_FEEL, DEFAULT_TEMPO } from "../api/client";
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

function renderInsights(id = "seed", axis = "energy", query = "") {
  return render(
    <PlayerProvider>
      <MemoryRouter initialEntries={[`/insights/${id}/${axis}${query}`]}>
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
    expect(screen.getByText(/not analyzed on the server yet/)).toBeInTheDocument(),
  );
});

test("a 409 from vizMap reads as re-analyzing, not as a failure", async () => {
  vi.mocked(api.seed).mockResolvedValue({ track_id: "42", status: "ready" });
  vi.mocked(api.vizMap).mockRejectedValue(
    new ApiError(409, "queued for re-analysis"));
  renderInsights();
  await waitFor(() =>
    expect(screen.getByText(/not analyzed on the server yet/)).toBeInTheDocument(),
  );
  expect(screen.queryByText("Something went wrong")).toBeNull();
});

test("shows the not-analyzed message when vizMap 404s", async () => {
  vi.mocked(api.seed).mockResolvedValue({ track_id: "seed", status: "ready" });
  vi.mocked(api.vizMap).mockRejectedValue(new ApiError(404, "not found"));
  renderInsights();

  await waitFor(() =>
    expect(screen.getByText(/not analyzed on the server yet/)).toBeInTheDocument(),
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


// ---- the feel weight ----
//
// Insights has to explain the list the user actually SAW, so the weight it
// asks the map for must be the one the recommendations screen ranked with.
// The link from that screen carries ?feel=; a bookmark or a direct load does
// not, and falling back to the default there would draw a different list.

test("uses the feel weight from the query string", async () => {
  vi.mocked(api.seed).mockResolvedValue({ track_id: "seed", status: "ready" });
  vi.mocked(api.vizMap).mockResolvedValue(makeMap());
  localStorage.setItem("essentia.feel", "1.5");

  renderInsights("seed", "energy", "?feel=0.8");
  await waitFor(() => expect(api.vizMap).toHaveBeenCalled());
  expect(api.vizMap).toHaveBeenCalledWith("seed", "energy", 10, undefined, 0.8, DEFAULT_TEMPO);
  localStorage.clear();
});

test("falls back to the stored slider position, then to the default", async () => {
  vi.mocked(api.seed).mockResolvedValue({ track_id: "seed", status: "ready" });
  vi.mocked(api.vizMap).mockResolvedValue(makeMap());
  localStorage.setItem("essentia.feel", "1.5");

  const stored = renderInsights();
  await waitFor(() => expect(api.vizMap).toHaveBeenCalled());
  expect(api.vizMap).toHaveBeenCalledWith("seed", "energy", 10, undefined, 1.5, DEFAULT_TEMPO);
  stored.unmount();

  localStorage.clear();
  vi.mocked(api.vizMap).mockClear();
  renderInsights();
  await waitFor(() => expect(api.vizMap).toHaveBeenCalled());
  expect(api.vizMap).toHaveBeenCalledWith("seed", "energy", 10, undefined, DEFAULT_FEEL, DEFAULT_TEMPO);
});

test("a hand-edited weight in the query string is clamped", async () => {
  vi.mocked(api.seed).mockResolvedValue({ track_id: "seed", status: "ready" });
  vi.mocked(api.vizMap).mockResolvedValue(makeMap());

  renderInsights("seed", "energy", "?feel=99");
  await waitFor(() => expect(api.vizMap).toHaveBeenCalled());
  expect(api.vizMap).toHaveBeenCalledWith("seed", "energy", 10, undefined, 2, DEFAULT_TEMPO);
});

test("uses the tempo weight from the query string, and clamps a hand-edited one", async () => {
  vi.mocked(api.seed).mockResolvedValue({ track_id: "seed", status: "ready" });
  vi.mocked(api.vizMap).mockResolvedValue(makeMap());

  const first = renderInsights("seed", "energy", "?feel=0.8&tempo=1.4");
  await waitFor(() => expect(api.vizMap).toHaveBeenCalled());
  expect(api.vizMap).toHaveBeenCalledWith("seed", "energy", 10, undefined, 0.8, 1.4);
  first.unmount();

  vi.mocked(api.vizMap).mockClear();
  renderInsights("seed", "energy", "?tempo=99");
  await waitFor(() => expect(api.vizMap).toHaveBeenCalled());
  expect(api.vizMap).toHaveBeenCalledWith("seed", "energy", 10, undefined, DEFAULT_FEEL, 2);
});
