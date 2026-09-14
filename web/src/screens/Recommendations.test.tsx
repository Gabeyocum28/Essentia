import { render, screen, waitFor, fireEvent } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { Recommendations } from "./Recommendations";
import { api, DEFAULT_FEEL, DEFAULT_TEMPO } from "../api/client";
import { PlayerProvider } from "../player/usePlayer";

vi.mock("../api/client", async () => {
  const actual = await vi.importActual<typeof import("../api/client")>("../api/client");
  return {
    ...actual,
    api: {
      ...actual.api,
      recommend: vi.fn(),
    },
  };
});

function track(id: string, title: string) {
  return { track_id: id, title, artist: "Artist", album: "Album", artwork_url: null, preview_url: null, score: 0.9 };
}

function renderRecs(id = "42", axis = "energy") {
  return render(
    <PlayerProvider>
      <MemoryRouter initialEntries={[`/recommend/${id}/${axis}`]}>
        <Routes>
          <Route path="/recommend/:id/:axis" element={<Recommendations />} />
        </Routes>
      </MemoryRouter>
    </PlayerProvider>,
  );
}

beforeEach(() => {
  localStorage.clear();
  vi.mocked(api.recommend).mockResolvedValue({
    seed_track_id: "42",
    axis: "energy",
    results: [track("a", "Track A")],
  });
});

test("fetches recommendations with the default feel on load", async () => {
  renderRecs();

  await waitFor(() => expect(screen.getByText("Track A")).toBeInTheDocument());
  expect(DEFAULT_FEEL).toBe(0.3);
  expect(DEFAULT_TEMPO).toBe(0.2);
  expect(api.recommend).toHaveBeenCalledWith("42", "energy", 10, DEFAULT_FEEL, DEFAULT_TEMPO);
});

test("a stored weight outside the slider's range is clamped on load", async () => {
  localStorage.setItem("essentia.feel", "42");
  renderRecs();

  await waitFor(() => expect(screen.getByText("Track A")).toBeInTheDocument());
  expect(api.recommend).toHaveBeenCalledWith("42", "energy", 10, 2, DEFAULT_TEMPO);
});

test("moving the feel slider triggers a debounced refetch with the new value", async () => {
  vi.useFakeTimers({ shouldAdvanceTime: true });
  try {
    renderRecs();
    await vi.waitFor(() => expect(screen.getByText("Track A")).toBeInTheDocument());

    vi.mocked(api.recommend).mockClear();
    const slider = screen.getByLabelText("Match the feel");
    fireEvent.change(slider, { target: { value: "1.2" } });

    // Not yet fetched before the debounce window elapses.
    expect(api.recommend).not.toHaveBeenCalled();

    await vi.advanceTimersByTimeAsync(250);

    expect(api.recommend).toHaveBeenCalledWith("42", "energy", 10, 1.2, DEFAULT_TEMPO);
    expect(localStorage.getItem("essentia.feel")).toBe("1.2");
  } finally {
    vi.useRealTimers();
  }
});

test("the insights link carries the current feel value", async () => {
  renderRecs();
  await waitFor(() => expect(screen.getByText("Track A")).toBeInTheDocument());

  const link = screen.getByText("See the math ✦") as HTMLAnchorElement;
  expect(link.getAttribute("href")).toBe(
    `/insights/42/energy?feel=${DEFAULT_FEEL}&tempo=${DEFAULT_TEMPO}`);
});

test("moving the tempo slider refetches with the new weight and stores it", async () => {
  vi.useFakeTimers({ shouldAdvanceTime: true });
  try {
    renderRecs();
    await vi.waitFor(() => expect(screen.getByText("Track A")).toBeInTheDocument());

    vi.mocked(api.recommend).mockClear();
    fireEvent.change(screen.getByLabelText("Match the tempo"), { target: { value: "1.5" } });
    expect(api.recommend).not.toHaveBeenCalled();   // debounced, like the feel slider

    await vi.advanceTimersByTimeAsync(250);

    expect(api.recommend).toHaveBeenCalledWith("42", "energy", 10, DEFAULT_FEEL, 1.5);
    expect(localStorage.getItem("essentia.tempo")).toBe("1.5");
  } finally {
    vi.useRealTimers();
  }
});

test("a stored tempo weight outside the slider's range is clamped on load", async () => {
  localStorage.setItem("essentia.tempo", "-3");
  renderRecs();

  await waitFor(() => expect(screen.getByText("Track A")).toBeInTheDocument());
  expect(api.recommend).toHaveBeenCalledWith("42", "energy", 10, DEFAULT_FEEL, 0);
});

test("both sliders moved inside one debounce window make one request carrying both", async () => {
  vi.useFakeTimers({ shouldAdvanceTime: true });
  try {
    renderRecs();
    await vi.waitFor(() => expect(screen.getByText("Track A")).toBeInTheDocument());

    vi.mocked(api.recommend).mockClear();
    fireEvent.change(screen.getByLabelText("Match the feel"), { target: { value: "1.2" } });
    fireEvent.change(screen.getByLabelText("Match the tempo"), { target: { value: "0.8" } });
    await vi.advanceTimersByTimeAsync(250);

    expect(api.recommend).toHaveBeenCalledTimes(1);
    expect(api.recommend).toHaveBeenCalledWith("42", "energy", 10, 1.2, 0.8);
  } finally {
    vi.useRealTimers();
  }
});
