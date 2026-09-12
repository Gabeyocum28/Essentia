import { render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { Seed } from "./Seed";
import { api, ApiError } from "../api/client";
import { PlayerProvider } from "../player/usePlayer";

vi.mock("../api/client", async () => {
  const actual = await vi.importActual<typeof import("../api/client")>("../api/client");
  return {
    ...actual,
    api: {
      ...actual.api,
      seed: vi.fn(),
      axes: vi.fn(),
    },
  };
});

function renderSeed(id = "42") {
  return render(
    <PlayerProvider>
      <MemoryRouter initialEntries={[`/seed/${id}`]}>
        <Routes>
          <Route path="/seed/:id" element={<Seed />} />
        </Routes>
      </MemoryRouter>
    </PlayerProvider>,
  );
}

beforeEach(() => {
  vi.mocked(api.axes).mockResolvedValue({
    axes: [
      { id: "energy", label: "Energy" },
      { id: "mood", label: "Mood" },
    ],
  });
});

test("renders axis buttons after a ready seed", async () => {
  vi.mocked(api.seed).mockResolvedValue({ track_id: "42", status: "ready" });
  renderSeed();

  await waitFor(() => expect(screen.getByText("Energy")).toBeInTheDocument());
  expect(screen.getByText("Mood")).toBeInTheDocument();
});

test("shows retry message on a 502 ApiError", async () => {
  vi.mocked(api.seed).mockRejectedValue(new ApiError(502, "undecodable preview"));
  renderSeed();

  await waitFor(() =>
    expect(screen.getByText("Couldn't prepare this track (undecodable preview).")).toBeInTheDocument(),
  );
  expect(screen.getByText("Try again")).toBeInTheDocument();
});
