import { render, screen, waitFor, fireEvent } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { Search } from "./Search";
import { api, ApiError } from "../api/client";
import { PlayerProvider } from "../player/usePlayer";

vi.mock("../api/client", async () => {
  const actual = await vi.importActual<typeof import("../api/client")>("../api/client");
  return {
    ...actual,
    api: { ...actual.api, search: vi.fn(), searchText: vi.fn() },
  };
});

function track(id: string, title: string) {
  return { track_id: id, title, artist: "Artist", album: "Album",
           artwork_url: null, preview_url: null };
}

function renderSearch() {
  return render(
    <PlayerProvider>
      <MemoryRouter><Search /></MemoryRouter>
    </PlayerProvider>,
  );
}

function submit(text: string) {
  fireEvent.change(screen.getByLabelText("Search"), { target: { value: text } });
  fireEvent.submit(screen.getByLabelText("Search"));
}

beforeEach(() => {
  vi.mocked(api.search).mockResolvedValue({ results: [track("a", "By Name")] });
  vi.mocked(api.searchText).mockResolvedValue({ results: [track("b", "By Description")] });
});

test("searches the catalogue by name by default", async () => {
  renderSearch();
  submit("kind of blue");

  await waitFor(() => expect(screen.getByText("By Name")).toBeInTheDocument());
  expect(api.search).toHaveBeenCalledWith("kind of blue");
  expect(api.searchText).not.toHaveBeenCalled();
});

test("the by-description toggle searches the corpus instead", async () => {
  renderSearch();
  fireEvent.click(screen.getByLabelText("Search by description"));
  submit("hazy late-night trumpet");

  await waitFor(() => expect(screen.getByText("By Description")).toBeInTheDocument());
  expect(api.searchText).toHaveBeenCalledWith("hazy late-night trumpet");
  expect(api.search).not.toHaveBeenCalled();
});

test("toggling after a search re-asks the same question the other way", async () => {
  renderSearch();
  submit("piano");
  await waitFor(() => expect(screen.getByText("By Name")).toBeInTheDocument());

  fireEvent.click(screen.getByLabelText("Search by description"));

  await waitFor(() => expect(screen.getByText("By Description")).toBeInTheDocument());
  expect(api.searchText).toHaveBeenCalledWith("piano");
});

test("a 503 from the text search shows the server's own detail", async () => {
  vi.mocked(api.searchText).mockRejectedValue(
    new ApiError(503, "text search unavailable: CLAP could not be loaded"),
  );
  renderSearch();
  fireEvent.click(screen.getByLabelText("Search by description"));
  submit("jazz");

  await waitFor(() =>
    expect(screen.getByText(/CLAP could not be loaded/)).toBeInTheDocument());
});

test("an empty query searches nothing", async () => {
  renderSearch();
  submit("   ");
  expect(api.search).not.toHaveBeenCalled();
  expect(api.searchText).not.toHaveBeenCalled();
});
