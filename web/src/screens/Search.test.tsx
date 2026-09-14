import { render, screen, waitFor, fireEvent } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { Search } from "./Search";
import { api, ApiError } from "../api/client";
import { PlayerProvider } from "../player/usePlayer";

vi.mock("../api/client", async () => {
  const actual = await vi.importActual<typeof import("../api/client")>("../api/client");
  return {
    ...actual,
    api: { ...actual.api, search: vi.fn(), searchText: vi.fn(), axes: vi.fn() },
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
  // The toggle only exists where the host can answer a text search.
  vi.mocked(api.axes).mockResolvedValue({ axes: [], text_search: true });
});

async function toggle() {
  const box = await screen.findByLabelText("Search by description");
  fireEvent.click(box);
}

test("searches the catalogue by name by default", async () => {
  renderSearch();
  submit("kind of blue");

  await waitFor(() => expect(screen.getByText("By Name")).toBeInTheDocument());
  expect(api.search).toHaveBeenCalledWith("kind of blue");
  expect(api.searchText).not.toHaveBeenCalled();
});

test("the by-description toggle searches the corpus instead", async () => {
  renderSearch();
  await toggle();
  submit("hazy late-night trumpet");

  await waitFor(() => expect(screen.getByText("By Description")).toBeInTheDocument());
  expect(api.searchText).toHaveBeenCalledWith("hazy late-night trumpet");
  expect(api.search).not.toHaveBeenCalled();
});

test("toggling after a search re-asks the same question the other way", async () => {
  renderSearch();
  submit("piano");
  await waitFor(() => expect(screen.getByText("By Name")).toBeInTheDocument());

  await toggle();

  await waitFor(() => expect(screen.getByText("By Description")).toBeInTheDocument());
  expect(api.searchText).toHaveBeenCalledWith("piano");
});

test("a 503 from the text search shows the server's own detail", async () => {
  vi.mocked(api.searchText).mockRejectedValue(
    new ApiError(503, "text search unavailable: CLAP could not be loaded"),
  );
  renderSearch();
  await toggle();
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

test("toggling re-asks the query the current results answer, not the edited box", async () => {
  renderSearch();
  submit("piano");
  await waitFor(() => expect(screen.getByText("By Name")).toBeInTheDocument());

  // The user starts typing something else but never presses Search.
  fireEvent.change(screen.getByLabelText("Search"), { target: { value: "drums" } });
  await toggle();

  await waitFor(() => expect(api.searchText).toHaveBeenCalled());
  expect(api.searchText).toHaveBeenCalledWith("piano");
});

test("no toggle at all when the host cannot answer a text search", async () => {
  vi.mocked(api.axes).mockResolvedValue({ axes: [], text_search: false });
  renderSearch();
  submit("piano");
  await waitFor(() => expect(screen.getByText("By Name")).toBeInTheDocument());

  expect(screen.queryByLabelText("Search by description")).not.toBeInTheDocument();
});

test("an older server that does not send the flag offers no toggle", async () => {
  vi.mocked(api.axes).mockResolvedValue({ axes: [] });
  renderSearch();
  submit("piano");
  await waitFor(() => expect(screen.getByText("By Name")).toBeInTheDocument());

  expect(screen.queryByLabelText("Search by description")).not.toBeInTheDocument();
});
