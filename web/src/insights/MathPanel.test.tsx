import { render, screen } from "@testing-library/react";
import { MathPanel } from "./MathPanel";
import type { VizRec } from "../api/types";

function rec(math: VizRec["math"]): VizRec {
  return {
    track_id: "rec1",
    title: "Rec One",
    artist: "Artist",
    album: "Album",
    artwork_url: null,
    preview_url: null,
    x: 0,
    y: 0,
    score: 0.9,
    math,
  };
}

test("renders the cosine line and does not crash when distance and centrality are null", () => {
  const r = rec({ metric: "cosine", dot: 0.9, seed_norm: 1, rec_norm: 1, centrality: null, distance: null });
  render(<MathPanel rec={r} />);

  expect(screen.getByText(/cos = 0\.9000/)).toBeInTheDocument();
  expect(screen.queryByText(/centrality =/)).not.toBeInTheDocument();
  expect(screen.queryByText(/distance =/)).not.toBeInTheDocument();
});

test("shows the centrality line when centrality is a number", () => {
  const r = rec({ metric: "cosine", dot: 0.9, seed_norm: 1, rec_norm: 1, centrality: 0.12, distance: null });
  render(<MathPanel rec={r} />);

  expect(screen.getByText("centrality = 0.1200")).toBeInTheDocument();
});
