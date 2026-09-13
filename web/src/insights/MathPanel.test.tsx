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

const FEEL_KEYS = [
  "danceable", "happy", "sad", "aggressive", "relaxed", "party",
  "acoustic", "electronic", "bright", "tonal", "instrumental",
];

test("renders the per-dimension feel comparison when math.feel is present", () => {
  const seed = [0.9, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.1];
  const recVals = [0.8, 0.2, 0.1, 0.4, 0.3, 0.6, 0.5, 0.6, 0.7, 0.8, 0.2];
  const r = rec({
    metric: "cosine", dot: 0.9, seed_norm: 1, rec_norm: 1, centrality: null, distance: null,
    feel_dist: 0.12345, feel: { seed, rec: recVals },
  });
  render(<MathPanel rec={r} feelKeys={FEEL_KEYS} />);

  for (const key of FEEL_KEYS) {
    expect(screen.getByText(key)).toBeInTheDocument();
  }
  expect(screen.getByText("feel_dist = 0.123")).toBeInTheDocument();
});

test("does not render the feel comparison and does not crash when math.feel is null", () => {
  const r = rec({
    metric: "cosine", dot: 0.9, seed_norm: 1, rec_norm: 1, centrality: null, distance: null,
    feel_dist: null, feel: null,
  });
  render(<MathPanel rec={r} feelKeys={FEEL_KEYS} />);

  expect(screen.queryByText("danceable")).not.toBeInTheDocument();
  expect(screen.queryByText(/feel_dist =/)).not.toBeInTheDocument();
});
