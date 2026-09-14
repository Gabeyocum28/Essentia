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

// analysis/feel_v2.PROMPT_BANK order, which the server sends as feel_keys.
const FEEL_KEYS = [
  "energy", "valence", "tension", "acoustic",
  "danceable", "vocal", "bright", "density",
];

test("renders the per-dimension feel comparison when math.feel is present", () => {
  const seed = [0.9, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7];
  const recVals = [0.8, 0.2, 0.1, 0.4, 0.3, 0.6, 0.5, 0.6];
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

  expect(screen.queryByText("energy")).not.toBeInTheDocument();
  expect(screen.queryByText(/feel_dist =/)).not.toBeInTheDocument();
});

const RHYTHM = {
  seed: { tempo_bpm: 120, beat_strength: 0.9, loudness_lufs: -9.4,
          loudness_range: 5, key: 5, mode: "minor", key_strength: 0.7 },
  rec: { tempo_bpm: 160, beat_strength: 0.8, loudness_lufs: -12.25,
         loudness_range: 7.5, key: 0, mode: "major", key_strength: 0.6 },
};

test("renders the BPM, key and loudness rows for the seed and the pick", () => {
  const r = rec({
    metric: "cosine", dot: 0.9, seed_norm: 1, rec_norm: 1, centrality: null, distance: null,
    tempo_dist: 0.415, rhythm: RHYTHM,
  });
  render(<MathPanel rec={r} feelKeys={FEEL_KEYS} />);

  expect(screen.getByText("120.0 BPM")).toBeInTheDocument();
  expect(screen.getByText("160.0 BPM")).toBeInTheDocument();
  expect(screen.getByText("F minor")).toBeInTheDocument();
  expect(screen.getByText("C major")).toBeInTheDocument();
  expect(screen.getByText("-9.4 LUFS · LRA 5.0")).toBeInTheDocument();
  expect(screen.getByText("-12.3 LUFS · LRA 7.5")).toBeInTheDocument();
  expect(screen.getByText("tempo_dist = 0.415")).toBeInTheDocument();
});

test("says nothing about the key when key_strength says the estimate is meaningless", () => {
  const r = rec({
    metric: "cosine", dot: 0.9, seed_norm: 1, rec_norm: 1, centrality: null, distance: null,
    tempo_dist: 0, rhythm: { ...RHYTHM, rec: { ...RHYTHM.rec, key_strength: 0 } },
  });
  render(<MathPanel rec={r} feelKeys={FEEL_KEYS} />);

  expect(screen.getByText("—")).toBeInTheDocument();
  expect(screen.queryByText("C major")).not.toBeInTheDocument();
});

test("shows no tempo where a track has no beat, and no rhythm block at all when there is none", () => {
  const noBeat = rec({
    metric: "cosine", dot: 0.9, seed_norm: 1, rec_norm: 1, centrality: null, distance: null,
    tempo_dist: null, rhythm: { ...RHYTHM, rec: { ...RHYTHM.rec, tempo_bpm: 0 } },
  });
  const { unmount } = render(<MathPanel rec={noBeat} feelKeys={FEEL_KEYS} />);
  expect(screen.getByText("—")).toBeInTheDocument();
  expect(screen.queryByText(/tempo_dist =/)).not.toBeInTheDocument();
  unmount();

  const none = rec({
    metric: "cosine", dot: 0.9, seed_norm: 1, rec_norm: 1, centrality: null, distance: null,
    tempo_dist: null, rhythm: null,
  });
  render(<MathPanel rec={none} feelKeys={FEEL_KEYS} />);
  expect(screen.queryByText("tempo")).not.toBeInTheDocument();
});
