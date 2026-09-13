import { render, screen, waitFor } from "@testing-library/react";
import { Sound } from "./Sound";
import { loadTrackAudio } from "../audio/decode";
import { analyzeSound } from "../audio/analyze";
import type { Track } from "../api/types";

vi.mock("../audio/decode", () => ({ loadTrackAudio: vi.fn(), rememberAnalysis: vi.fn() }));
vi.mock("../audio/analyze", () => ({ analyzeSound: vi.fn() }));
vi.mock("../player/usePlayer", () => ({
  usePlayer: () => ({
    nowPlaying: null,
    isPlaying: false,
    progress: 0,
    errorMessage: null,
    play: vi.fn(),
    toggle: vi.fn(),
    stop: vi.fn(),
    seek: vi.fn(),
  }),
  attachGraph: () => null,
}));

const TRACK: Track = {
  track_id: "721063",
  title: "So What",
  artist: "Miles Davis",
  album: "Kind of Blue",
  artwork_url: "",
  preview_url: "",
};

const ANALYSIS = {
  mel: new Float32Array([-80, -20, -40, -10]),
  frameCount: 2,
  bands: 2,
  bandEdgesHz: Float64Array.from([20, 240, 1200, 6000]),
  peak: -10,
  sampleRate: 44100,
  hop: 1024,
  fftSize: 2048,
  ssm: Float32Array.from([1, 0.5, 0.5, 1]),
  ssmColumns: 2,
  duration: 30,
};

beforeEach(() => {
  vi.mocked(loadTrackAudio).mockReset();
  vi.mocked(analyzeSound).mockReset();
});

test("shows a skeleton, then the spectrogram, band strip and self-similarity", async () => {
  vi.mocked(loadTrackAudio).mockResolvedValue({
    samples: new Float32Array(4096),
    sampleRate: 44100,
    duration: 30,
  });
  vi.mocked(analyzeSound).mockResolvedValue(ANALYSIS);

  render(<Sound track={TRACK} />);
  expect(screen.getByTestId("sound-skeleton")).toBeInTheDocument();

  expect(await screen.findByLabelText("Mel spectrogram")).toBeInTheDocument();
  expect(screen.getByLabelText("Self-similarity matrix")).toBeInTheDocument();
  expect(screen.getByLabelText("Band solo")).toBeInTheDocument();
  expect(screen.getByRole("button", { name: "Solo off" })).toBeInTheDocument();
  expect(screen.getByText("So What")).toBeInTheDocument();

  expect(vi.mocked(loadTrackAudio).mock.calls[0][0]).toBe("721063");
});

test("no playhead when this track isn't the one playing", async () => {
  vi.mocked(loadTrackAudio).mockResolvedValue({
    samples: new Float32Array(4096),
    sampleRate: 44100,
    duration: 30,
  });
  vi.mocked(analyzeSound).mockResolvedValue(ANALYSIS);

  render(<Sound track={TRACK} />);
  await screen.findByLabelText("Mel spectrogram");
  expect(screen.queryByTestId("spectrogram-playhead")).not.toBeInTheDocument();
});

test("a cached analysis skips the worker entirely", async () => {
  vi.mocked(loadTrackAudio).mockResolvedValue({
    samples: new Float32Array(4096),
    sampleRate: 44100,
    duration: 30,
    analysis: ANALYSIS,
  });

  render(<Sound track={TRACK} />);
  expect(await screen.findByLabelText("Mel spectrogram")).toBeInTheDocument();
  expect(analyzeSound).not.toHaveBeenCalled();
});

test("switching tracks aborts the in-flight load", async () => {
  const signals: (AbortSignal | undefined)[] = [];
  vi.mocked(loadTrackAudio).mockImplementation(async (_id, deps) => {
    signals.push(deps?.signal);
    return new Promise(() => {}) as unknown as Awaited<ReturnType<typeof loadTrackAudio>>;
  });

  const view = render(<Sound track={TRACK} />);
  expect(signals[0]?.aborted).toBe(false);
  view.rerender(<Sound track={{ ...TRACK, track_id: "other" }} />);
  expect(signals[0]?.aborted).toBe(true); // the first load is cancelled
  expect(signals[1]?.aborted).toBe(false); // the new one is live
});

test("a failed fetch shows an error instead of the panels", async () => {
  vi.mocked(loadTrackAudio).mockRejectedValue(new Error("audio fetch failed: 404"));

  render(<Sound track={TRACK} />);
  await waitFor(() => expect(screen.getByText(/Couldn’t load/)).toBeInTheDocument());
  expect(screen.queryByLabelText("Mel spectrogram")).not.toBeInTheDocument();
});
