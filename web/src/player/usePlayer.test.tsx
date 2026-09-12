import { renderHook, act } from "@testing-library/react";
import type { ReactNode } from "react";
import { PlayerProvider, usePlayer } from "./usePlayer";
import type { Track } from "../api/types";

function track(id: string): Track {
  return { track_id: id, title: `Title ${id}`, artist: "Artist", album: "Album",
    artwork_url: "", preview_url: "" };
}

function wrapper({ children }: { children: ReactNode }) {
  return <PlayerProvider>{children}</PlayerProvider>;
}

beforeEach(() => {
  vi.spyOn(HTMLMediaElement.prototype, "play").mockResolvedValue(undefined);
  vi.spyOn(HTMLMediaElement.prototype, "pause").mockImplementation(() => {});
});

test("play(track) sets src to /api/preview/<id> and isPlaying", async () => {
  const srcSetter = vi.spyOn(HTMLMediaElement.prototype, "src", "set");
  const { result } = renderHook(() => usePlayer(), { wrapper });
  await act(async () => {
    result.current.play(track("42"));
  });
  expect(srcSetter).toHaveBeenCalledWith("/api/preview/42");
  expect(result.current.nowPlaying?.track_id).toBe("42");
  expect(result.current.isPlaying).toBe(true);
});

test("toggle() flips isPlaying", async () => {
  const { result } = renderHook(() => usePlayer(), { wrapper });
  await act(async () => {
    result.current.play(track("42"));
  });
  expect(result.current.isPlaying).toBe(true);

  act(() => {
    result.current.toggle();
  });
  expect(result.current.isPlaying).toBe(false);

  await act(async () => {
    result.current.toggle();
  });
  expect(result.current.isPlaying).toBe(true);
});

test("playing a second track replaces the src", async () => {
  const srcSetter = vi.spyOn(HTMLMediaElement.prototype, "src", "set");
  const { result } = renderHook(() => usePlayer(), { wrapper });
  await act(async () => {
    result.current.play(track("1"));
  });
  expect(result.current.nowPlaying?.track_id).toBe("1");

  await act(async () => {
    result.current.play(track("2"));
  });
  expect(result.current.nowPlaying?.track_id).toBe("2");
  expect(srcSetter).toHaveBeenLastCalledWith("/api/preview/2");
});
