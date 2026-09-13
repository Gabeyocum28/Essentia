import { createContext, useCallback, useContext, useEffect, useRef, useState } from "react";
import type { ReactNode } from "react";
import type { Track } from "../api/types";
import { previewUrl } from "../api/client";

let audio: HTMLAudioElement | null = null;

function getAudio(): HTMLAudioElement {
  if (!audio) audio = new Audio();
  return audio;
}

// ---- Web Audio graph (used by SOUND mode's band solo) ----
//
// The element plays directly until something needs to filter it. A
// MediaElementAudioSourceNode can only be created ONCE per element and, once
// created, the element's own output is silenced unless the graph routes it
// to destination — so this is lazy, cached, and only ever called from a user
// gesture (Safari refuses to start a context otherwise).

let graph: { context: AudioContext; source: MediaElementAudioSourceNode } | null = null;

export function attachGraph(): { context: AudioContext; source: MediaElementAudioSourceNode } | null {
  if (graph) {
    void graph.context.resume?.();
    return graph;
  }
  const Ctor =
    globalThis.AudioContext ??
    (globalThis as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
  if (!Ctor) return null;
  try {
    const context = new Ctor();
    const source = context.createMediaElementSource(getAudio());
    graph = { context, source };
    void context.resume?.();
    return graph;
  } catch {
    return null; // no Web Audio here; playback keeps working, unfiltered
  }
}

interface PlayerState {
  nowPlaying: Track | null;
  isPlaying: boolean;
  progress: number;
  errorMessage: string | null;
  play(track: Track): void;
  toggle(): void;
  stop(): void;
  /** Jump to a fraction (0…1) of the current track. */
  seek(progress: number): void;
}

const PlayerContext = createContext<PlayerState | null>(null);

export function PlayerProvider({ children }: { children: ReactNode }) {
  const [nowPlaying, setNowPlaying] = useState<Track | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [progress, setProgress] = useState(0);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const nowPlayingRef = useRef<Track | null>(null);
  const isPlayingRef = useRef(false);

  useEffect(() => {
    const el = getAudio();
    const onTimeUpdate = () => {
      if (el.duration > 0) setProgress(el.currentTime / el.duration);
    };
    const onEnded = () => {
      isPlayingRef.current = false;
      setIsPlaying(false);
      setProgress(0);
    };
    const onError = () => {
      setErrorMessage("Preview unavailable");
      isPlayingRef.current = false;
      setIsPlaying(false);
    };
    el.addEventListener("timeupdate", onTimeUpdate);
    el.addEventListener("ended", onEnded);
    el.addEventListener("error", onError);
    return () => {
      el.removeEventListener("timeupdate", onTimeUpdate);
      el.removeEventListener("ended", onEnded);
      el.removeEventListener("error", onError);
    };
  }, []);

  const play = useCallback((track: Track) => {
    const el = getAudio();
    const trackId = track.track_id;
    nowPlayingRef.current = track;
    setNowPlaying(track);
    setErrorMessage(null);
    setProgress(0);
    el.src = previewUrl(trackId);
    el.play().then(
      () => {
        if (nowPlayingRef.current?.track_id !== trackId) return; // superseded by a newer play()
        isPlayingRef.current = true;
        setIsPlaying(true);
      },
      (err) => {
        // A play() call superseded by another play() (or a stop()) rejects with an
        // AbortError; that's expected and shouldn't surface as a playback error.
        if (err instanceof DOMException && err.name === "AbortError") return;
        if (nowPlayingRef.current?.track_id !== trackId) return;
        isPlayingRef.current = false;
        setErrorMessage("Preview unavailable");
      },
    );
  }, []);

  const toggle = useCallback(() => {
    const el = getAudio();
    if (!nowPlayingRef.current) return;
    if (!isPlayingRef.current) {
      el.play().then(
        () => {
          isPlayingRef.current = true;
          setIsPlaying(true);
        },
        () => {
          isPlayingRef.current = false;
          setErrorMessage("Preview unavailable");
        },
      );
    } else {
      el.pause();
      isPlayingRef.current = false;
      setIsPlaying(false);
    }
  }, []);

  const seek = useCallback((next: number) => {
    const el = getAudio();
    if (!nowPlayingRef.current || !(el.duration > 0)) return;
    const clamped = Math.max(0, Math.min(1, next));
    el.currentTime = clamped * el.duration;
    setProgress(clamped);
  }, []);

  const stop = useCallback(() => {
    const el = getAudio();
    el.pause();
    el.currentTime = 0;
    nowPlayingRef.current = null;
    isPlayingRef.current = false;
    setNowPlaying(null);
    setIsPlaying(false);
    setProgress(0);
    setErrorMessage(null);
  }, []);

  return (
    <PlayerContext.Provider value={{ nowPlaying, isPlaying, progress, errorMessage, play, toggle, stop, seek }}>
      {children}
    </PlayerContext.Provider>
  );
}

export function usePlayer(): PlayerState {
  const ctx = useContext(PlayerContext);
  if (!ctx) throw new Error("usePlayer must be used within a PlayerProvider");
  return ctx;
}
