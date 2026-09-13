import { createContext, useCallback, useContext, useEffect, useRef, useState } from "react";
import type { ReactNode } from "react";
import type { Track } from "../api/types";
import { previewAudioUrl, previewUrl } from "../api/client";

let audio: HTMLAudioElement | null = null;
let currentTrackId: string | null = null;

function getAudio(): HTMLAudioElement {
  if (!audio) audio = new Audio();
  return audio;
}

// ---- Web Audio graph (used by SOUND mode's band solo) ----
//
// Two things have to be true before a MediaElementAudioSourceNode is worth
// anything, and both are easy to get wrong:
//
// 1. The media has to be same-origin (or CORS-clean). /api/preview/{id} is a
//    302 to Deezer's CDN, which sends no CORS header, so a source node over
//    it outputs SILENCE — and since the node can only be created once per
//    element, that silence would last the whole session. So the moment a
//    solo is asked for, the element is moved onto /api/preview/{id}/audio,
//    our own proxy, keeping its position and play state. Ordinary playback
//    stays on the redirect, which is what keeps the mp3 bytes off this host
//    until someone actually wants SOUND mode.
// 2. Creating the node detaches the element from the speakers, so the graph
//    connects source -> destination itself. That is the audible default;
//    band solo takes the connection over and gives it back on clear().
//
// Lazy and cached, and only ever called from a user gesture (Safari refuses
// to start a context otherwise).

export interface AudioGraph {
  context: AudioContext;
  source: MediaElementAudioSourceNode;
}

let graph: AudioGraph | null = null;
let attaching: Promise<AudioGraph | null> | null = null;

/** True when the element is already playing this track through the proxy. */
function isProxySrc(el: HTMLAudioElement, trackId: string): boolean {
  return Boolean(el.src) && el.src.endsWith(previewAudioUrl(trackId));
}

/** Swap the element onto the same-origin proxy, keeping position and state. */
async function moveToProxy(el: HTMLAudioElement, trackId: string): Promise<void> {
  const wasPlaying = !el.paused;
  const position = el.currentTime;
  el.src = previewAudioUrl(trackId);
  await new Promise<void>((resolve) => {
    const done = () => {
      el.removeEventListener("loadedmetadata", done);
      el.removeEventListener("error", done);
      resolve();
    };
    el.addEventListener("loadedmetadata", done);
    el.addEventListener("error", done);
    el.load?.();
  });
  try {
    el.currentTime = position;
  } catch {
    /* not seekable yet; playback just restarts from the top */
  }
  if (wasPlaying) void el.play();
}

async function createGraph(): Promise<AudioGraph | null> {
  const Ctor =
    globalThis.AudioContext ??
    (globalThis as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
  if (!Ctor) return null;
  const el = getAudio();
  // Harmless while same-origin; required the day the redirect target grows
  // CORS headers.
  el.crossOrigin = "anonymous";
  if (currentTrackId && !isProxySrc(el, currentTrackId)) {
    await moveToProxy(el, currentTrackId);
  }
  try {
    const context = new Ctor();
    const source = context.createMediaElementSource(el);
    source.connect(context.destination); // audible by default
    graph = { context, source };
    void context.resume?.();
    return graph;
  } catch {
    return null; // no Web Audio here; playback keeps working, unfiltered
  }
}

export function attachGraph(): Promise<AudioGraph | null> {
  if (graph) {
    void graph.context.resume?.();
    return Promise.resolve(graph);
  }
  if (!attaching) {
    attaching = createGraph().finally(() => {
      attaching = null;
    });
  }
  return attaching;
}

/** Whether SOUND mode has taken the element over; drives the URL play() uses. */
export const isGraphAttached = () => graph !== null;

/** Test seam: install a stub element and forget any attached graph. */
export function __resetPlayerForTests(el: HTMLAudioElement | null = null): void {
  audio = el;
  graph = null;
  attaching = null;
  currentTrackId = null;
}

interface PlayerState {
  nowPlaying: Track | null;
  isPlaying: boolean;
  progress: number;
  errorMessage: string | null;
  /** `startProgress` (0…1) jumps there as soon as the duration is known. */
  play(track: Track, startProgress?: number): void;
  toggle(): void;
  stop(): void;
  /** Jump to a fraction (0…1) of the current track. */
  seek(progress: number): void;
}

/** Seek to a fraction of the track, waiting for the duration if need be. */
function seekWhenReady(
  el: HTMLAudioElement,
  fraction: number,
  onProgress: (p: number) => void,
): void {
  const clamped = Math.max(0, Math.min(1, fraction));
  const apply = () => {
    if (!(el.duration > 0)) return;
    el.currentTime = clamped * el.duration;
    onProgress(clamped);
  };
  if (el.duration > 0) {
    apply();
    return;
  }
  const onMeta = () => {
    el.removeEventListener("loadedmetadata", onMeta);
    apply();
  };
  el.addEventListener("loadedmetadata", onMeta);
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

  const play = useCallback((track: Track, startProgress?: number) => {
    const el = getAudio();
    const trackId = track.track_id;
    nowPlayingRef.current = track;
    currentTrackId = trackId;
    setNowPlaying(track);
    setErrorMessage(null);
    setProgress(startProgress ?? 0);
    // Once SOUND mode owns the element it must stay same-origin, or the
    // source node goes silent on the next track.
    el.src = isGraphAttached() ? previewAudioUrl(trackId) : previewUrl(trackId);
    if (startProgress !== undefined) seekWhenReady(el, startProgress, setProgress);
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
