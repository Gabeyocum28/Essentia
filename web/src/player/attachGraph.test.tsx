// The band-solo graph has one job that is easy to get wrong: the element must
// be on the same-origin proxy BEFORE the source node exists, or the node is
// created over a CORS-tainted stream and outputs silence for the session.

import { act, renderHook } from "@testing-library/react";
import type { ReactNode } from "react";
import { PlayerProvider, __resetPlayerForTests, attachGraph, isGraphAttached, usePlayer } from "./usePlayer";
import type { Track } from "../api/types";

function track(id: string): Track {
  return { track_id: id, title: `Title ${id}`, artist: "Artist", album: "Album",
    artwork_url: "", preview_url: "" };
}

function wrapper({ children }: { children: ReactNode }) {
  return <PlayerProvider>{children}</PlayerProvider>;
}

function stubElement() {
  const listeners = new Map<string, Set<() => void>>();
  const el = {
    src: "",
    crossOrigin: null as string | null,
    currentTime: 0,
    duration: 30,
    paused: true,
    played: 0,
    loads: 0,
    addEventListener(type: string, fn: () => void) {
      (listeners.get(type) ?? listeners.set(type, new Set()).get(type)!).add(fn);
    },
    removeEventListener(type: string, fn: () => void) {
      listeners.get(type)?.delete(fn);
    },
    load() {
      el.loads++;
      // Real elements fire this asynchronously once headers land.
      queueMicrotask(() => {
        for (const fn of [...(listeners.get("loadedmetadata") ?? [])]) fn();
      });
    },
    play() {
      el.played++;
      el.paused = false;
      return Promise.resolve();
    },
    pause() {
      el.paused = true;
    },
    /** Dispatch a media event by hand, for the failure paths. */
    fire(type: string) {
      for (const fn of [...(listeners.get(type) ?? [])]) fn();
    },
  };
  return el;
}

function stubAudioContext() {
  const created: unknown[] = [];
  const connections: string[] = [];
  // Every source node ever made. The context itself is now constructed
  // BEFORE the proxy swap is awaited (Safari only starts one inside the
  // gesture), so "nothing was attached" has to be asserted on the nodes.
  const sources: unknown[] = [];
  const resumes: unknown[] = [];
  class Ctx {
    destination = { id: "destination" };
    currentTime = 0;
    constructor() {
      created.push(this);
    }
    createMediaElementSource(el: unknown) {
      const node = {
        el,
        // Recorded so the test can prove the default path is audible.
        connect: (to: { id?: string }) => connections.push(to.id ?? "?"),
        disconnect: () => {},
      };
      sources.push(node);
      return node;
    }
    resume() {
      resumes.push(this);
      return Promise.resolve();
    }
  }
  return { Ctx, created, connections, sources, resumes };
}

let el: ReturnType<typeof stubElement>;

beforeEach(() => {
  el = stubElement();
  __resetPlayerForTests(el as unknown as HTMLAudioElement);
});

afterEach(() => {
  __resetPlayerForTests(null);
  Reflect.deleteProperty(globalThis, "AudioContext");
});

test("with no Web Audio, attaching fails soft and playback is untouched", async () => {
  Reflect.deleteProperty(globalThis, "AudioContext");
  expect(await attachGraph()).toBeNull();
  expect(isGraphAttached()).toBe(false);
});

test("attaching connects the source straight to destination, so audio still plays", async () => {
  const { Ctx, connections } = stubAudioContext();
  globalThis.AudioContext = Ctx as unknown as typeof AudioContext;

  const graph = await attachGraph();
  expect(graph).not.toBeNull();
  expect(connections).toEqual(["destination"]);
  expect(el.crossOrigin).toBe("anonymous");
  expect(isGraphAttached()).toBe(true);
});

test("attaching twice reuses the one graph (a source node can't be made twice)", async () => {
  const { Ctx, created } = stubAudioContext();
  globalThis.AudioContext = Ctx as unknown as typeof AudioContext;

  const first = await attachGraph();
  const second = await attachGraph();
  expect(second).toBe(first);
  expect(created).toHaveLength(1);
});

test("concurrent attaches share one in-flight attempt", async () => {
  const { Ctx, created } = stubAudioContext();
  globalThis.AudioContext = Ctx as unknown as typeof AudioContext;

  const [a, b] = await Promise.all([attachGraph(), attachGraph()]);
  expect(a).toBe(b);
  expect(created).toHaveLength(1);
});

test("a redirect src is swapped to the proxy, in position, before the node exists", async () => {
  const { Ctx } = stubAudioContext();
  globalThis.AudioContext = Ctx as unknown as typeof AudioContext;
  const { result } = renderHook(() => usePlayer(), { wrapper });

  await act(async () => {
    result.current.play(track("42"));
  });
  expect(el.src).toBe("/api/preview/42"); // the cheap redirect, as usual
  el.currentTime = 12;
  el.paused = false;
  const playsBefore = el.played;

  let srcWhenNodeMade = "";
  const realCreate = Ctx.prototype.createMediaElementSource;
  Ctx.prototype.createMediaElementSource = function (element: unknown) {
    srcWhenNodeMade = (element as { src: string }).src;
    return realCreate.call(this, element);
  };

  await act(async () => {
    await attachGraph();
  });

  expect(el.src).toBe("/api/preview/42/audio");
  expect(srcWhenNodeMade).toBe("/api/preview/42/audio"); // swapped FIRST
  expect(el.currentTime).toBe(12); // position kept
  expect(el.played).toBe(playsBefore + 1); // it was playing, so it resumes
});

test("an already-proxied element is left alone", async () => {
  const { Ctx } = stubAudioContext();
  globalThis.AudioContext = Ctx as unknown as typeof AudioContext;
  const { result } = renderHook(() => usePlayer(), { wrapper });

  await act(async () => {
    result.current.play(track("42"));
  });
  await act(async () => {
    await attachGraph();
  });
  const loads = el.loads;

  __resetPlayerForTests(el as unknown as HTMLAudioElement); // forget the graph, keep the element
  // Re-play so the module knows which track is loaded, then re-attach.
  const second = renderHook(() => usePlayer(), { wrapper });
  await act(async () => {
    second.result.current.play(track("42"));
  });
  el.src = "/api/preview/42/audio";
  await act(async () => {
    await attachGraph();
  });
  expect(el.loads).toBe(loads); // no reload
});

test("a proxy that never answers times out, restores the src, and rejects", async () => {
  const { Ctx, sources } = stubAudioContext();
  globalThis.AudioContext = Ctx as unknown as typeof AudioContext;
  const { result } = renderHook(() => usePlayer(), { wrapper });

  await act(async () => {
    result.current.play(track("42"));
  });
  el.currentTime = 12;
  // A stalled proxy response: neither `loadedmetadata` nor `error` ever
  // fires, so without the timeout this await never settles.
  el.load = () => {
    el.loads++;
  };

  vi.useFakeTimers();
  try {
    const attaching = attachGraph();
    const settled = expect(attaching).rejects.toThrow(/timed out/);
    await vi.advanceTimersByTimeAsync(5000);
    await settled;
  } finally {
    vi.useRealTimers();
  }

  expect(el.src).toBe("/api/preview/42"); // put back the way it was
  expect(el.currentTime).toBe(12);
  expect(sources).toHaveLength(0); // no source node over a half-loaded element
  expect(isGraphAttached()).toBe(false);
});

test("once attached, play() keeps later tracks on the same-origin proxy", async () => {
  const { Ctx } = stubAudioContext();
  globalThis.AudioContext = Ctx as unknown as typeof AudioContext;
  const { result } = renderHook(() => usePlayer(), { wrapper });

  await act(async () => {
    result.current.play(track("42"));
  });
  await act(async () => {
    await attachGraph();
  });
  await act(async () => {
    result.current.play(track("99"));
  });
  expect(el.src).toBe("/api/preview/99/audio");
});

test("a proxy that errors rejects, restores the src, and leaves no source node", async () => {
  const { Ctx, sources } = stubAudioContext();
  globalThis.AudioContext = Ctx as unknown as typeof AudioContext;
  const { result } = renderHook(() => usePlayer(), { wrapper });

  await act(async () => {
    result.current.play(track("42"));
  });
  el.currentTime = 12;
  el.paused = false;
  const playsBefore = el.played;
  // The proxy answers, but with a 502: the element fires `error`, not
  // `loadedmetadata`. This used to RESOLVE, so a source node was built over
  // an element with no media and SOUND mode went silent for the session.
  el.load = () => {
    el.loads++;
    queueMicrotask(() => el.fire("error"));
  };

  await expect(attachGraph()).rejects.toThrow(/proxy/);

  expect(el.src).toBe("/api/preview/42"); // put back the way it was
  expect(el.currentTime).toBe(12);
  expect(el.played).toBe(playsBefore + 1); // it was playing, so it resumes
  expect(sources).toHaveLength(0);
  expect(isGraphAttached()).toBe(false);
});

test("the context is built and resumed before the src swap awaits (Safari's gesture)", async () => {
  const { Ctx, created, resumes } = stubAudioContext();
  globalThis.AudioContext = Ctx as unknown as typeof AudioContext;
  const { result } = renderHook(() => usePlayer(), { wrapper });

  await act(async () => {
    result.current.play(track("42"));
  });

  // Safari only lets a context start inside the gesture that asked for it,
  // and an `await` ends the gesture -- so the context must exist and have
  // been resumed by the time moveToProxy() starts loading.
  let contextsAtSwap = -1;
  let resumesAtSwap = -1;
  el.load = () => {
    el.loads++;
    contextsAtSwap = created.length;
    resumesAtSwap = resumes.length;
    queueMicrotask(() => el.fire("loadedmetadata"));
  };

  await act(async () => {
    await attachGraph();
  });

  expect(contextsAtSwap).toBe(1);
  expect(resumesAtSwap).toBeGreaterThanOrEqual(1);
  expect(isGraphAttached()).toBe(true);
});

test("play() and toggle() resume a suspended context", async () => {
  const { Ctx, resumes } = stubAudioContext();
  globalThis.AudioContext = Ctx as unknown as typeof AudioContext;
  const { result } = renderHook(() => usePlayer(), { wrapper });

  await act(async () => {
    result.current.play(track("42"));
  });
  await act(async () => {
    await attachGraph();
  });
  const after = resumes.length;

  await act(async () => {
    result.current.play(track("99"));
  });
  expect(resumes.length).toBeGreaterThan(after);

  const afterPlay = resumes.length;
  await act(async () => {
    result.current.toggle();
  });
  expect(resumes.length).toBeGreaterThan(afterPlay);
});
