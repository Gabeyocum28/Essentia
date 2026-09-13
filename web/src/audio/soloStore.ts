// One band-solo chain for the whole app: the drag strip in SOUND mode and
// the WhySimilar attribution bars in PROOF mode both drive the same filters
// on the same <audio> element, so whichever asked last wins and there is
// never a second MediaElementAudioSourceNode.

import { useSyncExternalStore } from "react";
import { createBandSolo, type BandSolo } from "./bandSolo";
import { attachGraph } from "../player/usePlayer";

let chain: BandSolo | null = null;
let pending: Promise<BandSolo | null> | null = null;
let band: [number, number] | null = null;
/** What the user last asked for; applied once the chain exists. */
let requested: [number, number] | null = null;
/** Set when attaching failed outright, for the views to show. */
let error: string | null = null;
const ATTACH_FAILED = "Couldn't enable band solo.";
const listeners = new Set<() => void>();

function emit() {
  for (const listener of listeners) listener();
}

function ensureChain(): Promise<BandSolo | null> {
  if (chain) return Promise.resolve(chain);
  if (!pending) {
    // Attaching is async: it may first have to move the element onto the
    // same-origin proxy, or a cross-origin source node would play silence.
    pending = attachGraph()
      .then((graph) => {
        if (!graph) return null;
        chain = createBandSolo(graph.context, graph.source);
        return chain;
      })
      .catch(() => {
        // attachGraph rejects when the element could not be moved onto the
        // same-origin proxy (it times out after 5 s). Ordinary playback is
        // untouched -- only the solo is unavailable -- so this is a message,
        // not a thrown error: soloBand() is called from click handlers that
        // do not await it, and a rejection there is an unhandled one.
        error = ATTACH_FAILED;
        return null;
      })
      .finally(() => {
        pending = null;
      });
  }
  return pending;
}

/** Solo [lo, hi] Hz. Must be called from a user gesture (Safari). */
export async function soloBand(lo: number, hi: number): Promise<void> {
  requested = [lo, hi];
  error = null;
  const solo = await ensureChain();
  // A drag fires many of these; whatever was asked for last wins, and a
  // soloOff() during the await must not be undone.
  if (!solo || requested === null) {
    if (error) emit();
    return;
  }
  solo.setBand(requested[0], requested[1]);
  band = solo.band();
  emit();
}

export function soloOff(): void {
  requested = null;
  error = null;
  chain?.clear();
  band = null;
  emit();
}

export function currentBand(): [number, number] | null {
  return band;
}

/** "Couldn't enable band solo." once an attach has failed, else null. */
export function soloError(): string | null {
  return error;
}

function subscribe(listener: () => void): () => void {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

/** The soloed band, or null. Re-renders whichever views show it. */
export function useSoloBand(): [number, number] | null {
  return useSyncExternalStore(subscribe, currentBand, currentBand);
}

/** The attach failure message, or null. Re-renders whichever views show it. */
export function useSoloError(): string | null {
  return useSyncExternalStore(subscribe, soloError, soloError);
}

/** Test seam: drop the chain so the next solo rebuilds it. */
export function resetSolo(): void {
  chain = null;
  pending = null;
  requested = null;
  band = null;
  error = null;
  emit();
}
