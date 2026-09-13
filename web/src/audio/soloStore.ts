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
      .finally(() => {
        pending = null;
      });
  }
  return pending;
}

/** Solo [lo, hi] Hz. Must be called from a user gesture (Safari). */
export async function soloBand(lo: number, hi: number): Promise<void> {
  requested = [lo, hi];
  const solo = await ensureChain();
  // A drag fires many of these; whatever was asked for last wins, and a
  // soloOff() during the await must not be undone.
  if (!solo || requested === null) return;
  solo.setBand(requested[0], requested[1]);
  band = solo.band();
  emit();
}

export function soloOff(): void {
  requested = null;
  chain?.clear();
  band = null;
  emit();
}

export function currentBand(): [number, number] | null {
  return band;
}

function subscribe(listener: () => void): () => void {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

/** The soloed band, or null. Re-renders whichever views show it. */
export function useSoloBand(): [number, number] | null {
  return useSyncExternalStore(subscribe, currentBand, currentBand);
}

/** Test seam: drop the chain so the next solo rebuilds it. */
export function resetSolo(): void {
  chain = null;
  pending = null;
  requested = null;
  band = null;
  emit();
}
