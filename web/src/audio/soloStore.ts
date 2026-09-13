// One band-solo chain for the whole app: the drag strip in SOUND mode and
// the WhySimilar attribution bars in PROOF mode both drive the same filters
// on the same <audio> element, so whichever asked last wins and there is
// never a second MediaElementAudioSourceNode.

import { useSyncExternalStore } from "react";
import { createBandSolo, type BandSolo } from "./bandSolo";
import { attachGraph } from "../player/usePlayer";

let chain: BandSolo | null = null;
let band: [number, number] | null = null;
const listeners = new Set<() => void>();

function emit() {
  for (const listener of listeners) listener();
}

function ensureChain(): BandSolo | null {
  if (chain) return chain;
  const graph = attachGraph();
  if (!graph) return null;
  chain = createBandSolo(graph.context, graph.source);
  return chain;
}

/** Solo [lo, hi] Hz. Must be called from a user gesture (Safari). */
export function soloBand(lo: number, hi: number): void {
  const solo = ensureChain();
  if (!solo) return;
  solo.setBand(lo, hi);
  band = solo.band();
  emit();
}

export function soloOff(): void {
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
  band = null;
  emit();
}
