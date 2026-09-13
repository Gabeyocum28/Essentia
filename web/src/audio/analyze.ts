// Main-thread entry point to the spectrogram worker, with an inline fallback
// so anything without Workers (jsdom, an old Safari) still works, just with
// a blocked frame.

import { computeSound, type SoundAnalysis } from "./sound";

export type { SoundAnalysis };

export function analyzeSound(samples: Float32Array, sampleRate: number): Promise<SoundAnalysis> {
  if (typeof Worker === "undefined") {
    return Promise.resolve(computeSound(samples, sampleRate));
  }
  return new Promise<SoundAnalysis>((resolve, reject) => {
    const worker = new Worker(new URL("./spectrogram.worker.ts", import.meta.url), {
      type: "module",
    });
    const done = (fn: () => void) => {
      worker.terminate();
      fn();
    };
    worker.onmessage = (event: MessageEvent<SoundAnalysis>) => done(() => resolve(event.data));
    worker.onerror = (event) => done(() => reject(new Error(event.message || "analysis failed")));
    // A copy, not a transfer: the decoded samples stay cached for the next
    // track switch.
    worker.postMessage({ samples, sampleRate });
  });
}
