// Spectrogram + self-similarity off the main thread: ~1300 FFTs per preview
// would otherwise stall the UI for the better part of a second. No DOM APIs
// in here — the real work lives in ./sound so it stays testable.

import { computeSound, type SoundAnalysis } from "./sound";

export interface SoundRequest {
  samples: Float32Array;
  sampleRate: number;
}

// `self` is typed as a Window under lib.dom; in a module worker it is a
// DedicatedWorkerGlobalScope, whose postMessage takes a transfer list.
const scope = self as unknown as {
  onmessage: ((event: MessageEvent<SoundRequest>) => void) | null;
  postMessage(message: unknown, transfer?: Transferable[]): void;
};

scope.onmessage = (event: MessageEvent<SoundRequest>) => {
  const { samples, sampleRate } = event.data;
  const result: SoundAnalysis = computeSound(samples, sampleRate);
  // Transfer the big buffers rather than structured-clone them.
  scope.postMessage(result, [
    result.mel.buffer as ArrayBuffer,
    result.ssm.buffer as ArrayBuffer,
    result.bandEdgesHz.buffer as ArrayBuffer,
  ]);
};
