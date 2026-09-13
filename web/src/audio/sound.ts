// The pure analysis the SOUND mode needs, in one call. Lives outside the
// worker entry so it can be unit-tested (and run inline when Workers are
// unavailable) without touching `self` or `postMessage`.

import { BANDS, DB_FLOOR, FFT_SIZE, HOP, melSpectrogram } from "./mel";
import { MAX_COLUMNS, selfSimilarity } from "./ssm";

export interface SoundAnalysis {
  /** Mel dB values, frame-major: frame f, band b at f * bands + b. */
  mel: Float32Array;
  frameCount: number;
  bands: number;
  /** bands + 2 Hz edges of the filterbank. */
  bandEdgesHz: Float64Array;
  peak: number;
  sampleRate: number;
  hop: number;
  fftSize: number;
  /** Row-major SSM, `ssmColumns × ssmColumns`. */
  ssm: Float32Array;
  ssmColumns: number;
  /** Seconds of audio the frames cover. */
  duration: number;
}

export function computeSound(samples: Float32Array, sampleRate: number): SoundAnalysis {
  const spec = melSpectrogram(samples, sampleRate, { fftSize: FFT_SIZE, hop: HOP, bands: BANDS });
  const frameCount = spec.frames.length;
  const mel = new Float32Array(frameCount * spec.bands);
  for (let f = 0; f < frameCount; f++) mel.set(spec.frames[f], f * spec.bands);
  const { columns, values } = selfSimilarity(spec.frames, MAX_COLUMNS);
  return {
    mel,
    frameCount,
    bands: spec.bands,
    bandEdgesHz: spec.bandEdgesHz,
    peak: frameCount ? spec.peak : DB_FLOOR,
    sampleRate,
    hop: spec.hop,
    fftSize: spec.fftSize,
    ssm: values,
    ssmColumns: columns,
    duration: samples.length / sampleRate,
  };
}
