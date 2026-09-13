// Mel spectrogram, matching the iOS SOUND mode exactly: 96 bands, FFT 2048,
// hop 1024, triangular filters on the HTK mel scale (the same 2595·log10
// formula MelSpectrogram.swift uses), dB floor −80.

import { hann, magnitudes } from "./fft";

export const DB_FLOOR = -80;
export const FFT_SIZE = 2048;
export const HOP = 1024;
export const BANDS = 96;
/** dB range the colormap spans below the track's own peak. */
export const DB_RANGE = 70;

export const melFromHz = (hz: number) => 2595 * Math.log10(1 + hz / 700);
export const hzFromMel = (mel: number) => 700 * (10 ** (mel / 2595) - 1);

export interface MelOptions {
  fftSize?: number;
  hop?: number;
  bands?: number;
  minHz?: number;
}

export interface MelResult {
  /** One Float32Array of `bands` dB values per hop. */
  frames: Float32Array[];
  bands: number;
  hop: number;
  fftSize: number;
  sampleRate: number;
  /** Loudest dB value anywhere in the spectrogram; the colormap anchors here. */
  peak: number;
  /** bands + 2 Hz edges: band b spans edges[b] … edges[b + 2]. */
  bandEdgesHz: Float64Array;
}

/** bands + 2 equally-mel-spaced Hz edges over [minHz, sampleRate / 2]. */
export function bandEdges(bands: number, sampleRate: number, minHz = 20): Float64Array {
  const melMin = melFromHz(minHz);
  const melMax = melFromHz(sampleRate / 2);
  const edges = new Float64Array(bands + 2);
  for (let i = 0; i < bands + 2; i++) {
    edges[i] = hzFromMel(melMin + ((melMax - melMin) * i) / (bands + 1));
  }
  return edges;
}

interface Filter {
  start: number;
  weights: Float64Array;
}

function triangularFilters(
  edges: Float64Array,
  bands: number,
  fftSize: number,
  sampleRate: number,
): Filter[] {
  const binHz = sampleRate / fftSize;
  const nBins = fftSize / 2;
  const filters: Filter[] = [];
  for (let band = 0; band < bands; band++) {
    const lo = edges[band];
    const mid = edges[band + 1];
    const hi = edges[band + 2];
    const start = Math.max(Math.floor(lo / binHz) + 1, 0);
    const end = Math.min(Math.floor(hi / binHz), nBins - 1);
    if (start > end) {
      filters.push({ start: 0, weights: new Float64Array(0) });
      continue;
    }
    const weights = new Float64Array(end - start + 1);
    for (let bin = start; bin <= end; bin++) {
      const hz = bin * binHz;
      const w = hz <= mid ? (hz - lo) / (mid - lo) : (hi - hz) / (hi - mid);
      weights[bin - start] = Math.max(w, 0);
    }
    filters.push({ start, weights });
  }
  return filters;
}

/** Mel spectrogram in dB. Frame count is floor((n − fftSize) / hop) + 1. */
export function melSpectrogram(
  samples: Float32Array,
  sampleRate: number,
  options: MelOptions = {},
): MelResult {
  const fftSize = options.fftSize ?? FFT_SIZE;
  const hop = options.hop ?? HOP;
  const bands = options.bands ?? BANDS;
  const edges = bandEdges(bands, sampleRate, options.minHz ?? 20);
  const base: Omit<MelResult, "frames" | "peak"> = {
    bands,
    hop,
    fftSize,
    sampleRate,
    bandEdgesHz: edges,
  };
  if (samples.length < fftSize) return { ...base, frames: [], peak: DB_FLOOR };

  const filters = triangularFilters(edges, bands, fftSize, sampleRate);
  const window = hann(fftSize);
  const frameCount = Math.floor((samples.length - fftSize) / hop) + 1;
  const frames: Float32Array[] = [];
  let peak = DB_FLOOR;

  for (let f = 0; f < frameCount; f++) {
    const mag = magnitudes(samples, f * hop, fftSize, window);
    const frame = new Float32Array(bands);
    for (let band = 0; band < bands; band++) {
      const { start, weights } = filters[band];
      let energy = 0;
      for (let i = 0; i < weights.length; i++) {
        const m = mag[start + i];
        energy += m * m * weights[i];
      }
      const db = Math.max(10 * Math.log10(Math.max(energy, 1e-10)), DB_FLOOR);
      frame[band] = db;
      if (db > peak) peak = db;
    }
    frames.push(frame);
  }
  return { ...base, frames, peak };
}

/** A frequency as e.g. "240" or "1.2k". */
export function formatHz(n: number): string {
  if (n < 1000) return String(Math.round(n));
  return `${(n / 1000).toFixed(1)}k`;
}

/** Hz span of band `index`, using the same triangular edges as the filterbank. */
export function bandHzRange(edges: Float64Array, index: number, bands: number): [number, number] {
  const i = Math.max(0, Math.min(index, bands - 1));
  return [edges[i], edges[i + 2]];
}
