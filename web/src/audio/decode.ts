// Fetch a preview's mp3 bytes from our own origin and decode them to mono
// Float32 samples. Same-origin matters: decodeAudioData needs the bytes, and
// the ordinary /preview 302 lands on a CDN that sends no CORS header — hence
// the server's /preview/{id}/audio proxy.

import { previewAudioUrl } from "../api/client";

export interface DecodedAudio {
  samples: Float32Array;
  sampleRate: number;
  duration: number;
}

/** Small LRU: flipping between a few recs shouldn't refetch or re-decode. */
const CACHE_LIMIT = 5;
const cache = new Map<string, DecodedAudio>();

export function clearAudioCache(): void {
  cache.clear();
}

function remember(id: string, audio: DecodedAudio): DecodedAudio {
  cache.set(id, audio);
  while (cache.size > CACHE_LIMIT) {
    const oldest = cache.keys().next().value;
    if (oldest === undefined) break;
    cache.delete(oldest);
  }
  return audio;
}

/** Average the channels down to mono; the spectrogram is a mono view. */
export function toMono(buffer: AudioBuffer): Float32Array {
  const channels = buffer.numberOfChannels;
  const out = new Float32Array(buffer.length);
  if (channels === 0) return out;
  for (let c = 0; c < channels; c++) {
    const data = buffer.getChannelData(c);
    for (let i = 0; i < out.length; i++) out[i] += data[i];
  }
  if (channels > 1) for (let i = 0; i < out.length; i++) out[i] /= channels;
  return out;
}

// One decoding context for the whole app: browsers cap the number of live
// AudioContexts at a handful, and a fresh one per track would burn through
// them. It only ever decodes — nothing is routed to its destination.
let decodeContext: BaseAudioContext | null = null;

function sharedContext(): BaseAudioContext {
  if (decodeContext) return decodeContext;
  const Ctor =
    globalThis.AudioContext ??
    (globalThis as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
  if (!Ctor) throw new Error("Web Audio is unavailable in this browser");
  decodeContext = new Ctor();
  return decodeContext;
}

interface DecodeDeps {
  /** Decoding context. Defaults to a throwaway one — an OfflineAudioContext
   * would need the length up front, and a plain context here never plays. */
  context?: BaseAudioContext;
  signal?: AbortSignal;
}

export async function loadTrackAudio(trackId: string, deps: DecodeDeps = {}): Promise<DecodedAudio> {
  const cached = cache.get(trackId);
  if (cached) {
    // Refresh recency.
    cache.delete(trackId);
    return remember(trackId, cached);
  }

  const res = await fetch(previewAudioUrl(trackId), { signal: deps.signal });
  if (!res.ok) throw new Error(`audio fetch failed: ${res.status}`);
  const bytes = await res.arrayBuffer();

  const context = deps.context ?? sharedContext();
  const buffer = await context.decodeAudioData(bytes);
  const audio: DecodedAudio = {
    samples: toMono(buffer),
    sampleRate: buffer.sampleRate,
    duration: buffer.duration ?? buffer.length / buffer.sampleRate,
  };
  return remember(trackId, audio);
}
