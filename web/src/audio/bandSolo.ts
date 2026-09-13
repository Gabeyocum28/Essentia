// Band solo: keep one frequency band of the playing preview audible.
//
// iOS resynthesizes the band from a masked STFT; in the browser the same
// effect is a filter chain on the live <audio> element, which costs nothing
// and stays in sync with playback for free. Two cascaded highpasses and two
// cascaded lowpasses (Q = 1/sqrt(2), so each pair is Butterworth) give
// 24 dB/oct skirts — steep enough that a soloed band reads as "only this".
//
// The graph is built once per AudioContext/source pair and then only
// re-tuned: a MediaElementAudioSourceNode can be created once per element,
// and rebuilding filters mid-playback clicks.

/** The subset of the Web Audio API this module needs; lets tests pass a stub. */
export interface AudioGraphContext {
  destination: AudioNode;
  createBiquadFilter(): BiquadFilterNode;
  createGain(): GainNode;
  currentTime?: number;
}

export const SOLO_Q = Math.SQRT1_2; // 0.7071

export interface BandSolo {
  /** Solo [lo, hi] Hz: the band path opens, the bypass closes. */
  setBand(lo: number, hi: number): void;
  /** Back to unfiltered playback. */
  clear(): void;
  /** The currently soloed band, or null. */
  band(): [number, number] | null;
}

type AnyNode = { connect(destination: unknown): unknown };

function setParam(param: AudioParam, value: number, now: number) {
  // setTargetAtTime where available: a step on a filter frequency during
  // playback is audible as a zip.
  if (typeof param.setTargetAtTime === "function") param.setTargetAtTime(value, now, 0.01);
  else param.value = value;
}

/**
 * Builds `source -> [hp, hp, lp, lp] -> bandGain -> destination` alongside
 * `source -> bypassGain -> destination`. Solo off = bypass 1, band 0.
 */
export function createBandSolo(context: AudioGraphContext, source: AnyNode): BandSolo {
  const hp1 = context.createBiquadFilter();
  const hp2 = context.createBiquadFilter();
  const lp1 = context.createBiquadFilter();
  const lp2 = context.createBiquadFilter();
  const bandGain = context.createGain();
  const bypassGain = context.createGain();

  for (const hp of [hp1, hp2]) {
    hp.type = "highpass";
    hp.Q.value = SOLO_Q;
    hp.frequency.value = 20;
  }
  for (const lp of [lp1, lp2]) {
    lp.type = "lowpass";
    lp.Q.value = SOLO_Q;
    lp.frequency.value = 20000;
  }
  bandGain.gain.value = 0;
  bypassGain.gain.value = 1;

  source.connect(hp1);
  hp1.connect(hp2);
  hp2.connect(lp1);
  lp1.connect(lp2);
  lp2.connect(bandGain);
  bandGain.connect(context.destination);

  source.connect(bypassGain);
  bypassGain.connect(context.destination);

  let current: [number, number] | null = null;

  return {
    setBand(lo: number, hi: number) {
      const low = Math.max(10, Math.min(lo, hi));
      const high = Math.max(low + 1, Math.max(lo, hi));
      const now = context.currentTime ?? 0;
      setParam(hp1.frequency, low, now);
      setParam(hp2.frequency, low, now);
      setParam(lp1.frequency, high, now);
      setParam(lp2.frequency, high, now);
      setParam(bandGain.gain, 1, now);
      setParam(bypassGain.gain, 0, now);
      current = [low, high];
    },
    clear() {
      const now = context.currentTime ?? 0;
      setParam(bandGain.gain, 0, now);
      setParam(bypassGain.gain, 1, now);
      current = null;
    },
    band: () => current,
  };
}
