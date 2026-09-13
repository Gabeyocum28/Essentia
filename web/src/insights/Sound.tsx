import { useEffect, useState } from "react";
import { analyzeSound, type SoundAnalysis } from "../audio/analyze";
import { loadTrackAudio, rememberAnalysis } from "../audio/decode";
import { soloOff, useSoloBand } from "../audio/soloStore";
import { usePlayer } from "../player/usePlayer";
import { BandStrip } from "./BandStrip";
import { SelfSimilarity } from "./SelfSimilarity";
import { SpectrogramView } from "./SpectrogramView";
import type { Track } from "../api/types";

interface Props {
  /** The selected rec, or the seed when nothing is selected. */
  track: Track;
}

type Status = "loading" | "ready" | "error";

/**
 * SOUND mode: what the track actually sounds like, as opposed to where it
 * sits in feature space. The mp3 comes through our own origin (the CDN sends
 * no CORS header, so decodeAudioData can't touch the redirect), is decoded
 * to mono, and the mel spectrogram + self-similarity are computed in a
 * worker. Parity with the iOS SOUND mode is deliberate: same 96 bands, same
 * FFT 2048 / hop 1024, same colormap, same 70 dB window.
 */
export function Sound({ track }: Props) {
  const [status, setStatus] = useState<Status>("loading");
  const [analysis, setAnalysis] = useState<SoundAnalysis | null>(null);
  const { nowPlaying, progress, play, seek } = usePlayer();
  const band = useSoloBand();

  useEffect(() => {
    let cancelled = false;
    const controller = new AbortController();
    setStatus("loading");
    setAnalysis(null);

    (async () => {
      try {
        const audio = await loadTrackAudio(track.track_id, { signal: controller.signal });
        // Re-selecting a rec that's still in the LRU costs nothing.
        const result =
          audio.analysis ??
          (await analyzeSound(audio.samples, audio.sampleRate, { signal: controller.signal }));
        if (cancelled) return;
        rememberAnalysis(track.track_id, result);
        setAnalysis(result);
        setStatus("ready");
      } catch {
        if (!cancelled) setStatus("error");
      }
    })();

    return () => {
      cancelled = true;
      controller.abort();
    };
  }, [track.track_id]);

  const isCurrent = nowPlaying?.track_id === track.track_id;
  // Clicking a spot in a track that isn't playing should start it there, not
  // at the top — the player seeks once the duration is known.
  const handleSeek = (fraction: number) => {
    if (isCurrent) seek(fraction);
    else play(track, fraction);
  };

  return (
    <div className="sound">
      <div className="sound-header">
        <div className="sound-title">
          {track.title} <span className="sound-artist">{track.artist}</span>
        </div>
        <button type="button" className="sound-solo-off" onClick={soloOff} disabled={!band}>
          Solo off
        </button>
      </div>

      {status === "loading" && <div className="skeleton sound-skeleton" data-testid="sound-skeleton" />}
      {status === "error" && <p className="error-box">Couldn’t load this preview’s audio.</p>}

      {status === "ready" && analysis && (
        <>
          <div className="sound-row">
            <SpectrogramView
              analysis={analysis}
              playhead={isCurrent ? progress : null}
              onSeek={handleSeek}
            />
            <BandStrip
              bandEdgesHz={analysis.bandEdgesHz}
              bands={analysis.bands}
              band={band}
            />
          </div>
          <SelfSimilarity analysis={analysis} onSeek={handleSeek} />
        </>
      )}
    </div>
  );
}
