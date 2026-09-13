import { useEffect, useRef } from "react";
import { heatBytes } from "../audio/colormap";
import { DB_RANGE } from "../audio/mel";
import type { SoundAnalysis } from "../audio/analyze";

interface Props {
  analysis: SoundAnalysis;
  /** 0…1 playhead position, or null when this track isn't playing. */
  playhead: number | null;
  onSeek: (progress: number) => void;
}

/**
 * Mel spectrogram: time to the right, low frequency at the bottom, colors
 * anchored so the track's own loudest moment is the top of the ramp and
 * everything 70 dB below it is the floor.
 *
 * The canvas is drawn at its natural resolution (one pixel per frame per
 * band) and stretched by CSS, so a resize costs nothing and the analysis is
 * only rasterized once per track.
 */
export function SpectrogramView({ analysis, playhead, onSeek }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const { frameCount, bands, mel, peak } = analysis;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || frameCount === 0) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const image = ctx.createImageData(frameCount, bands);
    for (let f = 0; f < frameCount; f++) {
      for (let b = 0; b < bands; b++) {
        const row = bands - 1 - b; // row 0 is the top = the highest band
        const t = (mel[f * bands + b] - peak + DB_RANGE) / DB_RANGE;
        const [r, g, bl] = heatBytes(t);
        const idx = (row * frameCount + f) * 4;
        image.data[idx] = r;
        image.data[idx + 1] = g;
        image.data[idx + 2] = bl;
        image.data[idx + 3] = 255;
      }
    }
    ctx.putImageData(image, 0, 0);
  }, [frameCount, bands, mel, peak]);

  const handleClick = (e: React.MouseEvent<HTMLDivElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    if (rect.width === 0) return;
    onSeek(Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width)));
  };

  return (
    <div className="spectrogram" onClick={handleClick}>
      <canvas
        ref={canvasRef}
        className="spectrogram-canvas"
        width={Math.max(1, frameCount)}
        height={Math.max(1, bands)}
        aria-label="Mel spectrogram"
      />
      {playhead !== null && (
        <div
          className="spectrogram-playhead"
          data-testid="spectrogram-playhead"
          style={{ left: `${Math.max(0, Math.min(1, playhead)) * 100}%` }}
        />
      )}
    </div>
  );
}
