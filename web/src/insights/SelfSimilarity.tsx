import { useEffect, useRef } from "react";
import { heatBytes } from "../audio/colormap";
import type { SoundAnalysis } from "../audio/analyze";

interface Props {
  analysis: SoundAnalysis;
  onSeek: (progress: number) => void;
}

/**
 * Cosine self-similarity of the pooled mel frames. Both axes are time, so a
 * bright off-diagonal block means "this stretch sounds like that stretch" —
 * choruses, refrains, restated heads. Clicking column x seeks there.
 */
export function SelfSimilarity({ analysis, onSeek }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const { ssm, ssmColumns } = analysis;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || ssmColumns === 0) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const image = ctx.createImageData(ssmColumns, ssmColumns);
    for (let i = 0; i < ssmColumns * ssmColumns; i++) {
      const [r, g, b] = heatBytes((ssm[i] + 1) / 2); // cosine is [−1, 1]
      image.data[i * 4] = r;
      image.data[i * 4 + 1] = g;
      image.data[i * 4 + 2] = b;
      image.data[i * 4 + 3] = 255;
    }
    ctx.putImageData(image, 0, 0);
  }, [ssm, ssmColumns]);

  const handleClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    if (rect.width === 0) return;
    onSeek(Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width)));
  };

  return (
    <div className="self-similarity">
      <canvas
        ref={canvasRef}
        className="self-similarity-canvas"
        width={Math.max(1, ssmColumns)}
        height={Math.max(1, ssmColumns)}
        aria-label="Self-similarity matrix"
        onClick={handleClick}
      />
    </div>
  );
}
