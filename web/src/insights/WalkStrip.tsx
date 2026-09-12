import { Artwork } from "../components/Artwork";
import { usePlayer } from "../player/usePlayer";
import type { VizWalk } from "../api/types";

interface Props {
  walk: VizWalk;
  revealCount: number;
}

export function WalkStrip({ walk, revealCount }: Props) {
  const { play } = usePlayer();

  return (
    <div className="walk-strip">
      <div className="walk-strip-header">
        <span className="mono walk-strip-badge">detour ×{walk.detour.toFixed(1)}</span>
        <span className="mono walk-strip-metric">geodesic {walk.geodesic.toFixed(3)}</span>
        <span className="mono walk-strip-metric">ambient {walk.ambient.toFixed(3)}</span>
      </div>
      <div className="walk-strip-steps">
        {walk.path.slice(0, revealCount).map((step, i) => (
          <button
            key={`${step.track_id}-${i}`}
            type="button"
            className="walk-strip-item"
            onClick={() => play(step)}
            aria-label={`Play ${step.title}`}
          >
            <Artwork url={step.artwork_url} size={40} />
            <div className="walk-strip-info">
              <div className="walk-strip-title">{step.title}</div>
              <div className="walk-strip-artist">{step.artist}</div>
            </div>
          </button>
        ))}
      </div>
    </div>
  );
}
