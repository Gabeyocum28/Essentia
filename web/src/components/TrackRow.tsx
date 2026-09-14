import { Artwork } from "./Artwork";
import { Attribution } from "./Attribution";
import { usePlayer } from "../player/usePlayer";
import type { Track } from "../api/types";

interface Props {
  track: Track;
  onSelect?: (track: Track) => void;
  showScore?: boolean;
}

export function TrackRow({ track, onSelect, showScore }: Props) {
  const { play } = usePlayer();

  return (
    <div className="track-row">
      <button
        type="button"
        className="track-row-play"
        aria-label={`Play ${track.title}`}
        onClick={(e) => {
          e.stopPropagation();
          play(track);
        }}
      >
        ▶
      </button>
      <Artwork url={track.artwork_url} size={56} />
      {/* The credit is a sibling of the select button, not a child: a link
          inside a button is invalid HTML and the browser would swallow the
          click on one of them. */}
      <div className="track-row-body">
        <button
          type="button"
          className="track-row-info"
          onClick={() => onSelect?.(track)}
          disabled={!onSelect}
        >
          <div className="track-row-title">{track.title}</div>
          <div className="track-row-artist">{track.artist}</div>
        </button>
        <Attribution track={track} />
      </div>
      {showScore && track.score !== undefined && (
        <span className="mono track-row-score">{track.score.toFixed(4)}</span>
      )}
    </div>
  );
}
