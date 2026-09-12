import { Artwork } from "../components/Artwork";
import { usePlayer } from "../player/usePlayer";
import type { VizPoint, VizRec } from "../api/types";

interface Props {
  seed: VizPoint;
  recs: VizRec[];
  selectedId: string | null;
  onSelect: (id: string | null) => void;
}

export function RecStrip({ seed, recs, selectedId, onSelect }: Props) {
  const { play } = usePlayer();

  return (
    <div className="rec-strip">
      <button
        type="button"
        className="rec-strip-item rec-strip-seed"
        onClick={() => {
          play(seed);
          // Nothing selected means "the seed" — keeps seed selection consistent with Galaxy,
          // where clicking the seed dot also clears the selection.
          onSelect(null);
        }}
        aria-label={`Play ${seed.title}`}
      >
        <Artwork url={seed.artwork_url} size={56} />
        <div className="rec-strip-title">{seed.title}</div>
      </button>
      {recs.map((rec) => (
        <button
          key={rec.track_id}
          type="button"
          className={`rec-strip-item${selectedId === rec.track_id ? " rec-strip-item-selected" : ""}`}
          onClick={() => {
            play(rec);
            onSelect(rec.track_id);
          }}
          aria-label={`Play ${rec.title}`}
        >
          <Artwork url={rec.artwork_url} size={56} />
          <div className="rec-strip-title">{rec.title}</div>
        </button>
      ))}
    </div>
  );
}
