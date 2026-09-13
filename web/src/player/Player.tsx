import { Artwork } from "../components/Artwork";
import { usePlayer } from "./usePlayer";

export function Player() {
  const { nowPlaying, isPlaying, progress, errorMessage, toggle, stop } = usePlayer();

  if (!nowPlaying) return null;

  return (
    <div className="player-bar">
      {nowPlaying.artwork_url && (
        <div
          className="player-backdrop"
          aria-hidden="true"
          style={{ backgroundImage: `url(${nowPlaying.artwork_url})` }}
        />
      )}
      <div className="player-scrim" aria-hidden="true" />
      <div className="player-progress" style={{ width: `${Math.min(1, Math.max(0, progress)) * 100}%` }} />
      <div className="player-row">
        <Artwork url={nowPlaying.artwork_url} size={44} />
        <div className="player-info">
          <div className="player-title">{nowPlaying.title}</div>
          <div className="player-artist">{errorMessage ?? nowPlaying.artist}</div>
        </div>
        <button type="button" className="player-toggle" onClick={toggle} aria-label={isPlaying ? "Pause" : "Play"}>
          {isPlaying ? "⏸" : "▶"}
        </button>
        <button type="button" className="player-stop" onClick={stop} aria-label="Stop">
          ✕
        </button>
      </div>
    </div>
  );
}
