import { useCallback, useEffect, useRef, useState } from "react";
import { Link, useLocation, useParams } from "react-router-dom";
import {
  api, clampFeel, loadStoredFeel, storeFeel, FEEL_MAX, FEEL_MIN,
  clampTempo, loadStoredTempo, storeTempo, TEMPO_MAX, TEMPO_MIN,
  isUnanalyzed,
} from "../api/client";
import { Artwork } from "../components/Artwork";
import { TrackRow } from "../components/TrackRow";
import { Card } from "../components/Card";
import { SkeletonTrackList } from "../components/Skeleton";
import type { Track } from "../api/types";

type Status = "loading" | "ready" | "error" | "reanalyzing";

// Both sliders share it: a drag is a stream of change events and each one
// would otherwise be a request.
const FEEL_DEBOUNCE_MS = 250;

// A 409 means the seed was analyzed by a superseded version of the audio
// model and the worker is redoing it -- a wait, not a fault. /seed pushes
// such a row to the FRONT of the re-analysis queue, so the wait is one
// track's analysis (a few seconds), not the backlog's.
//
// Bounded, because "it will be along shortly" stops being true at some
// point: five tries over fifteen seconds, then the user gets a button and
// the choice of what to do next.
export const REANALYZE_RETRY_MS = 3000;
export const REANALYZE_MAX_RETRIES = 5;
const REANALYZING_MESSAGE =
  "We're re-analyzing this track with the new audio model. This usually takes a few seconds.";

export function Recommendations() {
  const { id = "", axis = "" } = useParams();
  const location = useLocation();
  const seedTrack = (location.state as { track?: Track } | null)?.track;

  const [status, setStatus] = useState<Status>("loading");
  const [results, setResults] = useState<Track[]>([]);
  const [feel, setFeel] = useState(loadStoredFeel);
  const [tempo, setTempo] = useState(loadStoredTempo);
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const retryRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const attemptsRef = useRef(0);

  const run = useCallback(async (feelValue: number, tempoValue: number,
                                 isRetry = false) => {
    if (retryRef.current) clearTimeout(retryRef.current);
    if (!isRetry) attemptsRef.current = 0;
    setStatus(isRetry ? "reanalyzing" : "loading");
    try {
      const res = await api.recommend(id, axis, 10, feelValue, tempoValue);
      attemptsRef.current = 0;
      setResults(res.results);
      setStatus("ready");
    } catch (err) {
      if (isUnanalyzed(err) && attemptsRef.current < REANALYZE_MAX_RETRIES) {
        attemptsRef.current += 1;
        setStatus("reanalyzing");
        retryRef.current = setTimeout(() => {
          void run(feelValue, tempoValue, true);
        }, REANALYZE_RETRY_MS);
        return;
      }
      setStatus("error");
    }
  }, [id, axis]);

  useEffect(() => {
    void run(feel, tempo);
    // Only re-run immediately on id/axis change; slider changes are debounced separately.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [id, axis]);

  // One debounce timer for both sliders: moving one while the other is
  // pending should produce ONE request carrying both values, not two.
  const debounced = (nextFeel: number, nextTempo: number) => {
    if (debounceRef.current) clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(() => {
      void run(nextFeel, nextTempo);
    }, FEEL_DEBOUNCE_MS);
  };

  const handleFeelChange = (raw: number) => {
    const value = clampFeel(raw);
    setFeel(value);
    storeFeel(value);
    debounced(value, tempo);
  };

  const handleTempoChange = (raw: number) => {
    const value = clampTempo(raw);
    setTempo(value);
    storeTempo(value);
    debounced(feel, value);
  };

  useEffect(() => {
    return () => {
      if (debounceRef.current) clearTimeout(debounceRef.current);
      if (retryRef.current) clearTimeout(retryRef.current);
    };
  }, []);

  return (
    <div className="screen recs-screen">
      <Card className="seed-header seed-header-compact">
        <Artwork url={seedTrack?.artwork_url} size={72} />
        <div className="seed-header-info">
          <div className="seed-eyebrow">{axis || "Seed"}</div>
          <div className="seed-title">{seedTrack?.title ?? id}</div>
          {seedTrack && <div className="seed-artist">{seedTrack.artist}</div>}
        </div>
      </Card>

      <label className="feel-slider">
        <span className="feel-slider-label">Match the feel</span>
        <input
          type="range"
          min={FEEL_MIN}
          max={FEEL_MAX}
          step={0.1}
          value={feel}
          onChange={(e) => handleFeelChange(Number(e.target.value))}
          className="feel-slider-input"
          aria-label="Match the feel"
        />
        <span className="mono feel-slider-value">{feel.toFixed(1)}</span>
      </label>

      <label className="feel-slider">
        <span className="feel-slider-label">Match the tempo</span>
        <input
          type="range"
          min={TEMPO_MIN}
          max={TEMPO_MAX}
          step={0.1}
          value={tempo}
          onChange={(e) => handleTempoChange(Number(e.target.value))}
          className="feel-slider-input"
          aria-label="Match the tempo"
        />
        <span className="mono feel-slider-value">{tempo.toFixed(1)}</span>
      </label>

      {status === "loading" && (
        <>
          <p className="hint">Loading recommendations…</p>
          <SkeletonTrackList rows={6} />
        </>
      )}

      {status === "reanalyzing" && (
        <>
          <p className="hint">{REANALYZING_MESSAGE}</p>
          <SkeletonTrackList rows={6} />
        </>
      )}

      {status === "error" && (
        <div className="error-box">
          <p>Something went wrong</p>
          <button type="button" onClick={() => void run(feel, tempo)}>
            Try again
          </button>
        </div>
      )}

      {status === "ready" && (
        <>
          <div className="recs-header-row">
            <span className="recs-count">
              {results.length} {results.length === 1 ? "track" : "tracks"}
            </span>
            <Link className="insights-link" to={`/insights/${id}/${axis}?feel=${feel}&tempo=${tempo}`}>
              See the math ✦
            </Link>
          </div>
          <div className="track-list">
            {results.map((t) => (
              <TrackRow key={t.track_id} track={t} showScore />
            ))}
          </div>
        </>
      )}
    </div>
  );
}
