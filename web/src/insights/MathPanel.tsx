import type { Rhythm, VizRec } from "../api/types";

// contract/features.py: `key` is 0-11 with 0 = C. Sharps only -- the server
// sends a pitch class, not a spelling, so there is no enharmonic to choose.
const NOTES = ["C", "C♯", "D", "D♯", "E", "F", "F♯", "G", "G♯", "A", "A♯", "B"];

/** "F minor", or "—" when key_strength says the estimate means nothing. */
function keyLabel(r: Rhythm): string {
  if (!r.key_strength) return "—";
  return `${NOTES[((r.key % 12) + 12) % 12] ?? "?"} ${r.mode}`;
}

const bpmLabel = (r: Rhythm) => (r.tempo_bpm > 0 ? `${r.tempo_bpm.toFixed(1)} BPM` : "—");
const loudnessLabel = (r: Rhythm) =>
  `${r.loudness_lufs.toFixed(1)} LUFS · LRA ${r.loudness_range.toFixed(1)}`;

/** One "seed / pick" row of the rhythm table. */
function RhythmRow({ label, seed, rec }: { label: string; seed: string; rec: string }) {
  return (
    <div className="rhythm-row">
      <span className="rhythm-row-label">{label}</span>
      <span className="mono rhythm-row-seed">{seed}</span>
      <span className="mono rhythm-row-rec">{rec}</span>
    </div>
  );
}

interface Props {
  rec: VizRec;
  feelKeys?: string[];
}

/** Shows the arithmetic behind a rec's score, using the raw numbers the server returned. */
export function MathPanel({ rec, feelKeys }: Props) {
  const { math } = rec;
  const cos = math.dot / (math.seed_norm * math.rec_norm);

  return (
    <div className="math-panel">
      {math.metric === "cosine" ? (
        <>
          <span className="math-panel-key">similarity</span>
          <p className="mono math-panel-line">
            cos = {math.dot.toFixed(4)} / ({math.seed_norm.toFixed(4)} · {math.rec_norm.toFixed(4)}) ={" "}
            {cos.toFixed(4)}
          </p>
        </>
      ) : (
        math.distance != null && (
          <>
            <span className="math-panel-key">metric</span>
            <p className="mono math-panel-line">distance = {math.distance.toFixed(4)}</p>
          </>
        )
      )}
      {math.centrality != null && (
        <>
          <span className="math-panel-key">graph</span>
          <p className="mono math-panel-line">centrality = {math.centrality.toFixed(4)}</p>
        </>
      )}
      {math.feel && (
        <>
          <span className="math-panel-key">feel</span>
          <div className="feel-compare">
            {math.feel.seed.map((seedVal, i) => {
              const recVal = math.feel!.rec[i] ?? 0;
              const label = feelKeys?.[i] ?? `dim ${i}`;
              return (
                <div className="feel-compare-row" key={label}>
                  <span className="feel-compare-label">{label}</span>
                  <div className="feel-compare-bars">
                    <div className="feel-compare-bar-track">
                      <div
                        className="feel-compare-bar feel-compare-bar-seed"
                        style={{ width: `${Math.max(0, Math.min(1, seedVal)) * 100}%` }}
                      />
                    </div>
                    <div className="feel-compare-bar-track">
                      <div
                        className="feel-compare-bar feel-compare-bar-rec"
                        style={{ width: `${Math.max(0, Math.min(1, recVal)) * 100}%` }}
                      />
                    </div>
                  </div>
                </div>
              );
            })}
            {math.feel_dist != null && (
              <p className="mono math-panel-line feel-compare-dist">feel_dist = {math.feel_dist.toFixed(3)}</p>
            )}
          </div>
        </>
      )}
      {math.rhythm && (
        <>
          <span className="math-panel-key">rhythm</span>
          <div className="rhythm-compare">
            <RhythmRow label="tempo" seed={bpmLabel(math.rhythm.seed)}
                       rec={bpmLabel(math.rhythm.rec)} />
            <RhythmRow label="key" seed={keyLabel(math.rhythm.seed)}
                       rec={keyLabel(math.rhythm.rec)} />
            <RhythmRow label="loudness" seed={loudnessLabel(math.rhythm.seed)}
                       rec={loudnessLabel(math.rhythm.rec)} />
            {math.tempo_dist != null && (
              <p className="mono math-panel-line feel-compare-dist">
                tempo_dist = {math.tempo_dist.toFixed(3)}
              </p>
            )}
          </div>
        </>
      )}
    </div>
  );
}
