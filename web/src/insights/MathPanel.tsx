import type { VizRec } from "../api/types";

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
    </div>
  );
}
