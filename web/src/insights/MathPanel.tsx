import type { VizRec } from "../api/types";

interface Props {
  rec: VizRec;
}

/** Shows the arithmetic behind a rec's score, using the raw numbers the server returned. */
export function MathPanel({ rec }: Props) {
  const { math } = rec;
  const cos = math.dot / (math.seed_norm * math.rec_norm);

  return (
    <div className="math-panel">
      {math.metric === "cosine" ? (
        <p className="mono math-panel-line">
          cos = {math.dot.toFixed(4)} / ({math.seed_norm.toFixed(4)} · {math.rec_norm.toFixed(4)}) ={" "}
          {cos.toFixed(4)}
        </p>
      ) : (
        math.distance != null && (
          <p className="mono math-panel-line">distance = {math.distance.toFixed(4)}</p>
        )
      )}
      {math.centrality != null && (
        <p className="mono math-panel-line">centrality = {math.centrality.toFixed(4)}</p>
      )}
    </div>
  );
}
