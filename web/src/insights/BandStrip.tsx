import { useRef, useState } from "react";
import { bandHzRange, formatHz } from "../audio/mel";
import { soloBand } from "../audio/soloStore";

interface Props {
  bandEdgesHz: Float64Array;
  bands: number;
  /** The Hz band currently soloed, or null. */
  band: [number, number] | null;
}

/**
 * The strip beside the spectrogram: drag across it to pick a range of mel
 * bands and hear only those. Low frequency at the bottom, so it lines up
 * with the spectrogram it sits next to.
 */
export function BandStrip({ bandEdgesHz, bands, band }: Props) {
  const stripRef = useRef<HTMLDivElement>(null);
  const anchorRef = useRef<number | null>(null);
  const [range, setRange] = useState<[number, number] | null>(null);

  const bandAt = (clientY: number): number | null => {
    const rect = stripRef.current?.getBoundingClientRect();
    if (!rect || rect.height === 0) return null;
    const fromBottom = 1 - (clientY - rect.top) / rect.height;
    return Math.max(0, Math.min(bands - 1, Math.floor(fromBottom * bands)));
  };

  const commit = (lo: number, hi: number) => {
    const [loHz] = bandHzRange(bandEdgesHz, lo, bands);
    const [, hiHz] = bandHzRange(bandEdgesHz, hi, bands);
    soloBand(loHz, hiHz);
  };

  const onPointerDown = (e: React.PointerEvent<HTMLDivElement>) => {
    const idx = bandAt(e.clientY);
    if (idx === null) return;
    anchorRef.current = idx;
    setRange([idx, idx]);
    stripRef.current?.setPointerCapture(e.pointerId);
    commit(idx, idx);
  };

  const onPointerMove = (e: React.PointerEvent<HTMLDivElement>) => {
    const anchor = anchorRef.current;
    if (anchor === null) return;
    const idx = bandAt(e.clientY);
    if (idx === null) return;
    const lo = Math.min(anchor, idx);
    const hi = Math.max(anchor, idx);
    setRange([lo, hi]);
    commit(lo, hi);
  };

  const onPointerUp = () => {
    anchorRef.current = null;
  };

  const label = band ? `${formatHz(band[0])} – ${formatHz(band[1])} Hz` : "drag to solo";
  const selection = range && band
    ? { bottom: `${(range[0] / bands) * 100}%`, height: `${((range[1] - range[0] + 1) / bands) * 100}%` }
    : null;

  return (
    <div className="band-strip">
      <div
        className="band-strip-track"
        ref={stripRef}
        role="slider"
        aria-label="Band solo"
        aria-valuetext={label}
        aria-valuemin={0}
        aria-valuemax={bands - 1}
        aria-valuenow={range ? range[1] : 0}
        tabIndex={0}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerCancel={onPointerUp}
      >
        {selection && <div className="band-strip-selection" style={selection} />}
      </div>
      <div className="mono band-strip-label">{label}</div>
    </div>
  );
}
