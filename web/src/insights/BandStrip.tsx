import { useRef, useState } from "react";
import { bandHzRange, formatHz } from "../audio/mel";
import { soloBand, useSoloError } from "../audio/soloStore";

interface Props {
  bandEdgesHz: Float64Array;
  bands: number;
  /** The Hz band currently soloed, or null. */
  band: [number, number] | null;
}

/** Arrow keys move by one band; with Shift, by this many. */
const COARSE_STEP = 8;

/**
 * The strip beside the spectrogram: drag across it to pick a range of mel
 * bands and hear only those. Low frequency at the bottom, so it lines up
 * with the spectrogram it sits next to.
 *
 * Under 560px the strip is laid out horizontally (see `.band-strip` in
 * styles.css), which is the phone case and therefore the common one. The
 * axis is read off the measured rect rather than a media query, so the two
 * never disagree: wider than tall means horizontal, and then low frequency
 * is at the LEFT.
 */
export function BandStrip({ bandEdgesHz, bands, band }: Props) {
  const stripRef = useRef<HTMLDivElement>(null);
  const anchorRef = useRef<number | null>(null);
  const [range, setRange] = useState<[number, number] | null>(null);
  const error = useSoloError();

  /** True when the strip is laid out left-to-right (the ≤560px layout). */
  const isHorizontal = (rect: DOMRect) => rect.width > rect.height;

  const bandAt = (clientX: number, clientY: number): number | null => {
    const rect = stripRef.current?.getBoundingClientRect();
    if (!rect) return null;
    const horizontal = isHorizontal(rect);
    const span = horizontal ? rect.width : rect.height;
    if (span === 0) return null;
    // Vertical: low frequency at the bottom. Horizontal: low at the left.
    const fraction = horizontal
      ? (clientX - rect.left) / span
      : 1 - (clientY - rect.top) / span;
    return Math.max(0, Math.min(bands - 1, Math.floor(fraction * bands)));
  };

  const commit = (lo: number, hi: number) => {
    const [loHz] = bandHzRange(bandEdgesHz, lo, bands);
    const [, hiHz] = bandHzRange(bandEdgesHz, hi, bands);
    soloBand(loHz, hiHz);
  };

  const onPointerDown = (e: React.PointerEvent<HTMLDivElement>) => {
    const idx = bandAt(e.clientX, e.clientY);
    if (idx === null) return;
    anchorRef.current = idx;
    setRange([idx, idx]);
    stripRef.current?.setPointerCapture(e.pointerId);
    commit(idx, idx);
  };

  const onPointerMove = (e: React.PointerEvent<HTMLDivElement>) => {
    const anchor = anchorRef.current;
    if (anchor === null) return;
    const idx = bandAt(e.clientX, e.clientY);
    if (idx === null) return;
    const lo = Math.min(anchor, idx);
    const hi = Math.max(anchor, idx);
    setRange([lo, hi]);
    commit(lo, hi);
  };

  const onPointerUp = () => {
    anchorRef.current = null;
  };

  // role="slider" without key handling is a lie to a screen reader, and the
  // strip is otherwise unreachable without a pointer. Up/Right raise the
  // band, Down/Left lower it, matching both layouts' "low first" direction.
  const onKeyDown = (e: React.KeyboardEvent<HTMLDivElement>) => {
    let delta = 0;
    if (e.key === "ArrowUp" || e.key === "ArrowRight") delta = 1;
    else if (e.key === "ArrowDown" || e.key === "ArrowLeft") delta = -1;
    else return;
    e.preventDefault();
    if (e.shiftKey) delta *= COARSE_STEP;
    const current = range ? range[1] : 0;
    const next = Math.max(0, Math.min(bands - 1, current + delta));
    setRange([next, next]);
    commit(next, next);
  };

  const label = band ? `${formatHz(band[0])} – ${formatHz(band[1])} Hz` : "drag to solo";
  const rect = stripRef.current?.getBoundingClientRect();
  const horizontal = rect ? isHorizontal(rect) : false;
  const start = range ? `${(range[0] / bands) * 100}%` : "0%";
  const extent = range ? `${((range[1] - range[0] + 1) / bands) * 100}%` : "0%";
  const selection = range && band
    ? horizontal
      ? { left: start, width: extent }
      : { bottom: start, height: extent }
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
        onKeyDown={onKeyDown}
      >
        {selection && <div className="band-strip-selection" style={selection} />}
      </div>
      <div className={`mono band-strip-label${error ? " band-strip-label-error" : ""}`}>
        {error ?? label}
      </div>
    </div>
  );
}
