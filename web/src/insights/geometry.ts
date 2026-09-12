export interface Transform {
  sx: number;
  sy: number;
  tx: number;
  ty: number;
}

/**
 * Fit data-space [minX,maxX] x [minY,maxY] into a width x height canvas with
 * `pad` pixels of margin, preserving aspect ratio (uniform scale, centered).
 * min -> pad, max -> width - pad (for whichever axis is limiting).
 */
export function fitTransform(
  xs: number[],
  ys: number[],
  width: number,
  height: number,
  pad: number,
): Transform {
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);

  const dx = maxX - minX || 1;
  const dy = maxY - minY || 1;

  const availW = Math.max(width - 2 * pad, 1);
  const availH = Math.max(height - 2 * pad, 1);

  const scale = Math.min(availW / dx, availH / dy);

  const usedW = dx * scale;
  const usedH = dy * scale;
  const extraX = (availW - usedW) / 2;
  const extraY = (availH - usedH) / 2;

  // screenX = tx + sx * x ; choose sx so that minX -> pad + extraX and maxX -> width - pad - extraX
  const sx = scale;
  const tx = pad + extraX - minX * scale;
  // y is often flipped in screen space (down = positive); keep same orientation as x by default,
  // callers can flip via a negative height convention if desired. We keep math axis-consistent:
  // minY -> pad + extraY, maxY -> height - pad - extraY
  const sy = scale;
  const ty = pad + extraY - minY * scale;

  return { sx, sy, tx, ty };
}

/** Apply the fit transform to a data point, producing CSS-pixel screen coordinates. */
export function toScreen(t: Transform, x: number, y: number): [number, number] {
  return [t.tx + t.sx * x, t.ty + t.sy * y];
}

/**
 * Compose a zoom (about a screen-space centre (cx, cy)) and pan (panX, panY, in screen pixels)
 * on top of an existing transform, returning a new transform.
 */
export function applyZoomPan(
  t: Transform,
  zoom: number,
  panX: number,
  panY: number,
  cx: number,
  cy: number,
): Transform {
  // screen' = cx + zoom * (screen - cx) + panX
  //         = cx + zoom * (t.tx + t.sx * x - cx) + panX
  //         = (cx - zoom * cx + panX + zoom * t.tx) + (zoom * t.sx) * x
  const sx = t.sx * zoom;
  const sy = t.sy * zoom;
  const tx = cx - zoom * cx + panX + zoom * t.tx;
  const ty = cy - zoom * cy + panY + zoom * t.ty;
  return { sx, sy, tx, ty };
}

export interface Pan {
  x: number;
  y: number;
}

/**
 * Compute the new zoom and pan that result from zooming by `factor` about a canvas-local
 * cursor point (mx, my), given the current zoom/pan and the screen-space zoom centre (cx, cy)
 * used by applyZoomPan. Zoom is clamped to [1, 8]. The pan is derived so that whatever data
 * point currently sits under the cursor stays under the cursor after the zoom is applied:
 * screen = cx + zoom * (fit(data) - cx) + pan, so holding (mx - cx - pan) / zoom constant
 * across the zoom change gives pan' = (mx - cx) - (zoom'/zoom) * ((mx - cx) - pan).
 *
 * `t` (the base fit transform) isn't needed for the derivation since it cancels out, but is
 * accepted for interface symmetry with the other geometry helpers.
 */
export function zoomAbout(
  t: Transform,
  zoom: number,
  pan: Pan,
  factor: number,
  mx: number,
  my: number,
  cx: number,
  cy: number,
): { zoom: number; pan: Pan } {
  void t;
  const nextZoom = Math.min(8, Math.max(1, zoom * factor));
  const ratio = nextZoom / zoom;
  const panX = mx - cx - ratio * (mx - cx - pan.x);
  const panY = my - cy - ratio * (my - cy - pan.y);
  return { zoom: nextZoom, pan: { x: panX, y: panY } };
}

/**
 * Find the index of the nearest point (in screen space, after applying transform t) to
 * (px, py), within maxDist pixels. Returns -1 if none within range or arrays are empty.
 */
export function nearestIndex(
  xs: number[],
  ys: number[],
  t: Transform,
  px: number,
  py: number,
  maxDist: number,
): number {
  let best = -1;
  let bestDist = Infinity;
  for (let i = 0; i < xs.length; i++) {
    const [sx, sy] = toScreen(t, xs[i], ys[i]);
    const dx = sx - px;
    const dy = sy - py;
    const dist = Math.sqrt(dx * dx + dy * dy);
    if (dist < bestDist) {
      bestDist = dist;
      best = i;
    }
  }
  if (best === -1 || bestDist > maxDist) return -1;
  return best;
}
