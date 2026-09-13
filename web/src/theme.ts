/**
 * Canvas drawing colours read from the CSS token layer, so the visualisations
 * follow `styles.css` instead of carrying their own hex values. Falls back to
 * the literal token value when there is no document (tests, workers).
 */
export function token(name: string, fallback: string): string {
  if (typeof document === "undefined") return fallback;
  const value = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return value || fallback;
}

/** The palette the canvases use, resolved once per draw setup.
 *
 * Every fallback is the literal value of the matching token in styles.css.
 * They drifted once (the accent fallbacks were still iOS blue after the
 * design system moved to #5b9cff), which is invisible in the browser and
 * wrong everywhere getComputedStyle isn't available. */
export function canvasPalette() {
  return {
    bg: token("--canvas-bg", "#0a0c12"),
    accent: token("--accent", "#5b9cff"),
    accentSoft: token("--accent-soft", "rgba(91,156,255,.35)"),
    accentLine: token("--accent-line", "rgba(91,156,255,.55)"),
    seed: token("--seed", "#ffd60a"),
    seedSoft: token("--seed-soft", "rgba(255,214,10,.6)"),
    seedSofter: token("--seed-softer", "rgba(255,214,10,.25)"),
    cyan: token("--cyan", "#64d2ff"),
    dot: token("--dot", "rgba(255,255,255,.55)"),
    dotBright: token("--dot-bright", "rgba(255,255,255,.7)"),
    edge: token("--edge", "rgba(255,255,255,.14)"),
  };
}
