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

/** The palette the canvases use, resolved once per draw setup. */
export function canvasPalette() {
  return {
    bg: token("--canvas-bg", "#0a0c12"),
    accent: token("--accent", "#0A84FF"),
    accentSoft: token("--accent-soft", "rgba(10,132,255,.35)"),
    accentLine: token("--accent-line", "rgba(10,132,255,.6)"),
    seed: token("--seed", "#FFD60A"),
    seedSoft: token("--seed-soft", "rgba(255,214,10,.6)"),
    seedSofter: token("--seed-softer", "rgba(255,214,10,.25)"),
    cyan: token("--cyan", "#64D2FF"),
    dot: token("--dot", "rgba(255,255,255,.55)"),
    dotBright: token("--dot-bright", "rgba(255,255,255,.7)"),
    edge: token("--edge", "rgba(255,255,255,.14)"),
  };
}
