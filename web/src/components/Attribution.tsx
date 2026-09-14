import type { Track } from "../api/types";

/** "CC BY-SA 3.0" from a Creative Commons deed URL, or null.
 *
 * The licence is not sent as a label, only as the deed URL, because the URL
 * is the thing the credit has to link to and a label derived from it can
 * never drift out of step with it. Anything that is not a recognisable
 * `/licenses/<code>/<version>/` path gets null rather than a guess. */
export function licenceLabel(url: string | null | undefined): string | null {
  if (!url) return null;
  const match = /\/licenses\/([a-z-]+)(?:\/([0-9.]+))?/i.exec(url);
  if (!match) return null;
  const code = match[1].toUpperCase();
  return match[2] ? `CC ${code} ${match[2]}` : `CC ${code}`;
}

function sourceLabel(source: string | undefined): string {
  if (!source) return "the source";
  return source.charAt(0).toUpperCase() + source.slice(1);
}

/** The credit a Creative Commons licence obliges us to show.
 *
 * Renders nothing when the server sent no `attribution_url` -- which is the
 * Deezer case, where the licence asks for no credit and a "via Deezer" line
 * on every row would be noise. */
export function Attribution({ track }: { track: Track }) {
  const url = track.attribution_url;
  if (!url) return null;
  const licence = licenceLabel(url);
  return (
    <a
      className="track-row-attribution"
      href={url}
      target="_blank"
      rel="noopener noreferrer license"
      onClick={(e) => e.stopPropagation()}
    >
      via {sourceLabel(track.source)}
      {licence ? ` · ${licence}` : ""}
    </a>
  );
}
