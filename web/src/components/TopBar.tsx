import { Link, useLocation } from "react-router-dom";

/** The section label for a path — purely cosmetic orientation. */
function sectionFor(pathname: string): string {
  if (pathname.startsWith("/insights/")) return "Insights";
  if (pathname.startsWith("/recs/")) return "Recommendations";
  if (pathname.startsWith("/seed/")) return "Seed";
  return "Search";
}

export function TopBar() {
  const location = useLocation();
  const isHome = location.pathname === "/";

  return (
    <div className="top-bar">
      <div className="top-bar-inner">
        <Link to="/" className="top-bar-title">
          <span className="top-bar-mark" aria-hidden="true" />
          Essentia
        </Link>
        <span className="top-bar-section">{sectionFor(location.pathname)}</span>
        {!isHome && (
          <Link to="/" className="top-bar-back">
            ← Back
          </Link>
        )}
      </div>
    </div>
  );
}
