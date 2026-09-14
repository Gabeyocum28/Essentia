import { render, screen } from "@testing-library/react";
import { TrackRow } from "./TrackRow";
import { PlayerProvider } from "../player/usePlayer";
import { licenceLabel } from "./Attribution";
import type { Track } from "../api/types";

function renderRow(track: Track) {
  return render(
    <PlayerProvider>
      <TrackRow track={track} />
    </PlayerProvider>,
  );
}

const DEEZER: Track = {
  track_id: "2711778",
  title: "So What",
  artist: "Miles Davis",
  album: "Kind of Blue",
  artwork_url: null,
  preview_url: null,
  source: "deezer",
};

const JAMENDO: Track = {
  track_id: "jamendo:168",
  title: "Sunrise",
  artist: "Dee Yan-Key",
  album: "Morning",
  artwork_url: null,
  preview_url: null,
  source: "jamendo",
  attribution_url: "https://www.jamendo.com/track/168/sunrise",
};

test("a Creative Commons track credits its source and links back", () => {
  renderRow(JAMENDO);
  const link = screen.getByRole("link", { name: /via Jamendo/ });
  expect(link).toHaveAttribute("href", JAMENDO.attribution_url);
  expect(link).toHaveAttribute("rel", expect.stringContaining("license"));
});

test("the licence label is read off the deed URL", () => {
  renderRow({
    ...JAMENDO,
    attribution_url: "http://creativecommons.org/licenses/by-sa/3.0/",
  });
  expect(screen.getByText("via Jamendo · CC BY-SA 3.0")).toBeInTheDocument();
});

test("a track with no attribution shows no credit line", () => {
  const { container } = renderRow(DEEZER);
  expect(container.querySelector(".track-row-attribution")).toBeNull();
  expect(screen.queryByText(/via /)).not.toBeInTheDocument();
});

test("licenceLabel reads the code and version, or gives up", () => {
  expect(licenceLabel("http://creativecommons.org/licenses/by-sa/3.0/")).toBe(
    "CC BY-SA 3.0",
  );
  expect(licenceLabel("https://creativecommons.org/licenses/by/4.0/")).toBe(
    "CC BY 4.0",
  );
  // A share page, not a deed: no licence can be read off it, and guessing
  // one would be a false claim about how the track may be used.
  expect(licenceLabel("https://www.jamendo.com/track/168/sunrise")).toBeNull();
  expect(licenceLabel(null)).toBeNull();
});
