//
//  TrackDecodingTests.swift
//  HackathonTests
//
//  Verifies the uniform Track shape decodes identically with and without the
//  recommend-only `score`. Sample track is a real entry from
//  contract/fixture.json (Miles Davis — So What).
//

import Foundation
import Testing
@testable import Hackathon

struct TrackDecodingTests {

    /// A real fixture track (no score) — the shape returned by /search.
    static let fixtureTrackJSON = """
    {
      "track_id": "2711778",
      "title": "So What (feat. John Coltrane, Cannonball Adderley & Bill Evans)",
      "artist": "Miles Davis",
      "album": "Kind Of Blue (Legacy Edition)",
      "artwork_url": "https://cdn-images.dzcdn.net/images/cover/fb9fbd93db667e24d4f5dee781e83f7d/250x250-000000-80-0-0.jpg",
      "preview_url": "https://cdnt-preview.dzcdn.net/api/1/1/d/4/2/0/d42ba522e8be4541d2224beb3ae4644c.mp3"
    }
    """

    @Test func decodesTrackWithoutScore() throws {
        let track = try JSONDecoder().decode(Track.self, from: Data(Self.fixtureTrackJSON.utf8))
        #expect(track.trackID == "2711778")
        #expect(track.artist == "Miles Davis")
        #expect(track.album == "Kind Of Blue (Legacy Edition)")
        #expect(track.artworkURL != nil)
        #expect(track.previewURL != nil)
        #expect(track.score == nil)
        #expect(track.id == track.trackID)
    }

    @Test func decodesTrackWithScore() throws {
        let json = """
        {
          "track_id": "2711778",
          "title": "So What",
          "artist": "Miles Davis",
          "album": "Kind Of Blue",
          "artwork_url": "https://example.com/a.jpg",
          "preview_url": "https://example.com/a.mp3",
          "score": 0.91
        }
        """
        let track = try JSONDecoder().decode(Track.self, from: Data(json.utf8))
        #expect(track.score == 0.91)
    }

    /// The two contract TRACK_OPTIONAL_FIELDS. The server omits the keys
    /// entirely when there is nothing to say, so "absent" is the normal case
    /// and must decode, not throw.
    @Test func decodesTrackWithoutSourceOrAttribution() throws {
        let track = try JSONDecoder().decode(Track.self, from: Data(Self.fixtureTrackJSON.utf8))
        #expect(track.source == nil)
        #expect(track.attributionURL == nil)
    }

    @Test func decodesJamendoSourceAndAttribution() throws {
        let json = """
        {
          "track_id": "jamendo:168",
          "title": "Sunrise",
          "artist": "Jasmine",
          "album": "Dawn",
          "artwork_url": "https://example.com/a.jpg",
          "preview_url": "https://example.com/a.mp3",
          "source": "jamendo",
          "attribution_url": "https://www.jamendo.com/track/168/sunrise"
        }
        """
        let track = try JSONDecoder().decode(Track.self, from: Data(json.utf8))
        #expect(track.source == "jamendo")
        #expect(track.attributionURL?.absoluteString == "https://www.jamendo.com/track/168/sunrise")
    }

    /// The credit line's label comes off the deed URL, never off a field, so
    /// it cannot drift out of step with the link it points at.
    @Test func readsTheLicenceOffTheDeedURL() {
        #expect(licenceLabel(URL(string: "http://creativecommons.org/licenses/by-sa/3.0/")!)
                == "CC BY-SA 3.0")
        #expect(licenceLabel(URL(string: "http://creativecommons.org/licenses/by/4.0/")!)
                == "CC BY 4.0")
        #expect(licenceLabel(URL(string: "https://www.jamendo.com/track/168/sunrise")!) == nil)
        #expect(licenceLabel(nil) == nil)
    }

    @Test func decodesAxis() throws {
        let json = """
        { "id": "groove", "label": "Keep the groove" }
        """
        let axis = try JSONDecoder().decode(Axis.self, from: Data(json.utf8))
        #expect(axis.id == "groove")
        #expect(axis.label == "Keep the groove")
    }
}
