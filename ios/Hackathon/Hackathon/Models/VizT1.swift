//
//  VizT1.swift
//  Hackathon
//
//  Additive, non-contract payloads for Walk and Proof modes.
//

import Foundation

nonisolated struct VizWalk: Decodable {
    struct Step: Decodable, Identifiable {
        let trackID: String
        let title: String
        let artist: String
        let album: String
        let artworkURL: URL?
        let previewURL: URL?
        let x: Double
        let y: Double
        var source: String? = nil
        var attributionURL: URL? = nil

        var id: String { trackID }
        var track: Track {
            var t = Track(trackID: trackID, title: title, artist: artist, album: album,
                          artworkURL: artworkURL, previewURL: previewURL, score: nil)
            t.source = source
            t.attributionURL = attributionURL
            return t
        }

        enum CodingKeys: String, CodingKey {
            case trackID = "track_id"
            case title, artist, album
            case artworkURL = "artwork_url"
            case previewURL = "preview_url"
            case x, y
            case source
            case attributionURL = "attribution_url"
        }
    }

    let path: [Step]
    let geodesic: Double
    let ambient: Double
    let detour: Double
    let k: Int
}

nonisolated struct VizHistogram: Decodable {
    struct Distribution: Decodable {
        let mean: Double
        let sd: Double
    }

    let bins: [Double]
    let counts: [Int]
    let recScores: [Double]
    let percentile: Double
    let null: Distribution
    let corpus: Distribution

    enum CodingKeys: String, CodingKey {
        case bins, counts, percentile, null, corpus
        case recScores = "rec_scores"
    }
}

nonisolated struct VizHubs: Decodable {
    struct HubEntry: Decodable, Identifiable {
        let trackID: String
        let title: String
        let artist: String
        let album: String
        let artworkURL: URL?
        let previewURL: URL?
        let count: Int
        var source: String? = nil
        var attributionURL: URL? = nil

        var id: String { trackID }
        var track: Track {
            var t = Track(trackID: trackID, title: title, artist: artist, album: album,
                          artworkURL: artworkURL, previewURL: previewURL, score: nil)
            t.source = source
            t.attributionURL = attributionURL
            return t
        }

        enum CodingKeys: String, CodingKey {
            case trackID = "track_id"
            case title, artist, album, count
            case artworkURL = "artwork_url"
            case previewURL = "preview_url"
            case source
            case attributionURL = "attribution_url"
        }
    }

    struct CentralityEntry: Decodable, Identifiable {
        let trackID: String
        let title: String
        let artist: String
        let album: String
        let artworkURL: URL?
        let previewURL: URL?
        let centrality: Double
        var source: String? = nil
        var attributionURL: URL? = nil

        var id: String { trackID }
        var track: Track {
            var t = Track(trackID: trackID, title: title, artist: artist, album: album,
                          artworkURL: artworkURL, previewURL: previewURL, score: nil)
            t.source = source
            t.attributionURL = attributionURL
            return t
        }

        enum CodingKeys: String, CodingKey {
            case trackID = "track_id"
            case title, artist, album, centrality
            case artworkURL = "artwork_url"
            case previewURL = "preview_url"
            case source
            case attributionURL = "attribution_url"
        }
    }

    struct Count: Decodable {
        let trackID: String
        let count: Int

        enum CodingKeys: String, CodingKey {
            case trackID = "track_id"
            case count
        }
    }

    let hubs: [HubEntry]
    let central: [CentralityEntry]
    let isolated: [CentralityEntry]
    let expectedK: Int
    let allCounts: [Count]

    enum CodingKeys: String, CodingKey {
        case hubs, central, isolated
        case expectedK = "expected_k"
        case allCounts = "all_counts"
    }
}
