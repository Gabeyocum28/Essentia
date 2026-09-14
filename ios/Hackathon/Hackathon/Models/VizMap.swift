//
//  VizMap.swift
//  Hackathon
//
//  Payload of GET /viz/map — the demo/debug endpoint behind the Insights
//  screen. Not part of contract/contract.md: the whole corpus as 2D points
//  (PCA of embedding space), the seed, and the recs with the numbers behind
//  each score.
//

import Foundation

nonisolated struct VizMap: Decodable {
    struct Points: Decodable {
        let ids: [String]
        let x: [Double]
        let y: [Double]
        let tracks: [Track]
    }

    /// The seed's contract fields plus its position and groove vector.
    struct SeedPoint: Decodable {
        let trackID: String
        let title: String
        let artist: String
        let album: String
        let artworkURL: URL?
        let previewURL: URL?
        let x: Double
        let y: Double
        let groove: [Double]?
        var source: String? = nil
        var attributionURL: URL? = nil

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
            case x, y, groove
            case source
            case attributionURL = "attribution_url"
        }
    }

    /// One recommendation with its position, groove, and score arithmetic.
    struct Rec: Decodable, Identifiable {
        let trackID: String
        let title: String
        let artist: String
        let album: String
        let artworkURL: URL?
        let previewURL: URL?
        let score: Double
        let x: Double
        let y: Double
        let groove: [Double]?
        let math: ScoreMath
        var source: String? = nil
        var attributionURL: URL? = nil

        var id: String { trackID }

        var track: Track {
            var t = Track(trackID: trackID, title: title, artist: artist, album: album,
                          artworkURL: artworkURL, previewURL: previewURL, score: score)
            t.source = source
            t.attributionURL = attributionURL
            return t
        }

        enum CodingKeys: String, CodingKey {
            case trackID = "track_id"
            case title, artist, album
            case artworkURL = "artwork_url"
            case previewURL = "preview_url"
            case score, x, y, groove, math
            case source
            case attributionURL = "attribution_url"
        }
    }

    /// cosine: score = dot / (seedNorm * recNorm)
    /// euclidean: score = 1 / (1 + distance)
    /// blend: score is a weighted mean of the `parts` percentiles, so no
    /// single formula equals it - the panel shows the parts instead.
    struct ScoreMath: Decodable {
        let metric: String
        let dot: Double
        let seedNorm: Double
        let recNorm: Double
        let distance: Double?
        let centrality: Double?
        /// Per-feature-key percentile, present only on a blended axis.
        var parts: [String: Double]? = nil

        enum CodingKeys: String, CodingKey {
            case metric, dot
            case seedNorm = "seed_norm"
            case recNorm = "rec_norm"
            case distance, centrality, parts
        }
    }

    struct AxisInfo: Decodable {
        let id: String
        let metric: String
        let direction: Int
        /// Weight per feature key, present only on a blended axis.
        var weights: [String: Double]? = nil
    }

    let points: Points
    let seed: SeedPoint
    let recs: [Rec]
    let axis: AxisInfo
}
