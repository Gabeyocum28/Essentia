//
//  VizT2.swift
//  Hackathon
//
//  Additive, non-contract payloads for the Tour, Topology, and
//  eigen-listening panels: GET /viz/tour, GET /viz/mst, GET /viz/extremes.
//

import Foundation

/// GET /viz/tour — the whole corpus's first 8 PCA dimensions, for the Grand
/// Tour to rotate through on-device. `coords8` arrives as base64 of
/// little-endian float32, row-major, n rows x 8 cols, rows aligned with ids.
nonisolated struct VizTour: Decodable {
    let ids: [String]
    /// n x 8, one row per id, decoded from `coords8`.
    let coords: [[Float]]
    /// Fraction of total variance held by each of the 8 PCs.
    let variance: [Double]

    enum CodingKeys: String, CodingKey {
        case ids, coords8, variance
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let ids = try container.decode([String].self, forKey: .ids)
        let variance = try container.decode([Double].self, forKey: .variance)
        let base64 = try container.decode(String.self, forKey: .coords8)

        guard let data = Data(base64Encoded: base64) else {
            throw DecodingError.dataCorruptedError(
                forKey: .coords8, in: container,
                debugDescription: "coords8 is not valid base64"
            )
        }

        let expectedFloats = ids.count * 8
        let expectedBytes = expectedFloats * MemoryLayout<Float>.size
        guard data.count == expectedBytes else {
            throw DecodingError.dataCorruptedError(
                forKey: .coords8, in: container,
                debugDescription: "coords8 has \(data.count) bytes, expected "
                    + "\(expectedBytes) for \(ids.count) ids x 8 floats"
            )
        }

        var flat = [Float](repeating: 0, count: expectedFloats)
        _ = flat.withUnsafeMutableBytes { destination in
            data.copyBytes(to: destination)
        }

        var rows: [[Float]] = []
        rows.reserveCapacity(ids.count)
        for row in 0..<ids.count {
            let start = row * 8
            rows.append(Array(flat[start..<(start + 8)]))
        }

        self.ids = ids
        self.coords = rows
        self.variance = variance
    }
}

/// GET /viz/mst — n-1 edges of the minimum spanning tree over cosine
/// distance, ascending by distance. Single-linkage = MST = H0 persistence:
/// an edge's distance is the birth-death (merge) time of its component.
nonisolated struct VizMST: Decodable {
    /// One MST edge: `i < j` are indices into `ids`, `d` is cosine distance.
    struct Edge: Decodable {
        let i: Int
        let j: Int
        let d: Double

        init(from decoder: Decoder) throws {
            var container = try decoder.unkeyedContainer()
            i = try container.decode(Int.self)
            j = try container.decode(Int.self)
            d = try container.decode(Double.self)
        }
    }

    let ids: [String]
    let edges: [Edge]
}

/// GET /viz/extremes?pc=1..8&limit=4 — the most negative and most positive
/// tracks along one principal component, for "what does this PC sound like?"
nonisolated struct VizExtremesResponse: Decodable {
    /// Same shape as VizWalk.Step minus x/y: playable, but not positioned.
    struct ExtremeTrack: Decodable, Identifiable {
        let trackID: String
        let title: String
        let artist: String
        let album: String
        let artworkURL: URL?
        let previewURL: URL?
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
            case source
            case attributionURL = "attribution_url"
        }
    }

    let pc: Int
    let variancePct: Double
    /// Most negative coordinate first.
    let low: [ExtremeTrack]
    /// Most positive coordinate first.
    let high: [ExtremeTrack]

    enum CodingKeys: String, CodingKey {
        case pc
        case variancePct = "variance_pct"
        case low, high
    }
}

/// Payload of GET /viz/attribute (T2.6) — which frequency bands carry a
/// pair's similarity, measured by deleting each band from the seed and
/// pushing the counterfactual back through the model.
///
/// The route is a mailbox: the first request queues the pair and answers
/// `pending`, so the client polls until `ready` (or `failed`).
nonisolated struct VizAttribution: Decodable {
    struct Band: Decodable, Identifiable {
        let loHz: Double
        let hiHz: Double
        /// How far the pair's cosine fell when this band was deleted.
        let delta: Double

        var id: String { "\(loHz)-\(hiHz)" }

        enum CodingKeys: String, CodingKey {
            case loHz = "lo_hz"
            case hiHz = "hi_hz"
            case delta
        }
    }

    let status: String
    /// The unoccluded cosine; absent while pending.
    let base: Double?
    let bands: [Band]?
    let error: String?

    var isReady: Bool { status == "ready" }
    var isPending: Bool { status == "pending" }

    /// Deltas are not additive — bands interact inside the network — so the
    /// bars are scaled against the largest drop, not against their sum.
    var largestDrop: Double {
        max(bands?.map(\.delta).max() ?? 0, 0)
    }
}
