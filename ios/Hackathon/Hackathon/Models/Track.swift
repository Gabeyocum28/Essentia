//
//  Track.swift
//  Hackathon
//
//  The uniform track shape from contract/contract.md. One struct, decoded
//  identically at every endpoint. `score` is present on /recommend results
//  only and is ignored by the v1 UI.
//
//  `source` and `attributionURL` are contract TRACK_OPTIONAL_FIELDS: the
//  server omits the key entirely when there is nothing to say (Deezer asks
//  for no credit line), so both decode as optionals and a missing key is the
//  normal case, not an error. Where attributionURL IS present the licence
//  obliges us to show it — see AttributionLine in Shared/Components.swift.
//

import Foundation

nonisolated struct Track: Identifiable, Decodable, Hashable {
    let trackID: String
    let title: String
    let artist: String
    let album: String
    let artworkURL: URL?
    let previewURL: URL?
    let score: Double?
    // `var … = nil` rather than `let`: it keeps the memberwise initializer
    // usable from the call sites that build a Track out of a viz payload
    // (Models/VizMap.swift and friends) without listing these two, and the
    // synthesized decoder still fills them in when the keys are there.
    var source: String? = nil
    var attributionURL: URL? = nil

    var id: String { trackID }

    enum CodingKeys: String, CodingKey {
        case trackID = "track_id"
        case title
        case artist
        case album
        case artworkURL = "artwork_url"
        case previewURL = "preview_url"
        case score
        case source
        case attributionURL = "attribution_url"
    }
}
