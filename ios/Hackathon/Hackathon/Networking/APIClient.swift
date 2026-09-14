//
//  APIClient.swift
//  Hackathon
//
//  The only thing that talks to the server. Mirrors contract/contract.md
//  exactly: GET /search, POST /seed, GET /axes, GET /recommend. All responses
//  decode the same uniform `Track`.
//

import Foundation

enum APIError: Error, LocalizedError {
    case invalidURL
    case invalidResponse
    case status(Int)
    /// HTTP 409 from a ranking endpoint: the seed was analyzed by a
    /// superseded version of the audio model and the worker is redoing it.
    /// Not a failure the user caused or can fix — a wait. The server has
    /// already pushed this track to the front of its re-analysis queue, so
    /// the wait is one track's analysis rather than the whole backlog's.
    case reanalyzing
    case decoding(Error)

    var errorDescription: String? {
        switch self {
        case .invalidURL: return "Invalid request URL."
        case .invalidResponse: return "The server sent an unexpected response."
        case .status(let code): return "The server returned status \(code)."
        case .reanalyzing:
            return "We're re-analyzing this track with the new audio model. "
                 + "This usually takes a few seconds."
        case .decoding: return "The server response could not be read."
        }
    }
}

actor APIClient {
    static let shared = APIClient(baseURL: AppConfig.baseURL)

    private let baseURL: URL
    private let session: URLSession
    private let decoder = JSONDecoder()

    init(baseURL: URL, session: URLSession = .shared) {
        self.baseURL = baseURL
        self.session = session
    }

    // MARK: - Contract endpoints

    /// GET /search?q=...
    func search(query: String) async throws -> [Track] {
        try await get("search", query: [URLQueryItem(name: "q", value: query)], as: TracksResponse.self).results
    }

    /// POST /seed { track_id }. Blocking on the server: warm instant, cold
    /// up to ~20s while the embed worker analyzes — hence the long timeout.
    @discardableResult
    func seed(trackID: String) async throws -> SeedResponse {
        try await post("seed", body: ["track_id": trackID], as: SeedResponse.self, timeout: 30)
    }

    /// GET /axes
    func axes() async throws -> [Axis] {
        try await get("axes", as: AxesResponse.self).axes
    }

    /// GET /recommend?track_id=...&axis=...&limit=...
    func recommend(trackID: String, axis: String, limit: Int = 10) async throws -> [Track] {
        try await get("recommend", query: [
            URLQueryItem(name: "track_id", value: trackID),
            URLQueryItem(name: "axis", value: axis),
            URLQueryItem(name: "limit", value: String(limit)),
        ], as: RecommendResponse.self).results
    }

    /// GET /viz/map?track_id=...&axis=...&limit=... — demo/debug endpoint
    /// behind the Insights screen, NOT part of contract/contract.md.
    func vizMap(trackID: String, axis: String, limit: Int = 10,
                correction: Bool? = nil) async throws -> VizMap {
        var query = [
            URLQueryItem(name: "track_id", value: trackID),
            URLQueryItem(name: "axis", value: axis),
            URLQueryItem(name: "limit", value: String(limit)),
        ]
        if let correction {
            query.append(URLQueryItem(name: "correction",
                                      value: correction ? "on" : "off"))
        }
        return try await get("viz/map", query: query, as: VizMap.self)
    }

    func vizWalk(from: String, to: String, k: Int = 8) async throws -> VizWalk {
        try await get("viz/walk", query: [
            URLQueryItem(name: "from", value: from),
            URLQueryItem(name: "to", value: to),
            URLQueryItem(name: "k", value: String(k)),
        ], as: VizWalk.self)
    }

    func vizHistogram(trackID: String) async throws -> VizHistogram {
        try await get("viz/histogram", query: [
            URLQueryItem(name: "track_id", value: trackID),
        ], as: VizHistogram.self)
    }

    /// GET /viz/hubs?track_id=...&recs=... — whole-corpus hub tracks, narrowed
    /// to the seed's subset when a seed and its recs are supplied.
    func vizHubs(seed: String? = nil, recs: [String] = []) async throws -> VizHubs {
        var query: [URLQueryItem] = []
        if let seed { query.append(URLQueryItem(name: "track_id", value: seed)) }
        if !recs.isEmpty { query.append(URLQueryItem(name: "recs", value: recs.joined(separator: ","))) }
        return try await get("viz/hubs", query: query, as: VizHubs.self)
    }

    /// GET /viz/tour?track_id=...&recs=... — 8-d PCA coordinates over the
    /// seed's subset for the Grand Tour.
    func vizTour(seed: String? = nil, recs: [String] = []) async throws -> VizTour {
        var query: [URLQueryItem] = []
        if let seed { query.append(URLQueryItem(name: "track_id", value: seed)) }
        if !recs.isEmpty { query.append(URLQueryItem(name: "recs", value: recs.joined(separator: ","))) }
        return try await get("viz/tour", query: query, as: VizTour.self)
    }

    /// GET /viz/attribute — per-band occlusion attribution for one pair.
    /// Answers `pending` until the Mac worker has run the counterfactuals.
    func vizAttribute(seed: String, rec: String) async throws -> VizAttribution {
        try await get("viz/attribute", query: [
            URLQueryItem(name: "seed", value: seed),
            URLQueryItem(name: "rec", value: rec),
        ], as: VizAttribution.self)
    }

    /// GET /viz/mst?track_id=...&recs=... — minimum spanning tree over cosine
    /// distance (H0 barcode), narrowed to the seed's subset when supplied.
    func vizMST(seed: String? = nil, recs: [String] = []) async throws -> VizMST {
        var query: [URLQueryItem] = []
        if let seed { query.append(URLQueryItem(name: "track_id", value: seed)) }
        if !recs.isEmpty { query.append(URLQueryItem(name: "recs", value: recs.joined(separator: ","))) }
        return try await get("viz/mst", query: query, as: VizMST.self)
    }

    /// GET /viz/extremes?pc=...&limit=...&track_id=...&recs=... — what a
    /// principal component sounds like: its most negative and most positive
    /// tracks, narrowed to the seed's subset when supplied.
    func vizExtremes(pc: Int, limit: Int = 4, seed: String? = nil, recs: [String] = []) async throws -> VizExtremesResponse {
        var query = [
            URLQueryItem(name: "pc", value: String(pc)),
            URLQueryItem(name: "limit", value: String(limit)),
        ]
        if let seed { query.append(URLQueryItem(name: "track_id", value: seed)) }
        if !recs.isEmpty { query.append(URLQueryItem(name: "recs", value: recs.joined(separator: ","))) }
        return try await get("viz/extremes", query: query, as: VizExtremesResponse.self)
    }

    // MARK: - Transport

    private func get<T: Decodable>(_ path: String, query: [URLQueryItem] = [], as type: T.Type) async throws -> T {
        guard var components = URLComponents(url: baseURL.appendingPathComponent(path), resolvingAgainstBaseURL: false) else {
            throw APIError.invalidURL
        }
        if !query.isEmpty { components.queryItems = query }
        guard let url = components.url else { throw APIError.invalidURL }
        var request = URLRequest(url: url)
        request.httpMethod = "GET"
        return try await send(request, as: type)
    }

    private func post<T: Decodable>(_ path: String, body: [String: String], as type: T.Type, timeout: TimeInterval? = nil) async throws -> T {
        var request = URLRequest(url: baseURL.appendingPathComponent(path))
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try JSONEncoder().encode(body)
        if let timeout { request.timeoutInterval = timeout }
        return try await send(request, as: type)
    }

    private func send<T: Decodable>(_ request: URLRequest, as type: T.Type) async throws -> T {
        var request = request
        for (field, value) in AppConfig.extraHeaders {
            request.setValue(value, forHTTPHeaderField: field)
        }
        do {
            let (data, response) = try await session.data(for: request)
            guard let http = response as? HTTPURLResponse else { throw APIError.invalidResponse }
            guard (200..<300).contains(http.statusCode) else {
                print("[API] \(request.httpMethod ?? "GET") \(request.url?.absoluteString ?? "<invalid url>") -> \(http.statusCode)")
                if http.statusCode == 409 { throw APIError.reanalyzing }
                throw APIError.status(http.statusCode)
            }
            do {
                return try decoder.decode(T.self, from: data)
            } catch {
                print("[API] decode failed for \(request.url?.absoluteString ?? "<invalid url>"): \(error)")
                throw APIError.decoding(error)
            }
        } catch {
            if !(error is APIError) {
                print("[API] request failed for \(request.httpMethod ?? "GET") \(request.url?.absoluteString ?? "<invalid url>"): \(error)")
            }
            throw error
        }
    }

    // MARK: - Response envelopes

    private struct TracksResponse: Decodable {
        let results: [Track]
    }

    private struct AxesResponse: Decodable {
        let axes: [Axis]
    }

    private struct RecommendResponse: Decodable {
        let seedTrackID: String
        let axis: String
        let results: [Track]

        enum CodingKeys: String, CodingKey {
            case seedTrackID = "seed_track_id"
            case axis
            case results
        }
    }
}

nonisolated struct SeedResponse: Decodable {
    let trackID: String
    let status: String

    enum CodingKeys: String, CodingKey {
        case trackID = "track_id"
        case status
    }
}
