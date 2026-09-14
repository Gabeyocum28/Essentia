//
//  APIClientTests.swift
//  HackathonTests
//
//  Exercises the four contract endpoints against a stubbed URLSession, so the
//  client can be validated without a live server.
//

import Foundation
import Testing
@testable import Hackathon

/// Intercepts requests and returns a canned response supplied per test.
/// The intercepted request is recorded so tests can assert on it *after* the
/// awaited call, on the test's own thread (never inside `startLoading`).
final class MockURLProtocol: URLProtocol {
    nonisolated(unsafe) static var handler: ((URLRequest) -> (Int, Data))?
    nonisolated(unsafe) static var lastRequest: URLRequest?

    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

    override func startLoading() {
        MockURLProtocol.lastRequest = request
        guard let handler = MockURLProtocol.handler else {
            client?.urlProtocol(self, didFailWithError: URLError(.badServerResponse))
            return
        }
        let (status, data) = handler(request)
        let response = HTTPURLResponse(url: request.url!, statusCode: status, httpVersion: nil, headerFields: nil)!
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        client?.urlProtocol(self, didLoad: data)
        client?.urlProtocolDidFinishLoading(self)
    }

    override func stopLoading() {}
}

/// Serialized: the stub handler is shared static state, so tests must not run
/// in parallel and clobber each other's response.
@Suite(.serialized)
struct APIClientTests {

    private func makeClient() -> APIClient {
        MockURLProtocol.lastRequest = nil
        let config = URLSessionConfiguration.ephemeral
        config.protocolClasses = [MockURLProtocol.self]
        return APIClient(baseURL: URL(string: "http://test.local")!, session: URLSession(configuration: config))
    }

    @Test func searchDecodesResults() async throws {
        MockURLProtocol.handler = { _ in
            let body = """
            { "results": [
              { "track_id": "1", "title": "A", "artist": "X", "album": "Z",
                "artwork_url": "https://e.com/a.jpg", "preview_url": "https://e.com/a.mp3" }
            ] }
            """
            return (200, Data(body.utf8))
        }
        let results = try await makeClient().search(query: "miles")
        #expect(MockURLProtocol.lastRequest?.url?.path == "/search")
        #expect(results.count == 1)
        #expect(results.first?.title == "A")
    }

    @Test func axesDecodeList() async throws {
        MockURLProtocol.handler = { _ in
            let body = """
            { "axes": [ { "id": "groove", "label": "Keep the groove" } ] }
            """
            return (200, Data(body.utf8))
        }
        let axes = try await makeClient().axes()
        #expect(axes.first?.id == "groove")
    }

    @Test func recommendDecodesScoredResults() async throws {
        MockURLProtocol.handler = { _ in
            let body = """
            { "seed_track_id": "1", "axis": "groove", "results": [
              { "track_id": "2", "title": "B", "artist": "Y", "album": "Z",
                "artwork_url": "https://e.com/b.jpg", "preview_url": "https://e.com/b.mp3", "score": 0.8 }
            ] }
            """
            return (200, Data(body.utf8))
        }
        let results = try await makeClient().recommend(trackID: "1", axis: "groove")
        #expect(MockURLProtocol.lastRequest?.url?.path == "/recommend")
        #expect(results.first?.score == 0.8)
    }

    @Test func seedDecodesStatus() async throws {
        MockURLProtocol.handler = { _ in
            (200, Data(#"{ "track_id": "1", "status": "ready" }"#.utf8))
        }
        let response = try await makeClient().seed(trackID: "1")
        #expect(MockURLProtocol.lastRequest?.httpMethod == "POST")
        #expect(response.status == "ready")
    }

    @Test func non200Throws() async throws {
        MockURLProtocol.handler = { _ in (500, Data()) }
        await #expect(throws: APIError.self) {
            _ = try await makeClient().search(query: "x")
        }
    }

    /// 409 is not a server fault: the seed's vectors came from a superseded
    /// audio model and the worker is redoing it, so the message the user sees
    /// has to say that rather than "the server returned status 409".
    @Test func aConflictReadsAsReanalyzing() async throws {
        MockURLProtocol.handler = { _ in
            (409, Data(#"{ "status": "unanalyzed", "track_id": "1", "reason": "queued" }"#.utf8))
        }
        do {
            _ = try await makeClient().recommend(trackID: "1", axis: "sounds_like")
            Issue.record("a 409 must throw")
        } catch let error as APIError {
            guard case .reanalyzing = error else {
                Issue.record("expected .reanalyzing, got \(error)")
                return
            }
            #expect(error.errorDescription?.contains("re-analyzing") == true)
        }
    }

    @Test func seedDecodesUnanalyzedStatus() async throws {
        MockURLProtocol.handler = { _ in
            (200, Data(#"{ "track_id": "3135556", "status": "unanalyzed" }"#.utf8))
        }
        let response = try await makeClient().seed(trackID: "3135556")
        #expect(response.status == "unanalyzed")
    }

    @Test func seedRequestAllowsThirtySecondWait() async throws {
        MockURLProtocol.handler = { _ in
            (200, Data(#"{ "track_id": "1", "status": "ready" }"#.utf8))
        }
        _ = try await makeClient().seed(trackID: "1")
        #expect(MockURLProtocol.lastRequest?.timeoutInterval == 30)
    }

    @Test func vizWalkSendsBothEndpointsAndK() async throws {
        MockURLProtocol.handler = { _ in
            let body = """
            { "path": [
              { "track_id": "1", "title": "A", "artist": "X", "album": "Z",
                "artwork_url": null, "preview_url": null, "x": 0, "y": 0 },
              { "track_id": "2", "title": "B", "artist": "Y", "album": "Z",
                "artwork_url": null, "preview_url": null, "x": 1, "y": 1 }
            ], "geodesic": 1.2, "ambient": 1.0, "detour": 1.2, "k": 6 }
            """
            return (200, Data(body.utf8))
        }

        _ = try await makeClient().vizWalk(from: "1", to: "2", k: 6)

        let requestURL = try #require(MockURLProtocol.lastRequest?.url)
        let components = try #require(
            URLComponents(url: requestURL, resolvingAgainstBaseURL: false)
        )
        let query = Dictionary(uniqueKeysWithValues:
            (components.queryItems ?? []).map { ($0.name, $0.value) }
        )
        #expect(components.path == "/viz/walk")
        #expect(query["from"] == "1")
        #expect(query["to"] == "2")
        #expect(query["k"] == "6")
    }
}
