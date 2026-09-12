//
//  AppConfig.swift
//  Hackathon
//
//  The single place the server address is configured. Points at the mock
//  server during development; flip `baseURL` to the tunnel URL at integration
//  (design spec hour 16). Nothing else in the app needs to change.
//

import Foundation

enum AppConfig {
    /// Live Essencia server: the Oracle VM, reached through the shared Caddy
    /// instance that terminates TLS for essentia.gabeyocum.com. Caddy strips
    /// the `/api` prefix before proxying to the container, so every request
    /// this app makes must include it. Override without editing code by
    /// setting ESSENTIA_BASE_URL.
    nonisolated static let baseURL: URL = {
        let env = ProcessInfo.processInfo.environment["ESSENTIA_BASE_URL"]?.trimmingCharacters(in: .whitespacesAndNewlines)
        let raw = (env?.isEmpty == false ? env : nil) ?? "https://essentia.gabeyocum.com/api"
        return URL(string: raw)!
    }()

    /// Extra headers applied to every request. (ngrok needed a skip-warning
    /// header; Cloudflare doesn't — kept as a hook for whatever tunnel is used.)
    nonisolated static let extraHeaders: [String: String] = [:]
}
