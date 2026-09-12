//
//  AppConfig.swift
//  Hackathon
//
//  The single place the server address is configured. Points at the live
//  Essentia backend on the Oracle VM, behind Caddy (deploy/). Override
//  without editing code by setting ESSENTIA_BASE_URL. Nothing else in the
//  app needs to change.
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

    /// Extra headers applied to every request. Empty: the VM is reached
    /// directly through Caddy over HTTPS, so nothing extra is needed. Kept
    /// as a hook in case a future backend requires one.
    nonisolated static let extraHeaders: [String: String] = [:]
}
