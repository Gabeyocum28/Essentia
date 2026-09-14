//
//  Components.swift
//  Hackathon
//
//  Small shared views used across the feature screens.
//

import SwiftUI

/// Square album artwork with a graceful placeholder.
struct Artwork: View {
    let url: URL?

    var body: some View {
        AsyncImage(url: url) { phase in
            switch phase {
            case .success(let image):
                image.resizable().scaledToFill()
            default:
                Rectangle()
                    .fill(.quaternary)
                    .overlay {
                        Image(systemName: "music.note")
                            .foregroundStyle(.secondary)
                    }
            }
        }
        .clipShape(.rect(cornerRadius: 8))
    }
}

/// "CC BY-SA 3.0" from a Creative Commons deed URL, or nil.
///
/// The licence arrives only as the deed URL, never as a label: the URL is
/// what the credit has to link to, and a label derived from it cannot drift
/// out of step with it. Anything that is not a recognisable
/// `/licenses/<code>/<version>/` path gets nil rather than a guess. Mirrors
/// `licenceLabel` in web/src/components/Attribution.tsx.
nonisolated func licenceLabel(_ url: URL?) -> String? {
    guard let path = url?.path else { return nil }
    let parts = path.split(separator: "/").map(String.init)
    guard let index = parts.firstIndex(of: "licenses"), index + 1 < parts.count
    else { return nil }
    let code = parts[index + 1].uppercased()
    guard !code.isEmpty else { return nil }
    if index + 2 < parts.count, parts[index + 2].first?.isNumber == true {
        return "CC \(code) \(parts[index + 2])"
    }
    return "CC \(code)"
}

/// The credit a Creative Commons licence obliges us to show.
///
/// Renders nothing when the server sent no `attribution_url` — which is the
/// Deezer case, where the licence asks for no credit and a "via Deezer" line
/// on every row would be noise. Jamendo may only be switched on for the
/// phone once this ships, because for Jamendo the credit is not optional.
struct AttributionLine: View {
    let track: Track

    var body: some View {
        if let url = track.attributionURL {
            Link(destination: url) {
                Text(creditText)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
            }
            // The row itself is a tap target (play, or drill in); the credit
            // must open its backlink without also firing that.
            .buttonStyle(.plain)
        }
    }

    private var creditText: String {
        let source = track.source.map {
            $0.prefix(1).uppercased() + String($0.dropFirst())
        } ?? "the source"
        if let licence = licenceLabel(track.attributionURL) {
            return "via \(source) · \(licence)"
        }
        return "via \(source)"
    }
}

/// Compact title + artist row with small artwork, used in lists.
struct TrackRow: View {
    let track: Track

    var body: some View {
        HStack(spacing: 12) {
            Artwork(url: track.artworkURL)
                .frame(width: 56, height: 56)
            VStack(alignment: .leading, spacing: 2) {
                Text(track.title)
                    .lineLimit(1)
                Text(track.artist)
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
                AttributionLine(track: track)
            }
        }
    }
}

/// Large seed header shown at the top of the axis-selection screen.
struct TrackHeader: View {
    let track: Track

    var body: some View {
        VStack(spacing: 12) {
            Artwork(url: track.artworkURL)
                .frame(width: 160, height: 160)
            VStack(spacing: 4) {
                Text(track.title)
                    .font(.title3.weight(.semibold))
                    .multilineTextAlignment(.center)
                Text(track.artist)
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                AttributionLine(track: track)
            }
        }
    }
}

/// Simple error + retry affordance.
struct RetryView: View {
    let message: String
    let retry: () -> Void

    var body: some View {
        VStack(spacing: 12) {
            Text(message)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
            Button("Try Again", action: retry)
                .buttonStyle(.bordered)
        }
    }
}
