//
//  InsightsView.swift
//  Hackathon
//
//  The wow screen: where the corpus lies (galaxy map of embedding space), what
//  the playing track looks like (mel-spectrogram), and the actual arithmetic
//  behind the selected recommendation's score.
//

import SwiftUI

enum InsightMode: String, CaseIterable, Identifiable {
    case galaxy = "Galaxy"
    case sound = "Sound"
    case proof = "Proof"

    var id: String { rawValue }
}

enum GalaxyMode: String, CaseIterable, Identifiable {
    case explore = "Explore"
    case walk = "Walk"
    case tour = "Tour"
    case topo = "Topo"

    var id: String { rawValue }
}

struct WalkEndpoints: Equatable {
    let from: String
    let to: String
}

private enum InsightsLoadError: LocalizedError {
    case unanalyzedCorpus
    case mapUnavailable(Error)

    var errorDescription: String? {
        switch self {
        case .unanalyzedCorpus:
            return "This track is not analyzed on the server yet."
        case .mapUnavailable(let error):
            return "The math view could not load: \(error.localizedDescription)"
        }
    }
}

@MainActor
@Observable
final class InsightsModel {
    let seed: Track
    let axis: Axis
    var map: VizMap?
    var proofMap: VizMap?
    var selected: VizMap.Rec?
    var focusedPoint: GalaxySelection?
    var mode: InsightMode = .galaxy
    var galaxyMode: GalaxyMode = .explore
    var walkStart: GalaxySelection?
    var walk: VizWalk?
    var walkProgress: CGFloat = 0
    var histogram: VizHistogram?
    var hubs: VizHubs?
    var correctionEnabled = true
    /// True while the seed itself is the selection (no rec chosen).
    var seedSelected = false
    var isLoading = true
    var errorMessage: String?
    var proofError: String?

    // T2: Tour, Topology, eigen-listening — additive, fetched lazily so a
    // seed that never opens these modes never pays for them.
    var tour: VizTour?
    var tourError: String?
    var mst: VizMST?
    var topoError: String?
    var extremesByPC: [Int: VizExtremesResponse] = [:]
    var extremesErrors: [Int: String] = [:]

    /// T2.6: the seed-vs-selected-rec band attribution, the pair it belongs
    /// to, and whether we gave up waiting for the worker.
    var attribution: VizAttribution?
    var attributionPairID: String?
    var attributionSilent = false
    /// Set when a bar is tapped; SpectrogramView watches it and solos.
    var soloRequest: BandSoloRequest?

    private let api: APIClient
    private var correctionRequestGeneration = 0
    private var soloToken = 0

    init(seed: Track, axis: Axis, api: APIClient = .shared) {
        self.seed = seed
        self.axis = axis
        self.api = api
    }

    func load() async {
        isLoading = true
        errorMessage = nil
        proofError = nil
        do {
            try await loadPrimaryMap()
            histogram = try? await api.vizHistogram(trackID: seed.trackID)
            hubs = try? await api.vizHubs(seed: seed.trackID, recs: map?.recs.map(\.trackID) ?? [])
            proofMap = try? await api.vizMap(
                trackID: seed.trackID, axis: "surprise", correction: true
            )
        } catch {
            print("[Insights] load failed for seed \(seed.trackID) axis \(axis.id): \(error)")
            errorMessage = (error as? LocalizedError)?.errorDescription
                ?? "The math needs an analyzed corpus — is the server up?"
        }
        isLoading = false
    }

    private func loadPrimaryMap() async throws {
        let seedResponse = try await api.seed(trackID: seed.trackID)
        print("[Insights] seed status for \(seed.trackID): \(seedResponse.status)")
        guard seedResponse.status == "ready" else {
            throw InsightsLoadError.unanalyzedCorpus
        }
        do {
            let map = try await api.vizMap(trackID: seed.trackID, axis: axis.id)
            applyMap(map)
        } catch {
            throw InsightsLoadError.mapUnavailable(error)
        }
    }

    private func applyMap(_ map: VizMap) {
        self.map = map
        selected = map.recs.first
        if let rec = map.recs.first {
            focusedPoint = GalaxySelection(track: rec.track, x: rec.x, y: rec.y)
        }
    }

    // MARK: - T2.6 attribution

    /// Poll cadence and ceiling for /viz/attribute: the worker runs ~10
    /// forward passes per pair, so a first answer takes seconds, and a
    /// worker that is simply not running must not poll forever.
    private static let attributionPollInterval = Duration.milliseconds(1500)
    private static let attributionDeadline: Duration = .seconds(60)

    /// Fetches the seed-vs-rec band attribution, polling while the worker
    /// computes it.
    ///
    /// The loop is deliberately structured — no detached Task — so the
    /// `.task(id:)` that starts it also cancels it. Wrapping it in an
    /// unstructured Task and awaiting the value would keep polling for the
    /// full deadline after the view is gone, because cancellation does not
    /// cross that boundary.
    func loadAttribution(for rec: VizMap.Rec?) async {
        guard let rec else {
            attribution = nil
            attributionPairID = nil
            return
        }
        let pairID = "\(seed.trackID)|\(rec.trackID)"
        guard pairID != attributionPairID || attribution?.isReady != true else { return }

        attributionPairID = pairID
        attribution = nil
        attributionSilent = false

        let deadline = ContinuousClock.now + Self.attributionDeadline
        while !Task.isCancelled {
            let answer = try? await api.vizAttribute(seed: seed.trackID,
                                                     rec: rec.trackID)
            guard !Task.isCancelled, attributionPairID == pairID else { return }
            if let answer, !answer.isPending {
                attribution = answer
                return
            }
            if ContinuousClock.now >= deadline {
                attributionSilent = true
                return
            }
            do {
                try await Task.sleep(for: Self.attributionPollInterval)
            } catch {
                return          // cancelled mid-wait
            }
        }
    }

    /// Tapping a bar solos that band in whatever the spectrogram is showing.
    func solo(_ band: VizAttribution.Band) {
        soloToken += 1
        soloRequest = BandSoloRequest(loHz: band.loHz, hiHz: band.hiHz,
                                      token: soloToken)
    }

    /// One intent keeps the visual selection and audible selection together.
    func selectAndPlay(_ rec: VizMap.Rec, play: (Track) -> Void) {
        seedSelected = false
        selected = rec
        focusedPoint = GalaxySelection(track: rec.track, x: rec.x, y: rec.y)
        play(rec.track)
    }

    /// The seed is a track like any other on this screen — you need its own
    /// spectrogram to compare a recommendation against. It has no score of
    /// its own, so the math and attribution panels stand down while it is
    /// the selection.
    func selectSeedAndPlay(play: (Track) -> Void) {
        seedSelected = true
        selected = nil
        let track = map?.seed.track ?? seed
        if let point = map?.seed {
            focusedPoint = GalaxySelection(track: track, x: point.x, y: point.y)
        }
        play(track)
    }

    func focus(_ point: GalaxySelection) {
        focusedPoint = point
        if let rec = displayMap?.recs.first(where: { $0.trackID == point.id }) {
            seedSelected = false
            selected = rec
        }
    }

    func focus(_ track: Track) {
        guard let map = displayMap,
              let index = map.points.ids.firstIndex(of: track.trackID) else { return }
        focus(GalaxySelection(track: track, x: map.points.x[index],
                              y: map.points.y[index]))
    }

    var displayMap: VizMap? {
        // Always the axis the user picked. proofMap is a second map on the
        // surprise axis, loaded only so the hubness toggle has something to
        // compare; it stays inside that section, where it is labelled as
        // deliberately unlike the seed. Letting it lead the screen made the
        // galaxy and rec strip show songs the user never asked for.
        map
    }

    func activate(_ mode: InsightMode) {
        self.mode = mode
        guard !seedSelected else { return }
        let source = displayMap
        if let rec = source?.recs.first {
            selected = rec
            focusedPoint = GalaxySelection(track: rec.track, x: rec.x, y: rec.y)
        }
    }

    func setGalaxyMode(_ mode: GalaxyMode) {
        galaxyMode = mode
        walkStart = nil
        if mode == .explore {
            walk = nil
            walkProgress = 0
        }
    }

    /// First tap arms the walk; the second distinct tap yields the request.
    func selectWalkPoint(_ point: GalaxySelection) -> WalkEndpoints? {
        if walkStart?.id == point.id { return nil }
        guard let start = walkStart else {
            walkStart = point
            return nil
        }
        walkStart = nil
        return WalkEndpoints(from: start.id, to: point.id)
    }

    func selectGalaxyPoint(_ point: GalaxySelection) async {
        focus(point)
        guard galaxyMode == .walk,
              let endpoints = selectWalkPoint(point) else { return }
        do {
            walk = try await api.vizWalk(from: endpoints.from, to: endpoints.to)
            walkProgress = 0
            withAnimation(.easeInOut(duration: max(0.8, Double(walk?.path.count ?? 1) * 0.22))) {
                walkProgress = 1
            }
        } catch {
            proofError = "No connected walk for that pair. Try two closer stars."
        }
    }

    func setCorrection(_ enabled: Bool) async {
        correctionEnabled = enabled
        correctionRequestGeneration += 1
        let requestGeneration = correctionRequestGeneration
        do {
            let updated = try await api.vizMap(
                trackID: seed.trackID, axis: "surprise", correction: enabled
            )
            guard requestGeneration == correctionRequestGeneration else { return }
            // Only the comparison list changes: the galaxy, rec strip and
            // math panel stay on the axis the user actually chose.
            withAnimation(.spring(duration: 0.45)) {
                proofMap = updated
            }
            proofError = nil
        } catch {
            guard requestGeneration == correctionRequestGeneration else { return }
            correctionEnabled = !enabled
            proofError = "The correction comparison could not be loaded."
        }
    }

    // MARK: - T2 lazy loads

    func loadTourIfNeeded() async {
        guard tour == nil, tourError == nil else { return }
        do {
            tour = try await api.vizTour(seed: seed.trackID, recs: map?.recs.map(\.trackID) ?? [])
        } catch {
            tourError = "The grand tour could not be loaded."
        }
    }

    func loadTopologyIfNeeded() async {
        await loadTourIfNeeded()
        guard mst == nil, topoError == nil else { return }
        do {
            mst = try await api.vizMST(seed: seed.trackID, recs: map?.recs.map(\.trackID) ?? [])
        } catch {
            topoError = "The topology graph could not be loaded."
        }
    }

    func loadExtremesIfNeeded(pc: Int) async {
        guard extremesByPC[pc] == nil, extremesErrors[pc] == nil else { return }
        do {
            extremesByPC[pc] = try await api.vizExtremes(pc: pc, limit: 4, seed: seed.trackID, recs: map?.recs.map(\.trackID) ?? [])
        } catch {
            extremesErrors[pc] = "unavailable"
        }
    }
}

struct InsightsView: View {
    @State private var model: InsightsModel
    @Environment(PlaybackController.self) private var playback

    init(seed: Track, axis: Axis) {
        _model = State(initialValue: InsightsModel(seed: seed, axis: axis))
    }

    var body: some View {
        Group {
            if let errorMessage = model.errorMessage {
                RetryView(message: errorMessage) {
                    Task { await model.load() }
                }
            } else if let map = model.map {
                loaded(map)
            } else {
                ProgressView("Projecting \u{2026}")
            }
        }
        .navigationTitle("The Math")
        .navigationBarTitleDisplayMode(.inline)
        .task { await model.load() }
    }

    private func loaded(_ map: VizMap) -> some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                Picker("Insight mode", selection: $model.mode) {
                    ForEach(InsightMode.allCases) { mode in
                        Text(mode.rawValue.uppercased()).tag(mode)
                    }
                }
                .pickerStyle(.segmented)
                .onChange(of: model.mode) { _, mode in model.activate(mode) }

                // The strip picks which track everything below is about, so
                // it sits above the analysis rather than under a screenful
                // of it — on Sound especially, the spectrogram and the
                // matrix pushed it off the bottom of the page.
                recPicker(model.displayMap ?? map)

                switch model.mode {
                case .galaxy:
                    galaxyMode(model.displayMap ?? map)
                case .sound:
                    soundMode(map)
                case .proof:
                    proofMode(model.displayMap ?? map)
                }
            }
            .padding()
        }
    }

    private func galaxyMode(_ map: VizMap) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            Picker("Galaxy interaction", selection: $model.galaxyMode) {
                ForEach(GalaxyMode.allCases) { mode in Text(mode.rawValue).tag(mode) }
            }
            .pickerStyle(.segmented)
            .onChange(of: model.galaxyMode) { _, mode in
                model.setGalaxyMode(mode)
                switch mode {
                case .tour: Task { await model.loadTourIfNeeded() }
                case .topo: Task { await model.loadTopologyIfNeeded() }
                case .explore, .walk: break
                }
            }

            galaxyCaption(map)

            switch model.galaxyMode {
            case .explore, .walk:
                GalaxyMapView(
                    map: map,
                    selectedTrackID: model.focusedPoint?.id,
                    walk: model.walk,
                    walkProgress: model.walkProgress,
                    onSelect: { point in Task { await model.selectGalaxyPoint(point) } }
                )
                .aspectRatio(1, contentMode: .fit)
                .background(.black, in: .rect(cornerRadius: 12))

                if let point = model.focusedPoint { galaxyCallout(point) }
                if let walk = model.walk {
                    WalkStrip(walk: walk) { step in
                        let point = GalaxySelection(track: step.track, x: step.x, y: step.y)
                        model.focus(point)
                        playback.toggle(step.track)
                    }
                }

            case .tour:
                TourView(tour: model.tour, errorMessage: model.tourError)

            case .topo:
                TopologyView(
                    tour: model.tour, mst: model.mst,
                    tourError: model.tourError, mstError: model.topoError
                )
            }
        }
    }

    private func galaxyCaption(_ map: VizMap) -> some View {
        let text: String
        switch model.galaxyMode {
        case .walk:
            text = model.walkStart == nil ? "Tap a start star" : "Tap a destination star"
        case .tour:
            text = "Rotating an orthonormal 2-frame through 8-d PCA space \u{2014} clusters that persist are real"
        case .topo:
            text = "Single-linkage clustering = MST = H0 persistence \u{2014} drag the threshold"
        case .explore:
            text = "\(map.points.ids.count) tracks · pinch, pan, tap any star"
        }
        return Text(text)
            .font(.caption)
            .foregroundStyle(.secondary)
    }

    private func soundMode(_ map: VizMap) -> some View {
        VStack(alignment: .leading, spacing: 16) {
            SpectrogramView(track: spotlightTrack(map),
                            soloRequest: model.soloRequest)
            if let rec = model.selected {
                WhySimilarView(
                    recTitle: rec.title,
                    attribution: model.attribution,
                    isWorkerSilent: model.attributionSilent,
                    onSolo: { band in model.solo(band) }
                )
                .task(id: rec.trackID) { await model.loadAttribution(for: rec) }
            }
            EigenListeningView(model: model) { track in
                model.focus(track)
                playback.toggle(track)
            }
            if let rec = model.selected {
                MathPanel(seed: map.seed, rec: rec, axis: map.axis)
            }
        }
    }

    private func proofMode(_ map: VizMap) -> some View {
        VStack(alignment: .leading, spacing: 18) {
            ProofModeView(
                seedTitle: map.seed.title,
                histogram: model.histogram,
                hubs: model.hubs,
                surpriseRecs: model.proofMap?.recs ?? [],
                correctionEnabled: model.correctionEnabled,
                errorMessage: model.proofError,
                onCorrectionChanged: { enabled in
                    Task { await model.setCorrection(enabled) }
                },
                onPlay: { track in
                    model.focus(track)
                    playback.toggle(track)
                }
            )
            if let rec = model.selected {
                MathPanel(seed: map.seed, rec: rec, axis: map.axis)
            }
        }
    }

    /// The spectrogram follows whatever is playing, else the selection, else
    /// the seed.
    private func spotlightTrack(_ map: VizMap) -> Track {
        if let playing = playback.nowPlaying { return playing }
        return model.selected?.track ?? map.seed.track
    }

    private func recPicker(_ map: VizMap) -> some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 10) {
                seedChip(map)
                Divider().frame(height: 52)
                ForEach(map.recs) { rec in
                    Button {
                        model.selectAndPlay(rec) { playback.toggle($0) }
                    } label: {
                        VStack(spacing: 4) {
                            Artwork(url: rec.artworkURL)
                                .frame(width: 52, height: 52)
                                .overlay {
                                    if rec.trackID == model.selected?.trackID {
                                        RoundedRectangle(cornerRadius: 8)
                                            .stroke(Color.accentColor, lineWidth: 3)
                                    }
                                }
                            Text(rec.title)
                                .font(.caption2)
                                .lineLimit(1)
                                .frame(width: 60)
                        }
                    }
                    .buttonStyle(.plain)
                }
            }
        }
    }

    /// The seed leads the strip, ringed in the same yellow it wears on the
    /// galaxy, so "which of these am I looking at?" has one answer.
    private func seedChip(_ map: VizMap) -> some View {
        Button {
            model.selectSeedAndPlay { playback.toggle($0) }
        } label: {
            VStack(spacing: 4) {
                Artwork(url: map.seed.artworkURL)
                    .frame(width: 52, height: 52)
                    .overlay {
                        if model.seedSelected {
                            RoundedRectangle(cornerRadius: 8)
                                .stroke(Color.yellow, lineWidth: 3)
                        }
                    }
                Text("Seed")
                    .font(.caption2.weight(.semibold))
                    .foregroundStyle(.yellow)
                    .frame(width: 60)
            }
        }
        .buttonStyle(.plain)
        .accessibilityLabel("Play the seed, \(map.seed.title) by \(map.seed.artist)")
    }

    private func galaxyCallout(_ point: GalaxySelection) -> some View {
        Button {
            model.focus(point)
            playback.toggle(point.track)
        } label: {
            HStack(spacing: 12) {
                Artwork(url: point.track.artworkURL)
                    .frame(width: 54, height: 54)
                VStack(alignment: .leading, spacing: 2) {
                    Text(point.track.title)
                        .font(.headline)
                        .lineLimit(1)
                    Text(point.track.artist)
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                }
                Spacer()
                Image(systemName: playback.nowPlayingID == point.id && playback.isPlaying
                      ? "pause.fill" : "play.fill")
                    .font(.title3)
                    .foregroundStyle(Color.accentColor)
            }
            .padding(10)
            .background(.thinMaterial, in: .rect(cornerRadius: 12))
        }
        .buttonStyle(.plain)
        .accessibilityLabel("Play \(point.track.title) by \(point.track.artist)")
    }
}

/// The actual numbers behind the selected recommendation's score.
private struct MathPanel: View {
    let seed: VizMap.SeedPoint
    let rec: VizMap.Rec
    let axis: VizMap.AxisInfo

    // Analysis stores groove already normalized to [0, 1] (tempo on a log
    // scale), so the bars plot the values directly.
    private static let grooveLabels = ["Tempo", "Beat conf.", "Onsets", "Dance"]

    /// The blend's feature keys, in the words the screen already uses.
    private static func partLabel(_ key: String) -> String {
        switch key {
        case "embedding": return "sound"
        case "genre":     return "style"
        default:          return key
        }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("Why \u{201C}\(rec.title)\u{201D} scored \(rec.score, specifier: "%.3f")")
                .font(.headline)

            scoreFormula
                .font(.system(.footnote, design: .monospaced))
                .padding(10)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(.quaternary.opacity(0.5), in: .rect(cornerRadius: 8))

            if let seedGroove = seed.groove, let recGroove = rec.groove,
               seedGroove.count == 4, recGroove.count == 4 {
                grooveComparison(seed: seedGroove, rec: recGroove)
            }
        }
        .padding()
        .background(.background.secondary, in: .rect(cornerRadius: 12))
    }

    @ViewBuilder private var scoreFormula: some View {
        let m = rec.math
        VStack(alignment: .leading, spacing: 4) {
            if let parts = m.parts, let weights = axis.weights {
                // A blend has no single formula: the score is the weighted
                // mean of these percentiles, so show the terms being averaged
                // rather than a cosine that isn't the number above.
                Text("score = weighted mean of percentiles")
                ForEach(parts.keys.sorted(), id: \.self) { key in
                    Text("      " + Self.partLabel(key) + ": "
                         + String(format: "%.0fth \u{00D7} %.2f",
                                  parts[key] ?? 0, weights[key] ?? 0))
                }
                Text("      = \(rec.score, specifier: "%.3f")")
            } else if m.metric == "cosine" {
                Text("cos \u{03B8} = a\u{00B7}b / (\u{2016}a\u{2016}\u{2016}b\u{2016})")
                Text("      = \(m.dot, specifier: "%.2f") / (\(m.seedNorm, specifier: "%.2f") \u{00D7} \(m.recNorm, specifier: "%.2f"))")
                Text("      = \(m.dot / (m.seedNorm * m.recNorm), specifier: "%.3f")")
            } else if let d = m.distance {
                Text("score = 1 / (1 + \u{2016}a \u{2212} b\u{2016})")
                Text("      = 1 / (1 + \(d, specifier: "%.2f"))")
                Text("      = \(1 / (1 + d), specifier: "%.3f")")
            }
            if axis.direction == -1, let c = m.centrality {
                Text("ranked by distance, minus \(c, specifier: "%.3f") centrality")
                Text("(so \u{201C}weird for everything\u{201D} can\u{2019}t win every seed)")
            }
        }
    }

    private func grooveComparison(seed: [Double], rec: [Double]) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Groove: seed vs. this track")
                .font(.subheadline.weight(.medium))
            HStack(alignment: .bottom, spacing: 18) {
                ForEach(0..<4, id: \.self) { i in
                    VStack(spacing: 4) {
                        HStack(alignment: .bottom, spacing: 4) {
                            bar(seed[i], color: .yellow)
                            bar(rec[i], color: .accentColor)
                        }
                        Text(Self.grooveLabels[i])
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                        Text(String(format: "%.2f / %.2f", seed[i], rec[i]))
                            .font(.system(.caption2, design: .monospaced))
                    }
                }
            }
        }
    }

    private func bar(_ fraction: Double, color: Color) -> some View {
        RoundedRectangle(cornerRadius: 2)
            .fill(color)
            .frame(width: 14, height: max(4, min(fraction, 1) * 56))
    }
}
