import Foundation
import Combine

// MARK: - Model Download Service
// Downloads Chatterbox Turbo ONNX models from HuggingFace

enum ModelVariant: String, CaseIterable {
    case fp32 = "fp32"
    case fp16 = "fp16"
    case q8 = "q8"
    case q4 = "q4"
    case q4f16 = "q4f16"

    var suffix: String {
        switch self {
        case .fp32: return ""
        case .fp16: return "_fp16"
        case .q8: return "_quantized"
        case .q4: return "_q4"
        case .q4f16: return "_q4f16"
        }
    }

    var displayName: String {
        switch self {
        case .fp32: return "Full Precision (3.3 GB)"
        case .fp16: return "Half Precision (1.6 GB)"
        case .q8: return "8-bit Quantized (1.1 GB)"
        case .q4: return "4-bit Quantized (720 MB)"
        case .q4f16: return "4-bit + FP16 (558 MB)"
        }
    }
}

enum ModelComponent: String, CaseIterable {
    case speechEncoder = "speech_encoder"
    case embedTokens = "embed_tokens"
    case languageModel = "language_model"
    case conditionalDecoder = "conditional_decoder"

    var displayName: String {
        switch self {
        case .speechEncoder: return "Speech Encoder"
        case .embedTokens: return "Token Embeddings"
        case .languageModel: return "Language Model"
        case .conditionalDecoder: return "Audio Decoder"
        }
    }
}

struct DownloadProgress {
    var component: ModelComponent
    var bytesDownloaded: Int64
    var totalBytes: Int64
    var fraction: Double { totalBytes > 0 ? Double(bytesDownloaded) / Double(totalBytes) : 0 }
}

@MainActor
final class ModelDownloadService: NSObject, ObservableObject {
    static let shared = ModelDownloadService()

    private static let repoBaseURL = "https://huggingface.co/ResembleAI/chatterbox-turbo-ONNX/resolve/main/onnx"
    // Tokenizer MUST come from the same repo as the models (GPT-2 byte-level BPE, 50257 vocab).
    // The onnx-community tokenizer (704-token char-BPE) is for a different Llama model and
    // produces garbled speech when used with the turbo (GPT-2-architecture) models.
    private static let tokenizerURL = "https://huggingface.co/ResembleAI/chatterbox-turbo-ONNX/resolve/main/tokenizer.json"
    private static let defaultVoiceURL = "https://huggingface.co/ResembleAI/chatterbox-turbo-ONNX/resolve/main/default_voice.wav"

    @Published var isDownloading = false
    @Published var overallProgress: Double = 0
    @Published var currentComponent: ModelComponent?
    @Published var componentProgress: [ModelComponent: Double] = [:]
    @Published var isModelReady = false
    @Published var errorMessage: String?

    private var downloadTasks: [URLSessionDownloadTask] = []
    private var urlSession: URLSession!

    private override init() {
        super.init()
        let config = URLSessionConfiguration.default
        config.timeoutIntervalForResource = 3600 // 1 hour for large models
        urlSession = URLSession(configuration: config)
    }

    // MARK: - Directory Management

    /// Documents directory for models
    var modelsDirectory: URL {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        let dir = docs.appendingPathComponent("ChatterboxModels", isDirectory: true)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    /// Bundle models directory - check both possible locations
    var bundleModelsDirectory: URL {
        // First try subdirectory
        let subdir = Bundle.main.bundleURL.appendingPathComponent("ChatterboxModels")
        if FileManager.default.fileExists(atPath: subdir.path) {
            return subdir
        }
        // Fall back to root (where they actually are)
        return Bundle.main.bundleURL
    }

    /// List of built-in voice file names (without extension)
    static let builtInVoiceFiles = ["Abigail", "Emily", "Alexander", "Jade", "Henry", "default_voice"]

    /// Copy voice files from bundle to Documents directory
    func copyVoiceFilesIfNeeded() {
        let docsDir = modelsDirectory
        let bundleDir = bundleModelsDirectory

        NSLog("[ModelDownload] Copying voice files from bundle to Documents...")

        for voiceFile in Self.builtInVoiceFiles {
            let wavFile = "\(voiceFile).wav"
            let dstPath = docsDir.appendingPathComponent(wavFile).path

            if FileManager.default.fileExists(atPath: dstPath) {
                NSLog("[ModelDownload] \(wavFile) already exists in Documents")
                continue
            }

            let srcPath = bundleDir.appendingPathComponent(wavFile).path
            if FileManager.default.fileExists(atPath: srcPath) {
                do {
                    try FileManager.default.copyItem(atPath: srcPath, toPath: dstPath)
                    NSLog("[ModelDownload] Successfully copied \(wavFile)")
                } catch {
                    NSLog("[ModelDownload] Failed to copy \(wavFile): \(error)")
                }
            } else {
                NSLog("[ModelDownload] \(wavFile) not found in bundle!")
            }
        }
    }

    /// Get voice file URL from Documents directory
    func voiceFilePath(for voiceId: String) -> URL? {
        // Map voice ID to voice file name
        let voiceFileName: String
        switch voiceId {
        case "default": voiceFileName = "Abigail"
        case "warm": voiceFileName = "Emily"
        case "energetic": voiceFileName = "Alexander"
        case "calm": voiceFileName = "Jade"
        case "deep": voiceFileName = "Henry"
        default: voiceFileName = "Abigail"
        }

        let path = modelsDirectory.appendingPathComponent("\(voiceFileName).wav")
        if FileManager.default.fileExists(atPath: path.path) {
            return path
        }
        return nil
    }

    /// Check if models exist, copy from bundle if needed
    /// NOTE: ONNX model files (.onnx, .onnx_data) are NOT copied to Documents -
    /// they are always loaded from the bundle where the external data file
    /// references work correctly. Only voice files and tokenizer are copied.
    func ensureModelsExist() {
        let docsDir = modelsDirectory
        let bundleDir = bundleModelsDirectory

        // List only non-ONNX files to copy to Documents
        // Voice files and tokenizer can be copied, but ONNX models must stay in bundle
        // because the external .onnx_data references break when copied
        let filesToCopy = ["tokenizer.json", "default_voice.wav"]

        NSLog("[ModelDownload] Checking models in: \(docsDir.path)")
        NSLog("[ModelDownload] Checking bundle: \(bundleDir.path)")

        // Check what exists in Documents
        let existingInDocs = Set((try? FileManager.default.contentsOfDirectory(atPath: docsDir.path)) ?? [])
        NSLog("[ModelDownload] Files in Documents: \(existingInDocs)")

        // Check what exists in bundle
        let bundleFiles = Set((try? FileManager.default.contentsOfDirectory(atPath: bundleDir.path)) ?? [])
        NSLog("[ModelDownload] Files in bundle: \(bundleFiles)")

        // Copy any missing files from bundle to Documents (only non-ONNX files)
        for file in filesToCopy {
            let dstPath = docsDir.appendingPathComponent(file).path

            if existingInDocs.contains(file) {
                NSLog("[ModelDownload] \(file) already exists in Documents")
                continue
            }

            if bundleFiles.contains(file) {
                let srcPath = bundleDir.appendingPathComponent(file).path
                NSLog("[ModelDownload] Copying \(file) from bundle to Documents...")

                do {
                    try FileManager.default.copyItem(atPath: srcPath, toPath: dstPath)
                    NSLog("[ModelDownload] Successfully copied \(file)")
                } catch {
                    NSLog("[ModelDownload] Failed to copy \(file): \(error)")
                }
            } else {
                NSLog("[ModelDownload] \(file) not found in bundle!")
            }
        }

        // Clean up any broken ONNX files from Documents to ensure fresh bundle load
        let variant = ModelVariant.q4f16
        let onnxFiles = ModelComponent.allCases.flatMap { component -> [String] in
            ["\(component.rawValue)\(variant.suffix).onnx", "\(component.rawValue)\(variant.suffix).onnx_data"]
        }

        for file in onnxFiles {
            let filePath = docsDir.appendingPathComponent(file).path
            if existingInDocs.contains(file) {
                NSLog("[ModelDownload] Removing broken ONNX file from Documents: \(file)")
                try? FileManager.default.removeItem(atPath: filePath)
            }
        }

        // Verify final state
        let finalFiles = Set((try? FileManager.default.contentsOfDirectory(atPath: docsDir.path)) ?? [])
        NSLog("[ModelDownload] Final files in Documents: \(finalFiles)")
    }

    /// Copy models from app bundle to Documents on first launch (async)
    func copyBundleModelsIfNeeded() async {
        NSLog("[ModelDownload] copyBundleModelsIfNeeded started")

        // Run file operations on background to avoid blocking UI
        await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in
            Task.detached(priority: .background) {
                // Use Task { } to properly move off main actor
                Task {
                    await self.performBundleCopy()
                    NSLog("[ModelDownload] copyBundleModelsIfNeeded completed")
                    continuation.resume()
                }
            }
        }
    }

    /// Actual work of copying files - runs off main actor
    private func performBundleCopy() async {
        await MainActor.run {
            self.ensureModelsExist()
            self.copyVoiceFilesIfNeeded()
            self.checkModelAvailability()
        }
    }

    func modelPath(for component: ModelComponent, variant: ModelVariant = .q4f16) -> URL {
        let filename = "\(component.rawValue)\(variant.suffix).onnx"

        // First check bundle - models in bundle are properly structured
        let bundlePath = bundleModelsDirectory.appendingPathComponent(filename)
        if FileManager.default.fileExists(atPath: bundlePath.path) {
            return bundlePath
        }

        // Fall back to Documents directory
        return modelsDirectory.appendingPathComponent(filename)
    }

    func modelDataPath(for component: ModelComponent, variant: ModelVariant = .q4f16) -> URL {
        let filename = "\(component.rawValue)\(variant.suffix).onnx_data"
        // Check bundle first - if model weights are bundled, use them directly
        // Use FileManager instead of Bundle.main.path because files are in bundle root
        let bundlePath = bundleModelsDirectory.appendingPathComponent(filename)
        if FileManager.default.fileExists(atPath: bundlePath.path) {
            return bundlePath
        }
        // Fall back to Documents directory (for downloaded models)
        return modelsDirectory.appendingPathComponent(filename)
    }

    var tokenizerPath: URL {
        // Check bundle first - tokenizer in bundle is properly structured
        let bundlePath = bundleModelsDirectory.appendingPathComponent("tokenizer.json")
        if FileManager.default.fileExists(atPath: bundlePath.path) {
            return bundlePath
        }
        // Fall back to Documents directory
        return modelsDirectory.appendingPathComponent("tokenizer.json")
    }

    var defaultVoicePath: URL {
        // Check bundle first
        let bundlePath = bundleModelsDirectory.appendingPathComponent("default_voice.wav")
        if FileManager.default.fileExists(atPath: bundlePath.path) {
            return bundlePath
        }
        // Fall back to Documents directory
        return modelsDirectory.appendingPathComponent("default_voice.wav")
    }

    // MARK: - Status Check

    /// Async version of checkModelAvailability that runs file checks on background thread
    func checkModelAvailabilityAsync(variant: ModelVariant = .q4f16) async {
        // Run expensive file operations on background thread
        await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in
            Task.detached(priority: .background) {
                Task { @MainActor in
                    self.performModelAvailabilityCheck(variant: variant)
                    continuation.resume()
                }
            }
        }
    }

    /// Actual model availability check implementation
    private func performModelAvailabilityCheck(variant: ModelVariant = .q4f16) {
        NSLog("[ModelDownload] checkModelAvailabilityAsync: starting background check")

        // Check each model component
        var missingFiles: [String] = []
        for component in ModelComponent.allCases {
            let modelPathURL = self.modelPath(for: component, variant: variant)
            let dataPathURL = self.modelDataPath(for: component, variant: variant)

            let modelExists = FileManager.default.fileExists(atPath: modelPathURL.path)
            let dataExists = FileManager.default.fileExists(atPath: dataPathURL.path)

            NSLog("[ModelDownload] checkModelAvailabilityAsync: \(component.rawValue) model exists=\(modelExists), data exists=\(dataExists)")

            if !modelExists {
                missingFiles.append(modelPathURL.lastPathComponent)
            }
            if !dataExists {
                missingFiles.append(dataPathURL.lastPathComponent)
            }
        }

        let tokenizerExists = FileManager.default.fileExists(atPath: self.tokenizerPath.path)
        NSLog("[ModelDownload] checkModelAvailabilityAsync: tokenizer exists=\(tokenizerExists)")

        // Check bundle for ONNX files (lightweight - just log presence)
        let bundleFiles = (try? FileManager.default.contentsOfDirectory(atPath: self.bundleModelsDirectory.path)) ?? []
        let onnxFiles = bundleFiles.filter { $0.hasSuffix(".onnx") }
        NSLog("[ModelDownload] checkModelAvailabilityAsync: bundle has \(onnxFiles.count) ONNX files")

        // Determine if models are ready (can use bundle files)
        let allPresent = ModelComponent.allCases.allSatisfy { component in
            FileManager.default.fileExists(atPath: self.modelPath(for: component, variant: variant).path) &&
            FileManager.default.fileExists(atPath: self.modelDataPath(for: component, variant: variant).path)
        }

        self.isModelReady = allPresent || !onnxFiles.isEmpty  // Ready if we have bundle models
        NSLog("[ModelDownload] checkModelAvailabilityAsync: isModelReady=\(self.isModelReady)")
    }

    func checkModelAvailability(variant: ModelVariant = .q4f16) {
        // Check each model component
        var missingFiles: [String] = []
        for component in ModelComponent.allCases {
            let modelPathURL = modelPath(for: component, variant: variant)
            let dataPathURL = modelDataPath(for: component, variant: variant)

            let modelExists = FileManager.default.fileExists(atPath: modelPathURL.path)
            let dataExists = FileManager.default.fileExists(atPath: dataPathURL.path)

            NSLog("[ModelDownload] checkModelAvailability: \(component.rawValue) model=\(modelPathURL.lastPathComponent) exists=\(modelExists), data=\(dataPathURL.lastPathComponent) exists=\(dataExists)")

            if !modelExists {
                missingFiles.append(modelPathURL.lastPathComponent)
            }
            if !dataExists {
                missingFiles.append(dataPathURL.lastPathComponent)
            }
        }

        let tokenizerExists = FileManager.default.fileExists(atPath: tokenizerPath.path)
        NSLog("[ModelDownload] checkModelAvailability: tokenizer exists=\(tokenizerExists), path=\(tokenizerPath.lastPathComponent)")

        // Also check what files exist in bundle
        let bundleFiles = (try? FileManager.default.contentsOfDirectory(atPath: bundleModelsDirectory.path)) ?? []
        NSLog("[ModelDownload] checkModelAvailability: bundle contains ONNX files: \(bundleFiles.filter { $0.hasSuffix(".onnx") })")

        let allPresent = ModelComponent.allCases.allSatisfy { component in
            FileManager.default.fileExists(atPath: modelPath(for: component, variant: variant).path) &&
            FileManager.default.fileExists(atPath: modelDataPath(for: component, variant: variant).path)
        }
        isModelReady = allPresent && tokenizerExists
        NSLog("[ModelDownload] checkModelAvailability: isModelReady=\(isModelReady)")
    }

    /// Fix topological sort in any already-downloaded ONNX graph files.
    /// Safe to call on every launch — ONNXGraphFixer only writes if a change is needed.
    func fixExistingModelsIfNeeded(variant: ModelVariant = .q4f16) {
        for component in ModelComponent.allCases {
            let url = modelPath(for: component, variant: variant)
            guard FileManager.default.fileExists(atPath: url.path) else { continue }
            try? ONNXGraphFixer.fixIfNeeded(at: url)
        }
    }

    // MARK: - Download

    func downloadModels(variant: ModelVariant = .q4f16) async {
        guard !isDownloading else { return }

        // Check if models are bundled - if so, skip download
        checkModelAvailability(variant: variant)
        if isModelReady {
            NSLog("[ModelDownload] Models bundled, skipping download")
            return
        }

        isDownloading = true
        errorMessage = nil
        overallProgress = 0

        do {
            // Download tokenizer and default reference voice (if not in bundle)
            if !FileManager.default.fileExists(atPath: tokenizerPath.path) {
                try await downloadFile(
                    from: Self.tokenizerURL,
                    to: tokenizerPath,
                    label: "Tokenizer"
                )
            }
            if !FileManager.default.fileExists(atPath: defaultVoicePath.path) {
                try await downloadFile(
                    from: Self.defaultVoiceURL,
                    to: defaultVoicePath,
                    label: "Default Voice"
                )
            }

            // Download each model component (ONNX graph + weights)
            for (index, component) in ModelComponent.allCases.enumerated() {
                currentComponent = component
                componentProgress[component] = 0

                let onnxFilename = "\(component.rawValue)\(variant.suffix).onnx"
                let dataFilename = "\(component.rawValue)\(variant.suffix).onnx_data"

                // Download ONNX graph file
                let onnxURL = "\(Self.repoBaseURL)/\(onnxFilename)"
                try await downloadFile(
                    from: onnxURL,
                    to: modelPath(for: component, variant: variant),
                    label: "\(component.displayName) graph"
                )

                // Download ONNX data/weights file
                let dataURL = "\(Self.repoBaseURL)/\(dataFilename)"
                try await downloadFile(
                    from: dataURL,
                    to: modelDataPath(for: component, variant: variant),
                    label: "\(component.displayName) weights"
                ) { progress in
                    Task { @MainActor in
                        self.componentProgress[component] = progress
                        let base = Double(index) / Double(ModelComponent.allCases.count)
                        let step = 1.0 / Double(ModelComponent.allCases.count)
                        self.overallProgress = base + (progress * step)
                    }
                }

                componentProgress[component] = 1.0
            }

            isModelReady = true
            overallProgress = 1.0
        } catch {
            errorMessage = "Download failed: \(error.localizedDescription)"
        }

        isDownloading = false
        currentComponent = nil
    }

    private func downloadFile(
        from urlString: String,
        to destination: URL,
        label: String,
        progressHandler: ((Double) -> Void)? = nil
    ) async throws {
        // Skip if file already exists
        if FileManager.default.fileExists(atPath: destination.path) {
            progressHandler?(1.0)
            return
        }

        guard let url = URL(string: urlString) else {
            throw URLError(.badURL)
        }

        let (tempURL, response) = try await urlSession.download(from: url)

        guard let httpResponse = response as? HTTPURLResponse,
              (200...299).contains(httpResponse.statusCode) else {
            throw URLError(.badServerResponse)
        }

        // Move downloaded file to destination
        if FileManager.default.fileExists(atPath: destination.path) {
            try FileManager.default.removeItem(at: destination)
        }
        try FileManager.default.moveItem(at: tempURL, to: destination)

        // Fix topological sort bug in ONNX graph files exported by PyTorch 2.9+
        // (Only the small .onnx graph file needs fixing; .onnx_data weights are untouched)
        if destination.pathExtension == "onnx" {
            try ONNXGraphFixer.fixIfNeeded(at: destination)
        }

        progressHandler?(1.0)
    }

    // MARK: - Cleanup

    func deleteModels() {
        try? FileManager.default.removeItem(at: modelsDirectory)
        isModelReady = false
    }

    func modelSizeOnDisk() -> String {
        guard FileManager.default.fileExists(atPath: modelsDirectory.path) else { return "0 MB" }
        let size = directorySize(url: modelsDirectory)
        return ByteCountFormatter.string(fromByteCount: Int64(size), countStyle: .file)
    }

    private func directorySize(url: URL) -> UInt64 {
        let enumerator = FileManager.default.enumerator(at: url, includingPropertiesForKeys: [.fileSizeKey])
        var total: UInt64 = 0
        while let fileURL = enumerator?.nextObject() as? URL {
            let size = (try? fileURL.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0
            total += UInt64(size)
        }
        return total
    }
}
