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

    /// Check if models exist, copy from bundle if needed
    func ensureModelsExist() {
        let docsDir = modelsDirectory
        let bundleDir = bundleModelsDirectory

        // List all required files
        let variant = ModelVariant.q4f16
        let onnxFiles = ModelComponent.allCases.map { "\($0.rawValue)\(variant.suffix).onnx" }
        let dataFiles = ModelComponent.allCases.map { "\($0.rawValue)\(variant.suffix).onnx_data" }
        let allRequiredFiles = onnxFiles + dataFiles + ["tokenizer.json", "default_voice.wav"]

        NSLog("[ModelDownload] Checking models in: \(docsDir.path)")
        NSLog("[ModelDownload] Checking bundle: \(bundleDir.path)")

        // Check what exists in Documents
        let existingInDocs = Set((try? FileManager.default.contentsOfDirectory(atPath: docsDir.path)) ?? [])
        NSLog("[ModelDownload] Files in Documents: \(existingInDocs)")

        // Check what exists in bundle
        let bundleFiles = Set((try? FileManager.default.contentsOfDirectory(atPath: bundleDir.path)) ?? [])
        NSLog("[ModelDownload] Files in bundle: \(bundleFiles)")

        // Copy any missing files from bundle to Documents
        for file in allRequiredFiles {
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

        // Verify copy worked
        let finalFiles = Set((try? FileManager.default.contentsOfDirectory(atPath: docsDir.path)) ?? [])
        NSLog("[ModelDownload] Final files in Documents: \(finalFiles)")
    }

    /// Copy models from app bundle to Documents on first launch (async)
    func copyBundleModelsIfNeeded() async {
        await withCheckedContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async {
                self.ensureModelsExist()
                DispatchQueue.main.async {
                    self.checkModelAvailability()
                    continuation.resume()
                }
            }
        }
    }

    func modelPath(for component: ModelComponent, variant: ModelVariant = .q4f16) -> URL {
        let filename = "\(component.rawValue)\(variant.suffix).onnx"
        return modelsDirectory.appendingPathComponent(filename)
    }

    func modelDataPath(for component: ModelComponent, variant: ModelVariant = .q4f16) -> URL {
        let filename = "\(component.rawValue)\(variant.suffix).onnx_data"
        return modelsDirectory.appendingPathComponent(filename)
    }

    var tokenizerPath: URL {
        modelsDirectory.appendingPathComponent("tokenizer.json")
    }

    var defaultVoicePath: URL {
        modelsDirectory.appendingPathComponent("default_voice.wav")
    }

    // MARK: - Status Check

    func checkModelAvailability(variant: ModelVariant = .q4f16) {
        let allPresent = ModelComponent.allCases.allSatisfy { component in
            FileManager.default.fileExists(atPath: modelPath(for: component, variant: variant).path) &&
            FileManager.default.fileExists(atPath: modelDataPath(for: component, variant: variant).path)
        }
        let tokenizerPresent = FileManager.default.fileExists(atPath: tokenizerPath.path)
        isModelReady = allPresent && tokenizerPresent
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

        isDownloading = true
        errorMessage = nil
        overallProgress = 0

        do {
            // Download tokenizer and default reference voice
            try await downloadFile(
                from: Self.tokenizerURL,
                to: tokenizerPath,
                label: "Tokenizer"
            )
            try await downloadFile(
                from: Self.defaultVoiceURL,
                to: defaultVoicePath,
                label: "Default Voice"
            )

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
