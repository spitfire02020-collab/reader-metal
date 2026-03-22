import Foundation
import Metal

// MARK: - MetalLibraryLoader

/// Loads compiled Metal shader libraries from the app bundle.
///
/// On iOS, Xcode's build system automatically compiles `.metal` source files
/// that are part of the target's "Compile Sources" build phase. The resulting
/// compiled library is accessible via `device.makeDefaultLibrary()`.
///
/// The loader first attempts to use the default library (compiled by Xcode),
/// then falls back to explicitly loading a named `.metal` file from the bundle.
///
/// Usage:
///   let library = try MetalLibraryLoader.load()
///   let kernel = library.makeFunction(name: "tanh_gelu_kernel")!
final enum MetalLibraryLoader {

    /// Errors that can occur when loading a Metal library.
    enum LoadError: Error, LocalizedError {
        case deviceNotFound
        case defaultLibraryFailed
        case bundleResourceNotFound(String)

        var errorDescription: String? {
            switch self {
            case .deviceNotFound:
                return "No Metal-capable GPU device found"
            case .defaultLibraryFailed:
                return "Could not load default Metal library — ensure .metal files are added to the target's Compile Sources build phase"
            case .bundleResourceNotFound(let name):
                return "Metal resource '\(name).metal' not found in app bundle"
            }
        }
    }

    /// Shared default library, loaded once.
    private static var _cachedLibrary: MTLLibrary?

    /// Load the compiled Metal shader library.
    ///
    /// - Returns: The default `MTLLibrary` containing compiled kernel functions.
    ///
    /// - Note: On iOS Simulator (arm64), Metal is available but some features
    ///   may differ from physical hardware. Test on a real device for accurate
    ///   GPU performance validation.
    static func load() throws -> MTLLibrary {
        if let cached = _cachedLibrary {
            return cached
        }

        guard let device = MTLCreateSystemDefaultDevice() else {
            throw LoadError.deviceNotFound
        }

        // Primary: use the default library compiled by Xcode from .metal files
        // in the target's Compile Sources build phase.
        if let defaultLib = device.makeDefaultLibrary() {
            _cachedLibrary = defaultLib
            return defaultLib
        }

        // Fallback: explicitly load a named .metal file from the bundle.
        // This path is primarily useful for development/debugging with
        // pre-compiled .metallib files, or for loading library archives.
        let bundle = Bundle.main
        let resourceNames = [
            "LanguageModel",
            "GEMMMetal",
            "LMForward",
            "Attention",
            "TanhGelu",
            "MaskCompute",
        ]

        for resourceName in resourceNames {
            if let url = bundle.url(forResource: resourceName, withExtension: "metal") {
                let library = try device.makeLibrary(URL: url)
                _cachedLibrary = library
                return library
            }
        }

        throw LoadError.defaultLibraryFailed
    }

    /// Clear the cached library (useful for testing or hot-reload scenarios).
    static func clearCache() {
        _cachedLibrary = nil
    }
}
