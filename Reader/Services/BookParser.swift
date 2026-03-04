import Foundation
import PDFKit
import Compression

// MARK: - Book Parser
// Parses EPUB and PDF files into structured text content

final class BookParser {

    // MARK: - EPUB Parsing

    /// Parse an EPUB file into chapters
    static func parseEPUB(at url: URL) throws -> ParsedBook {
        // EPUB is a ZIP archive containing XHTML files
        let tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tempDir) }

        // Unzip EPUB
        try unzipFile(at: url, to: tempDir)

        // Parse container.xml to find content.opf
        let containerPath = tempDir
            .appendingPathComponent("META-INF")
            .appendingPathComponent("container.xml")

        guard FileManager.default.fileExists(atPath: containerPath.path) else {
            throw BookParserError.invalidFormat("Missing META-INF/container.xml")
        }

        let containerXML = try String(contentsOf: containerPath, encoding: .utf8)
        guard let opfPath = extractAttribute(from: containerXML, tag: "rootfile", attribute: "full-path") else {
            throw BookParserError.invalidFormat("Cannot find content.opf path")
        }

        let opfURL = tempDir.appendingPathComponent(opfPath)
        let opfDir = opfURL.deletingLastPathComponent()
        let opfXML = try String(contentsOf: opfURL, encoding: .utf8)

        // Parse OPF to get title, author, and spine order
        let title = extractBetweenTags(opfXML, tag: "dc:title") ?? "Untitled"
        let author = extractBetweenTags(opfXML, tag: "dc:creator")

        // Extract manifest (id -> href mapping)
        let manifest = extractManifestItems(from: opfXML)

        // Extract spine order
        let spineIDs = extractSpineOrder(from: opfXML)

        // Read chapters in spine order
        var chapters: [Chapter] = []
        for itemID in spineIDs {
            guard let href = manifest[itemID] else { continue }
            let chapterURL = opfDir.appendingPathComponent(href)
            guard FileManager.default.fileExists(atPath: chapterURL.path) else { continue }

            let chapterHTML = (try? String(contentsOf: chapterURL, encoding: .utf8)) ?? ""
            let chapterText = stripHTML(chapterHTML)

            if chapterText.trimmingCharacters(in: .whitespacesAndNewlines).count > 50 {
                let chapterTitle = extractBetweenTags(chapterHTML, tag: "title")
                    ?? extractBetweenTags(chapterHTML, tag: "h1")
                    ?? extractBetweenTags(chapterHTML, tag: "h2")
                    ?? "Chapter \(chapters.count + 1)"

                chapters.append(Chapter(
                    title: chapterTitle.trimmingCharacters(in: .whitespacesAndNewlines),
                    textContent: chapterText.trimmingCharacters(in: .whitespacesAndNewlines)
                ))
            }
        }

        // Try to extract cover image
        let coverData = extractCoverImage(from: opfXML, manifest: manifest, baseDir: opfDir)

        return ParsedBook(
            title: title,
            author: author,
            chapters: chapters,
            coverImageData: coverData
        )
    }

    // MARK: - PDF Parsing

    /// Parse a PDF file into chapters (one per page or by bookmark structure)
    static func parsePDF(at url: URL) throws -> ParsedBook {
        guard let document = PDFDocument(url: url) else {
            throw BookParserError.invalidFormat("Cannot open PDF")
        }

        let title = document.documentAttributes?[PDFDocumentAttribute.titleAttribute] as? String
            ?? url.deletingPathExtension().lastPathComponent
        let author = document.documentAttributes?[PDFDocumentAttribute.authorAttribute] as? String

        var chapters: [Chapter] = []

        // Try to use document outline (bookmarks) for chapter structure
        if let outline = document.outlineRoot, outline.numberOfChildren > 0 {
            chapters = extractChaptersFromOutline(outline, document: document)
        }

        // Fallback: treat each page as content, grouped into rough chapters
        if chapters.isEmpty {
            var currentText = ""
            var chapterCount = 1

            for pageIndex in 0..<document.pageCount {
                guard let page = document.page(at: pageIndex) else { continue }
                let pageText = page.string ?? ""

                currentText += pageText + "\n\n"

                // Create a new chapter every ~10 pages or if text is getting long
                if (pageIndex + 1) % 10 == 0 || pageIndex == document.pageCount - 1 {
                    if !currentText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                        chapters.append(Chapter(
                            title: document.pageCount > 20 ? "Section \(chapterCount)" : "Content",
                            textContent: currentText.trimmingCharacters(in: .whitespacesAndNewlines)
                        ))
                        chapterCount += 1
                        currentText = ""
                    }
                }
            }
        }

        return ParsedBook(
            title: title,
            author: author,
            chapters: chapters,
            coverImageData: nil
        )
    }

    // MARK: - Plain Text

    static func parsePlainText(_ text: String, title: String) -> ParsedBook {
        let paragraphs = text.components(separatedBy: "\n\n")
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }

        // Group paragraphs into chapters of reasonable size
        var chapters: [Chapter] = []
        var currentText = ""
        var chapterIndex = 1

        for paragraph in paragraphs {
            currentText += paragraph + "\n\n"

            if currentText.count > 3000 {
                chapters.append(Chapter(
                    title: "Part \(chapterIndex)",
                    textContent: currentText.trimmingCharacters(in: .whitespacesAndNewlines)
                ))
                chapterIndex += 1
                currentText = ""
            }
        }

        if !currentText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            chapters.append(Chapter(
                title: chapters.isEmpty ? "Full Text" : "Part \(chapterIndex)",
                textContent: currentText.trimmingCharacters(in: .whitespacesAndNewlines)
            ))
        }

        return ParsedBook(title: title, author: nil, chapters: chapters, coverImageData: nil)
    }

    // MARK: - PDF Outline Extraction

    private static func extractChaptersFromOutline(_ outline: PDFOutline, document: PDFDocument) -> [Chapter] {
        var chapters: [Chapter] = []

        for i in 0..<outline.numberOfChildren {
            guard let child = outline.child(at: i) else { continue }
            let title = child.label ?? "Chapter \(i + 1)"

            // Get page range for this outline item
            guard let dest = child.destination, let page = dest.page else { continue }
            let pageIndex = document.index(for: page)

            // Determine end page (next outline item's page or end of document)
            var endPageIndex = document.pageCount - 1
            if i + 1 < outline.numberOfChildren,
               let nextChild = outline.child(at: i + 1),
               let nextDest = nextChild.destination,
               let nextPage = nextDest.page {
                let nextIndex = document.index(for: nextPage)
                endPageIndex = max(pageIndex, nextIndex - 1)
            }

            // Extract text from page range
            var text = ""
            for pi in pageIndex...endPageIndex {
                if let p = document.page(at: pi) {
                    text += (p.string ?? "") + "\n\n"
                }
            }

            if !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                chapters.append(Chapter(
                    title: title,
                    textContent: text.trimmingCharacters(in: .whitespacesAndNewlines)
                ))
            }
        }

        return chapters
    }

    // MARK: - EPUB Helpers

    private static func extractManifestItems(from opf: String) -> [String: String] {
        var items: [String: String] = [:]
        let pattern = "<item[^>]*id=[\"']([^\"']*)[\"'][^>]*href=[\"']([^\"']*)[\"'][^>]*/?>|<item[^>]*href=[\"']([^\"']*)[\"'][^>]*id=[\"']([^\"']*)[\"'][^>]*/?>"
        guard let regex = try? NSRegularExpression(pattern: pattern, options: .caseInsensitive) else {
            return items
        }
        let nsOPF = opf as NSString
        let matches = regex.matches(in: opf, range: NSRange(location: 0, length: nsOPF.length))
        for match in matches {
            let id: String
            let href: String
            if match.range(at: 1).location != NSNotFound {
                id = nsOPF.substring(with: match.range(at: 1))
                href = nsOPF.substring(with: match.range(at: 2))
            } else {
                href = nsOPF.substring(with: match.range(at: 3))
                id = nsOPF.substring(with: match.range(at: 4))
            }
            items[id] = href
        }
        return items
    }

    private static func extractSpineOrder(from opf: String) -> [String] {
        var ids: [String] = []
        let pattern = "<itemref[^>]*idref=[\"']([^\"']*)[\"'][^>]*/?"
        guard let regex = try? NSRegularExpression(pattern: pattern, options: .caseInsensitive) else {
            return ids
        }
        let nsOPF = opf as NSString
        let matches = regex.matches(in: opf, range: NSRange(location: 0, length: nsOPF.length))
        for match in matches {
            ids.append(nsOPF.substring(with: match.range(at: 1)))
        }
        return ids
    }

    private static func extractCoverImage(from opf: String, manifest: [String: String], baseDir: URL) -> Data? {
        // Look for cover in metadata
        let coverPattern = "<meta[^>]*name=[\"']cover[\"'][^>]*content=[\"']([^\"']*)[\"']"
        if let regex = try? NSRegularExpression(pattern: coverPattern, options: .caseInsensitive) {
            let nsOPF = opf as NSString
            if let match = regex.firstMatch(in: opf, range: NSRange(location: 0, length: nsOPF.length)) {
                let coverID = nsOPF.substring(with: match.range(at: 1))
                if let href = manifest[coverID] {
                    let coverURL = baseDir.appendingPathComponent(href)
                    return try? Data(contentsOf: coverURL)
                }
            }
        }
        return nil
    }

    // MARK: - HTML / XML Helpers

    private static func stripHTML(_ html: String) -> String {
        var text = html
        // Remove script/style blocks
        text = text.replacingOccurrences(of: "<script[^>]*>.*?</script>", with: "", options: [.regularExpression, .caseInsensitive])
        text = text.replacingOccurrences(of: "<style[^>]*>.*?</style>", with: "", options: [.regularExpression, .caseInsensitive])
        // Convert block elements to newlines
        text = text.replacingOccurrences(of: "<(p|div|br|h[1-6]|li|tr)[^>]*>", with: "\n", options: [.regularExpression, .caseInsensitive])
        // Strip remaining tags
        text = text.replacingOccurrences(of: "<[^>]+>", with: "", options: .regularExpression)
        // Decode entities
        text = text.replacingOccurrences(of: "&amp;", with: "&")
        text = text.replacingOccurrences(of: "&lt;", with: "<")
        text = text.replacingOccurrences(of: "&gt;", with: ">")
        text = text.replacingOccurrences(of: "&quot;", with: "\"")
        text = text.replacingOccurrences(of: "&nbsp;", with: " ")
        text = text.replacingOccurrences(of: "&#39;", with: "'")
        // Clean whitespace
        text = text.replacingOccurrences(of: "\\n{3,}", with: "\n\n", options: .regularExpression)
        return text
    }

    private static func extractBetweenTags(_ xml: String, tag: String) -> String? {
        let pattern = "<\(tag)[^>]*>(.*?)</\(tag)>"
        guard let regex = try? NSRegularExpression(pattern: pattern, options: [.dotMatchesLineSeparators, .caseInsensitive]) else {
            return nil
        }
        let nsXML = xml as NSString
        guard let match = regex.firstMatch(in: xml, range: NSRange(location: 0, length: nsXML.length)) else {
            return nil
        }
        return nsXML.substring(with: match.range(at: 1))
    }

    private static func extractAttribute(from xml: String, tag: String, attribute: String) -> String? {
        let pattern = "<\(tag)[^>]*\(attribute)=[\"']([^\"']*)[\"']"
        guard let regex = try? NSRegularExpression(pattern: pattern, options: .caseInsensitive) else {
            return nil
        }
        let nsXML = xml as NSString
        guard let match = regex.firstMatch(in: xml, range: NSRange(location: 0, length: nsXML.length)) else {
            return nil
        }
        return nsXML.substring(with: match.range(at: 1))
    }

    // MARK: - ZIP Extraction (minimal pure-Swift for iOS)

    private static func unzipFile(at source: URL, to destination: URL) throws {
        let data = try Data(contentsOf: source)
        guard data.count > 22 else { throw BookParserError.unzipFailed }

        // Parse ZIP central directory from end of file
        var entries: [(filename: String, offset: UInt32, compressedSize: UInt32, uncompressedSize: UInt32, method: UInt16)] = []

        // Find End of Central Directory record (signature: 0x06054b50)
        var eocdOffset = -1
        let bytes = [UInt8](data)
        for i in stride(from: bytes.count - 22, through: max(0, bytes.count - 65557), by: -1) {
            if bytes[i] == 0x50 && bytes[i+1] == 0x4B && bytes[i+2] == 0x05 && bytes[i+3] == 0x06 {
                eocdOffset = i
                break
            }
        }
        guard eocdOffset >= 0 else { throw BookParserError.unzipFailed }

        let cdOffset = readUInt32(bytes, at: eocdOffset + 16)
        let cdEntryCount = readUInt16(bytes, at: eocdOffset + 10)

        // Parse central directory entries
        var pos = Int(cdOffset)
        for _ in 0..<cdEntryCount {
            guard pos + 46 <= bytes.count else { break }
            guard bytes[pos] == 0x50, bytes[pos+1] == 0x4B, bytes[pos+2] == 0x01, bytes[pos+3] == 0x02 else { break }

            let method = readUInt16(bytes, at: pos + 10)
            let compressedSize = readUInt32(bytes, at: pos + 20)
            let uncompressedSize = readUInt32(bytes, at: pos + 24)
            let nameLen = Int(readUInt16(bytes, at: pos + 28))
            let extraLen = Int(readUInt16(bytes, at: pos + 30))
            let commentLen = Int(readUInt16(bytes, at: pos + 32))
            let localHeaderOffset = readUInt32(bytes, at: pos + 42)

            guard pos + 46 + nameLen <= bytes.count else { break }
            let filename = String(bytes: bytes[(pos + 46)..<(pos + 46 + nameLen)], encoding: .utf8) ?? ""

            entries.append((filename: filename, offset: localHeaderOffset, compressedSize: compressedSize, uncompressedSize: uncompressedSize, method: method))
            pos += 46 + nameLen + extraLen + commentLen
        }

        // Extract each entry
        for entry in entries {
            guard !entry.filename.isEmpty, !entry.filename.hasSuffix("/") else { continue }

            let localPos = Int(entry.offset)
            guard localPos + 30 <= bytes.count else { continue }
            let localNameLen = Int(readUInt16(bytes, at: localPos + 26))
            let localExtraLen = Int(readUInt16(bytes, at: localPos + 28))
            let dataStart = localPos + 30 + localNameLen + localExtraLen
            let dataEnd = dataStart + Int(entry.compressedSize)
            guard dataEnd <= bytes.count else { continue }

            let compressedData = Data(bytes[dataStart..<dataEnd])
            let fileData: Data

            if entry.method == 0 {
                // Stored (no compression)
                fileData = compressedData
            } else if entry.method == 8 {
                // Deflate
                guard let inflated = inflateData(compressedData, uncompressedSize: Int(entry.uncompressedSize)) else { continue }
                fileData = inflated
            } else {
                continue
            }

            let filePath = destination.appendingPathComponent(entry.filename)
            let dir = filePath.deletingLastPathComponent()
            try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
            try fileData.write(to: filePath)
        }
    }

    private static func readUInt16(_ bytes: [UInt8], at offset: Int) -> UInt16 {
        UInt16(bytes[offset]) | (UInt16(bytes[offset + 1]) << 8)
    }

    private static func readUInt32(_ bytes: [UInt8], at offset: Int) -> UInt32 {
        UInt32(bytes[offset]) | (UInt32(bytes[offset + 1]) << 8) | (UInt32(bytes[offset + 2]) << 16) | (UInt32(bytes[offset + 3]) << 24)
    }

    private static func inflateData(_ data: Data, uncompressedSize: Int) -> Data? {
        // Use Compression framework for deflate decompression
        // Raw deflate needs the COMPRESSION_ZLIB algorithm with a raw inflate workaround
        let sourceSize = data.count
        var destBuffer = [UInt8](repeating: 0, count: max(uncompressedSize, sourceSize * 4))

        let decompressedSize = data.withUnsafeBytes { srcPtr -> Int in
            guard let baseAddress = srcPtr.baseAddress else { return 0 }
            return compression_decode_buffer(
                &destBuffer, destBuffer.count,
                baseAddress.assumingMemoryBound(to: UInt8.self), sourceSize,
                nil,
                COMPRESSION_ZLIB
            )
        }

        guard decompressedSize > 0 else { return nil }
        return Data(destBuffer[0..<decompressedSize])
    }
}

// MARK: - Types

struct ParsedBook {
    let title: String
    let author: String?
    let chapters: [Chapter]
    let coverImageData: Data?

    var fullText: String {
        chapters.map(\.textContent).joined(separator: "\n\n")
    }

    var wordCount: Int {
        fullText.split(separator: " ").count
    }
}

enum BookParserError: LocalizedError {
    case invalidFormat(String)
    case unzipFailed
    case noContent

    var errorDescription: String? {
        switch self {
        case .invalidFormat(let msg): return "Invalid book format: \(msg)"
        case .unzipFailed: return "Failed to extract EPUB archive. Add ZIPFoundation package for iOS support."
        case .noContent: return "No readable content found in the book."
        }
    }
}
