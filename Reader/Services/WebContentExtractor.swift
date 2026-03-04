import Foundation
import WebKit

// MARK: - Web Content Extractor
// Extracts readable text content from web pages using WKWebView + JavaScript

@MainActor
final class WebContentExtractor: ObservableObject {
    @Published var isExtracting = false
    @Published var extractionError: String?

    /// Extract readable content from a URL - server-side first, then WKWebView fallback
    func extract(from url: URL) async throws -> ExtractedContent {
        isExtracting = true
        defer { isExtracting = false }

        // Try server-side HTML parsing first (works for RoyalRoad)
        do {
            let content = try await extractServerSide(from: url)
            if !content.text.isEmpty && content.text.count > 500 {
                return content
            }
        } catch {
            print("Server-side extraction failed: \(error)")
        }

        // Fallback: try WKWebView for JavaScript-rendered content
        do {
            let content = try await extractWithWebView(from: url)
            if !content.text.isEmpty && content.text.count > 500 {
                return content
            }
        } catch {
            print("WebView extraction failed: \(error)")
        }

        throw ExtractionError.noContentFound
    }

    /// Extract using server-side HTML parsing
    private func extractServerSide(from url: URL) async throws -> ExtractedContent {
        let (data, response) = try await URLSession.shared.data(from: url)

        guard let httpResponse = response as? HTTPURLResponse,
              (200...299).contains(httpResponse.statusCode) else {
            throw ExtractionError.fetchFailed
        }

        let encoding = httpResponse.textEncodingName.flatMap { String.Encoding(cfEncoding: $0) } ?? .utf8
        guard let html = String(data: data, encoding: encoding) ?? String(data: data, encoding: .utf8) else {
            throw ExtractionError.decodingFailed
        }

        return extractReadableContent(from: html, sourceURL: url)
    }

    /// Extract content using WKWebView - handles JavaScript-rendered pages
    private func extractWithWebView(from url: URL) async throws -> ExtractedContent {
        // Create a WebView configuration with disabled caching
        let config = WKWebViewConfiguration()
        config.websiteDataStore = .nonPersistent()

        let webView = WKWebView(frame: CGRect(x: 0, y: 0, width: 375, height: 812), configuration: config)

        // Load the page
        webView.load(URLRequest(url: url))

        // Wait for page to load (max 20 seconds)
        try await Task.sleep(nanoseconds: 5_000_000_000) // 5 seconds initial load

        // Try multiple times with increasing waits for lazy content
        for attempt in 0..<3 {
            let waitTime = UInt64((attempt + 1) * 2_000_000_000)
            try await Task.sleep(nanoseconds: waitTime)

            // Extract using JavaScript - focus on RoyalRoad selectors
            let jsScript = """
            (function() {
                // Try RoyalRoad chapter content first
                var content = null;

                // RoyalRoad specific - chapter content is usually in these selectors
                var rrSelectors = [
                    'div.chapter-inner',
                    'div.chapter-content',
                    'div[class*="chapter-inner"]',
                    'section.chapter-content',
                    'div[class*="portlet-body"]'
                ];

                for (var i = 0; i < rrSelectors.length; i++) {
                    var el = document.querySelector(rrSelectors[i]);
                    if (el) {
                        var text = el.textContent.trim();
                        // Check if this looks like chapter content (substantial length)
                        if (text.length > 500) {
                            content = el;
                            break;
                        }
                    }
                }

                if (!content) {
                    // Fallback: find largest text block in <p> tags
                    var paragraphs = document.querySelectorAll('p');
                    var allText = [];
                    for (var i = 0; i < paragraphs.length; i++) {
                        var p = paragraphs[i].textContent.trim();
                        if (p.length > 20) {
                            allText.push(p);
                        }
                    }
                    return allText.join('\\n\\n');
                }

                // Clean and return the content
                return content.textContent.trim();
            })()
            """

            if let text = try await webView.evaluateJavaScript(jsScript) as? String,
               text.count > 500 {
                // Got valid content, extract other metadata
                let title = (try? await webView.evaluateJavaScript("document.title")) as? String ?? url.host ?? "Untitled"

                // Try to get author
                let authorScript = """
                (function() {
                    var el = document.querySelector('a[href*="/author/"], span.author-name, span[itemprop="author"]');
                    return el ? el.textContent.trim() : null;
                })()
                """
                let author = try? await webView.evaluateJavaScript(authorScript) as? String

                let cleanedText = cleanExtractedText(text)

                if cleanedText.count > 500 {
                    return ExtractedContent(
                        title: title.trimmingCharacters(in: .whitespacesAndNewlines),
                        author: author?.trimmingCharacters(in: .whitespacesAndNewlines),
                        description: nil,
                        text: cleanedText,
                        sourceURL: url.absoluteString,
                        imageURL: nil
                    )
                }
            }
        }

        throw ExtractionError.noContentFound
    }

    /// Clean up extracted text
    private func cleanExtractedText(_ text: String) -> String {
        var cleaned = text

        // Remove excessive whitespace
        cleaned = cleaned.replacingOccurrences(
            of: "[ \\t]+",
            with: " ",
            options: .regularExpression
        )

        // Normalize newlines
        cleaned = cleaned.replacingOccurrences(
            of: "\\n{3,}",
            with: "\n\n",
            options: .regularExpression
        )

        // Remove common noise patterns
        let lines = cleaned.components(separatedBy: "\n")
            .map { $0.trimmingCharacters(in: .whitespaces) }
            .filter { line in
                !line.isEmpty &&
                line.count > 15 &&
                !line.lowercased().contains("advertisement") &&
                !line.lowercased().contains("cookie") &&
                !line.lowercased().contains("sign up") &&
                !line.lowercased().contains("subscribe") &&
                !line.lowercased().contains("patreon") &&
                !line.lowercased().contains("chapter") ||
                (line.lowercased().contains("chapter") && line.count > 50) // Keep chapter titles
            }

        return lines.joined(separator: "\n\n")
    }

    // MARK: - Content Extraction (Readability-style)

    private func extractReadableContent(from html: String, sourceURL: URL) -> ExtractedContent {
        // Extract title – raw HTML may contain entities like &#x27; or &amp;
        let rawTitle = extractBetweenTags(html, tag: "title")
            ?? extractMetaContent(html, property: "og:title")
            ?? sourceURL.host ?? "Untitled"
        let title = decodeHTMLEntities(rawTitle)

        // Extract author
        let rawAuthor = extractMetaContent(html, name: "author")
            ?? extractMetaContent(html, property: "article:author")
        let author = rawAuthor.map { decodeHTMLEntities($0) }

        // Extract description
        let description = extractMetaContent(html, name: "description")
            ?? extractMetaContent(html, property: "og:description")

        // Extract og:image for cover
        let imageURL = extractMetaContent(html, property: "og:image")

        // Extract main text content
        let text = extractMainText(from: html)

        return ExtractedContent(
            title: title.trimmingCharacters(in: .whitespacesAndNewlines),
            author: author?.trimmingCharacters(in: .whitespacesAndNewlines),
            description: description,
            text: text,
            sourceURL: sourceURL.absoluteString,
            imageURL: imageURL
        )
    }

    /// Extract main readable text, stripping navigation, ads, scripts, styles
    private func extractMainText(from html: String) -> String {
        var text = html

        // Remove script and style blocks
        text = removeBlocks(text, tag: "script")
        text = removeBlocks(text, tag: "style")
        text = removeBlocks(text, tag: "nav")
        text = removeBlocks(text, tag: "header")
        text = removeBlocks(text, tag: "footer")
        text = removeBlocks(text, tag: "aside")
        text = removeBlocks(text, tag: "noscript")
        text = removeBlocks(text, tag: "iframe")
        text = removeBlocks(text, tag: "form")

        // RoyalRoad specific - chapter content is in div.chapter-content or div.chapter-inner
        var articleContent = extractDivByClass(text, classNames: [
            "chapter-content", "chapter-inner", "chapter-inner chapter-content"
        ])

        // Fallback to standard selectors
        if articleContent == nil || articleContent?.count ?? 0 < 100 {
            articleContent = extractBetweenTags(text, tag: "article")
                ?? extractBetweenTags(text, tag: "main")
                ?? extractDivByClass(text, classNames: [
                    "article-body", "article-content", "post-content",
                    "entry-content", "story-body", "content-body"
                ])
                ?? text
        }

        // Strip remaining HTML tags
        var cleaned = articleContent?.replacingOccurrences(
            of: "<[^>]+>",
            with: "\n",
            options: .regularExpression
        ) ?? ""

        // Decode HTML entities
        cleaned = decodeHTMLEntities(cleaned)

        // Clean up whitespace
        cleaned = cleaned.replacingOccurrences(
            of: "\\n{3,}",
            with: "\n\n",
            options: .regularExpression
        )
        cleaned = cleaned.replacingOccurrences(
            of: "[ \\t]+",
            with: " ",
            options: .regularExpression
        )

        // Remove common non-content patterns
        let lines = cleaned.components(separatedBy: "\n")
            .map { $0.trimmingCharacters(in: .whitespaces) }
            .filter { line in
                !line.isEmpty &&
                line.count > 20 && // Filter out short fragments (nav items, etc.)
                !line.lowercased().hasPrefix("advertisement") &&
                !line.lowercased().hasPrefix("cookie") &&
                !line.lowercased().contains("sign up for") &&
                !line.lowercased().contains("subscribe to")
            }

        return lines.joined(separator: "\n\n")
    }

    // MARK: - HTML Parsing Helpers

    private func extractBetweenTags(_ html: String, tag: String) -> String? {
        let pattern = "<\(tag)[^>]*>(.*?)</\(tag)>"
        guard let regex = try? NSRegularExpression(pattern: pattern, options: [.dotMatchesLineSeparators, .caseInsensitive]) else {
            return nil
        }
        let nsHTML = html as NSString
        guard let match = regex.firstMatch(in: html, range: NSRange(location: 0, length: nsHTML.length)) else {
            return nil
        }
        return nsHTML.substring(with: match.range(at: 1))
    }

    private func extractMetaContent(_ html: String, name: String? = nil, property: String? = nil) -> String? {
        let attr: String
        let value: String
        if let name {
            attr = "name"
            value = name
        } else if let property {
            attr = "property"
            value = property
        } else {
            return nil
        }

        let pattern = "<meta[^>]*\(attr)=[\"']\(value)[\"'][^>]*content=[\"']([^\"']*)[\"'][^>]*/?>|<meta[^>]*content=[\"']([^\"']*)[\"'][^>]*\(attr)=[\"']\(value)[\"'][^>]*/?>"
        guard let regex = try? NSRegularExpression(pattern: pattern, options: .caseInsensitive) else {
            return nil
        }
        let nsHTML = html as NSString
        guard let match = regex.firstMatch(in: html, range: NSRange(location: 0, length: nsHTML.length)) else {
            return nil
        }
        let group1 = match.range(at: 1)
        let group2 = match.range(at: 2)
        if group1.location != NSNotFound {
            return nsHTML.substring(with: group1)
        }
        if group2.location != NSNotFound {
            return nsHTML.substring(with: group2)
        }
        return nil
    }

    private func extractDivByClass(_ html: String, classNames: [String]) -> String? {
        for className in classNames {
            let pattern = "<div[^>]*class=[\"'][^\"']*\(className)[^\"']*[\"'][^>]*>(.*?)</div>"
            if let regex = try? NSRegularExpression(pattern: pattern, options: [.dotMatchesLineSeparators, .caseInsensitive]) {
                let nsHTML = html as NSString
                if let match = regex.firstMatch(in: html, range: NSRange(location: 0, length: nsHTML.length)) {
                    return nsHTML.substring(with: match.range(at: 1))
                }
            }
        }
        return nil
    }

    private func removeBlocks(_ html: String, tag: String) -> String {
        html.replacingOccurrences(
            of: "<\(tag)[^>]*>.*?</\(tag)>",
            with: "",
            options: [.regularExpression, .caseInsensitive]
        )
    }

    private func decodeHTMLEntities(_ text: String) -> String {
        var result = text
        let entities: [(String, String)] = [
            ("&amp;", "&"), ("&lt;", "<"), ("&gt;", ">"),
            ("&quot;", "\""), ("&#39;", "'"), ("&apos;", "'"),
            ("&nbsp;", " "), ("&mdash;", "—"), ("&ndash;", "–"),
            ("&lsquo;", "\u{2018}"), ("&rsquo;", "\u{2019}"),
            ("&ldquo;", "\u{201C}"), ("&rdquo;", "\u{201D}"),
            ("&hellip;", "..."), ("&copy;", "(c)"),
        ]
        for (entity, replacement) in entities {
            result = result.replacingOccurrences(of: entity, with: replacement)
        }
        // Decode decimal numeric entities  e.g. &#39; → '
        if let regex = try? NSRegularExpression(pattern: "&#(\\d+);") {
            let nsResult = result as NSString
            let matches = regex.matches(in: result, range: NSRange(location: 0, length: nsResult.length))
            for match in matches.reversed() {
                let numStr = nsResult.substring(with: match.range(at: 1))
                if let num = Int(numStr), let scalar = UnicodeScalar(num) {
                    result = (result as NSString).replacingCharacters(in: match.range, with: String(Character(scalar)))
                }
            }
        }
        // Decode hex numeric entities  e.g. &#x27; → '  &#x2019; → '
        if let regex = try? NSRegularExpression(pattern: "&#x([0-9a-fA-F]+);", options: .caseInsensitive) {
            let nsResult = result as NSString
            let matches = regex.matches(in: result, range: NSRange(location: 0, length: nsResult.length))
            for match in matches.reversed() {
                let hexStr = nsResult.substring(with: match.range(at: 1))
                if let num = UInt32(hexStr, radix: 16), let scalar = UnicodeScalar(num) {
                    result = (result as NSString).replacingCharacters(in: match.range, with: String(Character(scalar)))
                }
            }
        }
        return result
    }
}

// MARK: - Types

struct ExtractedContent {
    let title: String
    let author: String?
    let description: String?
    let text: String
    let sourceURL: String
    let imageURL: String?

    var wordCount: Int { text.split(separator: " ").count }
    var estimatedReadTime: String {
        let minutes = max(1, wordCount / 200)
        return "\(minutes) min read"
    }
}

enum ExtractionError: LocalizedError {
    case fetchFailed
    case decodingFailed
    case noContentFound
    case invalidURL

    var errorDescription: String? {
        switch self {
        case .fetchFailed: return "Failed to fetch the web page."
        case .decodingFailed: return "Failed to decode the page content."
        case .noContentFound: return "No readable content found on the page."
        case .invalidURL: return "The URL is invalid."
        }
    }
}

// MARK: - String.Encoding Helper

extension String.Encoding {
    init?(cfEncoding name: String) {
        let encoding = CFStringConvertIANACharSetNameToEncoding(name as CFString)
        guard encoding != kCFStringEncodingInvalidId else { return nil }
        self.init(rawValue: CFStringConvertEncodingToNSStringEncoding(encoding))
    }
}
