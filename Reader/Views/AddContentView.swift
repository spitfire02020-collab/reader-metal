import SwiftUI

// MARK: - Add Content View - Redesigned

struct AddContentView: View {
    @ObservedObject var viewModel: LibraryViewModel
    @Binding var showFileImporter: Bool
    @Environment(\.dismiss) private var dismiss

    @State private var selectedTab: AddTab = .url
    @State private var urlText = ""
    @State private var pasteTitle = ""
    @State private var pasteText = ""
    @FocusState private var isURLFocused: Bool
    @FocusState private var isTitleFocused: Bool
    @FocusState private var isTextFocused: Bool

    enum AddTab: String, CaseIterable {
        case url = "URL"
        case file = "File"
        case paste = "Paste"

        var icon: String {
            switch self {
            case .url: return "link"
            case .file: return "doc"
            case .paste: return "doc.on.clipboard"
            }
        }
    }

    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                // Tab selector - Redesigned
                tabSelector

                // Content based on selected tab
                ScrollView {
                    switch selectedTab {
                    case .url:
                        urlContent
                    case .file:
                        fileContent
                    case .paste:
                        pasteContent
                    }
                }
                .frame(maxHeight: .infinity)
            }
            .background(Color.appBackground)
            .noiseTexture(opacity: 0.025)
            .navigationTitle("Add Content")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarLeading) {
                    Button("Cancel") { dismiss() }
                        .foregroundStyle(Color.appTextSecondary)
                }
            }
        }
        .preferredColorScheme(.dark)
    }

    // MARK: - Tab Selector - Redesigned

    private var tabSelector: some View {
        HStack(spacing: 6) {
            ForEach(AddTab.allCases, id: \.self) { tab in
                Button {
                    withAnimation(AppAnimation.smooth) {
                        selectedTab = tab
                    }
                } label: {
                    HStack(spacing: 8) {
                        Image(systemName: tab.icon)
                            .font(.system(size: 14, weight: .medium))
                        Text(tab.rawValue)
                            .font(AppTypography.captionLarge)
                    }
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 12)
                    .background(
                        RoundedRectangle(cornerRadius: 10)
                            .fill(selectedTab == tab ? AnyShapeStyle(AppGradients.accent) : AnyShapeStyle(Color.clear))
                    )
                    .foregroundStyle(selectedTab == tab ? .black : Color.appTextSecondary)
                }
                .buttonStyle(.plain)
            }
        }
        .padding(6)
        .background(
            RoundedRectangle(cornerRadius: 12)
                .fill(Color.appSurfaceElevated)
        )
        .padding(.horizontal, 16)
        .padding(.vertical, 14)
    }

    // MARK: - URL Content - Redesigned

    private var urlContent: some View {
        VStack(spacing: 22) {
            VStack(alignment: .leading, spacing: 10) {
                Text("Web Page URL")
                    .font(AppTypography.captionLarge)
                    .foregroundStyle(Color.appTextSecondary)

                HStack(spacing: 12) {
                    Image(systemName: "globe")
                        .font(.system(size: 16))
                        .foregroundStyle(Color.appTextTertiary)

                    TextField("https://example.com/article", text: $urlText)
                        .textFieldStyle(.plain)
                        .keyboardType(.URL)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()
                        .font(AppTypography.bodyMedium)
                        .foregroundStyle(Color.appTextPrimary)
                        .focused($isURLFocused)

                    if !urlText.isEmpty {
                        Button {
                            urlText = ""
                        } label: {
                            Image(systemName: "xmark.circle.fill")
                                .foregroundStyle(Color.appTextTertiary)
                        }
                    }
                }
                .padding(16)
                .background(
                    RoundedRectangle(cornerRadius: 14)
                        .fill(Color.appSurfaceElevated)
                        .overlay(
                            RoundedRectangle(cornerRadius: 14)
                                .stroke(isURLFocused ? Color.appAccent : Color.appDivider, lineWidth: isURLFocused ? 2 : 1)
                        )
                )
                .shadow(color: isURLFocused ? Color.appAccent.opacity(0.2) : Color.clear, radius: 8)
            }

            // Quick paste from clipboard
            Button {
                if let clipboardString = UIPasteboard.general.string,
                   clipboardString.hasPrefix("http") {
                    urlText = clipboardString
                }
            } label: {
                HStack(spacing: 8) {
                    Image(systemName: "doc.on.clipboard")
                        .font(.system(size: 14))
                    Text("Paste from Clipboard")
                        .font(AppTypography.captionLarge)
                }
                .foregroundStyle(Color.appAccent)
                .frame(maxWidth: .infinity)
                .padding(.vertical, 14)
                .background(
                    RoundedRectangle(cornerRadius: 12)
                        .fill(Color.appAccentSubtle)
                )
            }
            .buttonStyle(.plain)

            // Add button
            Button {
                Task { await viewModel.addFromURL(urlText) }
            } label: {
                HStack {
                    if viewModel.isLoading {
                        ProgressView()
                            .tint(.black)
                            .scaleEffect(0.8)
                    } else {
                        Image(systemName: "plus.circle.fill")
                    }
                    Text(viewModel.isLoading ? "Extracting..." : "Add to Library")
                }
                .font(AppTypography.headlineSmall)
                .foregroundStyle(.black)
                .frame(maxWidth: .infinity)
                .padding(.vertical, 16)
                .background(
                    Capsule()
                        .fill(urlText.isEmpty ? AnyShapeStyle(Color.appSurfaceElevated) : AnyShapeStyle(AppGradients.accent))
                )
                .shadow(color: urlText.isEmpty ? Color.clear : Color.appAccent.opacity(0.3), radius: 10, y: 4)
            }
            .disabled(urlText.isEmpty || viewModel.isLoading)
            .buttonStyle(.plain)

            // Supported sites hint
            VStack(alignment: .leading, spacing: 10) {
                Text("Supported Sources")
                    .font(AppTypography.captionMedium)
                    .foregroundStyle(Color.appTextTertiary)

                HStack(spacing: 8) {
                    supportedSiteTag("Articles")
                    supportedSiteTag("Blog Posts")
                    supportedSiteTag("News")
                    supportedSiteTag("Docs")
                }
            }
            .padding(.top, 12)
        }
        .padding(.horizontal, 18)
        .padding(.top, 12)
    }

    private func supportedSiteTag(_ text: String) -> some View {
        Text(text)
            .font(AppTypography.captionSmall)
            .foregroundStyle(Color.appTextTertiary)
            .padding(.horizontal, 12)
            .padding(.vertical, 6)
            .background(
                Capsule()
                    .fill(Color.appSurfaceElevated)
            )
    }

    // MARK: - File Content - Redesigned

    private var fileContent: some View {
        VStack(spacing: 28) {
            // Drop zone
            Button {
                dismiss()
                DispatchQueue.main.asyncAfter(deadline: .now() + 0.3) {
                    showFileImporter = true
                }
            } label: {
                VStack(spacing: 18) {
                    ZStack {
                        // Glow effect
                        Circle()
                            .fill(Color.appAccent.opacity(0.15))
                            .frame(width: 90, height: 90)
                            .blur(radius: 15)

                        Circle()
                            .fill(Color.appAccentSubtle)
                            .frame(width: 72, height: 72)

                        Image(systemName: "arrow.up.doc.fill")
                            .font(.system(size: 28))
                            .foregroundStyle(Color.appAccent)
                    }

                    VStack(spacing: 6) {
                        Text("Choose a File")
                            .font(AppTypography.headlineMedium)
                            .foregroundStyle(Color.appTextPrimary)

                        Text("EPUB, PDF, or plain text files")
                            .font(AppTypography.bodySmall)
                            .foregroundStyle(Color.appTextSecondary)
                    }
                }
                .frame(maxWidth: .infinity)
                .padding(.vertical, 44)
                .background(
                    RoundedRectangle(cornerRadius: 18)
                        .fill(Color.appSurface)
                        .overlay(
                            RoundedRectangle(cornerRadius: 18)
                                .strokeBorder(
                                    style: StrokeStyle(lineWidth: 2, dash: [10, 8])
                                )
                                .foregroundStyle(Color.appTextTertiary.opacity(0.3))
                        )
                )
            }
            .buttonStyle(.plain)

            // Supported formats
            VStack(alignment: .leading, spacing: 16) {
                Text("Supported Formats")
                    .font(AppTypography.captionLarge)
                    .foregroundStyle(Color.appTextSecondary)

                ForEach([
                    ("book.closed", "EPUB", "eBooks with chapter structure"),
                    ("doc.text", "PDF", "Documents and research papers"),
                    ("text.alignleft", "TXT / MD", "Plain text and Markdown files"),
                ], id: \.1) { icon, format, desc in
                    HStack(spacing: 14) {
                        Image(systemName: icon)
                            .font(.system(size: 18))
                            .foregroundStyle(Color.appAccent)
                            .frame(width: 28)

                        VStack(alignment: .leading, spacing: 3) {
                            Text(format)
                                .font(AppTypography.headlineSmall)
                                .foregroundStyle(Color.appTextPrimary)
                            Text(desc)
                                .font(AppTypography.captionMedium)
                                .foregroundStyle(Color.appTextTertiary)
                        }
                    }
                    .padding(14)
                    .background(
                        RoundedRectangle(cornerRadius: 12)
                            .fill(Color.appSurfaceElevated)
                    )
                }
            }
        }
        .padding(.horizontal, 18)
        .padding(.top, 12)
    }

    // MARK: - Paste Content - Redesigned

    private var pasteContent: some View {
        VStack(spacing: 22) {
            VStack(alignment: .leading, spacing: 10) {
                Text("Title")
                    .font(AppTypography.captionLarge)
                    .foregroundStyle(Color.appTextSecondary)

                TextField("Give your content a title", text: $pasteTitle)
                    .textFieldStyle(.plain)
                    .font(AppTypography.bodyMedium)
                    .foregroundStyle(Color.appTextPrimary)
                    .focused($isTitleFocused)
                    .padding(16)
                    .background(
                        RoundedRectangle(cornerRadius: 14)
                            .fill(Color.appSurfaceElevated)
                            .overlay(
                                RoundedRectangle(cornerRadius: 14)
                                    .stroke(isTitleFocused ? Color.appAccent : Color.appDivider, lineWidth: isTitleFocused ? 2 : 1)
                            )
                    )
            }

            VStack(alignment: .leading, spacing: 10) {
                HStack {
                    Text("Text Content")
                        .font(AppTypography.captionLarge)
                        .foregroundStyle(Color.appTextSecondary)

                    Spacer()

                    if !pasteText.isEmpty {
                        Text("\(pasteText.split(separator: " ").count) words")
                            .font(AppTypography.captionMedium)
                            .foregroundStyle(Color.appTextTertiary)
                    }
                }

                TextEditor(text: $pasteText)
                    .scrollContentBackground(.hidden)
                    .font(AppTypography.bodyMedium)
                    .foregroundStyle(Color.appTextPrimary)
                    .focused($isTextFocused)
                    .frame(minHeight: 180)
                    .padding(14)
                    .background(
                        RoundedRectangle(cornerRadius: 14)
                            .fill(Color.appSurfaceElevated)
                            .overlay(
                                RoundedRectangle(cornerRadius: 14)
                                    .stroke(isTextFocused ? Color.appAccent : Color.appDivider, lineWidth: isTextFocused ? 2 : 1)
                            )
                    )
            }

            // Paste from clipboard
            Button {
                if let text = UIPasteboard.general.string {
                    pasteText = text
                    if pasteTitle.isEmpty {
                        pasteTitle = String(text.prefix(50))
                    }
                }
            } label: {
                HStack(spacing: 8) {
                    Image(systemName: "doc.on.clipboard")
                        .font(.system(size: 14))
                    Text("Paste from Clipboard")
                        .font(AppTypography.captionLarge)
                }
                .foregroundStyle(Color.appAccent)
                .frame(maxWidth: .infinity)
                .padding(.vertical, 14)
                .background(
                    RoundedRectangle(cornerRadius: 12)
                        .fill(Color.appAccentSubtle)
                )
            }
            .buttonStyle(.plain)

            // Add button
            Button {
                viewModel.addFromText(pasteText, title: pasteTitle.isEmpty ? "Untitled" : pasteTitle)
                dismiss()
            } label: {
                HStack {
                    Image(systemName: "plus.circle.fill")
                    Text("Add to Library")
                }
                .font(AppTypography.headlineSmall)
                .foregroundStyle(.black)
                .frame(maxWidth: .infinity)
                .padding(.vertical, 16)
                .background(
                    Capsule()
                        .fill(pasteText.isEmpty ? AnyShapeStyle(Color.appSurfaceElevated) : AnyShapeStyle(AppGradients.accent))
                )
                .shadow(color: pasteText.isEmpty ? Color.clear : Color.appAccent.opacity(0.3), radius: 10, y: 4)
            }
            .disabled(pasteText.isEmpty)
            .buttonStyle(.plain)
        }
        .padding(.horizontal, 18)
        .padding(.top, 12)
    }
}
