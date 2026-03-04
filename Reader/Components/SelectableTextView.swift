import SwiftUI
import UIKit

// MARK: - Selectable Text View
// Enhanced version with chunk highlighting and beautiful floating button

struct SelectableTextView: View {
    let text: String
    let chunks: [String]
    @Binding var selectedText: String
    let onPlaySelection: () -> Void
    let isPlaying: Bool
    let currentChunkIndex: Int

    var body: some View {
        ZStack(alignment: .bottom) {
            // Text view with selection and highlighting
            UITextViewWrapper(
                text: text,
                chunks: chunks,
                selectedText: $selectedText,
                currentChunkIndex: currentChunkIndex
            )

            // Floating play button when text is selected
            if !selectedText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                floatingPlayButton
            }
        }
    }

    private var floatingPlayButton: some View {
        VStack(spacing: 0) {
            // Selected text preview
            Text(selectedText.prefix(50) + (selectedText.count > 50 ? "..." : ""))
                .font(.system(size: 12, weight: .medium))
                .foregroundStyle(Color.appTextSecondary)
                .lineLimit(1)
                .padding(.horizontal, 20)
                .padding(.top, 12)
                .padding(.bottom, 8)

            // Play button
            Button {
                onPlaySelection()
            } label: {
                HStack(spacing: 10) {
                    if isPlaying {
                        // Animated loading
                        HStack(spacing: 3) {
                            ForEach(0..<3, id: \.self) { index in
                                Circle()
                                    .fill(Color.white)
                                    .frame(width: 4, height: 4)
                                    .animation(
                                        .easeInOut(duration: 0.5)
                                        .repeatForever()
                                        .delay(Double(index) * 0.15),
                                        value: isPlaying
                                    )
                            }
                        }
                    } else {
                        Image(systemName: "play.fill")
                            .font(.system(size: 14, weight: .bold))
                    }
                    Text(isPlaying ? "Playing..." : "Play from Selection")
                        .font(.system(size: 14, weight: .semibold))
                }
                .foregroundStyle(.white)
                .padding(.horizontal, 24)
                .padding(.vertical, 14)
                .background(
                    Capsule()
                        .fill(AppGradients.accent)
                        .shadow(color: Color.appAccent.opacity(0.5), radius: 15, y: 5)
                )
            }
            .padding(.bottom, 32)
        }
        .transition(.move(edge: .bottom).combined(with: .opacity))
        .animation(.spring(response: 0.3, dampingFraction: 0.8), value: selectedText.isEmpty)
    }
}

// MARK: - UIKit Wrapper with Highlighting

struct UITextViewWrapper: UIViewRepresentable {
    let text: String
    let chunks: [String]
    @Binding var selectedText: String
    let currentChunkIndex: Int

    func makeUIView(context: Context) -> HighlightingTextView {
        let textView = HighlightingTextView()
        textView.isEditable = false
        textView.isSelectable = true
        textView.backgroundColor = .clear
        textView.font = UIFont.systemFont(ofSize: 16)
        textView.textColor = UIColor(Color.appTextPrimary)
        textView.textContainerInset = UIEdgeInsets(top: 20, left: 16, bottom: 20, right: 16)
        textView.delegate = context.coordinator
        textView.layoutManager.allowsNonContiguousLayout = false
        textView.chunks = chunks
        textView.currentHighlightedChunk = currentChunkIndex
        return textView
    }

    func updateUIView(_ textView: HighlightingTextView, context: Context) {
        if textView.text != text {
            textView.text = text
            textView.chunks = chunks
            textView.updateHighlighting()
        }

        // Update highlighting when chunk changes
        if textView.currentHighlightedChunk != currentChunkIndex {
            textView.currentHighlightedChunk = currentChunkIndex
            textView.updateHighlighting()
        }
    }

    func makeCoordinator() -> Coordinator {
        Coordinator(self)
    }

    class Coordinator: NSObject, UITextViewDelegate {
        var parent: UITextViewWrapper

        init(_ parent: UITextViewWrapper) {
            self.parent = parent
        }

        func textViewDidChangeSelection(_ textView: UITextView) {
            DispatchQueue.main.async {
                if let selectedRange = textView.selectedTextRange {
                    let selected = textView.text(in: selectedRange) ?? ""
                    self.parent.selectedText = selected
                } else {
                    self.parent.selectedText = ""
                }
            }
        }
    }
}

// MARK: - Highlighting Text View

class HighlightingTextView: UITextView {
    var chunks: [String] = []
    var currentHighlightedChunk: Int = -1

    // More visible highlight color
    private let highlightColor = UIColor(Color.appAccent.opacity(0.4))
    private let highlightTextColor = UIColor(Color.appAccent)

    func updateHighlighting() {
        let textStorage = self.textStorage

        // Reset all text to default style
        let fullRange = NSRange(location: 0, length: textStorage.length)
        textStorage.addAttribute(.foregroundColor, value: UIColor(Color.appTextPrimary), range: fullRange)
        textStorage.addAttribute(.backgroundColor, value: UIColor.clear, range: fullRange)

        // Highlight current chunk
        guard currentHighlightedChunk >= 0,
              currentHighlightedChunk < chunks.count,
              let textContent = self.text else { return }

        // Find the chunk in the text
        let chunkText = chunks[currentHighlightedChunk]

        guard let range = textContent.range(of: chunkText, options: .caseInsensitive) else { return }

        let nsRange = NSRange(range, in: textContent)
        if nsRange.location != NSNotFound {
            // Add background highlight
            textStorage.addAttribute(.backgroundColor, value: highlightColor, range: nsRange)
            // Slightly brighten the text
            textStorage.addAttribute(.foregroundColor, value: highlightTextColor, range: nsRange)
        }
    }
}
