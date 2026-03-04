import SwiftUI

// MARK: - Root Content View

struct ContentView: View {
    @EnvironmentObject var downloadService: ModelDownloadService
    @State private var showOnboarding = !UserDefaults.standard.bool(forKey: "has_completed_onboarding")

    var body: some View {
        ZStack {
            LibraryView()

            if showOnboarding {
                OnboardingView {
                    UserDefaults.standard.set(true, forKey: "has_completed_onboarding")
                    withAnimation(.easeInOut(duration: 0.4)) {
                        showOnboarding = false
                    }
                }
                .transition(.opacity)
            }
        }
        .task {
            // Copy models in background without blocking UI
            await ModelDownloadService.shared.copyBundleModelsIfNeeded()
        }
    }
}

// MARK: - Onboarding View

struct OnboardingView: View {
    let onComplete: () -> Void
    @State private var currentPage = 0

    private let pages: [(icon: String, title: String, subtitle: String)] = [
        ("headphones", "Listen to Everything", "Turn any article, book, or text into natural-sounding audio with AI"),
        ("waveform", "Powered by Chatterbox Turbo", "350M parameter TTS model runs entirely on your device — private and offline"),
        ("person.wave.2", "Clone Any Voice", "Record a short sample to create a personalized voice for your content")
    ]

    var body: some View {
        ZStack {
            Color.appBackground.ignoresSafeArea()

            VStack(spacing: 40) {
                Spacer()

                TabView(selection: $currentPage) {
                    ForEach(0..<pages.count, id: \.self) { index in
                        VStack(spacing: 24) {
                            Image(systemName: pages[index].icon)
                                .font(.system(size: 80))
                                .foregroundStyle(Color.appAccent)
                                .symbolRenderingMode(.hierarchical)

                            VStack(spacing: 8) {
                                Text(pages[index].title)
                                    .font(.title.bold())
                                    .foregroundColor(.appTextPrimary)

                                Text(pages[index].subtitle)
                                    .font(.body)
                                    .foregroundColor(.appTextSecondary)
                                    .multilineTextAlignment(.center)
                                    .padding(.horizontal, 40)
                            }
                        }
                        .tag(index)
                    }
                }
                .tabViewStyle(.page(indexDisplayMode: .never))
                .frame(height: 400)

                // Page indicator
                HStack(spacing: 8) {
                    ForEach(0..<pages.count, id: \.self) { index in
                        Circle()
                            .fill(currentPage == index ? Color.appAccent : Color.appTextTertiary)
                            .frame(width: 8, height: 8)
                            .animation(.easeInOut(duration: 0.2), value: currentPage)
                    }
                }

                Spacer()

                // Buttons
                VStack(spacing: 16) {
                    if currentPage < pages.count - 1 {
                        Button {
                            withAnimation {
                                currentPage += 1
                            }
                        } label: {
                            Text("Continue")
                                .font(.headline)
                                .foregroundColor(.white)
                                .frame(maxWidth: .infinity)
                                .padding(.vertical, 16)
                                .background(Color.appAccent)
                                .cornerRadius(12)
                        }
                        .padding(.horizontal, 24)
                    } else {
                        Button {
                            onComplete()
                        } label: {
                            Text("Get Started")
                                .font(.headline)
                                .foregroundColor(.white)
                                .frame(maxWidth: .infinity)
                                .padding(.vertical, 16)
                                .background(Color.appAccent)
                                .cornerRadius(12)
                        }
                        .padding(.horizontal, 24)
                    }

                    if currentPage > 0 {
                        Button {
                            withAnimation {
                                currentPage -= 1
                            }
                        } label: {
                            Text("Back")
                                .font(.subheadline)
                                .foregroundColor(.appTextSecondary)
                        }
                    }
                }
                .padding(.bottom, 50)
            }
        }
    }
}
