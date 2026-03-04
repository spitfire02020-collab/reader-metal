import SwiftUI

// MARK: - Reader App Entry Point

@main
struct ReaderApp: App {
    @StateObject private var downloadService = ModelDownloadService.shared

    init() {
        configureAppearance()
    }

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(downloadService)
                .onOpenURL { url in
                    handleDeepLink(url)
                }
        }
    }

    private func handleDeepLink(_ url: URL) {
        guard url.scheme == "reader" else { return }
        switch url.host {
        case "synthesize":
            Task { @MainActor in
                let defaults = UserDefaults.standard
                guard let data = defaults.data(forKey: "library_items"),
                      var items = try? JSONDecoder().decode([LibraryItem].self, from: data),
                      let firstPending = items.first(where: { $0.status == .pending }) else {
                    NSLog("[Reader] synthesize: no pending items found")
                    return
                }
                NSLog("[Reader] synthesize: starting for item '\(firstPending.title)'")
                let vm = PlayerViewModel(item: firstPending)
                await vm.startSynthesis()
                NSLog("[Reader] synthesize: done, status=\(vm.item.status)")
                if let idx = items.firstIndex(where: { $0.id == vm.item.id }) {
                    items[idx] = vm.item
                    if let encoded = try? JSONEncoder().encode(items) {
                        defaults.set(encoded, forKey: "library_items")
                    }
                }
            }
        default:
            NSLog("[Reader] unknown deep link host: \(url.host ?? "nil")")
        }
    }

    private func configureAppearance() {
        let navAppearance = UINavigationBarAppearance()
        navAppearance.configureWithOpaqueBackground()
        navAppearance.backgroundColor = UIColor(Color.appBackground)
        navAppearance.titleTextAttributes = [
            .foregroundColor: UIColor(Color.appTextPrimary),
            .font: UIFont.systemFont(ofSize: 17, weight: .semibold)
        ]
        navAppearance.largeTitleTextAttributes = [
            .foregroundColor: UIColor(Color.appTextPrimary),
            .font: UIFont.systemFont(ofSize: 34, weight: .bold)
        ]
        UINavigationBar.appearance().standardAppearance = navAppearance
        UINavigationBar.appearance().scrollEdgeAppearance = navAppearance
        UINavigationBar.appearance().compactAppearance = navAppearance
        UINavigationBar.appearance().tintColor = UIColor(Color.appAccent)

        let tabAppearance = UITabBarAppearance()
        tabAppearance.configureWithOpaqueBackground()
        tabAppearance.backgroundColor = UIColor(Color.appSurface)
        UITabBar.appearance().standardAppearance = tabAppearance
        UITabBar.appearance().scrollEdgeAppearance = tabAppearance

        UITextField.appearance(whenContainedInInstancesOf: [UISearchBar.self]).backgroundColor = UIColor(Color.appSurfaceElevated)
        UITableView.appearance().backgroundColor = UIColor(Color.appBackground)
        UITableViewCell.appearance().backgroundColor = UIColor(Color.appSurface)
    }
}
