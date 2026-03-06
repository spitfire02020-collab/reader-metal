import SwiftUI

// MARK: - Skeleton Loading Components

/// A skeleton row that mimics the LibraryItemRow shape
struct SkeletonRow: View {
    var body: some View {
        HStack(spacing: 14) {
            // Cover placeholder
            RoundedRectangle(cornerRadius: 10)
                .fill(Color.appSurfaceElevated)
                .frame(width: 56, height: 56)
                .shimmer()

            // Text placeholders
            VStack(alignment: .leading, spacing: 8) {
                RoundedRectangle(cornerRadius: 4)
                    .fill(Color.appSurfaceElevated)
                    .frame(height: 16)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .shimmer()

                RoundedRectangle(cornerRadius: 4)
                    .fill(Color.appSurfaceElevated)
                    .frame(width: 120, height: 12)
                    .shimmer()
            }

            Spacer()

            // Status indicator placeholder
            Circle()
                .fill(Color.appSurfaceElevated)
                .frame(width: 24, height: 24)
                .shimmer()
        }
        .padding(12)
        .background(
            RoundedRectangle(cornerRadius: 14)
                .fill(Color.appSurface)
        )
    }
}

/// Skeleton for the mini player
struct SkeletonMiniPlayer: View {
    var body: some View {
        HStack(spacing: 14) {
            RoundedRectangle(cornerRadius: 10)
                .fill(Color.appSurfaceElevated)
                .frame(width: 48, height: 48)
                .shimmer()

            VStack(alignment: .leading, spacing: 4) {
                RoundedRectangle(cornerRadius: 4)
                    .fill(Color.appSurfaceElevated)
                    .frame(height: 14)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .shimmer()

                RoundedRectangle(cornerRadius: 4)
                    .fill(Color.appSurfaceElevated)
                    .frame(width: 100, height: 10)
                    .shimmer()
            }

            Spacer()

            Circle()
                .fill(Color.appSurfaceElevated)
                .frame(width: 40, height: 40)
                .shimmer()
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 12)
        .background(
            RoundedRectangle(cornerRadius: 18)
                .fill(Color.appSurface)
        )
    }
}

/// Skeleton for filter tabs
struct SkeletonFilterTabs: View {
    var body: some View {
        HStack(spacing: 12) {
            ForEach(0..<4, id: \.self) { index in
                RoundedRectangle(cornerRadius: 8)
                    .fill(Color.appSurfaceElevated)
                    .frame(width: CGFloat.random(in: 50...80), height: 32)
                    .shimmer()
            }
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 12)
    }
}
