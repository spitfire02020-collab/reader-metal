import Foundation
import UIKit

// MARK: - Haptic Feedback Helpers

/// Play impact haptic feedback
/// - Parameter style: The feedback style (.light, .medium, .heavy, .rigid, .soft)
func playHaptic(_ style: UIImpactFeedbackGenerator.FeedbackStyle = .medium) {
    let generator = UIImpactFeedbackGenerator(style: style)
    generator.prepare()
    generator.impactOccurred()
}

/// Play selection haptic feedback (for UI selection changes)
func playSelectionHaptic() {
    let generator = UISelectionFeedbackGenerator()
    generator.prepare()
    generator.selectionChanged()
}

/// Play notification haptic feedback
/// - Parameter type: The notification type (.success, .warning, .error)
func playNotificationHaptic(_ type: UINotificationFeedbackGenerator.FeedbackType) {
    let generator = UINotificationFeedbackGenerator()
    generator.prepare()
    generator.notificationOccurred(type)
}
