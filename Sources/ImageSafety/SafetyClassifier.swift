import Foundation
import CryptoKit

/// Offline sensitive-content screening. Call from a background worker.
/// No uploads, training, telemetry, input cache or automatic policy action.
public final class SafetyClassifier: @unchecked Sendable {
    public struct Decision: Codable, Sendable {
        public let sensitive: Bool
        public let modelVersion: String
    }
    public enum Failure: Error { case invalidImage, invalidBundle }
    private let engine: ImageSafetyCommentPhotoContentGuard
    public init(modelsBundleURL: URL) throws {
        guard !FileManager.default.fileExists(atPath: modelsBundleURL.appendingPathComponent("INCOMPLETE").path),
              let bundle = Bundle(url: modelsBundleURL) else { throw Failure.invalidBundle }
        engine = ImageSafetyCommentPhotoContentGuard(assets: try ImageSafetyCommentPhotoSafetyAssets.load(bundle: bundle))
    }
    /// Matches the frozen image preparation: oriented JPEG, 1600px maximum,
    /// 0.82 JPEG quality, 600 KiB maximum, without capture metadata.
    public func inspect(imageData: Data, cancelled: () -> Bool = { false }) throws -> Decision {
        guard let jpeg = PhotoEncoder.encodeComment(imageData) else { throw Failure.invalidImage }
        return try inspectPreparedJPEG(jpeg, cancelled: cancelled)
    }
    /// JPEG must satisfy the same preparation contract; SHA is computed locally.
    public func inspectPreparedJPEG(_ data: Data, cancelled: () -> Bool = { false }) throws -> Decision {
        let sha = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
        let result = try engine.inspectPreparedJPEG(data, expectedSHA256: sha, cancelled: cancelled)
        return Decision(sensitive: result.containsSensitiveContent, modelVersion: result.modelVersion)
    }
    public func releaseWhenIdle() { engine.releaseWhenIdle() }
    public func wakeWaitingOperations() { engine.wakeWaitingOperations() }
}
