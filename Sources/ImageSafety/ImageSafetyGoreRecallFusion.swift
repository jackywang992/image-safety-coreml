import Foundation

/// A context correction may add evidence, while strong local evidence still
/// contributes to the decision. Inputs and output are signed margins.
struct ImageSafetyGoreRecallFusion {
  enum Failure: Error { case invalidMargin }
  static func score(local: Float, context: Float) throws -> Float {
    guard local.isFinite, context.isFinite,
          (-1...1).contains(local), (-1...1).contains(context) else {
      throw Failure.invalidMargin
    }
    _ = try ImageSafetyPhotoInspectionContract.Score(Double(local),domain:.signedUnitMargin).checked(as:.signedUnitMargin)
    _ = try ImageSafetyPhotoInspectionContract.Score(Double(context),domain:.signedUnitMargin).checked(as:.signedUnitMargin)
    return max(context, (local + context) / 2)
  }
}
