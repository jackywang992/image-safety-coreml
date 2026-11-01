import Foundation
import CoreGraphics

/// Names and geometry are part of the frozen preprocessing contract. They do
/// not imply that different encoders share a crop, scale, or view order.
enum ImageSafetyPhotoInspectionContract {
  static let version = "image-safety-pixels-v1"
  enum Failure: Error { case invalidCrop, invalidScore, wrongScoreDomain }
  enum View: String, Codable {
    case whole, center70, centerSquare, cornerTopLeft, cornerTopRight, cornerBottomLeft, cornerBottomRight
    var normalizedBox: [Double]? {
      switch self {
      case .whole: return [0,0,1,1]
      case .center70: return [0.15,0.15,0.85,0.85]
      case .centerSquare: return nil
      case .cornerTopLeft: return [0,0,0.6,0.6]
      case .cornerTopRight: return [0.4,0,1,0.6]
      case .cornerBottomLeft: return [0,0.4,0.6,1]
      case .cornerBottomRight: return [0.4,0.4,1,1]
      }
    }
    func rect(width: Int, height: Int) throws -> CGRect {
      guard width > 0, height > 0 else { throw Failure.invalidCrop }
      if self == .centerSquare {
        let side = min(width,height)
        return CGRect(x:(width-side)/2,y:(height-side)/2,width:side,height:side)
      }
      return try ImageSafetyPhotoInspectionContract.rect(normalizedBox!,width:width,height:height)
    }
  }
  static let corners: [View] = [.cornerTopLeft,.cornerTopRight,.cornerBottomLeft,.cornerBottomRight]
  static let semanticOrder: [View] = [.centerSquare,.whole]
  static func rect(_ box: [Double], width: Int, height: Int) throws -> CGRect {
    guard width > 0, height > 0, box.count == 4,
      box.allSatisfy({$0.isFinite && (0...1).contains($0)}), box[2] > box[0], box[3] > box[1] else { throw Failure.invalidCrop }
    let x=floor(Double(width)*box[0]),y=floor(Double(height)*box[1])
    let right=floor(Double(width)*box[2]),bottom=floor(Double(height)*box[3])
    guard right>x,bottom>y else { throw Failure.invalidCrop }
    return CGRect(x:x,y:y,width:right-x,height:bottom-y)
  }
  enum ScoreDomain: String, Codable {
    case probability, signedUnitMargin, decisionMargin, logit
    func accepts(_ value: Double) -> Bool {
      guard value.isFinite else { return false }
      switch self {
      case .probability: return (0...1).contains(value)
      case .signedUnitMargin: return (-1...1).contains(value)
      case .decisionMargin,.logit: return true
      }
    }
  }
  struct Score: Codable {
    let value: Double
    let domain: ScoreDomain
    init(_ value: Double, domain: ScoreDomain) throws {
      guard domain.accepts(value) else { throw Failure.invalidScore }
      self.value=value;self.domain=domain
    }
    /// Validation never remaps a score or changes Float/Double boundaries.
    func checked(as expected: ScoreDomain) throws -> Double {
      guard expected == domain else { throw Failure.wrongScoreDomain }
      return value
    }
  }
}

/// Opt-in, local, bounded diagnostic records. No image bytes, paths, thumbnails
/// or embeddings are retained. It is not the admission protocol or analytics.
final class ImageSafetyPhotoInspectionAudit: @unchecked Sendable {
  struct Metric: Codable {
    let name: String
    let value: Double
    let domain: ImageSafetyPhotoInspectionContract.ScoreDomain
  }
  struct Event: Codable {
    let stage: String
    let route: String
    let view: ImageSafetyPhotoInspectionContract.View?
    let pixelRect: [Double]?
    let metrics: [Metric]
    let milliseconds: Double
  }
  struct Report: Codable {
    let schema: String
    let jpegSHA256: String
    let modelVersion: String
    let preprocessingVersion: String
    let outcome: String
    let events: [Event]
    let droppedEvents: Int
    let milliseconds: Double
  }
  final class Session {
    fileprivate let sha: String, version: String
    fileprivate let generation: UInt64
    fileprivate let start = ProcessInfo.processInfo.systemUptime
    fileprivate var events: [Event] = []
    fileprivate var dropped = 0
    var outcome = "deferred"
    init(sha: String, version: String, generation: UInt64) { self.sha=sha;self.version=version;self.generation=generation }
    func record(_ stage: String, _ route: String,
                view: ImageSafetyPhotoInspectionContract.View? = nil, rect: CGRect? = nil,
                metrics: [Metric] = []) {
      guard events.count < 96 else { dropped += 1; return }
      // A diagnostic must not serialize NaN or interfere with an inspection.
      let finite=metrics.filter{$0.domain.accepts($0.value)}
      events.append(Event(stage:stage,route:route,view:view,
        pixelRect:rect.map{[Double($0.minX),Double($0.minY),Double($0.width),Double($0.height)]},
        metrics:finite,milliseconds:(ProcessInfo.processInfo.systemUptime-start)*1000))
    }
  }
  private let lock = NSLock()
  private let capacity: Int
  private var enabled: Bool
  private var reports: [Report] = []
  private var generation: UInt64 = 0
  init(enabled: Bool = false, capacity: Int = 16) {
    self.enabled=enabled;self.capacity=min(32,max(1,capacity))
  }
  func configure(enabled: Bool) { lock.lock();defer{lock.unlock()};self.enabled=enabled;generation &+= 1;reports.removeAll() }
  func clear() {lock.lock();defer{lock.unlock()};generation &+= 1;reports.removeAll()}
  func begin(sha: String, version: String) -> Session? {
    lock.lock();defer{lock.unlock()}
    guard enabled else {return nil}
    let safeSHA=sha.range(of:"^[0-9a-f]{64}$",options:.regularExpression) != nil ? sha : "invalid"
    return Session(sha:safeSHA,version:version,generation:generation)
  }
  func finish(_ session: Session) {
    let report=Report(schema:"image-safety-trace-v1",jpegSHA256:session.sha,modelVersion:session.version,
      preprocessingVersion:ImageSafetyPhotoInspectionContract.version,outcome:session.outcome,
      events:session.events,droppedEvents:session.dropped,milliseconds:(ProcessInfo.processInfo.systemUptime-session.start)*1000)
    lock.lock();defer{lock.unlock()}
    guard enabled, generation == session.generation else {return}
    if reports.count >= capacity {reports.removeFirst()};reports.append(report)
  }
  func snapshot() -> [Report] {lock.lock();defer{lock.unlock()};return reports}
}
