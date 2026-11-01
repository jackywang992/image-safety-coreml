import Foundation
import CoreML
import Vision
#if canImport(UIKit)
import UIKit
#endif

final class InferenceGate<Resource: AnyObject> {
  enum Priority { case background, foreground }
  enum Failure: Error { case workerThreadRequired, cancelled, reentrantUse }

  private let condition = NSCondition()
  private let idleSeconds: TimeInterval
  private var occupied = false
  private var ownerThread: Thread?
  private var foregroundWaiters = 0
  private var clearAfterUse = false
  private var idleGeneration: UInt64 = 0
  private var cached: (key: String, resource: Resource)?

  init(idleSeconds: TimeInterval = 30) {
    precondition(idleSeconds.isFinite && idleSeconds >= 0)
    self.idleSeconds = idleSeconds
  }

  func withResource<Value>(key: String, priority: Priority,
                           cancelled: () -> Bool = { false },
                           load: () throws -> Resource,
                           operation: (Resource) throws -> Value) throws -> Value {
    // UI callers enqueue work on their worker; they never wait for another
    // prediction or a model load on the main thread.
    guard !Thread.isMainThread else { throw Failure.workerThreadRequired }
    try acquire(priority: priority, cancelled: cancelled)
    // Drain the operation's temporary CoreML/Vision references before waking
    // the next borrower, including when loading or prediction fails.
    let result: Result<Value, Error> = autoreleasepool {
      Result {
        if cancelled() { throw Failure.cancelled }
        if cached?.key != key {
          autoreleasepool { cached = nil }
        }
        if cached == nil { cached = (key, try load()) }
        if cancelled() { throw Failure.cancelled }
        let value = try operation(cached!.resource)
        if cancelled() { throw Failure.cancelled }
        return value
      }
    }
    finish()
    return try result.get()
  }

  /// The cancellation flag must be nonblocking and must not re-enter the gate.
  /// The wake-up is optional: queued operations also check it every 50 ms.
  func wakeWaitingOperations() {
    condition.lock()
    condition.broadcast()
    condition.unlock()
  }

  /// Memory warning handlers only enqueue a release. Active inference keeps its
  /// resource until that operation ends; teardown always happens off the UI.
  func releaseWhenIdle() {
    DispatchQueue.global(qos: .utility).async { [weak self] in
      guard let self else { return }
      self.condition.lock()
      if self.occupied {
        self.clearAfterUse = true
      } else {
        autoreleasepool { self.cached = nil }
        self.idleGeneration &+= 1
      }
      self.condition.unlock()
    }
  }

  private func acquire(priority: Priority, cancelled: () -> Bool) throws {
    condition.lock()
    defer { condition.unlock() }
    if priority == .foreground { foregroundWaiters += 1 }
    defer {
      if priority == .foreground {
        foregroundWaiters -= 1
        condition.broadcast()
      }
    }
    if occupied && ownerThread === Thread.current { throw Failure.reentrantUse }
    while occupied || (priority == .background && foregroundWaiters > 0) {
      if cancelled() { throw Failure.cancelled }
      _ = condition.wait(until: Date(timeIntervalSinceNow: 0.05))
    }
    if cancelled() { throw Failure.cancelled }
    occupied = true
    ownerThread = Thread.current
  }

  private func finish() {
    condition.lock()
    if clearAfterUse {
      autoreleasepool { cached = nil }
      clearAfterUse = false
    }
    occupied = false
    ownerThread = nil
    idleGeneration &+= 1
    let generation = idleGeneration
    condition.broadcast()
    condition.unlock()
    DispatchQueue.global(qos: .utility).asyncAfter(deadline: .now() + idleSeconds) { [weak self] in
      guard let self else { return }
      self.condition.lock()
      if !self.occupied && self.idleGeneration == generation {
        autoreleasepool { self.cached = nil }
      }
      self.condition.unlock()
    }
  }

  #if DEBUG
  func lifecycleForTesting() -> (foregroundWaiters: Int, clearAfterUse: Bool) {
    condition.lock()
    defer { condition.unlock() }
    return (foregroundWaiters, clearAfterUse)
  }
  #endif
}


/// Owns one CoreML model and its optional Vision wrapper across photo tasks.
/// Album inference keeps its chosen backend; a foreground CPU borrower does not
/// change an album index's CPU-fallback preference.
final class ModelBroker {
  static let shared = ModelBroker()

  private final class LoadedModel {
    let model: MLModel
    var vision: VNCoreMLModel?
    init(_ model: MLModel) { self.model = model }
  }

  private let gate = InferenceGate<LoadedModel>()
  private var memoryWarningObserver: NSObjectProtocol?

  private init() {
    #if canImport(UIKit)
    memoryWarningObserver = NotificationCenter.default.addObserver(
      forName: UIApplication.didReceiveMemoryWarningNotification,
      object: nil, queue: nil
    ) { [weak self] _ in self?.releaseWhenIdle() }
    #endif
  }

  deinit {
    if let memoryWarningObserver { NotificationCenter.default.removeObserver(memoryWarningObserver) }
  }

  func withVisionModel<Value>(at url: URL, computeUnits: MLComputeUnits,
                              cancelled: () -> Bool = { false },
                              operation: (VNCoreMLModel) throws -> Value) throws -> Value {
    try borrow(at: url, computeUnits: computeUnits, foreground: false,
               cancelled: cancelled) { loaded in
      if loaded.vision == nil { loaded.vision = try VNCoreMLModel(for: loaded.model) }
      return try operation(loaded.vision!)
    }
  }

  func withForegroundModel<Value>(at url: URL,
                                  cancelled: () -> Bool,
                                  operation: (MLModel) throws -> Value) throws -> Value {
    try borrow(at: url, computeUnits: .cpuOnly, foreground: true,
               cancelled: cancelled) { try operation($0.model) }
  }

  /// Publishing shares the cached Vision wrapper and model lifetime while
  /// taking foreground priority over the next album inference operation.
  func withForegroundVisionModel<Value>(at url: URL,
                                        cancelled: () -> Bool,
                                        operation: (VNCoreMLModel) throws -> Value) throws -> Value {
    try borrow(at: url, computeUnits: .cpuOnly, foreground: true,
               cancelled: cancelled) { loaded in
      if loaded.vision == nil { loaded.vision = try VNCoreMLModel(for: loaded.model) }
      return try operation(loaded.vision!)
    }
  }

  func wakeWaitingOperations() { gate.wakeWaitingOperations() }
  func releaseWhenIdle() { gate.releaseWhenIdle() }

  private func borrow<Value>(at url: URL, computeUnits: MLComputeUnits,
                             foreground: Bool, cancelled: () -> Bool,
                             operation: (LoadedModel) throws -> Value) throws -> Value {
    let key = url.standardizedFileURL.path + "|" + String(computeUnits.rawValue)
    return try gate.withResource(key: key,
      priority: foreground ? .foreground : .background,
      cancelled: cancelled, load: {
        let configuration = MLModelConfiguration()
        configuration.computeUnits = computeUnits
        return LoadedModel(try MLModel(contentsOf: url, configuration: configuration))
      }, operation: operation)
  }
}
