import Foundation
import Accelerate

struct ImageSafetyGoreCompactContext: Sendable {
  enum Failure: Error { case invalidModel, invalidFeatures, invalidResult, mainThread }
  struct Result: Sendable {
    let originalLogit: Float
    let correction: Float
    let score: Float
    let originalScore: Float
    let summary: [Float]
    var sensitive: Bool { score > Float(0.1) }
  }
  private let first: [Float], firstBias: [Float], last: [Float], lastBias: Float
  private let mean: [Float], scale: [Float], context: [Float], contextBias: Float
  private let originalContext: [Float], localSupports: [Float], localCoefficients: [Float]
  private let localGamma: Double, supportCount: Int
  init(data: Data) throws {
    guard data.count >= 51004, Array(data.prefix(8)) == Array("MQGC3031".utf8) else { throw Failure.invalidModel }
    func u32(_ offset: Int) -> UInt32 {
      data.withUnsafeBytes { UInt32(littleEndian: $0.loadUnaligned(fromByteOffset: offset, as: UInt32.self)) }
    }
    guard [u32(8),u32(12),u32(16),u32(20)] == [384,32,96,12642] else { throw Failure.invalidModel }
    let values = (0..<12642).map { Float(bitPattern: u32(24 + 4 * $0)) }
    guard values.allSatisfy({ $0.isFinite }) else { throw Failure.invalidModel }
    first = Array(values[0..<12288]); firstBias = Array(values[12288..<12320])
    last = Array(values[12320..<12352]); lastBias = values[12352]
    mean = Array(values[12353..<12449]); scale = Array(values[12449..<12545])
    context = Array(values[12545..<12641]); contextBias = values[12641]
    guard scale.allSatisfy({ $0 >= Float(0.01) }) else { throw Failure.invalidModel }
    let extensionOffset = 50592
    guard Array(data[extensionOffset..<(extensionOffset + 8)]) == Array("MQGL3031".utf8),
      u32(extensionOffset + 12) == 96, u32(extensionOffset + 20) == 0 else { throw Failure.invalidModel }
    let count = Int(u32(extensionOffset + 8))
    let gamma = Float(bitPattern:u32(extensionOffset + 16))
    guard (1...4096).contains(count), gamma.isFinite, gamma > 0, gamma <= 32,
      data.count == extensionOffset + 24 + (97 + count * 97) * 4 else { throw Failure.invalidModel }
    let local = (0..<(97 + count * 97)).map { Float(bitPattern:u32(extensionOffset + 24 + 4 * $0)) }
    guard local.allSatisfy({ $0.isFinite }) else { throw Failure.invalidModel }
    originalContext = Array(local[0..<97]); localSupports = Array(local[97..<(97 + count * 96)])
    localCoefficients = Array(local[(97 + count * 96)...]); localGamma = Double(gamma); supportCount = count
    for i in 0..<count {
      var norm = 0.0
      for j in 0..<96 { let value = Double(localSupports[i * 96 + j]); norm += value * value }
      guard abs(norm - 1) < 0.001 else { throw Failure.invalidModel }
    }

  }
  func predict(unitPatches: [Float]) throws -> Result {
    guard !Thread.isMainThread else { throw Failure.mainThread }
    guard unitPatches.count == 1536 * 384, unitPatches.allSatisfy({ $0.isFinite }) else { throw Failure.invalidFeatures }
    let normalized = unitPatches.withUnsafeBufferPointer { pointer in
      (0..<1536).allSatisfy { i in
        let p = pointer.baseAddress! + i * 384
        let norm = cblas_sdot(384, p, 1, p, 1)
        return norm.isFinite && abs(norm - 1) < Float(0.001)
      }
    }
    guard normalized else { throw Failure.invalidFeatures }
    var hidden = [Float](repeating: 0, count: 1536 * 32)
    unitPatches.withUnsafeBufferPointer { x in
      first.withUnsafeBufferPointer { w in
        hidden.withUnsafeMutableBufferPointer { z in
          cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 1536, 32, 384, 16, x.baseAddress!, 384, w.baseAddress!, 384, 0, z.baseAddress!, 32)
        }
      }
    }
    var sum = [Double](repeating: 0, count: 32), square = sum
    var maximum = [Float](repeating: 0, count: 32)
    var original = -Float.infinity
    for i in 0..<1536 {
      var logit = lastBias
      for j in 0..<32 {
        let value = max(Float(0), hidden[i * 32 + j] + firstBias[j])
        guard value.isFinite else { throw Failure.invalidResult }
        sum[j] += Double(value); square[j] += Double(value) * Double(value)
        maximum[j] = max(maximum[j], value); logit += value * last[j]
      }
      original = max(original, logit)
    }
    var summary = [Float](repeating: 0, count: 96)
    for j in 0..<32 {
      let average = sum[j] / 1536
      summary[j] = Float(average)
      summary[32 + j] = Float(sqrt(max(0, square[j] / 1536 - average * average)))
      summary[64 + j] = maximum[j]
    }
    return try contextResult(original:original,summary:summary)
  }
  private func contextResult(original:Float,summary:[Float]) throws -> Result {
    guard !Thread.isMainThread else { throw Failure.mainThread }
    guard original.isFinite, summary.count == 96, summary.allSatisfy({ $0.isFinite }) else { throw Failure.invalidFeatures }
    var residual = contextBias, originalResidual = originalContext[96]
    var unit = [Double](repeating:0,count:96), norm = 0.0
    for j in 0..<96 {
      let standardized = (summary[j] - mean[j]) / scale[j]
      residual += standardized * context[j]
      originalResidual += standardized * originalContext[j]
      unit[j] = (Double(summary[j]) - Double(mean[j])) / Double(scale[j]); norm += unit[j] * unit[j]
    }
    let length = max(1e-12,sqrt(norm))
    for j in 0..<96 { unit[j] /= length }
    var local = 0.0
    for i in 0..<supportCount {
      var dot = 0.0
      for j in 0..<96 { dot += unit[j] * Double(localSupports[i * 96 + j]) }
      local += exp(-localGamma * max(0,2 - 2 * dot)) * Double(localCoefficients[i])
    }
    guard local.isFinite else { throw Failure.invalidResult }
    let corrected = min(residual,originalResidual - Float(max(0,local)))
    let combined = original + corrected
    guard corrected.isFinite, combined.isFinite else { throw Failure.invalidResult }
    let result = Result(originalLogit:original, correction:corrected,
      score:tanhf(combined / 2), originalScore:tanhf(original / 2), summary:summary)
    guard result.score.isFinite, result.originalScore.isFinite else { throw Failure.invalidResult }
    return result
  }
}
