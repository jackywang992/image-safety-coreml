import Foundation
import Accelerate

struct ImageSafetySemanticRiskHead: Sendable {
 enum Failure: Error { case invalidModel, invalidFeature, invalidResult, mainThread }
 struct Result: Sendable {
  let score: Float, riskCosine: Float, normalCosine: Float
  let riskIndex: Int, normalIndex: Int
  var positive: Bool { score > Float(0.1) }
 }
 private let prototypes: [Float]
 private let scale: Float
 private let groupCount: Int
 init(data: Data) throws {
  guard data.count >= 32, Array(data.prefix(8)) == Array("MQTC2391".utf8) else { throw Failure.invalidModel }
  func u32(_ at: Int) -> UInt32 { data.withUnsafeBytes { UInt32(littleEndian: $0.loadUnaligned(fromByteOffset: at, as: UInt32.self)) } }
  let groups=Int(u32(8))
  guard (11...32).contains(groups), data.count == 32+groups*512*4, [u32(12),u32(16),u32(20)] == [512,4,0] else { throw Failure.invalidModel }
  let s=Float(bitPattern:u32(24)), cut=Float(bitPattern:u32(28))
  guard s.isFinite, s>0, s<=100, cut==Float(0.1) else { throw Failure.invalidModel }
  let p=(0..<(groups*512)).map { Float(bitPattern:u32(32+4*$0)) }
  guard p.allSatisfy({$0.isFinite}) else { throw Failure.invalidModel }
  for i in 0..<groups {
   let norm=p[(i*512)..<((i+1)*512)].reduce(Double(0)) {$0+Double($1)*Double($1)}
   guard abs(norm-1)<0.001 else { throw Failure.invalidModel }
  }
  prototypes=p;scale=s;groupCount=groups
 }
 func score(embedding: [Float]) throws -> Result {
  guard !Thread.isMainThread else { throw Failure.mainThread }
  guard embedding.count==512, embedding.allSatisfy({$0.isFinite}) else { throw Failure.invalidFeature }
  let norm=embedding.reduce(Double(0)) {$0+Double($1)*Double($1)}
  guard abs(norm-1)<0.001 else { throw Failure.invalidFeature }
  var risk = -Float.infinity, normal = -Float.infinity, riskIndex=0, normalIndex=4
  embedding.withUnsafeBufferPointer { x in prototypes.withUnsafeBufferPointer { p in
   for i in 0..<groupCount {
    let value=cblas_sdot(512,x.baseAddress!,1,p.baseAddress!+i*512,1)
    if i<4 { if value>risk {risk=value;riskIndex=i} }
    else if value>normal {normal=value;normalIndex=i}
   }
  } }
  let result=Result(score:tanhf((risk-normal)*scale/2),riskCosine:risk,normalCosine:normal,riskIndex:riskIndex,normalIndex:normalIndex)
  guard result.score.isFinite, risk.isFinite, normal.isFinite else {throw Failure.invalidResult}
  return result
 }
}
