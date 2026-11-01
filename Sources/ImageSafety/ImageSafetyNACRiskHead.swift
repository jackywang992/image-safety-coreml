import Foundation
import Accelerate
struct ImageSafetyNACRiskHead: Sendable {
 enum Failure: Error { case invalidModel, invalidFeatures, invalidResult, mainThread }
 private let first:[Float], bias:[Float], last:[Float], outputBias:Float
 init(data:Data) throws {
  guard data.count == 49436, Array(data.prefix(8)) == Array("MQNAC173".utf8) else { throw Failure.invalidModel }
  func u32(_ at:Int)->UInt32 { data.withUnsafeBytes { UInt32(littleEndian:$0.loadUnaligned(fromByteOffset:at,as:UInt32.self)) } }
  guard [u32(8),u32(12),u32(16),u32(20)] == [384,32,512,12353] else { throw Failure.invalidModel }
  let values=(0..<12353).map { Float(bitPattern:u32(24+4*$0)) }
  guard values.allSatisfy({$0.isFinite}) else { throw Failure.invalidModel }
  first=Array(values[0..<12288]);bias=Array(values[12288..<12320]);last=Array(values[12320..<12352]);outputBias=values[12352]
 }
 func predict(unitPatches:[Float]) throws ->Float {
  guard !Thread.isMainThread else { throw Failure.mainThread }
  guard unitPatches.count == 512*384, unitPatches.allSatisfy({$0.isFinite}) else { throw Failure.invalidFeatures }
  let valid=unitPatches.withUnsafeBufferPointer { ptr in (0..<512).allSatisfy { i in let p=ptr.baseAddress!+i*384;return abs(cblas_sdot(384,p,1,p,1)-1)<Float(0.001) } }
  guard valid else { throw Failure.invalidFeatures }
  var z=[Float](repeating:0,count:512*32)
  unitPatches.withUnsafeBufferPointer { x in first.withUnsafeBufferPointer { w in z.withUnsafeMutableBufferPointer { h in cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasTrans,512,32,384,16,x.baseAddress!,384,w.baseAddress!,384,0,h.baseAddress!,32) } } }
  var maximum = -Float.infinity
  for i in 0..<512 {
   var value=outputBias
   for j in 0..<32 { value += max(Float(0),z[i*32+j]+bias[j])*last[j] }
   maximum=max(maximum,value)
  }
  guard maximum.isFinite else { throw Failure.invalidResult };return maximum
 }
}
