import Foundation
import CryptoKit
import Accelerate

/// Frozen schema-12 four-context/two-view decoder. Signed scores are decision values, not probabilities.
/// Thread-confined by the future publishing/album inference worker.
struct ImageSafetySensitivePhotoHead {
  enum Failure: Error { case wrongSize, wrongDigest, unsupportedFormat, invalidParameters, invalidFeature, cancelled }
  struct Decision {
    let sensitive: Bool
    let route: String
    let globalScores: [Double]
    let priorScore: Double
    let localScore: Double
    let localByView: [Double]
    let winningView: Int
    let winningPairCenters: [Double]
  }
  static let version = "maoqiu-dino-sensitive-v283-normal-readout-context4-2view-20261030"
  static let binarySHA256 = "5015de281aec4e28ab5e0aec632231c382304580e6fffdda8a3caa8a2e4d1063"
  private static let supports = 5107, dimensions = 2304, experts = 8, priorCount = 3089
  private static let headerBytes = 120
  private static let alphaOffset = supports * dimensions
  private static let priorOffset = alphaOffset + experts * supports
  private static let localOffset = priorOffset + priorCount
  private static let residualCount = 12353
  private static let count = localOffset + 385 + residualCount + 12481
  static let binaryBytes = headerBytes + count * 4
  private let values: [Float]
  private let biases: [Double]
  private static let positive = Double(Float(0.1))
  private static let priorBoundary = -4.710996150970459

  init(data: Data) throws {
    guard data.count == Self.binaryBytes else { throw Failure.wrongSize }
    let digest = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    guard digest == Self.binarySHA256 else { throw Failure.wrongDigest }
    func u32(_ offset: Int) -> UInt32 {
      (0..<4).reduce(UInt32(0)) { $0 | UInt32(data[data.startIndex + offset + $1]) << UInt32($1 * 8) }
    }
    guard data.prefix(8) == Data("MQSPH012".utf8),
      (0..<12).map({ Int(u32(8 + $0 * 4)) }) == [12,5107,2304,8,3089,384,32,2,4,384,388,0] else {
      throw Failure.unsupportedFormat
    }
    let header = (0..<16).map { Float(bitPattern: u32(56 + $0 * 4)) }
    guard header.allSatisfy({ $0.isFinite }), header[0] == Float(0.1), header[1] == -0.5,
      header[2] == 0.5, Double(header[3]) == Self.priorBoundary, header[4] == 2,
      header[13...15].allSatisfy({ $0 == 0 }) else { throw Failure.invalidParameters }
    biases = header[5...12].map(Double.init)
    values = data.withUnsafeBytes { (bytes: UnsafeRawBufferPointer) in
      (0..<Self.count).map { i in
        Float(bitPattern: UInt32(littleEndian: bytes.loadUnaligned(fromByteOffset: Self.headerBytes + i * 4, as: UInt32.self)))
      }
    }
    guard values.allSatisfy({ $0.isFinite }) else { throw Failure.invalidParameters }
  }

  private func block(_ pooling: [Float]) throws -> [Double] {
    guard pooling.count == 1152, pooling.allSatisfy({ $0.isFinite }) else { throw Failure.invalidFeature }
    for offset in [0,384] {
      let squared = pooling[offset..<(offset+384)].reduce(0.0) { $0 + Double($1) * Double($1) }
      guard (0.99...1.01).contains(squared) else { throw Failure.invalidFeature }
    }
    let std = pooling[768..<1152].map(Double.init)
    guard std.allSatisfy({ $0 >= 0 }) else { throw Failure.invalidFeature }
    let norm = sqrt(std.reduce(0.0) { $0 + $1 * $1 })
    guard norm.isFinite, norm > 0 else { throw Failure.invalidFeature }
    return pooling[0..<768].map(Double.init) + std.map { $0 / norm }
  }

  func expertScores(centerPooling: [Float], wholePooling: [Float], cancelled: () -> Bool = { false }) throws -> [Double] {
    let feature = (try block(centerPooling) + block(wholePooling)).map { Float($0 / sqrt(6.0)) }
    return try expertScores(pairedFeature: feature, cancelled: cancelled)
  }

  func expertScores(pairedFeature: [Float], cancelled: () -> Bool = { false }) throws -> [Double] {
    if cancelled() { throw Failure.cancelled }
    guard pairedFeature.count == Self.dimensions, pairedFeature.allSatisfy({ $0.isFinite }) else { throw Failure.invalidFeature }
    let feature = pairedFeature.map(Double.init)
    let norm = feature.reduce(0.0) { $0 + $1 * $1 }
    guard (0.99...1.01).contains(norm) else { throw Failure.invalidFeature }
    var scores = biases
    try values.withUnsafeBufferPointer { p in
      for row in 0..<Self.supports {
        if row % 128 == 0 && cancelled() { throw Failure.cancelled }
        var distance = 0.0
        let offset = row * Self.dimensions
        for d in 0..<Self.dimensions {
          let delta = feature[d] - Double(p[offset+d]); distance += delta * delta
        }
        let kernel = exp(-2.0 * distance)
        for expert in 0..<Self.experts { scores[expert] += kernel * Double(p[Self.alphaOffset + expert * Self.supports + row]) }
      }
    }
    guard scores.allSatisfy({ $0.isFinite }) else { throw Failure.invalidFeature }
    return scores
  }

  func priorScore(wholePooling: [Float]) throws -> Double {
    _ = try block(wholePooling)
    return try priorScore(wholeCLS: Array(wholePooling.prefix(384)))
  }

  func priorScore(wholeCLS: [Float]) throws -> Double {
    guard wholeCLS.count == 384, wholeCLS.allSatisfy({ $0.isFinite }) else { throw Failure.invalidFeature }
    let norm = wholeCLS.reduce(0.0) { $0 + Double($1) * Double($1) }
    guard (0.99...1.01).contains(norm) else { throw Failure.invalidFeature }
    let offset = Self.priorOffset
    var result = Double(values[offset+3088])
    for h in 0..<8 {
      var hidden = Double(values[offset+3072+h])
      for d in 0..<384 { hidden += Double(wholeCLS[d]) * Double(values[offset+d*8+h]) }
      result += tanh(hidden) * Double(values[offset+3080+h])
    }
    guard result.isFinite else { throw Failure.invalidFeature }
    return result
  }

  func localGrid(unitPatches: [Float], globalScores: [Double], cancelled: () -> Bool = { false }) throws -> [Float] {
    if cancelled() { throw Failure.cancelled }
    guard unitPatches.count == 512 * 384, unitPatches.allSatisfy({ $0.isFinite }),
      globalScores.count == Self.experts, globalScores.allSatisfy({ $0.isFinite }) else { throw Failure.invalidFeature }
    let context = [Float(globalScores[6]),Float(globalScores[5]),Float(globalScores[3]),Float(globalScores[2])]
    guard context.allSatisfy({ $0.isFinite }) else { throw Failure.invalidFeature }
    for token in 0..<512 {
      var norm = 0.0
      for d in 0..<384 { let x = Double(unitPatches[token*384+d]); norm += x*x }
      guard (0.99...1.01).contains(norm) else { throw Failure.invalidFeature }
    }
    var expanded = [Float](repeating: 0,count:512*388)
    for token in 0..<512 {
      for d in 0..<384 { expanded[token*388+d] = unitPatches[token*384+d] }
      expanded[token*388+384] = context[0];expanded[token*388+385] = context[1]
      expanded[token*388+386] = context[2];expanded[token*388+387] = context[3]
    }
    var base = [Float](repeating: 0,count:512),outputs = [[Float]]()
    values.withUnsafeBufferPointer { p in
      unitPatches.withUnsafeBufferPointer { x in
        base.withUnsafeMutableBufferPointer { y in
          cblas_sgemv(CblasRowMajor,CblasNoTrans,512,384,1,x.baseAddress!,384,p.baseAddress!+Self.localOffset,1,0,y.baseAddress!,1)
        }
      }
      for i in base.indices { base[i] += p[Self.localOffset+384] }
      for head in 0..<2 {
        let width = head == 0 ? 384 : 388
        let offset = Self.localOffset+385+(head == 0 ? 0 : Self.residualCount)
        let b1 = offset+32*width,w2=b1+32,b2=w2+32
        let input = head == 0 ? unitPatches : expanded
        var hidden=[Float](repeating:0,count:512*32)
        input.withUnsafeBufferPointer { x in
          hidden.withUnsafeMutableBufferPointer { h in
            cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasTrans,512,32,Int32(width),1,x.baseAddress!,Int32(width),p.baseAddress!+offset,Int32(width),0,h.baseAddress!,32)
          }
        }
        for token in 0..<512 {
          for channel in 0..<32 { let at=token*32+channel;hidden[at]=max(0,hidden[at]+p[b1+channel]) }
        }
        var out=[Float](repeating:0,count:512)
        hidden.withUnsafeBufferPointer { h in
          out.withUnsafeMutableBufferPointer { y in
            cblas_sgemv(CblasRowMajor,CblasNoTrans,512,32,1,h.baseAddress!,32,p.baseAddress!+w2,1,0,y.baseAddress!,1)
          }
        }
        for token in 0..<512 { out[token]=base[token]+(out[token]+p[b2]) }
        outputs.append(out)
      }
    }
    if cancelled() { throw Failure.cancelled }
    let result=(0..<512).map { (outputs[0][$0]+outputs[1][$0])*0.5 }
    guard result.allSatisfy({ $0.isFinite }) else { throw Failure.invalidFeature }
    return result
  }

  static func pairScore(_ grid: [Float]) throws -> (Double,Int,[Double]) {
    guard grid.count == 512, grid.allSatisfy({ $0.isFinite }) else { throw Failure.invalidFeature }
    var winning = (-Double.infinity,0,[Double]())
    for view in 0..<2 {
      var local = (-Double.infinity,[Double]())
      // Horizontal first, strict comparisons: same tie order as the reference.
      for y in 0..<16 {
        for x in 0..<15 {
          let at = view*256+y*16+x, score = (Double(grid[at])+Double(grid[at+1]))/2
          if score>local.0 { local = (score,[(Double(x)+0.5)/16,(Double(y)+0.5)/16,(Double(x)+1.5)/16,(Double(y)+0.5)/16]) }
        }
      }
      for y in 0..<15 {
        for x in 0..<16 {
          let at = view*256+y*16+x, score = (Double(grid[at])+Double(grid[at+16]))/2
          if score>local.0 { local = (score,[(Double(x)+0.5)/16,(Double(y)+0.5)/16,(Double(x)+0.5)/16,(Double(y)+1.5)/16]) }
        }
      }
      if local.0>winning.0 { winning = (local.0,view,local.1) }
    }
    return winning
  }

  static func pairScoresByView(_ grid: [Float]) throws -> [Double] {
    guard grid.count == 512, grid.allSatisfy({ $0.isFinite }) else { throw Failure.invalidFeature }
    return try (0..<2).map { view in
      let one = Array(grid[(view*256)..<((view+1)*256)])
      return try pairScore(one + one).0
    }
  }

  static func fuse(scores: [Double], prior: Double, local: Double, localByView: [Double]) throws -> (Bool,String) {
    guard scores.count == 8, scores.allSatisfy({ $0.isFinite }), prior.isFinite, local.isFinite,
      localByView.count == 2, localByView.allSatisfy({ $0.isFinite }),
      abs(localByView.max()! - local) < 1e-4 else { throw Failure.invalidFeature }
    let b=scores[0],r=scores[1],oldPet=scores[2],pet=scores[3],n=scores[4],g=scores[5],pos=Self.positive
    var flag: Bool, route: String
    if b>pos { flag=true;route="basePositive" }
    else if b < -0.5 { flag=false;route="baseNegative" }
    else if r>pos { flag=true;route="refinementPositive" }
    else if r < -0.5 { flag=false;route="refinementNegative" }
    else if prior>Self.priorBoundary && oldPet>pos { flag=false;route="semanticPetNormal" }
    else { flag=prior>Self.priorBoundary;route="wholePrior" }
    let boundary = ["basePositive","refinementPositive"].contains(route) ? 0.5 : pos
    if flag && pet>boundary { flag=false;route=boundary==0.5 ? "addedStrongPetNormal" : "addedGrayPetNormal" }
    let v44Flag=flag,v44Route=route
    if b >= -0.5 && b<=pos {
      if n>pos { flag=true;route="newRefinementPositive";if pet>0.5 { flag=false;route="strongPetAfterNewRefinement" } }
      else if n < -0.5 && g < -0.5 { flag=false;route="genitalVetoQualifiedNormal" }
    }
    if route=="genitalVetoQualifiedNormal" && !(local < -0.5) { flag=v44Flag;route="localVetoRestoreV44-"+v44Route }
    if !flag && b >= -0.5 && b<=pos && n >= -0.5 && local>pos && pet<=pos { flag=true;route="localizedGenitalPositive" }
    if flag && route == "wholePrior" && scores[6] < -0.5 && g < -0.5 && local < -0.5 { flag=false;route="augmentedHumanQualifiedNormal" }
    if flag && scores[7] < -0.5 && local < -0.5 { flag=false;route="learnedNormalCorrectionStrongLocal" }
    if flag && pet>pos && oldPet > -0.5 && g<pos && local < -0.5 { flag=false;route="strongLocalPetConsensusNormal" }
    let humanMax = max(max(b,r),max(n,scores[6]))
    if flag && humanMax<0.5 && local<pos {
      if pet>pos && oldPet > -0.5 && g<pos { flag=false;route="qualifiedWeakPetNormal" }
      else if pet > -0.5 && oldPet > -0.5 && n < -0.5 && scores[6] < -0.5 && g < -0.5 { flag=false;route="qualifiedGrayPetNormal" }
    }
    if flag && humanMax<0.5 && pet>pos && oldPet > -0.5 && g<pos && local<0.5 && localByView.min()! < pos {
      flag=false;route="qualifiedPetViewNormal"
    }
    return (flag,route)
  }

  func classify(centerPooling: [Float], wholePooling: [Float], unitPatches: [Float], cancelled: () -> Bool = { false }) throws -> Decision {
    let global = try expertScores(centerPooling: centerPooling,wholePooling: wholePooling,cancelled: cancelled)
    let prior = try priorScore(wholePooling: wholePooling)
    let grid = try localGrid(unitPatches: unitPatches,globalScores: global,cancelled: cancelled)
    let (local,view,centers) = try Self.pairScore(grid)
    let perView = try Self.pairScoresByView(grid)
    let (flag,route) = try Self.fuse(scores: global,prior: prior,local: local,localByView: perView)
    if cancelled() { throw Failure.cancelled }
    return Decision(sensitive: flag,route: route,globalScores: global,priorScore: prior,localScore: local,localByView: perView,winningView: view,winningPairCenters: centers)
  }

  func classifyFeatures(pairedFeature: [Float], wholeCLS: [Float], unitPatches: [Float], cancelled: () -> Bool = { false }) throws -> Decision {
    let global=try expertScores(pairedFeature:pairedFeature,cancelled:cancelled)
    let prior=try priorScore(wholeCLS:wholeCLS)
    let grid=try localGrid(unitPatches:unitPatches,globalScores:global,cancelled:cancelled)
    let (local,view,centers)=try Self.pairScore(grid)
    let perView=try Self.pairScoresByView(grid)
    let (flag,route)=try Self.fuse(scores:global,prior:prior,local:local,localByView:perView)
    if cancelled() { throw Failure.cancelled }
    return Decision(sensitive:flag,route:route,globalScores:global,priorScore:prior,localScore:local,localByView:perView,winningView:view,winningPairCenters:centers)
  }

}
