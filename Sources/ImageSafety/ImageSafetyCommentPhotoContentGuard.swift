import Foundation
import CoreML
import Vision
import ImageIO
import CryptoKit
import CoreGraphics
#if canImport(UIKit)
import UIKit
#endif

/// Storage-only sharing of bit-identical FP32 rows. Restore the frozen binary
/// before its original SHA and decoder checks; inference never sees row indices.
enum ImageSafetySharedSafetyParameters {
  enum Failure: Error { case invalidStorage }
  static let bankBytes = 14_035_984
  static let bankSHA256 = "5ad50cdbd8e22908e7186f7eb6eb4d89907019503f848459e45635f06418bb93"
  static func restore(_ stored: Data, at url: URL, expectedBytes: Int) throws -> Data {
    guard expectedBytes > 0, expectedBytes <= 47_342_888 else { throw Failure.invalidStorage }
    if stored.count == expectedBytes { return stored }
    guard stored.count >= 40, stored.prefix(8) == Data("MQREF001".utf8),
      stored[8..<16] == Data("MQBG3601".utf8) else { throw Failure.invalidStorage }
    func u32(_ data: Data, _ offset: Int) -> Int {
      data.withUnsafeBytes { Int(UInt32(littleEndian: $0.loadUnaligned(fromByteOffset: offset, as: UInt32.self))) }
    }
    let count = u32(stored, 16)
    guard (1...8192).contains(count), u32(stored, 20) == 768,
      expectedBytes == 32 + count * 3081, stored.count == 40 + count * 13 else { throw Failure.invalidStorage }
    let bankURL = url.deletingLastPathComponent().appendingPathComponent("ImageSafetySharedSafetyRows.bin")
    let attributes = try FileManager.default.attributesOfItem(atPath: bankURL.path)
    guard (attributes[.size] as? NSNumber)?.intValue == bankBytes else { throw Failure.invalidStorage }
    let bank = try Data(contentsOf: bankURL, options: .mappedIfSafe)
    guard bank.count == bankBytes, bank.prefix(8) == Data("MQBANK01".utf8),
      u32(bank, 8) == 4569, u32(bank, 12) == 768,
      SHA256.hash(data: bank).map({ String(format: "%02x", $0) }).joined() == bankSHA256 else { throw Failure.invalidStorage }
    var restored = Data(count: expectedBytes)
    try restored.withUnsafeMutableBytes { (destination: UnsafeMutableRawBufferPointer) in
      try bank.withUnsafeBytes { (rows: UnsafeRawBufferPointer) in
        try stored.withUnsafeBytes { (source: UnsafeRawBufferPointer) in
          destination.baseAddress!.copyMemory(from: source.baseAddress! + 8, byteCount: 32)
          for i in 0..<count {
            let row = u32(stored, 40 + i * 4)
            guard row < 4569 else { throw Failure.invalidStorage }
            (destination.baseAddress! + 32 + i * 3072).copyMemory(from: rows.baseAddress! + 16 + row * 3072, byteCount: 3072)
          }
          (destination.baseAddress! + 32 + count * 3072).copyMemory(from: source.baseAddress! + 40 + count * 4, byteCount: count * 9)
        }
      }
    }
    return restored
  }
}


/// Thread-confined photo work; only immutable models/heads survive an operation.
/// The publishing adapter runs the independent code guard before this engine.
final class ImageSafetyCommentPhotoContentGuard: @unchecked Sendable {
  enum Failure: Error { case workerThreadRequired, cancelled, invalidPhoto, changedBytes, invalidAsset, invalidOutput }
  struct Binary {
    let url: URL
    let sha256: String
    let bytes: Int
  }
  struct Assets {
    let version: String
    let dino: URL
    let xs: URL
    let semantic: URL
    let body: Binary
    let nac: Binary
    let gore: Binary
    let semanticHead: Binary
    let upperbodySemantic: Binary
    let bloodSemantic: Binary
    let bloodContext: Binary
    let normalContext: Binary
    let reviewedContext: Binary
    let recall12Context: Binary
    let newRecallContext: Binary
    let recallContext: Binary
  }
  struct Inspection {
    let containsSensitiveContent: Bool
    let inspectedSHA256: String
    let modelVersion: String
  }
  private final class Heads {
    let body: ImageSafetySensitivePhotoHead
    let nac: ImageSafetyNACRiskHead
    let gore: ImageSafetyGoreCompactContext
    let semantic: ImageSafetySemanticRiskHead
    let upperbodySemantic: ImageSafetySemanticRiskHead
    let bloodSemantic: ImageSafetySemanticRiskHead
    init(_ assets: Assets) throws {
      upperbodySemantic = try ImageSafetySemanticRiskHead(data: ImageSafetyCommentPhotoContentGuard.read(assets.upperbodySemantic))
      body = try ImageSafetySensitivePhotoHead(data: ImageSafetyCommentPhotoContentGuard.read(assets.body))
      nac = try ImageSafetyNACRiskHead(data: ImageSafetyCommentPhotoContentGuard.read(assets.nac))
      gore = try ImageSafetyGoreCompactContext(data: ImageSafetyCommentPhotoContentGuard.read(assets.gore))
      semantic = try ImageSafetySemanticRiskHead(data: ImageSafetyCommentPhotoContentGuard.read(assets.semanticHead))
      bloodSemantic = try ImageSafetySemanticRiskHead(data: ImageSafetyCommentPhotoContentGuard.read(assets.bloodSemantic))
    }
  }
  private struct Features {
    let pooling: [Float]
    let patches: [Float]
  }
  private struct BaseContext {let image:CGImage;let whole:Features;let center:Features;let gore:Float;let xs:Float;let normalAllowed:Bool;let recallPositive:Bool;let ungatedRecall:Float}
  private final class BloodContextModel {let value:ImageSafetyBloodContextResiduals;init(data:Data)throws{value=try ImageSafetyBloodContextResiduals(data:data)}}
  private let recall12Residuals = InferenceGate<BloodContextModel>()
  private let newRecallResiduals = InferenceGate<BloodContextModel>()
  private let recallResiduals = InferenceGate<BloodContextModel>()
  private let reviewResiduals = InferenceGate<BloodContextModel>()
  private let normalResiduals = InferenceGate<BloodContextModel>()
  private let residuals = InferenceGate<BloodContextModel>()
  private final class InspectionPermit {}
  private let inspections = InferenceGate<InspectionPermit>()
  private var semanticCache: [[Float]]?
  private let diagnostics: ImageSafetyPhotoInspectionAudit?
  private var trace: ImageSafetyPhotoInspectionAudit.Session?
  private var researchEvidence: [String:Any] = [:]
  func researchBloodEvidence() -> [String:Any] { researchEvidence }
  private let assets: Assets
  private let heads = InferenceGate<Heads>()
  // One auxiliary encoder at a time, plus the existing shared album encoder.
  private let auxiliary = InferenceGate<MLModel>()
  private var memoryObserver: NSObjectProtocol?
  private static let cornerViews = ImageSafetyPhotoInspectionContract.corners

  init(assets: Assets, diagnostics: ImageSafetyPhotoInspectionAudit? = nil) {
    self.assets = assets
    self.diagnostics = diagnostics
    #if canImport(UIKit)
    memoryObserver = NotificationCenter.default.addObserver(
      forName: UIApplication.didReceiveMemoryWarningNotification, object: nil, queue: nil
    ) { [weak self] _ in self?.releaseWhenIdle() }
    #endif
  }
  deinit { if let memoryObserver { NotificationCenter.default.removeObserver(memoryObserver) } }

  func wakeWaitingOperations() {
    inspections.wakeWaitingOperations()
    recall12Residuals.wakeWaitingOperations()
    newRecallResiduals.wakeWaitingOperations()
    recallResiduals.wakeWaitingOperations()
    reviewResiduals.wakeWaitingOperations()
    normalResiduals.wakeWaitingOperations()
    residuals.wakeWaitingOperations()
    heads.wakeWaitingOperations()
    auxiliary.wakeWaitingOperations()
    ModelBroker.shared.wakeWaitingOperations()
  }
  func releaseWhenIdle() {
    inspections.releaseWhenIdle()
    recall12Residuals.releaseWhenIdle()
    newRecallResiduals.releaseWhenIdle()
    recallResiduals.releaseWhenIdle()
    reviewResiduals.releaseWhenIdle()
    normalResiduals.releaseWhenIdle()
    residuals.releaseWhenIdle()
    heads.releaseWhenIdle()
    auxiliary.releaseWhenIdle()
    ModelBroker.shared.releaseWhenIdle()
  }

  func inspectPreparedJPEG(_ data: Data, expectedSHA256: String,
                           cancelled: () -> Bool = { false }) throws -> Inspection {
    guard !Thread.isMainThread else { throw Failure.workerThreadRequired }
    return try inspections.withResource(key: assets.version, priority: .foreground,
      cancelled: cancelled, load: { InspectionPermit() }) { _ in
      try inspectSerialJPEG(data, expectedSHA256: expectedSHA256, cancelled: cancelled)
    }
  }

  private func inspectSerialJPEG(_ data: Data, expectedSHA256: String,
                                cancelled: () -> Bool) throws -> Inspection {
    trace = diagnostics?.begin(sha:expectedSHA256,version:assets.version)
    defer {
      if let trace {
        if trace.outcome == "deferred" { trace.record("failure","inspectionDidNotComplete") }
        diagnostics?.finish(trace)
      }
      trace = nil
    }
    trace?.record("input","preparedJPEGValidation")
    semanticCache = nil; researchEvidence = [:]; defer { semanticCache = nil; researchEvidence = [:] }
    guard !Thread.isMainThread else { throw Failure.workerThreadRequired }
    func check() throws { if cancelled() { throw Failure.cancelled } }
    try check()
    guard !data.isEmpty, data.count <= 614400,
      expectedSHA256.range(of: "^[0-9a-f]{64}$", options: .regularExpression) != nil else { throw Failure.invalidPhoto }
    let digest = Self.hash(data)
    guard digest == expectedSHA256 else { throw Failure.changedBytes }
    let source = try Self.source(data)
    if trace != nil, let p=CGImageSourceCopyPropertiesAtIndex(source,0,nil) as? [CFString:Any],
      let width=p[kCGImagePropertyPixelWidth] as? Int,let height=p[kCGImagePropertyPixelHeight] as? Int {
      trace?.record("input","validatedUprightJPEG",view:.whole,rect:CGRect(x:0,y:0,width:width,height:height))
    }
    try check()
    let baseContext = try makeBaseContext(data, cancelled:cancelled)
    // Shared two-view DINO context calibrates both blood branches; body bank remains lazy.
    let positive: Bool
    do {
      if baseContext.recallPositive { trace?.record("decision","recallPositive"); positive = true }
      else if try xsPositive(data, correction:baseContext.xs, pornRecall:baseContext.ungatedRecall, cancelled: cancelled), !baseContext.normalAllowed { trace?.record("decision","xsPositive"); positive = true }
      else {
        positive = try heads.withResource(key: assets.version, priority: .foreground,
          cancelled: cancelled, load: { try Heads(self.assets) }) { models in
          try check()
          if try dinoPositive(data, models: models, context:baseContext, cancelled: cancelled) { return true }
          try check()
          if baseContext.normalAllowed { trace?.record("decision","normalContextProtection"); return false }
          return try semanticPositive(data, head: models.semantic, cancelled: cancelled)
        }
      }
    } catch {
      // Resource gates have their own error types; expose one cancellation
      // outcome regardless of whether it happened while waiting or predicting.
      if cancelled() { throw Failure.cancelled }
      throw error
    }
    try check()
    trace?.outcome = positive ? "rejected" : "allowed"
    trace?.record("decision",positive ? "contentRejected" : "contentAllowed")
    researchEvidence["oldPolicyContentBlocked"] = positive || (researchEvidence["guardApplied"] as? Bool == true)
    researchEvidence["v2CounterfactualBlocked"] = positive || (researchEvidence["upperbodyCornerCorrected"] as? Bool == true)
    return Inspection(containsSensitiveContent: positive, inspectedSHA256: digest, modelVersion: assets.version)
  }

  private static func hash(_ data: Data) -> String {
    SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
  }
  private static func read(_ asset: Binary) throws -> Data {
    let attributes = try FileManager.default.attributesOfItem(atPath: asset.url.path)
    guard let storedBytes = (attributes[.size] as? NSNumber)?.intValue,
      storedBytes > 0, storedBytes <= asset.bytes else { throw Failure.invalidAsset }
    let stored = try Data(contentsOf: asset.url, options: .mappedIfSafe)
    let data = try ImageSafetySharedSafetyParameters.restore(stored, at: asset.url, expectedBytes: asset.bytes)
    guard data.count == asset.bytes, hash(data) == asset.sha256 else { throw Failure.invalidAsset }
    return data
  }
  private static func source(_ data: Data) throws -> CGImageSource {
    guard let source = CGImageSourceCreateWithData(data as CFData, nil),
      CGImageSourceGetType(source) as String? == "public.jpeg", CGImageSourceGetCount(source) == 1,
      let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString:Any],
      let width = properties[kCGImagePropertyPixelWidth] as? NSNumber,
      let height = properties[kCGImagePropertyPixelHeight] as? NSNumber,
      width.intValue > 0, height.intValue > 0, max(width.intValue,height.intValue) <= 1600,
      ((properties[kCGImagePropertyOrientation] as? NSNumber)?.intValue ?? 1) == 1 else { throw Failure.invalidPhoto }
    return source
  }
  private static func crop(_ image: CGImage, _ view: ImageSafetyPhotoInspectionContract.View) throws -> CGImage {
    let rect: CGRect
    do { rect = try view.rect(width:image.width,height:image.height) }
    catch { throw Failure.invalidPhoto }
    guard let crop = image.cropping(to: rect) else { throw Failure.invalidPhoto }
    return crop
  }
  private func checked(_ value: Float, _ domain: ImageSafetyPhotoInspectionContract.ScoreDomain) throws -> Float {
    do { _ = try ImageSafetyPhotoInspectionContract.Score(Double(value),domain:domain).checked(as:domain) }
    catch { throw Failure.invalidOutput }
    return value
  }
  private func metric(_ name: String, _ value: Float, _ domain: ImageSafetyPhotoInspectionContract.ScoreDomain) -> ImageSafetyPhotoInspectionAudit.Metric {
    .init(name:name,value:Double(value),domain:domain)
  }
  private func withAuxiliary<Value>(at url: URL, cancelled: () -> Bool,
                                     operation: (MLModel) throws -> Value) throws -> Value {
    try auxiliary.withResource(key: url.standardizedFileURL.path, priority: .foreground,
      cancelled: cancelled, load: {
        self.trace?.record("performance", "auxiliaryModelLoadStart")
        let config = MLModelConfiguration(); config.computeUnits = .cpuOnly
        let model = try MLModel(contentsOf: url, configuration: config)
        self.trace?.record("performance", "auxiliaryModelLoadEnd")
        return model
      }, operation: operation)
  }

  private func xsPositive(_ data: Data, correction:Float, pornRecall:Float=0, cancelled: () -> Bool) throws -> Bool {
    trace?.record("xs","start")
    var eligibleCornerOnly = false
    let original:Bool = try withAuxiliary(at: assets.xs, cancelled: cancelled) { model in
      guard model.modelDescription.inputDescriptionsByName.count == 1,
        let constraint = model.modelDescription.inputDescriptionsByName["image"]?.multiArrayConstraint,
        constraint.shape.map({$0.intValue}) == [1,3,224,224], constraint.dataType == .float32,
        model.modelDescription.outputDescriptionsByName["probabilities"] != nil else { throw Failure.invalidAsset }
      let source = try Self.source(data)
      guard let image = CGImageSourceCreateImageAtIndex(source,0,
        [kCGImageSourceShouldCacheImmediately:true] as CFDictionary) else { throw Failure.invalidPhoto }
      let center = try Self.crop(image,.center70)
      let input = try MLMultiArray(shape:[1,3,224,224],dataType:.float32)
      guard input.strides.map({$0.intValue}) == [3*224*224,224*224,224,1] else { throw Failure.invalidOutput }
      let provider = try MLDictionaryFeatureProvider(dictionary:["image":MLFeatureValue(multiArray:input)])
      var primaryNSFW=0.0
      func predict(_ view: CGImage, _ kind: ImageSafetyPhotoInspectionContract.View) throws -> Double {
        if cancelled() { throw Failure.cancelled }
        let count = 224*224
        var rgba = [UInt8](repeating:0,count:count*4)
        let drawn = rgba.withUnsafeMutableBytes { buffer -> Bool in
          guard let context = CGContext(data:buffer.baseAddress,width:224,height:224,bitsPerComponent:8,
            bytesPerRow:224*4,space:CGColorSpaceCreateDeviceRGB(),
            bitmapInfo:CGImageAlphaInfo.premultipliedLast.rawValue|CGBitmapInfo.byteOrder32Big.rawValue) else { return false }
          context.interpolationQuality = .high
          context.draw(view,in:CGRect(x:0,y:0,width:224,height:224)); return true
        }
        guard drawn else { throw Failure.invalidPhoto }
        let pointer = input.dataPointer.bindMemory(to:Float.self,capacity:count*3)
        for c in 0..<3 { for i in 0..<count { pointer[c*count+i] = Float(rgba[i*4+c]) } }
        if cancelled() { throw Failure.cancelled }
        let output = try model.prediction(from:provider)
        if cancelled() { throw Failure.cancelled }
        guard let probabilities = output.featureValue(for:"probabilities")?.multiArrayValue,
          probabilities.shape.map({$0.intValue}) == [1,3], probabilities.dataType == .float32 else { throw Failure.invalidOutput }
        let scores = (0..<3).map { probabilities[$0].doubleValue }
        guard scores.allSatisfy({$0.isFinite && $0>=0 && $0<=1}), abs(scores.reduce(0,+)-1)<1e-4 else { throw Failure.invalidOutput }
        primaryNSFW=max(primaryNSFW,scores[1])
        let p=min(1-1e-7,max(1e-7,scores[0]))
        let adjusted=1/(1+exp(-(log(p/(1-p))-Double(correction))))
        _ = try ImageSafetyPhotoInspectionContract.Score(adjusted,domain:.probability).checked(as:.probability)
        trace?.record("xs","viewProbability",view:kind,
          rect:try kind.rect(width:image.width,height:image.height),metrics:[
            .init(name:"rawNSFL",value:scores[0],domain:.probability),
            .init(name:"rawNSFW",value:scores[1],domain:.probability),
            .init(name:"correction",value:Double(correction),domain:.logit),
            .init(name:"adjustedNSFL",value:adjusted,domain:.probability)])
        return adjusted // Learned negative-only NSFL correction; cutoff remains0.1.
      }
      let base = max(try predict(image,.whole),try predict(center,.center70))
      if base >= 0.1 { trace?.record("xs","wholeOrCenterProbability>=0.1");return true }
      // Independent XS色情 evidence can confirm unseen-scene DINO recall at the same cutoff.
      // The caller retains the existing normal-context protection; corner expansion remains NSFL-only.
      // A low posterior is not independent confirmation. Use the same signed
      // tanh(logit/2) == 2*p-1 convention as the other DINO/semantic heads.
      let confirmation=min(2*Float(primaryNSFW)-1,2*pornRecall-1)
      _ = try checked(confirmation,.signedUnitMargin)
      trace?.record("xs","independentConfirmation",metrics:[metric("signedConfirmation",confirmation,.signedUnitMargin)])
      if confirmation>Float(0.1) {trace?.record("xs","confirmationSigned>0.1");return true}
      if base >= 0.03 && base < 0.1 {
        var marginalHit = false
        for view in Self.cornerViews {
          let score = try autoreleasepool { try predict(Self.crop(image,view),view) }
          if score >= 0.1 {
            // Whole/center positives and independent pornography confirmation above are untouched.
            // Check every corner; one stronger injury response keeps the original block.
            if score >= 0.2 || pornRecall >= Float(0.1) { trace?.record("xs","strongCornerOrRecall",view:view);return true }
            marginalHit = true
          }
        }
        if marginalHit { trace?.record("xs","marginalCornerOnly");eligibleCornerOnly = true; return true }
      }
      return false
    }
    if original && eligibleCornerOnly {
      let normal = try heads.withResource(key:assets.version,priority:.foreground,cancelled:cancelled,load:{try Heads(self.assets)}) { models in
        let views = try semanticEmbeddings(data,cancelled:cancelled).map {try models.upperbodySemantic.score(embedding:$0)}
        for (index,value) in views.enumerated() {
          _ = try checked(value.score,.signedUnitMargin)
          trace?.record("upperbodySemantic","normalGroup\(value.normalIndex)",view:ImageSafetyPhotoInspectionContract.semanticOrder[index],metrics:[metric("margin",value.score,.signedUnitMargin)])
        }
        researchEvidence["upperbodySemanticViews"] = views.map {v in ["score":v.score,"riskIndex":v.riskIndex,"normalIndex":v.normalIndex,"riskCosine":v.riskCosine,"normalCosine":v.normalCosine] as [String:Any]}
        return views.count==2 && views.allSatisfy { [4,5].contains($0.normalIndex) && $0.score < Float(-0.8) }
      }
      if normal { trace?.record("xs","upperbodyCornerProtection");researchEvidence["upperbodyCornerCorrected"] = true;return false }
    }
    return original
  }

  private func makeBaseContext(_ data:Data,cancelled:()->Bool)throws->BaseContext {
    trace?.record("context","start")
    return try residuals.withResource(key:assets.version+"-blood360",priority:.foreground,cancelled:cancelled,load:{
      try BloodContextModel(data:Self.read(assets.bloodContext))
    }) { residual in
      try ModelBroker.shared.withForegroundVisionModel(at:assets.dino,cancelled:cancelled) { model in
      let source = try Self.source(data)
      guard let image = CGImageSourceCreateThumbnailAtIndex(source,0,[
        kCGImageSourceCreateThumbnailFromImageAlways:true,
        kCGImageSourceCreateThumbnailWithTransform:true,
        kCGImageSourceThumbnailMaxPixelSize:1024,
        kCGImageSourceShouldCacheImmediately:true
      ] as CFDictionary) else { throw Failure.invalidPhoto }
      let center = try Self.crop(image,.center70)
      trace?.record("preprocessing","dinoThumbnail1024",view:.whole,rect:try ImageSafetyPhotoInspectionContract.View.whole.rect(width:image.width,height:image.height))
      trace?.record("preprocessing","dinoCenter70",view:.center70,rect:try ImageSafetyPhotoInspectionContract.View.center70.rect(width:image.width,height:image.height))
      let centerFeatures = try dinoFeatures(center, model:model, cancelled:cancelled)
      let wholeFeatures = try dinoFeatures(image, model:model, cancelled:cancelled)

        let c=try residual.value.correction(whole:wholeFeatures.pooling,center:centerFeatures.pooling)
        if cancelled(){throw Failure.cancelled}
        let normalScore = try normalResiduals.withResource(key:assets.version+"-normal377",priority:.foreground,cancelled:cancelled,load:{
          try BloodContextModel(data:Self.read(assets.normalContext))
        }) { contextModel in
          try contextModel.value.correction(whole:wholeFeatures.pooling,center:centerFeatures.pooling).gore
        }
        let reviewScore = try reviewResiduals.withResource(key:assets.version+"-review11",priority:.foreground,cancelled:cancelled,load:{
          try BloodContextModel(data:Self.read(assets.reviewedContext))
        }) { model in try model.value.correction(whole:wholeFeatures.pooling,center:centerFeatures.pooling).gore }
        let recall12Evidence = try recall12Residuals.withResource(key:assets.version+"-recall12",priority:.foreground,cancelled:cancelled,load:{
          try BloodContextModel(data:Self.read(assets.recall12Context))
        }) { model in try model.value.recallEvidence(whole:wholeFeatures.pooling,center:centerFeatures.pooling) }
        let recall12Score=recall12Evidence.gated
        let newRecallScore = try newRecallResiduals.withResource(key:assets.version+"-new33",priority:.foreground,cancelled:cancelled,load:{
          try BloodContextModel(data:Self.read(assets.newRecallContext))
        }) { model in try model.value.recallProbability(whole:wholeFeatures.pooling,center:centerFeatures.pooling) }
        let recallScore = try recallResiduals.withResource(key:assets.version+"-recall394",priority:.foreground,cancelled:cancelled,load:{
          try BloodContextModel(data:Self.read(assets.recallContext))
        }) { model in try model.value.recallProbability(whole:wholeFeatures.pooling,center:centerFeatures.pooling) }
        for value in [recall12Score,newRecallScore,recallScore,recall12Evidence.raw] { _ = try checked(value,.probability) }
        trace?.record("context","recallProbability>0.1",metrics:[metric("recall12",recall12Score,.probability),metric("newRecall",newRecallScore,.probability),metric("recall",recallScore,.probability),metric("ungatedRecall12",recall12Evidence.raw,.probability)])
        trace?.record("context","normalLogit>log9",metrics:[metric("normal",normalScore,.logit),metric("reviewedNormal",reviewScore,.logit),metric("goreCorrection",c.gore,.logit),metric("xsCorrection",c.xs,.logit)])
        return BaseContext(image:image,whole:wholeFeatures,center:centerFeatures,gore:c.gore,xs:c.xs,normalAllowed:(Double(normalScore)>log(9.0) || Double(reviewScore)>log(9.0)),recallPositive:(recallScore>Float(0.1) || newRecallScore>Float(0.1) || recall12Score>Float(0.1)),ungatedRecall:recall12Evidence.raw)
      }
    }
  }

  private func dinoPositive(_ data: Data, models: Heads, context:BaseContext, cancelled: () -> Bool) throws -> Bool {
    trace?.record("dino","start")
    return try ModelBroker.shared.withForegroundVisionModel(at:assets.dino,cancelled:cancelled) { model in
      let image=context.image,wholeFeatures=context.whole,centerFeatures=context.center
      var patches = wholeFeatures.patches + centerFeatures.patches
      let decision = try models.body.classify(centerPooling:centerFeatures.pooling,
        wholePooling:wholeFeatures.pooling,unitPatches:patches,cancelled:cancelled)
      if cancelled() { throw Failure.cancelled }
      trace?.record("body",decision.route,view:decision.winningView == 0 ? .whole : .center70,metrics:
        decision.globalScores.enumerated().map{.init(name:"expert\($0.offset)",value:$0.element,domain:.decisionMargin)} +
        [.init(name:"prior",value:decision.priorScore,domain:.decisionMargin),.init(name:"local",value:decision.localScore,domain:.decisionMargin)])
      if decision.sensitive, !context.normalAllowed { trace?.record("decision","bodyPositive");return true }
      let nac = try checked(models.nac.predict(unitPatches:patches),.decisionMargin)
      trace?.record("nac","decisionMargin>0.1",metrics:[metric("margin",nac,.decisionMargin)])
      if nac > Float(0.1), !context.normalAllowed { trace?.record("decision","nacPositive");return true }
      for view in Self.cornerViews {
        if cancelled() { throw Failure.cancelled }
        let features = try autoreleasepool { try dinoFeatures(Self.crop(image,view),model:model,cancelled:cancelled) }
        trace?.record("preprocessing","dinoCorner",view:view,rect:try view.rect(width:image.width,height:image.height))
        patches.append(contentsOf:features.patches)
      }
      if cancelled() { throw Failure.cancelled }
      let gore = try models.gore.predict(unitPatches:patches)
      let original=try ImageSafetyGoreRecallFusion.score(local:gore.originalScore,context:gore.score)
      trace?.record("gore","uncorrectedSignedFusion",metrics:[metric("local",gore.originalScore,.signedUnitMargin),metric("context",gore.score,.signedUnitMargin),metric("fusion",original,.signedUnitMargin)])
      if original<=Float(0.1){trace?.record("gore","uncorrectedFusion<=0.1");return false}
      let adjusted=tanhf((gore.originalLogit+gore.correction-context.gore)/2)
      let score = try ImageSafetyGoreRecallFusion.score(local:gore.originalScore,context:adjusted)
      _ = try checked(score,.signedUnitMargin)
      trace?.record("gore","signedFusion>0.1",metrics:[metric("local",gore.originalScore,.signedUnitMargin),metric("uncorrectedFusion",original,.signedUnitMargin),metric("rawLogit",gore.originalLogit,.logit),metric("headCorrection",gore.correction,.logit),metric("contextCorrection",context.gore,.logit),metric("adjustedContext",adjusted,.signedUnitMargin),metric("final",score,.signedUnitMargin)])
      if cancelled() { throw Failure.cancelled }
      researchEvidence["goreBefore"] = score
      researchEvidence["goreLocal"] = gore.originalScore
      if score > Float(0.1), !context.normalAllowed, gore.originalScore <= Float(0.9) {
        let embeddings = try semanticEmbeddings(data, cancelled:cancelled)
        let views = try embeddings.enumerated().map { index,embedding in
          let value = try models.bloodSemantic.score(embedding:embedding)
          _ = try checked(value.score,.signedUnitMargin)
          trace?.record("bloodSemantic","normalGroup\(value.normalIndex)",view:ImageSafetyPhotoInspectionContract.semanticOrder[index],metrics:[metric("signedMargin",value.score,.signedUnitMargin)])
          return value
        }
        researchEvidence["bloodSemanticViews"] = views.enumerated().map { index,v in ["viewIndex":index,"score":v.score,"normalIndex":v.normalIndex,"riskIndex":v.riskIndex,"normalCosine":v.normalCosine,"riskCosine":v.riskCosine] as [String:Any] }
        // Strict agreement of both views on an explicit healthy-pet group.
        // Negative terror evidence alone is never a release signal.
        if views.count == 2, views.allSatisfy({ (4...6).contains($0.normalIndex) && $0.score < Float(-0.8) }) {
          researchEvidence["guardApplied"] = true
          trace?.record("gore","healthyPetSemanticProtection")
          researchEvidence["goreAfter"] = min(score,views.map{$0.score}.max()!)
          return false
        }
      }
      researchEvidence["guardApplied"] = false
      researchEvidence["goreAfter"] = score
      let positive=score > Float(0.1) && !context.normalAllowed
      trace?.record("gore",positive ? "positive" : "negativeOrProtected")
      return positive
    }
  }
  private func dinoFeatures(_ image: CGImage, model: VNCoreMLModel,
                            cancelled: () -> Bool) throws -> Features {
    if cancelled() { throw Failure.cancelled }
    let request = VNCoreMLRequest(model:model);request.imageCropAndScaleOption = .scaleFill
    trace?.record("performance", "dinoPredictionStart")
    try VNImageRequestHandler(cgImage:image,options:[:]).perform([request])
    trace?.record("performance", "dinoPredictionEnd")
    if cancelled() { throw Failure.cancelled }
    guard let observations = request.results as? [VNCoreMLFeatureValueObservation],
      let descriptor = observations.first(where:{$0.featureName=="descriptor"})?.featureValue.multiArrayValue,
      descriptor.count == 384, descriptor.shape.map({$0.intValue}) == [1,384],
      let tokens = observations.first(where:{$0.featureName=="sequence_output_cast_fp16"})?.featureValue.multiArrayValue,
      tokens.count == 257*384, tokens.shape.map({$0.intValue}) == [1,257,384],
      tokens.dataType == .float16, tokens.strides.suffix(2).map({$0.intValue}) == [384,1] else { throw Failure.invalidOutput }
    // Keep Double normalization and accumulation order unchanged. Patch output
    // is contiguous FP16; avoid boxing every scalar through NSNumber.
    let tokenValues = tokens.dataPointer.bindMemory(to: Float16.self, capacity: 257*384)
    defer { withExtendedLifetime(tokens) {} }
    let rawDescriptor = (0..<384).map { descriptor[$0].doubleValue }
    guard rawDescriptor.allSatisfy({$0.isFinite}) else { throw Failure.invalidOutput }
    let descriptorNorm = sqrt(rawDescriptor.reduce(0) {$0+$1*$1})
    guard descriptorNorm.isFinite, descriptorNorm>0 else { throw Failure.invalidOutput }
    var mean = [Double](repeating:0,count:384), squares = mean, patches=[Float]()
    patches.reserveCapacity(256*384)
    for token in 1...256 {
      var row = [Double](repeating:0,count:384), norm = 0.0
      for channel in 0..<384 {
        let value = Double(tokenValues[token*384+channel])
        guard value.isFinite else { throw Failure.invalidOutput }
        mean[channel] += value/256; squares[channel] += value*value/256
        row[channel] = value; norm += value*value
      }
      let length = sqrt(norm)
      guard length.isFinite, length>0 else { throw Failure.invalidOutput }
      patches.append(contentsOf:row.map {Float($0/length)})
    }
    let meanNorm = sqrt(mean.reduce(0) {$0+$1*$1})
    guard meanNorm.isFinite, meanNorm>0 else { throw Failure.invalidOutput }
    let deviation = (0..<384).map {sqrt(max(0,squares[$0]-mean[$0]*mean[$0]))}
    let normalizedDescriptor: [Double] = rawDescriptor.map { $0 / descriptorNorm }
    let normalizedMean: [Double] = mean.map { $0 / meanNorm }
    let joined: [Double] = normalizedDescriptor + normalizedMean + deviation
    let pooling: [Float] = joined.map { Float($0) }
    guard pooling.allSatisfy({$0.isFinite}), patches.allSatisfy({$0.isFinite}) else { throw Failure.invalidOutput }
    trace?.record("performance", "dinoFeatureProcessingEnd")
    return Features(pooling:pooling,patches:patches)
  }

  private func semanticPositive(_ data: Data, head: ImageSafetySemanticRiskHead,
                                cancelled: () -> Bool) throws -> Bool {
    for (index,embedding) in try semanticEmbeddings(data,cancelled:cancelled).enumerated() {
      let value = try head.score(embedding:embedding)
      _ = try checked(value.score,.signedUnitMargin)
      trace?.record("semantic","signedMargin>0.1",view:ImageSafetyPhotoInspectionContract.semanticOrder[index],metrics:[metric("margin",value.score,.signedUnitMargin)])
      if value.positive { trace?.record("decision","semanticPositive",view:ImageSafetyPhotoInspectionContract.semanticOrder[index]);return true }
    }
    return false
  }
  private func semanticEmbeddings(_ data: Data, cancelled: () -> Bool) throws -> [[Float]] {
    if let cached=semanticCache { trace?.record("semanticFeatures","perInspectionCacheHit");return cached }
    trace?.record("semanticFeatures","computeCenterSquareThenWhole")
    if cancelled() { throw Failure.cancelled }
    let pixels = try ImageSafetySemanticPhotoPreprocessor.pixels(jpeg:data, geometry:trace == nil ? nil : { kind,rect in
      self.trace?.record("preprocessing","semantic224ScaleFill",view:kind,rect:rect)
    })
    guard pixels.count == 2 else { throw Failure.invalidOutput }
    let computed: [[Float]] = try withAuxiliary(at:assets.semantic,cancelled:cancelled) { model in
      guard model.modelDescription.inputDescriptionsByName.count==1,
        let constraint = model.modelDescription.inputDescriptionsByName["pixels"]?.multiArrayConstraint,
        constraint.shape.map({$0.intValue}) == [1,3,224,224], constraint.dataType == .float32,
        model.modelDescription.outputDescriptionsByName["embedding"] != nil else { throw Failure.invalidAsset }
      let input = try MLMultiArray(shape:[1,3,224,224],dataType:.float32)
      guard input.strides.map({$0.intValue}) == [3*224*224,224*224,224,1] else { throw Failure.invalidOutput }
      let provider = try MLDictionaryFeatureProvider(dictionary:["pixels":MLFeatureValue(multiArray:input)])
      var vectors: [[Float]] = []
      for values in pixels {
        if cancelled() { throw Failure.cancelled }
        guard values.count == 3*224*224 else { throw Failure.invalidOutput }
        values.withUnsafeBufferPointer { pointer in _ = memcpy(input.dataPointer,pointer.baseAddress!,values.count*4) }
        let output = try model.prediction(from:provider)
        if cancelled() { throw Failure.cancelled }
        guard let embedding = output.featureValue(for:"embedding")?.multiArrayValue,
          embedding.count == 512, embedding.shape.map({$0.intValue}) == [1,512],
          embedding.dataType == .float32 else { throw Failure.invalidOutput }
        vectors.append((0..<512).map {embedding[$0].floatValue})
      }
      return vectors
    }
    semanticCache = computed
    return computed
  }
  func researchXSOnly(_ data: Data) throws -> [[Double]] {
    let cancelled:()->Bool={false}
    return try withAuxiliary(at: assets.xs, cancelled: cancelled) { model in
      guard model.modelDescription.inputDescriptionsByName.count == 1,
        let constraint = model.modelDescription.inputDescriptionsByName["image"]?.multiArrayConstraint,
        constraint.shape.map({$0.intValue}) == [1,3,224,224], constraint.dataType == .float32,
        model.modelDescription.outputDescriptionsByName["probabilities"] != nil else { throw Failure.invalidAsset }
      let source = try Self.source(data)
      guard let image = CGImageSourceCreateImageAtIndex(source,0,
        [kCGImageSourceShouldCacheImmediately:true] as CFDictionary) else { throw Failure.invalidPhoto }
      let center = try Self.crop(image,.center70)
      let input = try MLMultiArray(shape:[1,3,224,224],dataType:.float32)
      guard input.strides.map({$0.intValue}) == [3*224*224,224*224,224,1] else { throw Failure.invalidOutput }
      let provider = try MLDictionaryFeatureProvider(dictionary:["image":MLFeatureValue(multiArray:input)])
      func predict(_ view: CGImage) throws -> [Double] {
        if cancelled() { throw Failure.cancelled }
        let count = 224*224
        var rgba = [UInt8](repeating:0,count:count*4)
        let drawn = rgba.withUnsafeMutableBytes { buffer -> Bool in
          guard let context = CGContext(data:buffer.baseAddress,width:224,height:224,bitsPerComponent:8,
            bytesPerRow:224*4,space:CGColorSpaceCreateDeviceRGB(),
            bitmapInfo:CGImageAlphaInfo.premultipliedLast.rawValue|CGBitmapInfo.byteOrder32Big.rawValue) else { return false }
          context.interpolationQuality = .high
          context.draw(view,in:CGRect(x:0,y:0,width:224,height:224)); return true
        }
        guard drawn else { throw Failure.invalidPhoto }
        let pointer = input.dataPointer.bindMemory(to:Float.self,capacity:count*3)
        for c in 0..<3 { for i in 0..<count { pointer[c*count+i] = Float(rgba[i*4+c]) } }
        if cancelled() { throw Failure.cancelled }
        let output = try model.prediction(from:provider)
        if cancelled() { throw Failure.cancelled }
        guard let probabilities = output.featureValue(for:"probabilities")?.multiArrayValue,
          probabilities.shape.map({$0.intValue}) == [1,3], probabilities.dataType == .float32 else { throw Failure.invalidOutput }
        let scores = (0..<3).map { probabilities[$0].doubleValue }
        guard scores.allSatisfy({$0.isFinite && $0>=0 && $0<=1}), abs(scores.reduce(0,+)-1)<1e-4 else { throw Failure.invalidOutput }
        return scores
      }
      return [try predict(image),try predict(center)]
    }
  }

 func researchConfirmationEvidence(_ data:Data)throws->[String:Any]{
  let c=try makeBaseContext(data,cancelled:{false});let views=try researchXSOnly(data);let p=Float(max(views[0][1],views[1][1]));let signedXS=2*p-1;let signedDINO=2*c.ungatedRecall-1;let score=min(signedXS,signedDINO)
  return ["xsViews":views,"ungatedDINOProbability":c.ungatedRecall,"xsSigned":signedXS,"dinoSigned":signedDINO,"confirmationScore":score,"normalAllowed":c.normalAllowed,"originalGatedRecall":c.recallPositive,"newConfirmationHit":score>Float(0.1) && !c.normalAllowed]
 }

}
