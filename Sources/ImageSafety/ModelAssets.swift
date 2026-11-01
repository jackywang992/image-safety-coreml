import Foundation

/// Resource names and identity are part of the admission protocol, not a remote
/// configuration. A missing asset must defer publishing rather than skip a head.
enum ImageSafetyCommentPhotoSafetyAssets {
  static let candidate = "c5c3f711aec21606f2f8d0eec92c8747df3f64efb2a36874e0d8e6fe8f6ee592"
  static let contentVersion = "image-safety-v1-" + candidate
  static let complianceVersion = "image-safety-v1-" + candidate

  static func load(bundle: Bundle = .main) throws -> ImageSafetyCommentPhotoContentGuard.Assets {
    func resource(_ name: String, _ ext: String) throws -> URL {
      guard let url = bundle.url(forResource: name, withExtension: ext) else {
        throw ImageSafetyCommentPhotoContentGuard.Failure.invalidAsset
      }
      return url
    }
    func binary(_ name: String, _ sha: String, _ bytes: Int) throws -> ImageSafetyCommentPhotoContentGuard.Binary {
      .init(url: try resource(name, "bin"), sha256: sha, bytes: bytes)
    }
    return try .init(version: contentVersion,
      dino: resource("ImageSafetyDINODescriptor", "mlmodelc"),
      xs: resource("ImageSafetySafetyXS-FP32", "mlmodelc"),
      semantic: resource("ImageSafetyTinyCLIPVision-FP32", "mlmodelc"),
      body: binary("ImageSafetySensitivePhotoHead", "5015de281aec4e28ab5e0aec632231c382304580e6fffdda8a3caa8a2e4d1063", 47342888),
      nac: binary("ImageSafetyNACRiskHead", "7539844435480b022c8e6f4a3814bf61eb38f0861c9f1a47a7677d6c47f64817", 49436),
      gore: binary("ImageSafetyGoreCompactContext", "aa9677338cdb1355197dc808264d2858affb1543124d37aeec0e3fffed997a11", 152660),
      semanticHead: binary("ImageSafetySemanticRiskHead", "03c0618575ac3255507acf5c67fd968986869b49455a0a14c02692b24d1df4a7", 55328),
      upperbodySemantic: binary("ImageSafetyUpperBodySemantic", "05db2af05ba1fa906040dd5b8adbebceab21a1c9c6c31bc670bdc19e7ee49633", 22560),
      bloodSemantic: binary("ImageSafetyBloodSemantic", "d095a1f72eebeb36f7c767d2fdafe9f61ad94495935a10fd098548804e68531d", 22560),
      bloodContext: binary("ImageSafetyBloodContextResiduals", "624d990dc559afa061e0487b85d27c4bac3da82d0c5d051604b568a3ed2e3cde", 13325357),
      normalContext: binary("ImageSafetyNormalContext", "f511c165e46a60ef8583a931ae4aeef6fa7ac5a687debeea9ece598226060fec", 13556432),
      reviewedContext: binary("ImageSafetyReviewedNormalContext", "62f087488071c9e70e61eec00935fb692495d94b72d61799d2e7ee2e1d5aee33", 13701239),
      recall12Context: binary("ImageSafetyRecall12Context", "08ff6c4157373e8bbb1b182fde0f605c2c2e218ef058847265798c53057010e6", 11331950),
      newRecallContext: binary("ImageSafetyNewRecallContext", "ea379e2cd8d06e317d3a3e23ebc07afe8a1efc73d96414a1de30ed26a9914d67", 11258006),
      recallContext: binary("ImageSafetyRecallContext", "3fdc8c43b4e870a17568b61871aa65d6b59f1e7a8b53905901de1246432a09e9", 11162495))
  }
}
