import Foundation
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers

enum PhotoEncoder {
  static let commentMaximumLongEdge = 1600
  static let commentMaximumBytes = 614400
  static func encodeComment(_ data: Data) -> Data? {
    guard let source = CGImageSourceCreateWithData(data as CFData, [kCGImageSourceShouldCache: false] as CFDictionary) else { return nil }
    for edge in [commentMaximumLongEdge, 1440, 1280] {
      guard let image = CGImageSourceCreateThumbnailAtIndex(source, 0, [
        kCGImageSourceCreateThumbnailFromImageAlways: true,
        kCGImageSourceCreateThumbnailWithTransform: true,
        kCGImageSourceThumbnailMaxPixelSize: edge,
        kCGImageSourceShouldCacheImmediately: true
      ] as CFDictionary), let bytes = jpeg(image, quality: 0.82, metadata: [:]) else { return nil }
      if bytes.count <= commentMaximumBytes { return bytes }
    }
    return nil
  }

  private static func jpeg(_ image: CGImage, quality: Double, metadata: [CFString: Any]) -> Data? {
    let data = NSMutableData()
    guard let destination = CGImageDestinationCreateWithData(
      data, UTType.jpeg.identifier as CFString, 1, nil
    ) else { return nil }
    var properties = metadata
    properties[kCGImageDestinationLossyCompressionQuality] = quality
    properties[kCGImageDestinationBackgroundColor] = CGColor(gray: 1, alpha: 1)
    CGImageDestinationAddImage(destination, image, properties as CFDictionary)
    guard CGImageDestinationFinalize(destination) else { return nil }
    return data as Data
  }
}
