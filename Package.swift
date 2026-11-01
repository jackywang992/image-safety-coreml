// swift-tools-version: 5.9
import PackageDescription
let package = Package(
    name: "ImageSafetyCoreML",
    platforms: [.macOS(.v14), .iOS(.v17)],
    products: [.library(name: "ImageSafety", targets: ["ImageSafety"]),
               .executable(name: "safety-check", targets: ["SafetyCheck"])],
    targets: [.target(name: "ImageSafety"),
              .executableTarget(name: "SafetyCheck", dependencies: ["ImageSafety"]),
              .testTarget(name: "ImageSafetyTests", dependencies: ["ImageSafety"])])
