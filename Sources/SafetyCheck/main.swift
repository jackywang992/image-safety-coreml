import Foundation
import ImageSafety

func run() -> Int32 {
    var arguments = Array(CommandLine.arguments.dropFirst())
    guard arguments.count >= 3, arguments.removeFirst() == "--models" else {
        fputs("Usage: safety-check --models Models.bundle IMAGE [IMAGE ...]\n       safety-check --models Models.bundle --input-list paths.txt\n", stderr)
        return 2
    }
    let models = URL(fileURLWithPath: arguments.removeFirst())
    do {
        if arguments.first == "--input-list" {
            guard arguments.count == 2 else { return 2 }
            arguments = try String(contentsOfFile: arguments[1], encoding: .utf8)
                .split(separator: "\n").map(String.init).filter { !$0.isEmpty }
        }
        let classifier = try SafetyClassifier(modelsBundleURL: models)
        defer { classifier.releaseWhenIdle() }
        for path in arguments {
            let output: [String: Any] = try autoreleasepool {
                let url = URL(fileURLWithPath: path)
                let start = ProcessInfo.processInfo.systemUptime
                let result = try classifier.inspect(imageData: Data(contentsOf: url))
                return ["file": url.lastPathComponent, "sensitive": result.sensitive,
                        "modelVersion": result.modelVersion,
                        "milliseconds": (ProcessInfo.processInfo.systemUptime - start) * 1000]
            }
            let data = try JSONSerialization.data(withJSONObject: output, options: [.sortedKeys])
            print(String(decoding: data, as: UTF8.self)); fflush(stdout)
        }
        return 0
    } catch {
        fputs("Screening failed (no allow decision): \(error)\n", stderr)
        return 1
    }
}
DispatchQueue.global(qos: .userInitiated).async { exit(run()) }
dispatchMain()
