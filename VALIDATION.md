# Export validation — v1.0.0

Validated on 2026-11-02, Apple Silicon macOS, Apple Swift 6.4.2, CPU Core ML/Vision path.

| Check | Result |
|---|---|
| Independent Swift package release build | Passed |
| Independent Core ML compilation of all three released `.mlpackage` models | Passed |
| Standalone Swift unit tests | 4 passed, 0 failed |
| Content-path historical image replay against the original implementation | 681/681 identical sensitive-content verdicts |
| Inputs excluded from content parity | 19 images short-circuited by the original application's QR module |
| Original raw parameter reconstruction before this export | 12/12 SHA256 identical after lossless shared-row storage |
| Original malformed/truncated storage checks | 7 cases rejected |
| Public source/release payload scan | No private filesystem paths, service endpoints, credentials or raw media found |

The public package intentionally omits QR detection. Comparing its content output
with an application's combined QR-or-content rejection would not be a valid parity
test. The 681-image replay compares **content to content**, on inputs which really
reached the original content model.

These are packaging/integration regression checks on previously used image sources,
not a new independently sampled accuracy benchmark. They do not establish 100%
recall, a deployment-wide false-positive rate, or performance on arbitrary unseen
images. Raw input photos, per-image outputs, identities and annotation records are
not published. Upstream SafetyXS accuracy numbers in its retained model card apply
to that upstream model/dataset, not this combined detector.

No numeric changes, quantization or retraining were made for the public export.
Changes are standalone packaging, neutral identifiers, extraction of the existing
resource gate, and a small public API/CLI. The existing image preparation, crop
orders, CPU configuration, model weights and per-head threshold arithmetic remain.

The Swift library targets iOS 17+ / macOS 14+. This export was replayed through the
macOS CLI; it is not a fresh iPhone device, thermal, memory or energy validation of
this standalone Swift package. Consumers should run their own domain-specific
quality and device tests before automatically rejecting user content.
