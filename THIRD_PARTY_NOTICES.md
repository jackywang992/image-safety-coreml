# Third-party attribution

The repository's MIT license covers the original integration code and locally learned
heads/reference parameters. It does not relicense third-party components or grant
rights to the original training images. No original dataset images are distributed.

| Component | Frozen upstream | License | Local changes |
|---|---|---|---|
| DINOv2-small, Meta / Hugging Face | [facebook/dinov2-small](https://huggingface.co/facebook/dinov2-small/tree/ed25f3a31f01632728cabb09d1542f84ab7b0056) | Apache-2.0, [full text](licenses/DINOv2-APACHE-2.0.txt) | Core ML conversion, FP16 weights, existing intermediate token output exposed; no fine-tuning of this backbone |
| SafetyXS, OwenElliott | [image-safety-classifier-xs](https://huggingface.co/OwenElliott/image-safety-classifier-xs/tree/54f4560bd9c5ee92d45dc30418a8f8680e80de6d) | Upstream model card declares MIT; SwiftFormer backbone declares Apache-2.0 | Core ML conversion and adapted NSFL readout; source [model card](licenses/SafetyXS-MODEL-CARD.md) retained |
| TinyCLIP, Microsoft / TinyCLIP contributors | [TinyCLIP-ViT-8M-16-Text-3M-YFCC15M](https://huggingface.co/wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M/tree/a2a8c6eaa2549ad66eb7c31b85022bf58273a26c) | MIT, [full text and notices](licenses/TinyCLIP-MIT.txt) | Core ML image encoder conversion; fixed local text prototypes/readout, no runtime text encoder |

MIT terms are reproduced in LICENSE; Apache-2.0 terms in licenses/DINOv2-APACHE-2.0.txt
also accompany the Apache-licensed backbone components. The SafetyXS upstream model
card is a third-party description: its reported upstream accuracy is not a measured
accuracy of this combined detector. Model licenses do not establish ownership of
underlying training media or independence of our evaluation data.

Apple Core ML, Vision, ImageIO and Accelerate are system dependencies; their binaries
are not redistributed here. A recent Apple SDK/Xcode is required to compile and run.
