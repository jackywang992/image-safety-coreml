# Image Safety Core ML

本地运行的 **色情/裸露与血腥、严重伤害图片筛查**。提供训练后的参数、完整 Core ML 权重、Swift 库及 macOS 命令行示例。**不需要 API Key，不上传图片。**

> 这是 DINOv2 + SafetyXS + TinyCLIP 和本地训练读出/参考库组成的检测链路，不是从零训练的单一基础模型。可能误报和漏检。输出 `sensitive` 是筛查结果，不代表合规结论或人工审核；错误时没有“安全放行”结果。

## 直接试用（macOS）

需要 macOS 14+、Xcode / Command Line Tools、Python 3.10+（标准库即可），以及能够访问 GitHub Releases 的网络。第一次下载权重后，推理可完全离线。

```bash
git clone https://github.com/jackywang992/image-safety-coreml.git
cd image-safety-coreml
python3 scripts/prepare_models.py
swift run -c release safety-check --models build/ImageSafetyModels.bundle /path/to/photo.jpg
```

输出为每张图片一行 JSON：

```json
{"file":"photo.jpg","sensitive":false,"modelVersion":"image-safety-v1-...","milliseconds":300}
```

示例中的耗时仅示意，并非性能承诺。`sensitive=true` 表示触发任一色情/裸露或血腥伤害规则；当前 API 不输出可靠的逐类归因或校准概率。

批量处理：

```bash
swift run -c release safety-check --models build/ImageSafetyModels.bundle --input-list paths.txt
```

`paths.txt` 每行一个本地路径。检测失败时命令退出非零，调用方应延后/复核，而不是自动视为正常。

## 完整权重下载

下载 [v1.0.0 Release](https://github.com/jackywang992/image-safety-coreml/releases/tag/v1.0.0) 中的 `image-safety-models-v1.0.0.zip`。它包含三个 `.mlpackage`、12 份训练参数及共享特征库、SHA256 清单和许可证，**不是占位文件或只有分类头的空壳**。

脚本自动校验下载大小、整包 SHA256、文件清单和逐文件 SHA256，再通过 Apple `coremlcompiler` 编译。也可手动下载后离线准备：

```bash
python3 scripts/prepare_models.py --archive image-safety-models-v1.0.0.zip
```

| 资源 | 约未压缩体积 |
|---|---:|
| DINOv2-small FP16 通用图像特征 | 41.4 MiB |
| SafetyXS Core ML 编码器/读出 | 11.7 MiB |
| TinyCLIP Core ML 图像编码器 | 31.7 MiB |
| 全部本地训练参数（无损共享存储） | 59.1 MiB |
| 模型和参数合计 | 约 144 MiB |

六份上下文参考库中的重复 FP32 行采用无损共享存储，参数约 116.33 → 59.12 MiB。使用时按原顺序恢复并核对原始 SHA256，**没有量化、删样本或改变权重**。安装资源变小不等于展开后的推理内存减半。

## Swift / iOS 集成

将本项目作为本地或 GitHub Swift Package 依赖，选择 `ImageSafety` 产品。为 iOS 准备独立的资源 Bundle，再加入 App 的 Copy Bundle Resources：

```bash
python3 scripts/prepare_models.py --platform iOS --output build/iOS/ImageSafetyModels.bundle
```

```swift
import ImageSafety
let url = Bundle.main.url(forResource: "ImageSafetyModels", withExtension: "bundle")!
let classifier = try SafetyClassifier(modelsBundleURL: url)
DispatchQueue.global(qos: .userInitiated).async {
    do {
        let result = try classifier.inspect(imageData: photoData)
        print(result.sensitive)
    } catch {
        // 延后或人工复核；不要将检测失败转换成“图片正常”。
    }
}
```

模型按需加载，调用必须在后台线程。可复用 classifier；不再需要时可调用 `releaseWhenIdle()`。默认 CPU 路径、裁剪和计算顺序固定；本项目未验证任意修改后的 GPU/ANE 或跨平台 ONNX 结果。

输入先按 EXIF 方向变换，生成最长边不超过 1600px、质量 0.82、最大 600 KiB 的无拍摄元数据 JPEG；过大图片会依次尝试 1440/1280。模型内部仍执行各自固定的缩放/裁剪。直接调用 `inspectPreparedJPEG` 时也必须遵守此契约。

## 效果、验证和边界

- 固定有符号判定阈值为 **0.1**，不是“10% 色情概率”。不同分支的概率、边距、logit 不可混用。
- 这套读出针对既定色情裸露和主要血腥严重伤害场景调整；仍可能误拦正常宠物、人物局部或医疗/历史图片。不是所有动物伤害、影视惊悚或复杂边界场景的完整检测器。
- 公开导出保留参数位模式和推理算术，另做独立包编译、资源完整性及与原链路的逐图对照。结果见 [VALIDATION.md](VALIDATION.md)。回归一致性不是未知图片准确率保证。
- **不包含**政治人物识别、二维码检测、宠物识别/同宠归类、OCR、App 账号与上传逻辑。该仓库专注色情裸露与血腥内容筛查。
- 只提供 Core ML / Apple 平台运行路径，没有声称已验证 Windows/Linux/CUDA、ONNX 或 Android 版本。

## 脱敏范围与数据

该公开项目从白名单文件单独导出，没有原 App 的 Git 历史。未分发原始训练/测试照片、逐图标注、相册资产 ID、私有目录、服务器地址、密钥、账号、上传接口或训练日志。源码命名和资源目录已中性化。

训练参数包含推理所需的数值参考特征；“移除私有元数据”不等于对模型进行差分隐私训练，也不构成对所有潜在模型隐私风险的证明。没有为了脱敏修改参数数值。

## 许可证

本项目自研代码及本地训练参数按 [MIT](LICENSE) 授权，可使用、修改和再分发，需保留声明。上游 DINOv2 / SwiftFormer / SafetyXS / TinyCLIP 保留各自的 Apache-2.0 或 MIT 许可，详见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。模型许可证不授予原始训练照片的再分发权。

## 开发

```bash
swift test
```

单元测试不下载权重。完整权重和数值对照须使用相同平台、预处理、资源版本和真实图片另行验证。
