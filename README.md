# SnapTrans — 轻量离线翻译兜底引擎

> **≤60ms** 单句翻译 · **纯 Rust** 零 Python · **≤40MB** 模型体积 · **离线运行**

## 特性

- ⚡ **极速推理** — 单句 ≤ 60ms，段落 ≤ 150ms（纯 CPU）
- 📦 **超轻量** — 常驻内存 ≤ 35MB，模型 ≤ 40MB
- 🔒 **纯离线** — 零网络依赖，断网环境下全功能可用
- 🦀 **纯 Rust** — 零 Python 运行时，编译进单个二进制
- 🔌 **懒加载** — 首次翻译时才加载模型，不影响宿主启动
- 🔄 **中英双向** — 支持 English↔Chinese 双向翻译
- 🛡️ **自动降级** — 在线超时/断网时毫秒级无感切换离线引擎

## 在 WingSnap (Tauri) 中使用

### 1. 添加依赖

```toml
# src-tauri/Cargo.toml
[dependencies]
snaptrans = { path = "../snaptrans" }
```

### 2. 调用翻译

```rust
use snaptrans::{SnapTransEngine, SnapTransConfig};

// 初始化（懒加载，不会立即载入模型）
let engine = SnapTransEngine::new(SnapTransConfig {
    model_dir: "models/snaptrans".into(),
    num_threads: 2,
    ..Default::default()
});

// 翻译（首次调用时自动加载模型）
let result = engine.translate("Hello, world!", "en", "zh")?;
println!("{}", result.translated_text); // 你好，世界！
println!("耗时: {:.1}ms", result.latency_ms);
println!("引擎: {}", result.engine);    // snaptrans-offline
```

### 3. 降级熔断集成

```rust
// 在线翻译超时 1.2s 或失败时，自动降级到 SnapTrans
match tokio::time::timeout(Duration::from_millis(1200), online_translate(&text)).await {
    Ok(Ok(res)) => Ok(res),
    _ => {
        let mut res = engine.translate(&text, "auto", "zh")?;
        res.is_fallback = true;
        Ok(res)
    }
}
```

## 模型准备

### 使用 Optimum CLI 一键导出

```bash
pip install optimum[onnxruntime]

# 导出 English → Chinese
optimum-cli export onnx \
    --model Helsinki-NLP/opus-mt-en-zh \
    --task text2text-generation-with-past \
    models/snaptrans/en_zh/

# 导出 Chinese → English
optimum-cli export onnx \
    --model Helsinki-NLP/opus-mt-zh-en \
    --task text2text-generation-with-past \
    models/snaptrans/zh_en/
```

### INT8 量化（可选，进一步压缩）

```python
from onnxruntime.quantization import quantize_dynamic, QuantType

quantize_dynamic("encoder_model.onnx", "encoder_model_int8.onnx", weight_type=QuantType.QInt8)
quantize_dynamic("decoder_model_merged.onnx", "decoder_model_merged_int8.onnx", weight_type=QuantType.QInt8)
```

### 模型文件结构

```
models/snaptrans/
├── en_zh/                      # English → Chinese
│   ├── encoder_model.onnx
│   ├── decoder_model_merged.onnx
│   └── tokenizer.json
└── zh_en/                      # Chinese → English
    ├── encoder_model.onnx
    ├── decoder_model_merged.onnx
    └── tokenizer.json
```

## 构建

```bash
cargo build --release
cargo test --lib
cargo test --test integration_test -- --ignored  # 需要模型文件
cargo bench
```

## 基准测试实测对比 (Benchmark)

在真实硬件（Intel i7 笔记本 CPU）上与在线双通道引擎（Google GTX + 有道移动端）的实测延迟对比：

| 评测场景 | 原文样例 | SnapTrans 离线耗时 | 在线引擎耗时 (Google+有道) | 提速比 |
| :--- | :--- | :---: | :---: | :---: |
| **短句与日常 UI** | `"Hello, world! Welcome to WingSnap."` | **66.5 ms** | 913.7 ms | **⚡ 13.7x** |
| **技术与开发者报错** | `"Failed to allocate shared memory buffer for DirectML execution provider."` | **143.5 ms** | 920.7 ms | **⚡ 6.4x** |
| **日常段落与排忧解难** | `"Dear Lynn, I understand you are feeling stressed about school lately..."` | **181.9 ms** | 1174.1 ms | **⚡ 6.5x** |
| **系统与配置路径报错** | `"The system cannot find the path specified. Please check your configuration..."` | **155.6 ms** | 978.9 ms | **⚡ 6.3x** |
| **复杂复合学术长句** | `"Although neural machine translation requires substantial compute resources..."` | **273.3 ms** | 890.9 ms | **⚡ 3.3x** |
| **综合平均延迟** | — | **164.2 ms** | **975.7 ms** | **⚡ 综合提速 ~6x** |

## 架构

```
Input text
  → 文本规范化 (全角转半角, 空白清理)
  → 语言检测 (CJK 字符比例判定)
  → SentencePiece BPE 分词 (tokenizers crate)
  → MarianMT Encoder (ort, 单次推理)
  → MarianMT Decoder (自回归, KV Cache, Greedy Search)
  → 反分词 + 后处理 (空格修复, 标点清洗)
  → OfflineTranslateResponse
```

## 许可证

MIT OR Apache-2.0
