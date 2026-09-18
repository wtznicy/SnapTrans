//! # SnapTrans — Lightweight Offline Translation Engine
//!
//! Ultra-lightweight, zero-Python, millisecond-latency offline neural machine
//! translation engine for Chinese-English bidirectional translation.
//!
//! ## Features
//! - ⚡ **Fast**: Single sentence ≤ 60ms, paragraph ≤ 150ms on CPU
//! - 📦 **Lightweight**: Models ≤ 40MB, resident memory ≤ 35MB
//! - 🔒 **Offline**: Works fully disconnected, zero network dependency
//! - 🦀 **Pure Rust**: Zero Python runtime, compiles into native binary
//! - 🔌 **Lazy Loading**: Models loaded on first use, zero startup cost
//!
//! ## Quick Start
//! ```no_run
//! use snaptrans::{SnapTransEngine, SnapTransConfig};
//!
//! let engine = SnapTransEngine::new(SnapTransConfig {
//!     model_dir: "models/snaptrans".into(),
//!     ..Default::default()
//! });
//!
//! let result = engine.translate("Hello, world!", "en", "zh").unwrap();
//! println!("{}", result.translated_text); // 你好，世界！
//! println!("Latency: {:.1}ms", result.latency_ms);
//! ```
//!
//! ## Architecture
//! ```text
//! Input text
//!   → Text normalization (full-width cleanup, whitespace collapse)
//!   → Language detection (CJK ratio heuristic for "auto" mode)
//!   → SentencePiece BPE tokenization (pure Rust tokenizers crate)
//!   → MarianMT Encoder (ONNX Runtime, single-pass)
//!   → MarianMT Decoder (autoregressive, KV Cache, greedy search)
//!   → Detokenization + post-processing
//!   → OfflineTranslateResponse { translated_text, latency_ms, ... }
//! ```

mod config;
mod decoder;
mod encoder;
mod engine;
mod error;
mod generation;
mod lang_detect;
mod postprocess;
mod text_normalize;
mod tokenizer;
mod types;

// Public API
pub use config::SnapTransConfig;
pub use engine::SnapTransEngine;
pub use error::SnapTransError;
pub use types::{OfflineTranslateRequest, OfflineTranslateResponse};

/// Convenience function for one-shot translation.
///
/// Creates a temporary engine, loads models, and translates.
/// For repeated use, create and reuse a `SnapTransEngine` instead.
///
/// # Arguments
/// * `text` — Text to translate.
/// * `source_lang` — `"en"`, `"zh"`, or `"auto"`.
/// * `target_lang` — `"zh"` or `"en"`.
pub fn translate(
    text: &str,
    source_lang: &str,
    target_lang: &str,
) -> Result<OfflineTranslateResponse, String> {
    let config = SnapTransConfig::default();
    let engine = SnapTransEngine::new(config);
    engine.translate(text, source_lang, target_lang).map_err(|e| e.to_string())
}
