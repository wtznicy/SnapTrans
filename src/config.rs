//! Configuration for the SnapTrans engine.

use std::path::PathBuf;

/// Configuration for initializing a `SnapTransEngine`.
pub struct SnapTransConfig {
    /// Path to the model directory containing:
    /// - `encoder_model.onnx` (or INT8 quantized variant)
    /// - `decoder_model_merged.onnx` (or INT8 quantized variant)
    /// - `tokenizer.json` (HuggingFace tokenizer config)
    pub model_dir: PathBuf,

    /// Number of ONNX Runtime inference threads.
    /// Recommended: 1~2 for minimal contention on desktop CPUs.
    pub num_threads: usize,

    /// Maximum number of tokens to generate.
    /// Acts as a safety valve to prevent runaway generation.
    pub max_length: usize,

    /// Repetition penalty factor (> 1.0 penalizes repeats).
    /// Applied to logits of already-generated tokens.
    pub repetition_penalty: f32,

    /// Decoder start token ID.
    /// For MarianMT, this is typically the `<pad>` token (58100 for en-zh).
    pub decoder_start_token_id: i64,

    /// End-of-sequence token ID.
    /// For MarianMT, this is typically `</s>` = 0.
    pub eos_token_id: i64,

    /// Pad token ID.
    /// For MarianMT, this is typically `<pad>` = 58100.
    pub pad_token_id: i64,

    /// Forced BOS (beginning-of-sequence) token ID, if any.
    /// Some models require a language tag token at the start of decoding.
    /// Set to `None` if not needed.
    pub forced_bos_token_id: Option<i64>,
}

impl Default for SnapTransConfig {
    fn default() -> Self {
        Self {
            model_dir: PathBuf::from("models/snaptrans"),
            num_threads: 2,
            max_length: 256,
            repetition_penalty: 1.15,
            decoder_start_token_id: 65000, // <pad> for MarianMT en-zh
            eos_token_id: 0,               // </s> for MarianMT
            pad_token_id: 65000,
            forced_bos_token_id: None,
        }
    }
}
