//! Public types for SnapTrans offline translation.
//!
//! Matches the interface contract specified in PRD §5.1.

use serde::{Deserialize, Serialize};

/// Request for offline translation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OfflineTranslateRequest {
    /// Text to translate.
    pub text: String,
    /// Source language: `"en"`, `"zh"`, or `"auto"` (auto-detect).
    pub source_lang: String,
    /// Target language: `"zh"` or `"en"`.
    pub target_lang: String,
}

/// Response from the offline translation engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OfflineTranslateResponse {
    /// The translated text.
    pub translated_text: String,
    /// Detected or specified source language.
    pub source_lang: String,
    /// Target language.
    pub target_lang: String,
    /// Engine identifier — always `"snaptrans-offline"`.
    pub engine: String,
    /// Inference latency in milliseconds.
    pub latency_ms: f64,
    /// Whether this response was triggered by automatic fallback
    /// due to network timeout or failure.
    pub is_fallback: bool,
}

impl OfflineTranslateResponse {
    /// Create a new response with default engine tag.
    pub fn new(
        translated_text: String,
        source_lang: String,
        target_lang: String,
        latency_ms: f64,
    ) -> Self {
        Self {
            translated_text,
            source_lang,
            target_lang,
            engine: "snaptrans-offline".to_string(),
            latency_ms,
            is_fallback: false,
        }
    }
}
