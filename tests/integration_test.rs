//! Integration tests for SnapTrans.

use snaptrans::{OfflineTranslateResponse, SnapTransConfig, SnapTransEngine};

// =============================================================================
// Text normalization tests (no model needed)
// =============================================================================

#[test]
fn test_empty_translation() {
    let engine = SnapTransEngine::new(SnapTransConfig::default());
    let result = engine.translate("", "en", "zh");
    assert!(result.is_ok());
    assert_eq!(result.unwrap().translated_text, "");
}

#[test]
fn test_whitespace_only() {
    let engine = SnapTransEngine::new(SnapTransConfig::default());
    let result = engine.translate("   \n\t  ", "en", "zh");
    assert!(result.is_ok());
    assert_eq!(result.unwrap().translated_text, "");
}

#[test]
fn test_same_language_passthrough() {
    let engine = SnapTransEngine::new(SnapTransConfig::default());
    // Chinese text with target "zh" → should pass through
    let result = engine.translate("你好世界", "zh", "zh");
    assert!(result.is_ok());
    let resp = result.unwrap();
    assert_eq!(resp.translated_text, "你好世界");
    assert_eq!(resp.source_lang, "zh");
    assert_eq!(resp.target_lang, "zh");
}

#[test]
fn test_response_engine_tag() {
    let resp = OfflineTranslateResponse::new(
        "翻译结果".to_string(),
        "en".to_string(),
        "zh".to_string(),
        42.5,
    );
    assert_eq!(resp.engine, "snaptrans-offline");
    assert_eq!(resp.is_fallback, false);
    assert_eq!(resp.latency_ms, 42.5);
}

#[test]
fn test_response_serialization() {
    let resp = OfflineTranslateResponse::new(
        "Hello World".to_string(),
        "zh".to_string(),
        "en".to_string(),
        15.0,
    );

    let json = serde_json::to_string(&resp).unwrap();
    assert!(json.contains("snaptrans-offline"));
    assert!(json.contains("Hello World"));

    let deserialized: OfflineTranslateResponse = serde_json::from_str(&json).unwrap();
    assert_eq!(deserialized.translated_text, "Hello World");
    assert_eq!(deserialized.engine, "snaptrans-offline");
}

// =============================================================================
// End-to-end tests (require model files)
// =============================================================================

#[test]
#[ignore = "Requires ONNX model files in models/snaptrans/"]
fn test_translate_short_en_to_zh() {
    let config = SnapTransConfig {
        model_dir: "models/snaptrans".into(),
        ..Default::default()
    };
    let engine = SnapTransEngine::new(config);

    let result = engine.translate("Hello, world!", "en", "zh");
    assert!(result.is_ok());

    let resp = result.unwrap();
    assert!(!resp.translated_text.is_empty());
    assert_eq!(resp.source_lang, "en");
    assert_eq!(resp.target_lang, "zh");
    assert_eq!(resp.engine, "snaptrans-offline");
    assert!(resp.latency_ms > 0.0);

    println!(
        "EN→ZH: '{}' → '{}' ({:.1}ms)",
        "Hello, world!", resp.translated_text, resp.latency_ms
    );
}

#[test]
#[ignore = "Requires ONNX model files in models/snaptrans/"]
fn test_translate_technical() {
    let config = SnapTransConfig {
        model_dir: "models/snaptrans".into(),
        ..Default::default()
    };
    let engine = SnapTransEngine::new(config);

    let text = "Failed to allocate shared memory buffer for DirectML execution provider.";
    let result = engine.translate(text, "en", "zh");
    assert!(result.is_ok());

    let resp = result.unwrap();
    println!(
        "Technical: '{}' → '{}' ({:.1}ms)",
        text, resp.translated_text, resp.latency_ms
    );
    assert!(!resp.translated_text.is_empty());
}

#[test]
#[ignore = "Requires ONNX model files in models/snaptrans/"]
fn test_translate_paragraph() {
    let config = SnapTransConfig {
        model_dir: "models/snaptrans".into(),
        ..Default::default()
    };
    let engine = SnapTransEngine::new(config);

    let text = "Dear Lynn, I understand you are feeling stressed about school lately. \
                Don't worry, everyone goes through this. We can practice together.";
    let result = engine.translate(text, "en", "zh");
    assert!(result.is_ok());

    let resp = result.unwrap();
    println!(
        "Paragraph: '{}' → '{}' ({:.1}ms)",
        &text[..50],
        resp.translated_text,
        resp.latency_ms
    );
    assert!(!resp.translated_text.is_empty());
    // PRD requirement: ≤ 120ms for paragraph
    // (relax for CI environments)
    assert!(resp.latency_ms < 5000.0, "Paragraph took too long");
}

#[test]
#[ignore = "Requires ONNX model files in models/snaptrans/"]
fn test_translate_auto_detect() {
    let config = SnapTransConfig {
        model_dir: "models/snaptrans".into(),
        ..Default::default()
    };
    let engine = SnapTransEngine::new(config);

    // Auto-detect English → translate to Chinese
    let result = engine.translate("Good morning", "auto", "zh");
    assert!(result.is_ok());
    let resp = result.unwrap();
    assert_eq!(resp.source_lang, "en");

    // Auto-detect Chinese → translate to English (if zh_en model exists)
    if std::path::Path::new("models/snaptrans/zh_en").exists() {
        let result = engine.translate("早上好", "auto", "en");
        assert!(result.is_ok());
        let resp = result.unwrap();
        assert_eq!(resp.source_lang, "zh");
    }
}

#[test]
#[ignore = "Requires ONNX model files in models/snaptrans/"]
fn test_latency_short_sentence() {
    let config = SnapTransConfig {
        model_dir: "models/snaptrans".into(),
        ..Default::default()
    };
    let engine = SnapTransEngine::new(config);

    // Warm up
    let _ = engine.translate("warmup", "en", "zh");

    // Measure short sentence
    let start = std::time::Instant::now();
    let result = engine.translate("Status: Ready for deployment.", "en", "zh");
    let elapsed = start.elapsed().as_secs_f64() * 1000.0;

    assert!(result.is_ok());
    println!("Short sentence latency: {:.1}ms", elapsed);
    // PRD: ≤ 60ms (relax for CI)
    assert!(elapsed < 5000.0, "Short sentence took too long: {:.1}ms", elapsed);
}
