//! Lightweight language detection for Chinese vs English.
//!
//! Used when `source_lang = "auto"` to select the correct
//! translation model direction (en→zh or zh→en).

/// Detected language.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DetectedLang {
    Chinese,
    English,
}

impl DetectedLang {
    /// Get the language code string.
    pub fn code(&self) -> &'static str {
        match self {
            DetectedLang::Chinese => "zh",
            DetectedLang::English => "en",
        }
    }
}

/// Detect whether text is primarily Chinese or English.
///
/// Algorithm: Count CJK Unicode characters vs total non-whitespace characters.
/// If CJK ratio > 30%, classify as Chinese; otherwise English.
///
/// This is intentionally simple and fast (< 0.01ms) — we only need to
/// distinguish Chinese from English for model routing, not handle
/// arbitrary language identification.
pub fn detect_language(text: &str) -> DetectedLang {
    let mut cjk_count = 0u32;
    let mut total_count = 0u32;

    for ch in text.chars() {
        if ch.is_whitespace() {
            continue;
        }
        total_count += 1;
        if is_cjk(ch) {
            cjk_count += 1;
        }
    }

    if total_count == 0 {
        return DetectedLang::English; // Default to English for empty text
    }

    let cjk_ratio = cjk_count as f32 / total_count as f32;
    if cjk_ratio > 0.3 {
        DetectedLang::Chinese
    } else {
        DetectedLang::English
    }
}

/// Resolve `source_lang` to a concrete `DetectedLang`.
///
/// If `source_lang` is `"auto"`, runs detection on the text.
/// Otherwise maps `"zh"` → Chinese, `"en"` → English.
pub fn resolve_source_lang(source_lang: &str, text: &str) -> DetectedLang {
    match source_lang {
        "zh" => DetectedLang::Chinese,
        "en" => DetectedLang::English,
        _ => detect_language(text), // "auto" or any other value
    }
}

/// Check if a character is in the CJK Unicode ranges.
#[inline]
fn is_cjk(ch: char) -> bool {
    matches!(ch,
        '\u{4E00}'..='\u{9FFF}' |   // CJK Unified Ideographs
        '\u{3400}'..='\u{4DBF}' |   // CJK Extension A
        '\u{F900}'..='\u{FAFF}' |   // CJK Compatibility Ideographs
        '\u{2E80}'..='\u{2EFF}' |   // CJK Radicals Supplement
        '\u{3000}'..='\u{303F}' |   // CJK Symbols and Punctuation
        '\u{FF00}'..='\u{FFEF}'     // Halfwidth and Fullwidth Forms
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_english() {
        assert_eq!(
            detect_language("Hello, this is a test sentence."),
            DetectedLang::English
        );
    }

    #[test]
    fn test_detect_chinese() {
        assert_eq!(
            detect_language("你好，这是一个测试句子。"),
            DetectedLang::Chinese
        );
    }

    #[test]
    fn test_detect_mixed_mostly_chinese() {
        assert_eq!(
            detect_language("这是一个包含English的中文句子测试"),
            DetectedLang::Chinese
        );
    }

    #[test]
    fn test_detect_mixed_mostly_english() {
        assert_eq!(
            detect_language("This is an English sentence with 中文"),
            DetectedLang::English
        );
    }

    #[test]
    fn test_detect_empty() {
        assert_eq!(detect_language(""), DetectedLang::English);
    }

    #[test]
    fn test_resolve_explicit() {
        assert_eq!(resolve_source_lang("zh", "hello"), DetectedLang::Chinese);
        assert_eq!(resolve_source_lang("en", "你好"), DetectedLang::English);
    }

    #[test]
    fn test_resolve_auto() {
        assert_eq!(
            resolve_source_lang("auto", "Hello world"),
            DetectedLang::English
        );
        assert_eq!(
            resolve_source_lang("auto", "你好世界"),
            DetectedLang::Chinese
        );
    }

    #[test]
    fn test_lang_code() {
        assert_eq!(DetectedLang::Chinese.code(), "zh");
        assert_eq!(DetectedLang::English.code(), "en");
    }
}
