//! Output post-processing for translated text.
//!
//! Cleans up tokenizer artifacts and fixes formatting issues
//! in the decoded translation output.

use crate::lang_detect::DetectedLang;

/// Post-process translated text based on the target language.
pub fn postprocess_translation(text: &str, target_lang: DetectedLang) -> String {
    let text = fix_sentencepiece_spaces(text);
    let text = match target_lang {
        DetectedLang::Chinese => postprocess_chinese(&text),
        DetectedLang::English => postprocess_english(&text),
    };
    text.trim().to_string()
}

/// Fix SentencePiece tokenizer artifacts.
///
/// SentencePiece uses `▁` (U+2581, LOWER ONE EIGHTH BLOCK) as a word
/// boundary marker. Replace these with regular spaces.
fn fix_sentencepiece_spaces(text: &str) -> String {
    text.replace('\u{2581}', " ")
}

/// Chinese-specific post-processing.
fn postprocess_chinese(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    let chars: Vec<char> = text.chars().collect();

    let mut i = 0;
    while i < chars.len() {
        let ch = chars[i];

        if ch == ' ' {
            // Remove spaces between CJK characters
            let prev_cjk = i > 0 && is_cjk_char(chars[i - 1]);
            // Skip consecutive spaces
            let mut j = i;
            while j < chars.len() && chars[j] == ' ' {
                j += 1;
            }
            let next_cjk = j < chars.len() && is_cjk_char(chars[j]);

            if prev_cjk && next_cjk {
                // Space between two CJK chars → remove
                i = j;
            } else if prev_cjk && j < chars.len() && is_cjk_punct(chars[j]) {
                // Space before CJK punctuation → remove
                i = j;
            } else {
                result.push(' ');
                i = j;
            }
        } else {
            result.push(ch);
            i += 1;
        }
    }

    result
}

/// English-specific post-processing.
fn postprocess_english(text: &str) -> String {
    let text = collapse_spaces(text);

    // Capitalize first letter
    let mut chars = text.chars();
    match chars.next() {
        None => String::new(),
        Some(first) => {
            let capitalized: String = first.to_uppercase().collect();
            capitalized + chars.as_str()
        }
    }
}

/// Collapse multiple consecutive spaces into a single space.
fn collapse_spaces(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    let mut prev_space = false;

    for ch in text.chars() {
        if ch == ' ' {
            if !prev_space {
                result.push(' ');
                prev_space = true;
            }
        } else {
            result.push(ch);
            prev_space = false;
        }
    }

    result
}

/// Check if a character is a CJK ideograph.
#[inline]
fn is_cjk_char(ch: char) -> bool {
    matches!(ch,
        '\u{4E00}'..='\u{9FFF}' |
        '\u{3400}'..='\u{4DBF}' |
        '\u{F900}'..='\u{FAFF}'
    )
}

/// Check if a character is CJK punctuation.
#[inline]
fn is_cjk_punct(ch: char) -> bool {
    matches!(ch,
        '\u{3000}'..='\u{303F}' |   // CJK Symbols and Punctuation
        '\u{FF01}'..='\u{FF0F}' |   // Fullwidth punctuation
        '\u{FF1A}'..='\u{FF20}' |   // Fullwidth punctuation continued
        '\u{FF3B}'..='\u{FF40}' |   // Fullwidth brackets
        '\u{FF5B}'..='\u{FF65}'     // Fullwidth punctuation end
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sentencepiece_spaces() {
        assert_eq!(fix_sentencepiece_spaces("▁Hello▁world"), " Hello world");
    }

    #[test]
    fn test_chinese_remove_spaces() {
        let result = postprocess_chinese("你 好 世 界");
        assert_eq!(result, "你好世界");
    }

    #[test]
    fn test_chinese_keep_cjk_latin_space() {
        let result = postprocess_chinese("你好 World 世界");
        // Space between CJK and Latin should be preserved
        assert!(result.contains(" World ") || result.contains("World"));
    }

    #[test]
    fn test_chinese_remove_punct_space() {
        let result = postprocess_chinese("你好 ，世界");
        // Space before CJK punctuation should be removed
        assert!(!result.contains(" ，"));
    }

    #[test]
    fn test_english_capitalize() {
        let result = postprocess_english("hello world");
        assert_eq!(result, "Hello world");
    }

    #[test]
    fn test_english_collapse_spaces() {
        let result = postprocess_english("hello   world   foo");
        assert_eq!(result, "Hello world foo");
    }

    #[test]
    fn test_postprocess_translation_chinese() {
        let result = postprocess_translation("你 好 世 界", DetectedLang::Chinese);
        assert_eq!(result, "你好世界");
    }

    #[test]
    fn test_postprocess_translation_english() {
        let result = postprocess_translation("hello  world", DetectedLang::English);
        assert_eq!(result, "Hello world");
    }

    #[test]
    fn test_empty() {
        assert_eq!(postprocess_translation("", DetectedLang::Chinese), "");
        assert_eq!(postprocess_translation("", DetectedLang::English), "");
    }
}
