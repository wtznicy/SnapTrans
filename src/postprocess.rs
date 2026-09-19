//! Output post-processing for translated text.
//!
//! Cleans up tokenizer artifacts and fixes formatting issues
//! in the decoded translation output:
//! - Removes SentencePiece artifacts
//! - Restores Pangu spacing (between CJK and Latin/digits)
//! - Ensures proper spacing after English punctuation
//! - Preserves paragraphs and newlines

use crate::lang_detect::DetectedLang;

/// Post-process translated text based on the target language.
pub fn postprocess_translation(text: &str, target_lang: DetectedLang) -> String {
    let text = fix_sentencepiece_spaces(text);
    let lines: Vec<String> = text
        .lines()
        .map(|line| match target_lang {
            DetectedLang::Chinese => postprocess_chinese_line(line),
            DetectedLang::English => postprocess_english_line(line),
        })
        .collect();
    lines.join("\n").trim().to_string()
}

/// Fix SentencePiece tokenizer artifacts.
///
/// SentencePiece uses `▁` (U+2581, LOWER ONE EIGHTH BLOCK) as a word
/// boundary marker. Replace these with regular spaces.
fn fix_sentencepiece_spaces(text: &str) -> String {
    text.replace('\u{2581}', " ")
}

/// Chinese-specific post-processing for a single line:
/// 1. Remove spaces between CJK characters and around CJK punctuation.
/// 2. Apply Pangu spacing (insert 1 space between CJK and Latin/digits).
fn postprocess_chinese_line(text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    let mut step1 = String::with_capacity(chars.len());
    let mut i = 0;

    // Step 1: Clean spaces between CJK characters and CJK punctuation
    while i < chars.len() {
        let ch = chars[i];

        if ch == ' ' || ch == '\t' {
            let prev_cjk = i > 0 && is_cjk_char(chars[i - 1]);
            let prev_cjk_punct = i > 0 && is_cjk_punct(chars[i - 1]);
            let mut j = i;
            while j < chars.len() && (chars[j] == ' ' || chars[j] == '\t') {
                j += 1;
            }
            let next_cjk = j < chars.len() && is_cjk_char(chars[j]);
            let next_cjk_punct = j < chars.len() && is_cjk_punct(chars[j]);

            if (prev_cjk && next_cjk) || (prev_cjk && next_cjk_punct) || (prev_cjk_punct && next_cjk) {
                // Space between two CJK characters or adjacent to CJK punctuation -> remove
                i = j;
            } else {
                step1.push(' ');
                i = j;
            }
        } else {
            step1.push(ch);
            i += 1;
        }
    }

    // Step 2: Apply Pangu spacing (ensure 1 space between CJK and Latin/digits)
    let s1_chars: Vec<char> = step1.chars().collect();
    let mut result = String::with_capacity(s1_chars.len() + 8);
    for (idx, &c) in s1_chars.iter().enumerate() {
        result.push(c);
        if idx + 1 < s1_chars.len() {
            let next = s1_chars[idx + 1];
            // CJK character followed by Latin letter or digit
            if is_cjk_char(c) && is_latin_or_digit(next) {
                result.push(' ');
            }
            // Latin letter or digit followed by CJK character
            else if is_latin_or_digit(c) && is_cjk_char(next) {
                result.push(' ');
            }
        }
    }

    result.trim().to_string()
}

/// English-specific post-processing for a single line:
/// 1. Collapse multiple consecutive spaces.
/// 2. Ensure exactly one space after punctuation marks (. , ! ? ; :) unless part of numbers/time.
/// 3. Capitalize first letter of line.
fn postprocess_english_line(text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    let mut result = String::with_capacity(chars.len() + 8);
    let mut prev_space = false;

    let mut i = 0;
    while i < chars.len() {
        let ch = chars[i];
        if ch == ' ' || ch == '\t' {
            if !prev_space && !result.is_empty() {
                result.push(' ');
                prev_space = true;
            }
            i += 1;
            continue;
        }

        prev_space = false;
        result.push(ch);

        // Ensure space after punctuation: . , ! ? ; :
        if matches!(ch, '.' | ',' | '!' | '?' | ';' | ':') {
            if i + 1 < chars.len() {
                let next = chars[i + 1];
                let is_prev_digit = i > 0 && chars[i - 1].is_ascii_digit();
                let is_next_digit = next.is_ascii_digit();

                // Do not insert space in numbers (3.14), thousands (1,000), or time (12:30)
                let is_numeric_token = is_prev_digit && is_next_digit && matches!(ch, '.' | ',' | ':');

                if !is_numeric_token
                    && next != ' '
                    && next != '\n'
                    && next != '\t'
                    && next != '"'
                    && next != '\''
                    && next != ')'
                    && next != ']'
                {
                    result.push(' ');
                    prev_space = true;
                }
            }
        }
        i += 1;
    }

    let trimmed = result.trim();
    let mut chars_iter = trimmed.chars();
    match chars_iter.next() {
        None => String::new(),
        Some(first) => {
            let capitalized: String = first.to_uppercase().collect();
            capitalized + chars_iter.as_str()
        }
    }
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

/// Check if a character is an ASCII Latin letter or digit.
#[inline]
fn is_latin_or_digit(ch: char) -> bool {
    ch.is_ascii_alphanumeric()
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
        let result = postprocess_chinese_line("你 好 世 界");
        assert_eq!(result, "你好世界");
    }

    #[test]
    fn test_chinese_pangu_spacing() {
        let result = postprocess_chinese_line("你好World世界");
        assert_eq!(result, "你好 World 世界");

        let result2 = postprocess_chinese_line("使用MarianMT进行100次测试");
        assert_eq!(result2, "使用 MarianMT 进行 100 次测试");
    }

    #[test]
    fn test_chinese_remove_punct_space() {
        let result = postprocess_chinese_line("你好 ，世界");
        assert_eq!(result, "你好，世界");
    }

    #[test]
    fn test_english_capitalize() {
        let result = postprocess_english_line("hello world");
        assert_eq!(result, "Hello world");
    }

    #[test]
    fn test_english_collapse_spaces() {
        let result = postprocess_english_line("hello   world   foo");
        assert_eq!(result, "Hello world foo");
    }

    #[test]
    fn test_english_punctuation_spacing() {
        let result = postprocess_english_line("hello,world!This is a test.");
        assert_eq!(result, "Hello, world! This is a test.");

        // Numbers should not be split
        let result2 = postprocess_english_line("pi is 3.14, count is 1,000, time is 12:30.");
        assert_eq!(result2, "Pi is 3.14, count is 1,000, time is 12:30.");
    }

    #[test]
    fn test_postprocess_multiline() {
        let input = "hello world\nthis is a test";
        let result = postprocess_translation(input, DetectedLang::English);
        assert_eq!(result, "Hello world\nThis is a test");

        let input_zh = "你 好\n世 界";
        let result_zh = postprocess_translation(input_zh, DetectedLang::Chinese);
        assert_eq!(result_zh, "你好\n世界");
    }

    #[test]
    fn test_empty() {
        assert_eq!(postprocess_translation("", DetectedLang::Chinese), "");
        assert_eq!(postprocess_translation("", DetectedLang::English), "");
    }
}
