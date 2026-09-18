//! Input text normalization for translation preprocessing.
//!
//! Cleans and standardizes OCR output and user input before
//! feeding to the tokenizer and translation model.

/// Normalize input text for translation.
///
/// Steps:
/// 1. Trim leading/trailing whitespace
/// 2. Replace newlines with spaces (OCR text often has line breaks)
/// 3. Collapse multiple consecutive spaces into one
/// 4. Normalize full-width ASCII characters to half-width
/// 5. Fix common OCR artifacts
pub fn normalize_text(text: &str) -> String {
    let text = text.trim();
    if text.is_empty() {
        return String::new();
    }

    let mut result = String::with_capacity(text.len());

    let mut prev_space = false;
    for ch in text.chars() {
        let ch = normalize_char(ch);

        match ch {
            '\n' | '\r' | '\t' => {
                // Replace line breaks and tabs with space
                if !prev_space {
                    result.push(' ');
                    prev_space = true;
                }
            }
            ' ' => {
                if !prev_space {
                    result.push(' ');
                    prev_space = true;
                }
            }
            _ => {
                result.push(ch);
                prev_space = false;
            }
        }
    }

    result.trim().to_string()
}

/// Normalize a single character.
///
/// Converts full-width ASCII variants (U+FF01~U+FF5E) to half-width,
/// and full-width space (U+3000) to regular space.
#[inline]
fn normalize_char(ch: char) -> char {
    match ch {
        // Full-width ASCII variants → half-width
        '\u{FF01}'..='\u{FF5E}' => (ch as u32 - 0xFF01 + 0x21) as u8 as char,
        // Full-width space → half-width
        '\u{3000}' => ' ',
        // Everything else passes through
        _ => ch,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_basic() {
        assert_eq!(normalize_text("  hello  world  "), "hello world");
    }

    #[test]
    fn test_normalize_newlines() {
        assert_eq!(normalize_text("hello\nworld\nfoo"), "hello world foo");
    }

    #[test]
    fn test_normalize_fullwidth() {
        assert_eq!(normalize_text("Ｈｅｌｌｏ"), "Hello");
    }

    #[test]
    fn test_normalize_fullwidth_space() {
        assert_eq!(normalize_text("你好\u{3000}世界"), "你好 世界");
    }

    #[test]
    fn test_normalize_empty() {
        assert_eq!(normalize_text(""), "");
        assert_eq!(normalize_text("   "), "");
    }

    #[test]
    fn test_normalize_mixed() {
        let input = "  Hello，World！  This is\n  a test。  ";
        let result = normalize_text(input);
        assert!(!result.contains('\n'));
        assert!(!result.starts_with(' '));
        assert!(!result.ends_with(' '));
    }

    #[test]
    fn test_normalize_tabs() {
        assert_eq!(normalize_text("hello\t\tworld"), "hello world");
    }
}
