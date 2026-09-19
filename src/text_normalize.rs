//! Input text normalization for translation preprocessing.
//!
//! Cleans and standardizes OCR output and user input before
//! feeding to the tokenizer and translation model while preserving
//! paragraph and line breaks.

/// Normalize a single line of text.
///
/// Steps:
/// 1. Converts full-width ASCII variants (U+FF01~U+FF5E) to half-width
/// 2. Converts full-width spaces (U+3000) to regular spaces
/// 3. Collapses multiple spaces and tabs into a single space
/// 4. Trims leading and trailing spaces on the line
pub fn normalize_line(line: &str) -> String {
    let mut result = String::with_capacity(line.len());
    let mut prev_space = false;
    for ch in line.chars() {
        let ch = normalize_char(ch);
        match ch {
            ' ' | '\t' => {
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

/// Normalize input text for translation.
///
/// Standardizes line endings, normalizes each line (preserving newlines),
/// and trims leading/trailing blank lines.
pub fn normalize_text(text: &str) -> String {
    let unified = text.replace("\r\n", "\n").replace('\r', "\n");
    let trimmed = unified.trim();
    if trimmed.is_empty() {
        return String::new();
    }

    let lines: Vec<String> = unified.lines().map(normalize_line).collect();

    let start = match lines.iter().position(|l| !l.is_empty()) {
        Some(pos) => pos,
        None => return String::new(),
    };
    let end = match lines.iter().rposition(|l| !l.is_empty()) {
        Some(pos) => pos + 1,
        None => return String::new(),
    };

    lines[start..end].join("\n")
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
    fn test_normalize_newlines_preserved() {
        assert_eq!(normalize_text("hello\nworld\nfoo"), "hello\nworld\nfoo");
        assert_eq!(normalize_text("hello\r\nworld\r\nfoo"), "hello\nworld\nfoo");
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
        let input = "  Hello, World!  \n  This is\n  a test.  ";
        let result = normalize_text(input);
        assert_eq!(result, "Hello, World!\nThis is\na test.");
    }

    #[test]
    fn test_normalize_tabs() {
        assert_eq!(normalize_text("hello\t\tworld"), "hello world");
    }
}
