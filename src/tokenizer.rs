//! Pure Rust tokenizer wrapper for MarianMT SentencePiece BPE.
//!
//! Wraps the HuggingFace `tokenizers` crate to provide encoding
//! and decoding for the translation pipeline.

use crate::error::SnapTransError;
use tokenizers::Tokenizer;

/// Tokenizer wrapper for SnapTrans.
///
/// Loads a `tokenizer.json` file (HuggingFace format) and provides
/// encode/decode operations compatible with MarianMT ONNX models.
pub(crate) struct SnapTokenizer {
    tokenizer: Tokenizer,
}

/// Encoded output ready for ONNX model input.
pub(crate) struct EncodedInput {
    /// Token IDs as i64 (required by ONNX Runtime).
    pub input_ids: Vec<i64>,
    /// Attention mask (1 for real tokens, 0 for padding).
    pub attention_mask: Vec<i64>,
}

impl SnapTokenizer {
    /// Load a tokenizer from a `tokenizer.json` file.
    pub fn from_file(path: &str) -> Result<Self, SnapTransError> {
        let tokenizer = Tokenizer::from_file(path).map_err(|e| SnapTransError::TokenizerLoad {
            reason: format!("Failed to load tokenizer from '{}': {}", path, e),
        })?;

        Ok(Self { tokenizer })
    }

    /// Encode text into token IDs and attention mask.
    ///
    /// Returns token IDs as `Vec<i64>` suitable for direct ONNX input.
    pub fn encode(&self, text: &str) -> Result<EncodedInput, SnapTransError> {
        let encoding = self
            .tokenizer
            .encode(text, true) // add_special_tokens = true
            .map_err(|e| SnapTransError::TokenizerLoad {
                reason: format!("Encoding failed: {}", e),
            })?;

        let input_ids: Vec<i64> = encoding.get_ids().iter().map(|&id| id as i64).collect();
        let attention_mask: Vec<i64> = encoding
            .get_attention_mask()
            .iter()
            .map(|&m| m as i64)
            .collect();

        Ok(EncodedInput {
            input_ids,
            attention_mask,
        })
    }

    /// Decode token IDs back into text.
    ///
    /// Skips special tokens (`<pad>`, `</s>`, `<unk>`, etc.).
    pub fn decode(&self, token_ids: &[i64]) -> Result<String, SnapTransError> {
        let ids: Vec<u32> = token_ids.iter().map(|&id| id as u32).collect();

        let text = self
            .tokenizer
            .decode(&ids, true) // skip_special_tokens = true
            .map_err(|e| SnapTransError::TokenizerLoad {
                reason: format!("Decoding failed: {}", e),
            })?;

        Ok(text)
    }

    /// Get the vocabulary size.
    #[allow(dead_code)]
    pub fn vocab_size(&self) -> usize {
        self.tokenizer.get_vocab_size(true)
    }
}

#[cfg(test)]
mod tests {
    // Tokenizer tests require a tokenizer.json file,
    // so they are placed in integration tests.
}
