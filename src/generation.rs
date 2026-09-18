//! Greedy search generation strategy for autoregressive decoding.
//!
//! Implements the token generation loop with repetition penalty
//! and EOS/max-length termination conditions.

use crate::decoder::{KVCacheTensor, MarianDecoder};
use crate::encoder::EncoderOutput;
use crate::error::SnapTransError;

/// Configuration for the generation strategy.
pub(crate) struct GenerationConfig {
    /// Maximum number of tokens to generate.
    pub max_length: usize,
    /// EOS (end-of-sequence) token ID.
    pub eos_token_id: i64,
    /// Decoder start token ID (first input to decoder).
    pub decoder_start_token_id: i64,
    /// Repetition penalty (> 1.0 penalizes repeated tokens).
    pub repetition_penalty: f32,
    /// Optional forced BOS token ID.
    pub forced_bos_token_id: Option<i64>,
}

/// Run greedy search generation.
///
/// # Algorithm
/// 1. Initialize decoder with `decoder_start_token_id`
/// 2. Optionally force a BOS token as the first generated token
/// 3. Loop:
///    a. Run decoder step with current token + KV cache
///    b. Apply repetition penalty to logits of already-generated tokens
///    c. Select token with highest logit (greedy argmax)
///    d. Break if EOS or max_length reached
///    e. Update KV cache for next step
/// 4. Return generated token IDs
pub(crate) fn greedy_search(
    decoder: &mut MarianDecoder,
    encoder_output: &EncoderOutput,
    encoder_attention_mask: &[i64],
    config: &GenerationConfig,
) -> Result<Vec<i64>, SnapTransError> {
    let mut generated_ids: Vec<i64> = Vec::with_capacity(config.max_length);
    let mut current_input = vec![config.decoder_start_token_id];
    let mut past_kv: Option<Vec<KVCacheTensor>> = None;

    // Handle forced BOS token
    if let Some(bos_id) = config.forced_bos_token_id {
        // First step: run decoder with start token to get KV cache
        let step_output = decoder.step(
            &current_input,
            encoder_output,
            encoder_attention_mask,
            None,
        )?;

        generated_ids.push(bos_id);
        current_input = vec![bos_id];
        past_kv = Some(step_output.present_kv);
    }

    // Main generation loop
    for _step in 0..config.max_length {
        let step_output = decoder.step(
            &current_input,
            encoder_output,
            encoder_attention_mask,
            past_kv.as_deref(),
        )?;

        // Apply repetition penalty and select next token
        let mut logits = step_output.logits;
        apply_repetition_penalty(&mut logits, &generated_ids, config.repetition_penalty);

        let next_token = argmax(&logits);

        // Check for EOS
        if next_token == config.eos_token_id {
            break;
        }

        generated_ids.push(next_token);
        current_input = vec![next_token];
        past_kv = Some(step_output.present_kv);
    }

    Ok(generated_ids)
}

/// Apply repetition penalty to logits.
///
/// For each token that has already been generated, its logit is divided
/// by the penalty factor (if positive) or multiplied (if negative).
/// This discourages the model from repeating the same tokens.
fn apply_repetition_penalty(logits: &mut [f32], generated_ids: &[i64], penalty: f32) {
    if penalty == 1.0 || generated_ids.is_empty() {
        return;
    }

    for &token_id in generated_ids {
        let idx = token_id as usize;
        if idx < logits.len() {
            if logits[idx] > 0.0 {
                logits[idx] /= penalty;
            } else {
                logits[idx] *= penalty;
            }
        }
    }
}

/// Find the index of the maximum value in a slice (greedy argmax).
#[inline]
fn argmax(logits: &[f32]) -> i64 {
    let mut max_idx = 0;
    let mut max_val = f32::NEG_INFINITY;
    for (i, &val) in logits.iter().enumerate() {
        if val > max_val {
            max_val = val;
            max_idx = i;
        }
    }
    max_idx as i64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_argmax() {
        let logits = vec![1.0, 3.0, 2.0, 0.5];
        assert_eq!(argmax(&logits), 1);
    }

    #[test]
    fn test_argmax_negative() {
        let logits = vec![-5.0, -1.0, -3.0, -2.0];
        assert_eq!(argmax(&logits), 1);
    }

    #[test]
    fn test_repetition_penalty() {
        let mut logits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let generated = vec![1, 3];
        apply_repetition_penalty(&mut logits, &generated, 1.5);

        // Token 1 and 3 should be penalized
        assert!((logits[1] - 2.0 / 1.5).abs() < 0.001);
        assert!((logits[3] - 4.0 / 1.5).abs() < 0.001);
        // Others unchanged
        assert_eq!(logits[0], 1.0);
        assert_eq!(logits[2], 3.0);
        assert_eq!(logits[4], 5.0);
    }

    #[test]
    fn test_repetition_penalty_negative_logits() {
        let mut logits = vec![-1.0, -2.0, 3.0];
        let generated = vec![0, 1];
        apply_repetition_penalty(&mut logits, &generated, 1.5);

        // Negative logits are multiplied by penalty (making them more negative)
        assert!((logits[0] - (-1.0 * 1.5)).abs() < 0.001);
        assert!((logits[1] - (-2.0 * 1.5)).abs() < 0.001);
    }

    #[test]
    fn test_no_penalty() {
        let mut logits = vec![1.0, 2.0, 3.0];
        let original = logits.clone();
        apply_repetition_penalty(&mut logits, &[0, 1], 1.0);
        assert_eq!(logits, original);
    }
}
