//! ONNX Decoder with KV Cache for MarianMT autoregressive generation.

use crate::encoder::EncoderOutput;
use crate::error::SnapTransError;
use ort::session::Session;
use ort::value::Value;

/// MarianMT Decoder wrapping an ONNX Runtime session.
pub(crate) struct MarianDecoder {
    session: Session,
    num_layers: usize,
    num_heads: usize,
    head_dim: usize,
    has_use_cache_branch: bool,
}

/// Output from a single decoder step.
pub(crate) struct DecoderStepOutput {
    /// Logits for the last token, shape `[vocab_size]`.
    pub logits: Vec<f32>,
    /// Vocabulary size.
    #[allow(dead_code)]
    pub vocab_size: usize,
    /// Present key values for KV cache chaining.
    pub present_kv: Vec<KVCacheTensor>,
}

/// A single KV cache tensor with its data and shape.
#[derive(Clone)]
pub(crate) struct KVCacheTensor {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
}

impl MarianDecoder {
    /// Load the decoder model.
    pub fn load(
        model_path: &str,
        num_threads: usize,
        num_layers: usize,
        num_heads: usize,
        head_dim: usize,
    ) -> Result<Self, SnapTransError> {
        let session = Session::builder()
            .map_err(|e| SnapTransError::ModelLoad {
                model_name: "decoder".into(),
                reason: format!("Failed to create session builder: {}", e),
            })?
            .with_intra_threads(num_threads)
            .map_err(|e| SnapTransError::ModelLoad {
                model_name: "decoder".into(),
                reason: format!("Failed to set thread count: {}", e),
            })?
            .commit_from_file(model_path)
            .map_err(|e| SnapTransError::ModelLoad {
                model_name: "decoder".into(),
                reason: format!("Failed to load model '{}': {}", model_path, e),
            })?;

        let has_use_cache_branch = session
            .inputs()
            .iter()
            .any(|input| input.name() == "use_cache_branch");

        Ok(Self {
            session,
            num_layers,
            num_heads,
            head_dim,
            has_use_cache_branch,
        })
    }

    /// Run a single decoder step.
    pub fn step(
        &mut self,
        decoder_input_ids: &[i64],
        encoder_output: &EncoderOutput,
        encoder_attention_mask: &[i64],
        past_kv: Option<&[KVCacheTensor]>,
    ) -> Result<DecoderStepOutput, SnapTransError> {
        let dec_seq_len = decoder_input_ids.len();
        let enc_seq_len = encoder_attention_mask.len();

        // Hidden size from encoder output shape [1, seq_len, hidden_size]
        let hidden_size = if encoder_output.shape.len() == 3 {
            encoder_output.shape[2]
        } else {
            512
        };

        // Build input tensors using ort's (shape, Vec<T>) tuple API
        let dec_ids_val = Value::from_array(
            ([1, dec_seq_len], decoder_input_ids.to_vec()),
        )
        .map_err(|e| SnapTransError::Inference {
            stage: "decoder_input".into(),
            reason: format!("decoder input_ids: {}", e),
        })?;

        let enc_mask_val = Value::from_array(
            ([1, enc_seq_len], encoder_attention_mask.to_vec()),
        )
        .map_err(|e| SnapTransError::Inference {
            stage: "decoder_input".into(),
            reason: format!("encoder attention_mask: {}", e),
        })?;

        let enc_hidden_val = Value::from_array(
            ([1, enc_seq_len, hidden_size], encoder_output.data.clone()),
        )
        .map_err(|e| SnapTransError::Inference {
            stage: "decoder_input".into(),
            reason: format!("encoder hidden states: {}", e),
        })?;

        // Build named input list
        let mut inputs: Vec<(String, ort::value::DynValue)> = Vec::new();

        inputs.push(("input_ids".to_string(), dec_ids_val.into_dyn()));
        inputs.push(("encoder_attention_mask".to_string(), enc_mask_val.into_dyn()));
        inputs.push(("encoder_hidden_states".to_string(), enc_hidden_val.into_dyn()));

        // KV cache: 4 tensors per layer (decoder.key/value, encoder.key/value)
        for layer in 0..self.num_layers {
            let kv_names = [
                format!("past_key_values.{}.decoder.key", layer),
                format!("past_key_values.{}.decoder.value", layer),
                format!("past_key_values.{}.encoder.key", layer),
                format!("past_key_values.{}.encoder.value", layer),
            ];

            for (i, name) in kv_names.iter().enumerate() {
                let tensor_idx = layer * 4 + i;
                let val = if let Some(kv) = past_kv {
                    if tensor_idx < kv.len() {
                        self.make_kv_value(&kv[tensor_idx])?
                    } else {
                        self.make_empty_kv_value()?
                    }
                } else {
                    self.make_empty_kv_value()?
                };
                inputs.push((name.clone(), val));
            }
        }

        let use_cache = past_kv.is_some();
        if self.has_use_cache_branch {
            let use_cache_val = Value::from_array(([1], vec![use_cache])).map_err(|e| {
                SnapTransError::Inference {
                    stage: "decoder_input".into(),
                    reason: format!("use_cache_branch: {}", e),
                }
            })?;
            inputs.push(("use_cache_branch".to_string(), use_cache_val.into_dyn()));
        }

        // Run inference
        let outputs = self
            .session
            .run(inputs)
            .map_err(|e| SnapTransError::Inference {
                stage: "decoder".into(),
                reason: e.to_string(),
            })?;

        // Extract logits: (&Shape, &[f32])
        let (logits_shape, logits_data) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| SnapTransError::Inference {
                stage: "decoder_logits".into(),
                reason: format!("Failed to extract logits: {}", e),
            })?;

        let vocab_size = *logits_shape.last().unwrap_or(&1) as usize;

        // Get logits for the last token only
        let last_token_logits = if logits_data.len() >= vocab_size {
            logits_data[logits_data.len() - vocab_size..].to_vec()
        } else {
            logits_data.to_vec()
        };

        // Extract present KV cache from remaining outputs
        let mut present_kv = Vec::with_capacity(outputs.len() - 1);
        for i in 1..outputs.len() {
            if let Ok((shape_ref, data_slice)) = outputs[i].try_extract_tensor::<f32>() {
                let shape: Vec<usize> = shape_ref.iter().map(|&d| d as usize).collect();
                let tensor_idx = i - 1;

                // For MarianMT, cross-attention encoder key/value (tensor_idx % 4 >= 2)
                // is only computed in step 0. In subsequent steps with use_cache_branch=true,
                // the ONNX graph outputs dummy shapes (e.g. [0, 8, 1, 64]).
                // We must preserve the original encoder KV from past_kv.
                if tensor_idx % 4 >= 2 && (shape.is_empty() || shape[0] == 0) {
                    if let Some(past) = past_kv {
                        if tensor_idx < past.len() {
                            present_kv.push(past[tensor_idx].clone());
                            continue;
                        }
                    }
                }

                let data: Vec<f32> = data_slice.to_vec();
                present_kv.push(KVCacheTensor { data, shape });
            }
        }

        Ok(DecoderStepOutput {
            logits: last_token_logits,
            vocab_size,
            present_kv,
        })
    }

    /// Create a DynValue from a KV cache tensor (4D).
    fn make_kv_value(&self, kv: &KVCacheTensor) -> Result<ort::value::DynValue, SnapTransError> {
        let s = &kv.shape;
        if s.len() != 4 {
            return Err(SnapTransError::Inference {
                stage: "decoder_kv".into(),
                reason: format!("Expected 4D KV cache, got {}D", s.len()),
            });
        }
        let val = Value::from_array(([s[0], s[1], s[2], s[3]], kv.data.clone())).map_err(|e| {
            SnapTransError::Inference {
                stage: "decoder_kv".into(),
                reason: format!("KV cache value: {}", e),
            }
        })?;
        Ok(val.into_dyn())
    }

    /// Create an empty DynValue for KV cache (4D: [1, heads, 0, dim]).
    fn make_empty_kv_value(&self) -> Result<ort::value::DynValue, SnapTransError> {
        let val = Value::from_array(
            ([1, self.num_heads, 0, self.head_dim], Vec::<f32>::new()),
        )
        .map_err(|e| SnapTransError::Inference {
            stage: "decoder_kv_init".into(),
            reason: format!("Empty KV cache: {}", e),
        })?;
        Ok(val.into_dyn())
    }
}
