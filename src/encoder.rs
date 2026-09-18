//! ONNX Encoder inference for MarianMT.

use crate::error::SnapTransError;
use ort::session::Session;
use ort::value::Value;

/// MarianMT Encoder wrapping an ONNX Runtime session.
pub(crate) struct MarianEncoder {
    session: Session,
}

impl MarianEncoder {
    /// Load the encoder model from an ONNX file.
    pub fn load(model_path: &str, num_threads: usize) -> Result<Self, SnapTransError> {
        let session = Session::builder()
            .map_err(|e| SnapTransError::ModelLoad {
                model_name: "encoder".into(),
                reason: format!("Failed to create session builder: {}", e),
            })?
            .with_intra_threads(num_threads)
            .map_err(|e| SnapTransError::ModelLoad {
                model_name: "encoder".into(),
                reason: format!("Failed to set thread count: {}", e),
            })?
            .commit_from_file(model_path)
            .map_err(|e| SnapTransError::ModelLoad {
                model_name: "encoder".into(),
                reason: format!("Failed to load model '{}': {}", model_path, e),
            })?;

        Ok(Self { session })
    }

    /// Run encoder inference.
    ///
    /// # Arguments
    /// * `input_ids` — Token IDs, shape `[seq_len]`.
    /// * `attention_mask` — Attention mask, shape `[seq_len]`.
    ///
    /// # Returns
    /// `EncoderOutput` containing the last hidden state.
    pub fn encode(
        &mut self,
        input_ids: &[i64],
        attention_mask: &[i64],
    ) -> Result<EncoderOutput, SnapTransError> {
        let seq_len = input_ids.len();

        // ort 2.0 RC: Value::from_array takes (shape, Vec<T>)
        let ids_val =
            Value::from_array(([1, seq_len], input_ids.to_vec())).map_err(|e| {
                SnapTransError::Inference {
                    stage: "encoder_input".into(),
                    reason: format!("input_ids tensor: {}", e),
                }
            })?;

        let mask_val =
            Value::from_array(([1, seq_len], attention_mask.to_vec())).map_err(|e| {
                SnapTransError::Inference {
                    stage: "encoder_input".into(),
                    reason: format!("attention_mask tensor: {}", e),
                }
            })?;

        // Run inference
        let outputs = self
            .session
            .run(ort::inputs![ids_val, mask_val])
            .map_err(|e| SnapTransError::Inference {
                stage: "encoder".into(),
                reason: e.to_string(),
            })?;

        // Extract last_hidden_state: try_extract_tensor returns (&Shape, &[T])
        let (shape_ref, data_slice) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| SnapTransError::Inference {
                stage: "encoder_output".into(),
                reason: format!("Failed to extract hidden state: {}", e),
            })?;

        let shape: Vec<usize> = shape_ref.iter().map(|&d| d as usize).collect();
        let data: Vec<f32> = data_slice.to_vec();

        Ok(EncoderOutput { data, shape })
    }
}

/// Encoder output containing the last hidden state.
pub(crate) struct EncoderOutput {
    /// Flat data of shape `[1, seq_len, hidden_size]`.
    pub data: Vec<f32>,
    /// Shape: `[batch_size, seq_len, hidden_size]`.
    pub shape: Vec<usize>,
}

impl EncoderOutput {
    /// Get the sequence length.
    #[allow(dead_code)]
    pub fn seq_len(&self) -> usize {
        if self.shape.len() >= 2 { self.shape[1] } else { 0 }
    }
}
