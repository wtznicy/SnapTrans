//! Error types for the SnapTrans engine.

use std::fmt;

/// All errors that can occur in the SnapTrans engine.
#[derive(Debug)]
pub enum SnapTransError {
    /// Failed to load an ONNX model file.
    ModelLoad { model_name: String, reason: String },
    /// Failed to load the tokenizer.
    TokenizerLoad { reason: String },
    /// Error during ONNX inference.
    Inference { stage: String, reason: String },
    /// Invalid input (empty text, unsupported language, etc.).
    InvalidInput { reason: String },
    /// Language detection failure.
    LangDetect { reason: String },
    /// I/O error.
    Io(std::io::Error),
}

impl fmt::Display for SnapTransError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ModelLoad { model_name, reason } => {
                write!(f, "Failed to load model '{}': {}", model_name, reason)
            }
            Self::TokenizerLoad { reason } => write!(f, "Tokenizer load error: {}", reason),
            Self::Inference { stage, reason } => {
                write!(f, "Inference error in '{}': {}", stage, reason)
            }
            Self::InvalidInput { reason } => write!(f, "Invalid input: {}", reason),
            Self::LangDetect { reason } => write!(f, "Language detection error: {}", reason),
            Self::Io(e) => write!(f, "I/O error: {}", e),
        }
    }
}

impl std::error::Error for SnapTransError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for SnapTransError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

// Convert to String for Tauri command compatibility
impl From<SnapTransError> for String {
    fn from(e: SnapTransError) -> String {
        e.to_string()
    }
}
