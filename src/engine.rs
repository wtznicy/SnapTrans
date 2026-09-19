//! Core translation engine orchestrating the full pipeline.

use std::path::Path;
use std::sync::Mutex;
use std::time::Instant;

use crate::config::SnapTransConfig;
use crate::decoder::MarianDecoder;
use crate::encoder::MarianEncoder;
use crate::error::SnapTransError;
use crate::generation::{greedy_search, GenerationConfig};
use crate::lang_detect::{resolve_source_lang, DetectedLang};
use crate::postprocess::postprocess_translation;
use crate::text_normalize::normalize_text;
use crate::tokenizer::SnapTokenizer;
use crate::types::OfflineTranslateResponse;

/// The main SnapTrans offline translation engine.
///
/// Supports lazy loading: models are only loaded on first use.
/// Thread-safe via internal `Mutex`.
pub struct SnapTransEngine {
    config: SnapTransConfig,
    /// Lazily-loaded en→zh translation components.
    en_zh: Mutex<Option<TranslationPair>>,
    /// Lazily-loaded zh→en translation components.
    zh_en: Mutex<Option<TranslationPair>>,
}

/// A loaded translation direction (encoder + decoder + tokenizer).
struct TranslationPair {
    tokenizer: SnapTokenizer,
    encoder: MarianEncoder,
    decoder: MarianDecoder,
}

impl SnapTransEngine {
    /// Create a new engine. Does NOT load models — they load lazily.
    pub fn new(config: SnapTransConfig) -> Self {
        Self {
            config,
            en_zh: Mutex::new(None),
            zh_en: Mutex::new(None),
        }
    }

    /// Translate text from source language to target language.
    pub fn translate(
        &self,
        text: &str,
        source_lang: &str,
        target_lang: &str,
    ) -> Result<OfflineTranslateResponse, SnapTransError> {
        let total_start = Instant::now();

        // 1. Input validation
        let text = text.trim();
        if text.is_empty() {
            return Ok(OfflineTranslateResponse::new(
                String::new(),
                source_lang.to_string(),
                target_lang.to_string(),
                0.0,
            ));
        }

        // 2. Text normalization
        let normalized = normalize_text(text);
        if normalized.is_empty() {
            return Ok(OfflineTranslateResponse::new(
                String::new(),
                source_lang.to_string(),
                target_lang.to_string(),
                0.0,
            ));
        }

        // 3. Language detection
        let detected_source = resolve_source_lang(source_lang, &normalized);
        let mut target = match target_lang {
            "zh" => DetectedLang::Chinese,
            "en" => DetectedLang::English,
            _ => match detected_source {
                DetectedLang::Chinese => DetectedLang::English,
                DetectedLang::English => DetectedLang::Chinese,
            },
        };

        // If source was auto-detected and matches target, automatically invert target to provide useful translation
        if source_lang == "auto" && detected_source == target {
            target = match detected_source {
                DetectedLang::Chinese => DetectedLang::English,
                DetectedLang::English => DetectedLang::Chinese,
            };
        }

        // If caller explicitly requested same language passthrough, return normalized text
        if detected_source == target {
            return Ok(OfflineTranslateResponse::new(
                normalized,
                detected_source.code().to_string(),
                target.code().to_string(),
                total_start.elapsed().as_secs_f64() * 1000.0,
            ));
        }

        // 4-9. Run translation pipeline line-by-line to preserve paragraphs and line breaks
        let lines: Vec<&str> = normalized.split('\n').collect();
        let mut translated_lines = Vec::with_capacity(lines.len());
        for line in lines {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                translated_lines.push(String::new());
            } else {
                let trans_line = self.run_pipeline(detected_source, target, trimmed)?;
                translated_lines.push(trans_line);
            }
        }
        let translated_text = translated_lines.join("\n");

        let latency_ms = total_start.elapsed().as_secs_f64() * 1000.0;

        log::info!(
            "Translation: {}→{}, {:.1}ms",
            detected_source.code(),
            target.code(),
            latency_ms,
        );

        Ok(OfflineTranslateResponse::new(
            translated_text,
            detected_source.code().to_string(),
            target.code().to_string(),
            latency_ms,
        ))
    }

    /// Run the core translation pipeline (tokenize → encode → decode → postprocess).
    /// Acquires the mutex for the appropriate translation direction.
    fn run_pipeline(
        &self,
        source: DetectedLang,
        target: DetectedLang,
        text: &str,
    ) -> Result<String, SnapTransError> {
        let mutex = match (source, target) {
            (DetectedLang::English, DetectedLang::Chinese) => &self.en_zh,
            (DetectedLang::Chinese, DetectedLang::English) => &self.zh_en,
            _ => {
                return Err(SnapTransError::InvalidInput {
                    reason: format!(
                        "Unsupported direction: {}→{}",
                        source.code(),
                        target.code()
                    ),
                });
            }
        };

        let direction = match (source, target) {
            (DetectedLang::English, DetectedLang::Chinese) => "en_zh",
            _ => "zh_en",
        };

        let mut guard = mutex.lock().map_err(|e| SnapTransError::Inference {
            stage: "lock".into(),
            reason: format!("Mutex poisoned: {}", e),
        })?;

        // Lazy init: load models if not yet loaded
        if guard.is_none() {
            let pair = self.load_pair(direction)?;
            *guard = Some(pair);
        }

        let pair = guard.as_mut().unwrap();

        // 5. Tokenize
        let encoded = pair.tokenizer.encode(text)?;

        log::debug!(
            "Tokenized: {} tokens from '{}'",
            encoded.input_ids.len(),
            &text[..text.len().min(50)],
        );

        // 6. Encoder inference
        let encoder_output = pair
            .encoder
            .encode(&encoded.input_ids, &encoded.attention_mask)?;

        // 7. Decoder autoregressive generation
        let gen_config = GenerationConfig {
            max_length: self.config.max_length,
            eos_token_id: self.config.eos_token_id,
            decoder_start_token_id: self.config.decoder_start_token_id,
            repetition_penalty: self.config.repetition_penalty,
            forced_bos_token_id: self.config.forced_bos_token_id,
        };

        let generated_ids = greedy_search(
            &mut pair.decoder,
            &encoder_output,
            &encoded.attention_mask,
            &gen_config,
        )?;

        // 8. Detokenize
        let raw_translation = pair.tokenizer.decode(&generated_ids)?;

        // 9. Post-process
        Ok(postprocess_translation(&raw_translation, target))
    }

    /// Load a translation pair (tokenizer + encoder + decoder).
    fn load_pair(&self, direction: &str) -> Result<TranslationPair, SnapTransError> {
        let model_dir = &self.config.model_dir;

        let dir = if model_dir.join(direction).exists() {
            model_dir.join(direction)
        } else {
            model_dir.to_path_buf()
        };

        log::info!("Loading {} translation models from {:?}", direction, dir);
        let load_start = Instant::now();

        let tokenizer_path = find_file(&dir, &["tokenizer.json"])?;
        let tokenizer = SnapTokenizer::from_file(&tokenizer_path)?;

        let encoder_path = find_file(
            &dir,
            &[
                "encoder_model.onnx",
                "encoder_model_int8.onnx",
                "encoder_model_quantized.onnx",
            ],
        )?;
        let encoder = MarianEncoder::load(&encoder_path, self.config.num_threads)?;

        let decoder_path = find_file(
            &dir,
            &[
                "decoder_model_merged.onnx",
                "decoder_model_merged_int8.onnx",
                "decoder_model_merged_quantized.onnx",
                "decoder_with_past_model.onnx",
                "decoder_model.onnx",
            ],
        )?;
        let decoder = MarianDecoder::load(
            &decoder_path,
            self.config.num_threads,
            6, 8, 64,
        )?;

        log::info!(
            "Loaded {} models in {:.1}ms",
            direction,
            load_start.elapsed().as_secs_f64() * 1000.0
        );

        Ok(TranslationPair { tokenizer, encoder, decoder })
    }
}

/// Find the first existing file from a list of candidates.
fn find_file(dir: &Path, candidates: &[&str]) -> Result<String, SnapTransError> {
    for name in candidates {
        let path = dir.join(name);
        if path.exists() {
            return Ok(path.to_string_lossy().to_string());
        }
    }
    Err(SnapTransError::ModelLoad {
        model_name: candidates.first().unwrap_or(&"unknown").to_string(),
        reason: format!("None of {:?} found in {:?}", candidates, dir),
    })
}
