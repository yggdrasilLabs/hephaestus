//! Automatic speech recognition pipeline (CTC and Whisper).

use std::path::Path;

use ndarray::Array2;
use ort::session::Session;
use ort::value::TensorRef;
use tokenizers::Tokenizer;

use crate::ep::ExecutionProvider;
use crate::error::CoreError;

use super::{build_onnx_session, check_outputs_nonempty, resolve_onnx_path, Pipeline, PreparedAudio};

/// Internal mode discriminator for the ASR pipeline.
///
/// CTC models use a single ONNX session with greedy decode.
/// EncoderDecoder models (Whisper) use separate encoder and decoder sessions
/// with autoregressive token generation.
enum AsrMode {
    /// CTC model (wav2vec2, HuBERT): single session, greedy decode.
    Ctc,
    /// Encoder-decoder model (Whisper): separate encoder + decoder sessions.
    EncoderDecoder,
}

/// Automatic speech recognition pipeline for CTC and Whisper models.
///
/// Follows the Ousterhout deep-module pattern: a single `prepare()` /
/// `execute()` interface hides the complexity of feature extraction,
/// ONNX inference, and decoding (CTC greedy or Whisper autoregressive).
pub struct AsrPipeline {
    /// Primary session: the CTC model or the Whisper encoder.
    encoder_session: Session,

    /// Optional decoder session for Whisper encoder-decoder models.
    decoder_session: Option<Session>,

    /// Optional tokenizer for Whisper (uses tokenizer.json for decode).
    tokenizer: Option<Tokenizer>,

    /// Vocabulary for CTC models (loaded from vocab.json).
    vocab: Vec<String>,

    /// CTC blank token index (typically 0 for wav2vec2).
    blank_id: usize,

    /// Whether this is a CTC or encoder-decoder model.
    mode: AsrMode,

    /// Feature extractor type: "mel" (Whisper) or "none" (raw waveform for CTC).
    feature_extractor: String,

    /// Number of mel filter banks (Whisper: 80 or 128).
    n_mels: usize,

    /// FFT window size (Whisper: 400).
    n_fft: usize,

    /// STFT hop length (Whisper: 160).
    hop_length: usize,

    /// Decoder start token ID for Whisper autoregressive generation.
    decoder_start_token_id: usize,

    /// End-of-sequence token ID for Whisper.
    eos_token_id: usize,

    /// Maximum number of decoder tokens (Whisper: 448).
    max_target_positions: usize,

    /// Name of the decoder input_ids input (varies: "input_ids" or "decoder_input_ids").
    decoder_input_name: String,

    /// Name of the encoder hidden states input to the decoder
    /// (varies: "encoder_hidden_states", "encoder_output", etc.).
    encoder_hidden_name: String,
}

impl AsrPipeline {
    /// Construct a new ASR pipeline from a model directory.
    ///
    /// Reads `config.json` to detect whether this is a CTC model or
    /// Whisper encoder-decoder model, then loads the appropriate ONNX
    /// session(s) and vocabulary/tokenizer.
    ///
    /// # Arguments
    ///
    /// * `model_dir` -- Path to the directory containing model files.
    /// * `ep` -- Execution provider for ONNX Runtime.
    /// * `feature_extractor` -- `"mel"` for Whisper, `"none"` for CTC.
    ///
    /// # Errors
    ///
    /// Returns [`CoreError`] if required files are missing or the model
    /// architecture is not a recognized ASR type.
    pub fn new(
        model_dir: &Path,
        ep: &ExecutionProvider,
        feature_extractor: &str,
    ) -> Result<Self, CoreError> {
        // Read config.json to detect model type.
        let config_path = model_dir.join("config.json");
        let config_text = std::fs::read_to_string(&config_path)?;
        let config: serde_json::Value = serde_json::from_str(&config_text)?;

        let archs = config
            .get("architectures")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();

        let is_ctc = archs.iter().any(|a| a.ends_with("ForCTC"));
        let is_whisper = archs.iter().any(|a| a == "WhisperForConditionalGeneration");

        if is_ctc {
            Self::new_ctc(model_dir, ep, feature_extractor)
        } else if is_whisper {
            Self::new_whisper(model_dir, ep, feature_extractor, &config)
        } else {
            Err(CoreError::ModelLoad(format!(
                "ASR pipeline requires ForCTC or WhisperForConditionalGeneration architecture, got: {archs:?}"
            )))
        }
    }

    /// Build a CTC-mode ASR pipeline.
    fn new_ctc(
        model_dir: &Path,
        ep: &ExecutionProvider,
        feature_extractor: &str,
    ) -> Result<Self, CoreError> {
        // Resolve ONNX model file with onnx/ subdirectory fallback.
        let model_path = resolve_onnx_path(model_dir, "model.onnx")?;
        let session = build_onnx_session(&model_path, ep)?;

        // T-11-07: Validate that the model has "input_values" input.
        let input_names: Vec<String> = session
            .inputs()
            .iter()
            .map(|i| i.name().to_string())
            .collect();
        if !input_names.iter().any(|n| n == "input_values") {
            return Err(CoreError::ModelValidation(format!(
                "CTC model must have 'input_values' input; got: {input_names:?}"
            )));
        }

        // Load vocab.json: map of token -> id.
        let vocab_path = model_dir.join("vocab.json");
        let vocab_text = std::fs::read_to_string(&vocab_path).map_err(|e| {
            CoreError::ModelLoad(format!("failed to read vocab.json: {e}"))
        })?;
        let vocab_map: std::collections::HashMap<String, usize> =
            serde_json::from_str(&vocab_text)?;

        // Invert to Vec<String> indexed by id.
        let max_id = vocab_map.values().copied().max().unwrap_or(0);
        if max_id > 1_000_000 {
            return Err(CoreError::ModelLoad(format!(
                "vocab.json max token ID {max_id} exceeds safety limit of 1,000,000"
            )));
        }
        let mut vocab = vec![String::new(); max_id + 1];
        for (token, id) in &vocab_map {
            vocab[*id] = token.clone();
        }

        // Determine blank_id: check for "<pad>" first, then "|", default 0.
        let blank_id = vocab_map
            .get("<pad>")
            .or_else(|| vocab_map.get("|"))
            .copied()
            .unwrap_or(0);

        Ok(Self {
            encoder_session: session,
            decoder_session: None,
            tokenizer: None,
            vocab,
            blank_id,
            mode: AsrMode::Ctc,
            feature_extractor: feature_extractor.to_string(),
            n_mels: 80,
            n_fft: 400,
            hop_length: 160,
            decoder_start_token_id: 0,
            eos_token_id: 0,
            max_target_positions: 0,
            decoder_input_name: String::new(),
            encoder_hidden_name: String::new(),
        })
    }

    /// Build a Whisper encoder-decoder ASR pipeline.
    fn new_whisper(
        model_dir: &Path,
        ep: &ExecutionProvider,
        feature_extractor: &str,
        config: &serde_json::Value,
    ) -> Result<Self, CoreError> {
        let onnx_dir = model_dir.join("onnx");

        // Load encoder session.
        let encoder_path = onnx_dir.join("encoder_model.onnx");
        if !encoder_path.exists() {
            return Err(CoreError::ModelLoad(format!(
                "Whisper encoder not found at '{}'",
                encoder_path.display(),
            )));
        }

        let encoder_session = build_onnx_session(&encoder_path, ep)?;

        // T-11-07: Validate encoder has "input_features" input.
        let encoder_inputs: Vec<String> = encoder_session
            .inputs()
            .iter()
            .map(|i| i.name().to_string())
            .collect();
        if !encoder_inputs.iter().any(|n| n == "input_features") {
            return Err(CoreError::ModelValidation(format!(
                "Whisper encoder must have 'input_features' input; got: {encoder_inputs:?}"
            )));
        }

        // Load decoder session (try merged first, then standalone).
        let decoder_merged = onnx_dir.join("decoder_model_merged.onnx");
        let decoder_plain = onnx_dir.join("decoder_model.onnx");
        let decoder_path = if decoder_merged.exists() {
            decoder_merged
        } else if decoder_plain.exists() {
            decoder_plain
        } else {
            return Err(CoreError::ModelLoad(
                "Whisper decoder not found (tried decoder_model_merged.onnx and decoder_model.onnx)".into(),
            ));
        };
        let decoder_session = build_onnx_session(&decoder_path, ep)?;

        // T-11-07: Validate decoder has "input_ids" or "decoder_input_ids".
        let decoder_inputs: Vec<String> = decoder_session
            .inputs()
            .iter()
            .map(|i| i.name().to_string())
            .collect();
        let decoder_input_name = if decoder_inputs.iter().any(|n| n == "input_ids") {
            "input_ids".to_string()
        } else if decoder_inputs.iter().any(|n| n == "decoder_input_ids") {
            "decoder_input_ids".to_string()
        } else {
            return Err(CoreError::ModelValidation(format!(
                "Whisper decoder must have 'input_ids' or 'decoder_input_ids' input; got: {decoder_inputs:?}"
            )));
        };

        // Validate encoder hidden states input name (varies across ONNX exports).
        let encoder_hidden_name =
            if decoder_inputs.iter().any(|n| n == "encoder_hidden_states") {
                "encoder_hidden_states".to_string()
            } else if decoder_inputs.iter().any(|n| n == "encoder_output") {
                "encoder_output".to_string()
            } else if decoder_inputs
                .iter()
                .any(|n| n == "encoder_last_hidden_state")
            {
                "encoder_last_hidden_state".to_string()
            } else {
                return Err(CoreError::ModelValidation(format!(
                    "Whisper decoder must have 'encoder_hidden_states', 'encoder_output', \
                     or 'encoder_last_hidden_state' input; got: {decoder_inputs:?}"
                )));
            };

        // Load tokenizer.
        let tokenizer_path = model_dir.join("tokenizer.json");
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| CoreError::Tokenization(e.to_string()))?;

        // Read preprocessor config for mel parameters.
        let preprocessor_path = model_dir.join("preprocessor_config.json");
        let (n_mels, n_fft, hop_length) = if preprocessor_path.exists() {
            let text = std::fs::read_to_string(&preprocessor_path)?;
            let preproc: serde_json::Value = serde_json::from_str(&text)?;
            let n_mels = preproc
                .get("feature_size")
                .and_then(|v| v.as_u64())
                .unwrap_or(80) as usize;
            let n_fft = preproc
                .get("n_fft")
                .and_then(|v| v.as_u64())
                .unwrap_or(400) as usize;
            let hop = preproc
                .get("hop_length")
                .and_then(|v| v.as_u64())
                .unwrap_or(160) as usize;
            (n_mels, n_fft, hop)
        } else {
            (80, 400, 160)
        };

        // Read decoder config.
        let decoder_start_token_id = config
            .get("decoder_start_token_id")
            .and_then(|v| v.as_u64())
            .unwrap_or(50258) as usize;
        let eos_token_id = config
            .get("eos_token_id")
            .and_then(|v| v.as_u64())
            .unwrap_or(50257) as usize;
        let max_target_positions = config
            .get("max_target_positions")
            .and_then(|v| v.as_u64())
            .unwrap_or(448) as usize;

        Ok(Self {
            encoder_session,
            decoder_session: Some(decoder_session),
            tokenizer: Some(tokenizer),
            vocab: Vec::new(),
            blank_id: 0,
            mode: AsrMode::EncoderDecoder,
            feature_extractor: feature_extractor.to_string(),
            n_mels,
            n_fft,
            hop_length,
            decoder_start_token_id,
            eos_token_id,
            max_target_positions,
            decoder_input_name,
            encoder_hidden_name,
        })
    }
}

/// The feature extraction settings of an ASR pipeline: everything needed
/// to turn audio samples into the model's input, without the sessions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct AudioFeatures {
    /// `"mel"` for a mel spectrogram, anything else for the raw waveform.
    feature_extractor: String,
    n_mels: usize,
    n_fft: usize,
    hop_length: usize,
}

impl AudioFeatures {
    /// Settings for a test: the given extractor with Whisper's mel sizes.
    #[cfg(test)]
    pub(super) fn for_test(feature_extractor: &str) -> Self {
        Self {
            feature_extractor: feature_extractor.to_string(),
            n_mels: 80,
            n_fft: 400,
            hop_length: 160,
        }
    }

    /// Extract the model input from mono samples.
    pub(super) fn extract(&self, input: Vec<f32>) -> Result<PreparedAudio, CoreError> {
        if self.feature_extractor == "mel" {
            let mel_features = crate::mel::compute_mel_spectrogram(
                &input,
                self.n_fft,
                self.hop_length,
                self.n_mels,
                16000,
            )?;
            Ok(PreparedAudio {
                features: mel_features,
            })
        } else {
            // Raw waveform: shape [1, num_samples] for CTC models.
            let num_samples = input.len();
            let features = Array2::from_shape_vec((1, num_samples), input)
                .map_err(|e| CoreError::Inference(format!("waveform reshape failed: {e}")))?;
            Ok(PreparedAudio {
                features,
            })
        }
    }
}

impl AsrPipeline {
    /// This pipeline's feature extraction settings.
    pub(super) fn features(&self) -> AudioFeatures {
        AudioFeatures {
            feature_extractor: self.feature_extractor.clone(),
            n_mels: self.n_mels,
            n_fft: self.n_fft,
            hop_length: self.hop_length,
        }
    }
}

impl Pipeline for AsrPipeline {
    type Input = Vec<f32>;
    type Prepared = PreparedAudio;
    type Output = String;

    fn prepare(&self, input: Vec<f32>) -> Result<PreparedAudio, CoreError> {
        self.features().extract(input)
    }

    fn execute(&mut self, prepared: PreparedAudio) -> Result<String, CoreError> {
        match self.mode {
            AsrMode::Ctc => self.execute_ctc(prepared),
            AsrMode::EncoderDecoder => self.execute_whisper(prepared),
        }
    }
}

impl AsrPipeline {
    /// Execute CTC model inference and greedy decode.
    fn execute_ctc(&mut self, prepared: PreparedAudio) -> Result<String, CoreError> {
        // Build input tensor: [1, num_samples] for raw waveform.
        let features = &prepared.features;
        let tensor = TensorRef::from_array_view(features.view())
            .map_err(|e| CoreError::Inference(e.to_string()))?;

        let outputs = self
            .encoder_session
            .run(ort::inputs!["input_values" => tensor])
            .map_err(|e| CoreError::Inference(e.to_string()))?;

        check_outputs_nonempty(&outputs)?;

        // Extract logits: shape [1, timesteps, vocab_size].
        let logits = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| CoreError::Inference(e.to_string()))?;
        let (shape, data) = logits;

        if shape.len() != 3 {
            return Err(CoreError::Inference(format!(
                "expected 3D CTC output (batch, timesteps, vocab_size), got {}-D",
                shape.len()
            )));
        }
        let num_timesteps = shape[1] as usize;
        let vocab_size = shape[2] as usize;

        Ok(crate::ctc::ctc_greedy_decode(
            data,
            num_timesteps,
            vocab_size,
            &self.vocab,
            self.blank_id,
        ))
    }

    /// Execute Whisper encoder-decoder inference with autoregressive decode.
    fn execute_whisper(&mut self, prepared: PreparedAudio) -> Result<String, CoreError> {
        let mel_features = &prepared.features;
        let (n_mels, num_frames) = mel_features.dim();

        // Build 3D input: [1, n_mels, num_frames].
        let flat: Vec<f32> = mel_features.iter().copied().collect();
        let input_3d = ndarray::Array3::from_shape_vec((1, n_mels, num_frames), flat)
            .map_err(|e| CoreError::Inference(format!("mel tensor reshape failed: {e}")))?;

        let input_tensor = TensorRef::from_array_view(input_3d.view())
            .map_err(|e| CoreError::Inference(e.to_string()))?;

        // Run encoder.
        let encoder_outputs = self
            .encoder_session
            .run(ort::inputs!["input_features" => input_tensor])
            .map_err(|e| CoreError::Inference(e.to_string()))?;
        check_outputs_nonempty(&encoder_outputs)?;

        // Extract encoder hidden state.
        let encoder_hidden = encoder_outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| CoreError::Inference(e.to_string()))?;
        let (enc_shape, enc_data) = encoder_hidden;

        // Reconstruct as a 3D array for repeated decoder input.
        let enc_3d = ndarray::Array3::from_shape_vec(
            (
                enc_shape[0] as usize,
                enc_shape[1] as usize,
                enc_shape[2] as usize,
            ),
            enc_data.to_vec(),
        )
        .map_err(|e| CoreError::Inference(format!("encoder output reshape failed: {e}")))?;

        // Autoregressive decode loop (T-11-08: bounded by max_target_positions).
        let decoder_session = self.decoder_session.as_mut().ok_or_else(|| {
            CoreError::Inference("Whisper pipeline missing decoder session".into())
        })?;

        let mut tokens: Vec<i64> = vec![self.decoder_start_token_id as i64];

        for _ in 0..self.max_target_positions {
            let seq_len = tokens.len();
            let token_view =
                ndarray::ArrayView2::from_shape((1, seq_len), &tokens)
                    .map_err(|e| CoreError::Inference(format!("token tensor failed: {e}")))?;

            let token_tensor = TensorRef::from_array_view(token_view)
                .map_err(|e| CoreError::Inference(e.to_string()))?;
            let enc_tensor = TensorRef::from_array_view(enc_3d.view())
                .map_err(|e| CoreError::Inference(e.to_string()))?;

            let decoder_outputs = decoder_session
                .run(ort::inputs![
                    self.decoder_input_name.as_str() => token_tensor,
                    self.encoder_hidden_name.as_str() => enc_tensor,
                ])
                .map_err(|e| CoreError::Inference(e.to_string()))?;
            check_outputs_nonempty(&decoder_outputs)?;

            // Extract logits: [1, seq_len, vocab_size].
            let logits = decoder_outputs[0]
                .try_extract_tensor::<f32>()
                .map_err(|e| CoreError::Inference(e.to_string()))?;
            let (logit_shape, logit_data) = logits;

            if logit_shape.len() != 3 {
                return Err(CoreError::Inference(format!(
                    "expected 3D decoder output (batch, seq, vocab), got {}-D",
                    logit_shape.len()
                )));
            }
            let vocab_size = logit_shape[2] as usize;

            // Take logits for the last position.
            let last_pos_start = (seq_len - 1) * vocab_size;
            let expected_len = last_pos_start + vocab_size;
            if logit_data.len() < expected_len {
                return Err(CoreError::Inference(format!(
                    "decoder logit data too short: expected at least {expected_len} elements, got {}",
                    logit_data.len()
                )));
            }
            let last_pos_logits = &logit_data[last_pos_start..expected_len];

            // Argmax.
            let next_token = last_pos_logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx)
                .unwrap_or(self.eos_token_id);

            if next_token == self.eos_token_id {
                break;
            }

            tokens.push(next_token as i64);
        }

        // Decode tokens to text (skip the start token and special tokens).
        let tokenizer = self.tokenizer.as_ref().ok_or_else(|| {
            CoreError::Inference("Whisper pipeline missing tokenizer".into())
        })?;

        let output_ids: Vec<u32> = tokens
            .iter()
            .skip(1) // Skip decoder_start_token_id.
            .map(|&t| {
                u32::try_from(t).map_err(|_| {
                    CoreError::Inference(format!(
                        "invalid token ID {t} in Whisper decoder output"
                    ))
                })
            })
            .collect::<Result<Vec<u32>, CoreError>>()?;

        tokenizer
            .decode(&output_ids, true)
            .map_err(|e| CoreError::Inference(e.to_string()))
    }
}
