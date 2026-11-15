//! Core inference pipeline trait, classifier, and embeddings implementations.

mod asr;
mod classifier;
mod embeddings;
mod encoder_decoder;
mod seq2seq;
mod token_classifier;

pub use asr::AsrPipeline;
pub use classifier::ClassifierPipeline;
pub use embeddings::EmbeddingsPipeline;
pub use encoder_decoder::{SEQ2SEQ_MAX_NEW_TOKENS_LIMIT, SEQ2SEQ_MIN_INPUT_TOKENS};
pub use seq2seq::{Seq2SeqOptions, Seq2SeqPipeline};
pub use token_classifier::TokenClassifierPipeline;

use std::path::{Path, PathBuf};

use ndarray::Array2;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::TensorRef;
use tokenizers::Tokenizer;

use serde::Serialize;

use crate::ep::ExecutionProvider;
use crate::error::CoreError;

/// A named entity span extracted from token classification output.
///
/// Contains the surface text, entity label, confidence score, and
/// character offsets in the original input string.
#[derive(Debug, Clone, Serialize)]
pub struct Entity {
    /// The surface text of the entity (original word, not subword).
    pub word: String,
    /// The entity label (e.g., "PER", "LOC", "ORG").
    pub entity: String,
    /// The confidence score from the model's softmax output.
    pub score: f32,
    /// Start character offset in the original input.
    pub start: usize,
    /// End character offset in the original input (exclusive).
    pub end: usize,
}

/// Output from a classifier inference pass.
///
/// Contains the top predicted label and its confidence score.
#[derive(Debug, Clone)]
pub struct ClassifierOutput {
    /// The predicted label (e.g., "POSITIVE", "NEGATIVE").
    pub label: String,
    /// The confidence score in the range [0.0, 1.0].
    pub score: f32,
}

/// Input accepted by [`PipelineKind::prepare`] (D-01).
///
/// Text-based pipelines consume `Text(String)`; ASR pipelines
/// consume `Audio(Vec<f32>)`. The `From<String>` impl enables
/// backward-compatible calls -- existing callers that pass a `String`
/// continue to work without modification.
pub enum InferenceInput {
    /// Text input for classifiers, embeddings, seq2seq, and token classifiers.
    Text(String),
    /// Raw audio samples (mono, f32, model-specific sample rate) for ASR.
    Audio(Vec<f32>),
}

impl From<String> for InferenceInput {
    fn from(text: String) -> Self {
        Self::Text(text)
    }
}

/// Prepared audio features ready for ONNX inference.
///
/// Holds mel spectrogram (or reshaped raw waveform) as a single
/// tensor ready for the ONNX session.
pub struct PreparedAudio {
    /// Mel spectrogram or reshaped waveform tensor (time_steps x features).
    pub(crate) features: Array2<f32>,
}

impl PreparedAudio {
    /// Construct a `PreparedAudio` for testing purposes.
    ///
    /// Mirrors `PreparedInput::new_for_test` -- bypasses the `pub(crate)`
    /// restriction so downstream crates can create test instances.
    pub fn new_for_test(features: Array2<f32>) -> Self {
        Self { features }
    }
}

/// Generic prepared data wrapping text or audio preprocessing output.
///
/// Returned by [`PipelineKind::prepare`], consumed by
/// [`PipelineKind::execute`] and [`PipelineKind::execute_batch`].
pub enum PreparedData {
    /// Tokenized text input.
    Text(PreparedInput),
    /// Preprocessed audio features.
    Audio(PreparedAudio),
}

impl PreparedData {
    /// Extract the inner [`PreparedInput`] if this is a `Text` variant.
    ///
    /// Returns `None` for `Audio`. Used by the batcher path to unwrap
    /// text inputs for batch tensor construction.
    pub fn into_text(self) -> Option<PreparedInput> {
        match self {
            Self::Text(t) => Some(t),
            Self::Audio(_) => None,
        }
    }
}

/// Typed output from [`PipelineKind::execute`] (D-02).
///
/// Replaces raw `serde_json::Value` with compile-time-checked variants
/// for each model profile. The [`to_json`](PipelineOutput::to_json)
/// method converts to JSON for REST/gRPC responses.
pub enum PipelineOutput {
    /// Classifier result with predicted label and confidence score.
    Classifier { label: String, score: f32 },
    /// Embedding vector (L2-normalized, unit length).
    Embeddings(Vec<f32>),
    /// Generated text from a seq2seq model (fused or encoder-decoder).
    Seq2Seq(String),
    /// Token-level entity spans from NER/POS models.
    TokenClassifier(Vec<Entity>),
    /// Transcribed text from an ASR model.
    Asr(String),
}

impl PipelineOutput {
    /// Convert typed output to a JSON value for HTTP/gRPC responses.
    ///
    /// Each variant produces the same JSON shape that the old
    /// `serde_json::json!` calls in `PipelineKind::execute` produced,
    /// ensuring backward compatibility with existing API consumers.
    pub fn to_json(&self) -> serde_json::Value {
        match self {
            Self::Classifier { label, score } => {
                serde_json::json!({ "label": label, "score": score })
            }
            Self::Embeddings(vec) => {
                serde_json::json!({ "embedding": vec })
            }
            Self::Seq2Seq(text) => {
                serde_json::json!({ "generated_text": text })
            }
            Self::TokenClassifier(entities) => {
                serde_json::json!({ "entities": entities })
            }
            Self::Asr(text) => {
                serde_json::json!({ "text": text })
            }
        }
    }
}

/// Prepared input ready for batch collection or immediate execution.
///
/// This type is opaque to callers outside the crate -- construct it
/// via [`Pipeline::prepare`] and pass it to [`Pipeline::execute`].
/// Fields are not publicly accessible.
pub struct PreparedInput {
    pub(crate) input_ids: Vec<i64>,
    pub(crate) attention_mask: Vec<i64>,
    pub(crate) sequence_length: usize,
    /// Optional tokenizer encoding preserved for token classification.
    ///
    /// Set to `Some` by `TokenClassifierPipeline::prepare()` so that
    /// `execute()` can access word IDs and offsets for BIO span merging.
    /// All other pipelines set this to `None`.
    pub(crate) encoding: Option<tokenizers::Encoding>,
}

impl PreparedInput {
    /// Construct a `PreparedInput` for testing purposes.
    ///
    /// This bypasses the `pub(crate)` restriction on fields so that
    /// tests in downstream crates (e.g., `hephaestus-api`) can create
    /// instances without a real tokenizer. Not intended for production use.
    pub fn new_for_test(
        input_ids: Vec<i64>,
        attention_mask: Vec<i64>,
        sequence_length: usize,
    ) -> Self {
        Self {
            input_ids,
            attention_mask,
            sequence_length,
            encoding: None,
        }
    }
}

/// Core inference pipeline trait.
///
/// Each model profile (classifier, embeddings, etc.) implements this trait.
/// Follows the Ousterhout deep module pattern: two methods hide tokenization,
/// tensor construction, ONNX inference, and post-processing complexity.
///
/// The two-step API (prepare then execute) enables future batching --
/// collect prepared inputs, then execute as a batch (Phase 4).
#[cfg_attr(test, mockall::automock(
    type Input = String;
    type Prepared = PreparedInput;
    type Output = ClassifierOutput;
))]
pub trait Pipeline {
    /// The raw input type accepted by this pipeline (e.g., `String` for text).
    type Input;
    /// The prepared representation after tokenization/preprocessing.
    type Prepared;
    /// The output produced by inference and post-processing.
    type Output;

    /// Tokenize and prepare input for inference.
    ///
    /// # Errors
    ///
    /// Returns [`CoreError`] if tokenization or input preparation fails.
    fn prepare(&self, input: Self::Input) -> Result<Self::Prepared, CoreError>;

    /// Run inference on prepared input and return results.
    ///
    /// # Errors
    ///
    /// Returns [`CoreError`] if inference or post-processing fails.
    fn execute(&mut self, prepared: Self::Prepared) -> Result<Self::Output, CoreError>;
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// Check that session outputs are non-empty before tensor extraction (WR-05).
///
/// Returns [`CoreError::Inference`] if the model produced zero output tensors,
/// which would otherwise cause a panic via direct `outputs[0]` indexing.
fn check_outputs_nonempty(outputs: &ort::session::SessionOutputs<'_>) -> Result<(), CoreError> {
    if outputs.len() == 0 {
        return Err(CoreError::Inference("model produced no output tensors".into()));
    }
    Ok(())
}

/// Resolve an ONNX model file path with `onnx/` subdirectory fallback.
///
/// Checks `{model_dir}/onnx/{filename}` first, then `{model_dir}/{filename}`.
/// Returns the first path that exists on disk.
///
/// # Errors
///
/// Returns [`CoreError::ModelLoad`] if neither path exists.
fn resolve_onnx_path(model_dir: &Path, filename: &str) -> Result<PathBuf, CoreError> {
    let onnx_subdir = model_dir.join("onnx").join(filename);
    let flat_path = model_dir.join(filename);
    if onnx_subdir.exists() {
        Ok(onnx_subdir)
    } else if flat_path.exists() {
        Ok(flat_path)
    } else {
        Err(CoreError::ModelLoad(format!(
            "ONNX model not found; tried '{}' and '{}'",
            onnx_subdir.display(),
            flat_path.display(),
        )))
    }
}

/// Build an ONNX Runtime session from a model file path.
///
/// Creates a session with Level3 graph optimization and the requested
/// execution providers. Shared by all pipeline constructors.
///
/// # Errors
///
/// Returns [`CoreError::ModelLoad`] if the session cannot be created.
fn build_onnx_session(path: &Path, ep: &ExecutionProvider) -> Result<Session, CoreError> {
    let providers = ep.to_ort_providers()?;

    let mut builder = Session::builder()
        .map_err(|e| CoreError::ModelLoad(e.to_string()))?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|e| CoreError::ModelLoad(e.to_string()))?;

    if !providers.is_empty() {
        builder = builder
            .with_execution_providers(providers)
            .map_err(|e| CoreError::ModelLoad(e.to_string()))?;
    }

    builder
        .commit_from_file(path)
        .map_err(|e| CoreError::ModelLoad(e.to_string()))
}

/// Load an ONNX session from a model directory.
///
/// Resolves the model file with `onnx/` subdirectory fallback, loads
/// with Level3 optimization, registers the requested execution
/// providers, loads the tokenizer, and validates inputs.
fn load_session_and_tokenizer(
    model_dir: &Path,
    ep: &ExecutionProvider,
) -> Result<(Session, Tokenizer), CoreError> {
    // 1. Resolve model file and build ONNX session.
    let model_path = resolve_onnx_path(model_dir, "model.onnx")?;
    let session = build_onnx_session(&model_path, ep)?;

    // 2. Load tokenizer.
    let tokenizer_path = model_dir.join("tokenizer.json");
    let mut tokenizer = Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| CoreError::Tokenization(e.to_string()))?;

    // 3. Configure truncation to mitigate T-01-02 DoS threat.
    tokenizer
        .with_truncation(Some(tokenizers::TruncationParams {
            max_length: 512,
            ..Default::default()
        }))
        .map_err(|e| CoreError::Tokenization(e.to_string()))?;

    // 4. Validate tokenizer-model compatibility (TOKN-03).
    let model_input_names: Vec<String> = session
        .inputs()
        .iter()
        .map(|input| input.name().to_string())
        .collect();
    let model_input_name_refs: Vec<&str> =
        model_input_names.iter().map(String::as_str).collect();
    let required_inputs = ["input_ids", "attention_mask"];
    for required in &required_inputs {
        if !model_input_name_refs.contains(required) {
            return Err(CoreError::ModelValidation(format!(
                "model does not accept input '{required}'; model inputs are: {model_input_name_refs:?}",
            )));
        }
    }

    Ok((session, tokenizer))
}

/// Tokenize text into prepared input (shared by all text-based pipelines).
fn tokenize_text(
    tokenizer: &Tokenizer,
    input: String,
) -> Result<PreparedInput, CoreError> {
    let encoding = tokenizer
        .encode(input.as_str(), true)
        .map_err(|e| CoreError::Tokenization(e.to_string()))?;

    let input_ids: Vec<i64> = encoding.get_ids().iter().map(|&id| i64::from(id)).collect();
    let attention_mask: Vec<i64> = encoding
        .get_attention_mask()
        .iter()
        .map(|&m| i64::from(m))
        .collect();
    let sequence_length = encoding.len();

    Ok(PreparedInput {
        input_ids,
        attention_mask,
        sequence_length,
        encoding: None,
    })
}

/// Check whether an ONNX session declares a `token_type_ids` input.
///
/// BERT-family models require this input (a zeros tensor of the same
/// shape as `input_ids`); DistilBERT-family models omit it. The check
/// lets us conditionally provide it for backward compatibility.
fn session_expects_token_type_ids(session: &Session) -> bool {
    session
        .inputs()
        .iter()
        .any(|input| input.name() == "token_type_ids")
}

/// Run ONNX inference with input_ids, attention_mask, and optional
/// token_type_ids tensors.
///
/// Conditionally includes a `token_type_ids` zeros tensor when the
/// session expects it (BERT-family models). DistilBERT models that
/// omit `token_type_ids` from their inputs are handled without it.
fn run_onnx_inference<'s>(
    session: &'s mut Session,
    prepared: &PreparedInput,
) -> Result<ort::session::SessionOutputs<'s>, CoreError> {
    let seq_len = prepared.sequence_length;
    let needs_token_type_ids = session_expects_token_type_ids(session);

    let input_ids_view =
        ndarray::ArrayView2::from_shape((1, seq_len), &prepared.input_ids)
            .map_err(|e| CoreError::Inference(e.to_string()))?;
    let attention_mask_view =
        ndarray::ArrayView2::from_shape((1, seq_len), &prepared.attention_mask)
            .map_err(|e| CoreError::Inference(e.to_string()))?;
    let token_type_ids_array = Array2::<i64>::zeros((1, seq_len));

    let input_ids_tensor = TensorRef::from_array_view(input_ids_view)
        .map_err(|e| CoreError::Inference(e.to_string()))?;
    let attention_mask_tensor =
        TensorRef::from_array_view(attention_mask_view)
            .map_err(|e| CoreError::Inference(e.to_string()))?;

    if needs_token_type_ids {
        let token_type_ids_tensor =
            TensorRef::from_array_view(token_type_ids_array.view())
                .map_err(|e| CoreError::Inference(e.to_string()))?;
        session
            .run(ort::inputs![
                "input_ids" => input_ids_tensor,
                "attention_mask" => attention_mask_tensor,
                "token_type_ids" => token_type_ids_tensor,
            ])
            .map_err(|e| CoreError::Inference(e.to_string()))
    } else {
        session
            .run(ort::inputs![
                "input_ids" => input_ids_tensor,
                "attention_mask" => attention_mask_tensor,
            ])
            .map_err(|e| CoreError::Inference(e.to_string()))
    }
}

// ---------------------------------------------------------------------------
// PipelineKind enum dispatch (D-03)
// ---------------------------------------------------------------------------

/// Enum dispatch wrapper for all pipeline types (D-03).
///
/// `AppState` holds `RwLock<PipelineKind>` instead of a concrete pipeline.
/// The handler matches on the variant to dispatch prepare/execute calls.
/// No trait objects, no dynamic dispatch overhead.
pub enum PipelineKind {
    /// Text classification pipeline.
    Classifier(ClassifierPipeline),
    /// Sentence/document embeddings pipeline.
    Embeddings(EmbeddingsPipeline),
    /// Seq2seq pipeline: fused single-pass or encoder-decoder (D-10).
    Seq2Seq(Seq2SeqPipeline),
    /// Token classification pipeline (NER, POS).
    TokenClassifier(TokenClassifierPipeline),
    /// Automatic speech recognition pipeline (CTC or Whisper).
    Asr(AsrPipeline),
}

impl PipelineKind {
    /// Prepare input for any profile (D-01).
    ///
    /// Accepts `String` (backward compat via `From<String>`) or
    /// `InferenceInput` directly. Text pipelines reject audio input
    /// with [`CoreError::InvalidInput`]; the ASR variant will be
    /// added in Plan 11-03.
    pub fn prepare(&self, input: impl Into<InferenceInput>) -> Result<PreparedData, CoreError> {
        let input = input.into();
        match (self, input) {
            (Self::Classifier(p), InferenceInput::Text(text)) => {
                Ok(PreparedData::Text(p.prepare(text)?))
            }
            (Self::Embeddings(p), InferenceInput::Text(text)) => {
                Ok(PreparedData::Text(p.prepare(text)?))
            }
            (Self::Seq2Seq(p), InferenceInput::Text(text)) => {
                Ok(PreparedData::Text(p.prepare(text)?))
            }
            (Self::TokenClassifier(p), InferenceInput::Text(text)) => {
                Ok(PreparedData::Text(p.prepare(text)?))
            }
            (Self::Asr(p), InferenceInput::Audio(audio)) => {
                Ok(PreparedData::Audio(p.prepare(audio)?))
            }
            (Self::Asr(_), InferenceInput::Text(_)) => Err(CoreError::InvalidInput(
                "ASR pipeline requires audio input".to_string(),
            )),
            (_, InferenceInput::Audio(_)) => Err(CoreError::InvalidInput(
                "text pipeline requires text input, got audio".to_string(),
            )),
        }
    }

    /// Execute batched inference for multiple prepared inputs.
    ///
    /// Pads all inputs to the maximum sequence length, constructs batch
    /// tensors, runs a single `session.run()` call, then splits the
    /// output by sample and applies profile-specific post-processing.
    ///
    /// Returns one `Result` per input sample. The method matches on the
    /// variant to access session, tokenizer, and id2label directly,
    /// avoiding borrow conflicts between the mutable session borrow
    /// (needed for `session.run()`) and the immutable reads for
    /// post-processing.
    pub fn execute_batch(
        &mut self,
        batch: Vec<PreparedData>,
    ) -> Vec<Result<PipelineOutput, CoreError>> {
        if batch.is_empty() {
            return Vec::new();
        }

        // Encoder-decoder seq2seq generates token by token, so its inputs
        // cannot share the single padded forward pass built below; that
        // mode works through the batch itself. Every other pipeline,
        // fused seq2seq included, continues unchanged.
        if let Self::Seq2Seq(p) = self
            && p.is_encoder_decoder()
        {
            return p.execute_batch_sequential(batch);
        }

        // Extract PreparedInput from each PreparedData::Text item.
        // Audio items in a text-pipeline batch produce per-item errors.
        let batch_size = batch.len();
        let mut text_batch: Vec<PreparedInput> = Vec::with_capacity(batch_size);
        let mut audio_indices: Vec<usize> = Vec::new();
        for (i, item) in batch.into_iter().enumerate() {
            match item {
                PreparedData::Text(t) => text_batch.push(t),
                PreparedData::Audio(_) => {
                    audio_indices.push(i);
                    // Push a dummy to keep indexing aligned.
                    text_batch.push(PreparedInput::new_for_test(vec![0], vec![0], 1));
                }
            }
        }

        // If all items are audio, short-circuit with errors.
        if audio_indices.len() == batch_size {
            return (0..batch_size)
                .map(|_| {
                    Err(CoreError::InvalidInput(
                        "text pipeline received audio prepared data in batch".to_string(),
                    ))
                })
                .collect();
        }

        let batch = text_batch;
        let max_seq_len = batch.iter().map(|p| p.sequence_length).max().unwrap_or(0);

        // Check if the session expects token_type_ids (BERT vs DistilBERT).
        // ASR models do not use token_type_ids.
        let needs_tti = match self {
            Self::Classifier(p) => session_expects_token_type_ids(&p.session),
            Self::Embeddings(p) => session_expects_token_type_ids(&p.session),
            Self::Seq2Seq(p) => session_expects_token_type_ids(&p.session),
            Self::TokenClassifier(p) => session_expects_token_type_ids(&p.session),
            Self::Asr(_) => false,
        };

        // Pad and stack into batch tensors.
        let (input_ids_array, attention_mask_array) =
            match pad_and_stack(&batch, batch_size, max_seq_len) {
                Ok(arrays) => arrays,
                Err(e) => {
                    return (0..batch_size)
                        .map(|_| Err(CoreError::Inference(e.to_string())))
                        .collect();
                }
            };
        let token_type_ids_array = Array2::<i64>::zeros((batch_size, max_seq_len));

        let input_ids_tensor = match TensorRef::from_array_view(input_ids_array.view()) {
            Ok(t) => t,
            Err(e) => {
                return (0..batch_size)
                    .map(|_| Err(CoreError::Inference(e.to_string())))
                    .collect();
            }
        };
        let attention_mask_tensor = match TensorRef::from_array_view(attention_mask_array.view()) {
            Ok(t) => t,
            Err(e) => {
                return (0..batch_size)
                    .map(|_| Err(CoreError::Inference(e.to_string())))
                    .collect();
            }
        };

        let ort_inputs = if needs_tti {
            let tti_tensor = match TensorRef::from_array_view(token_type_ids_array.view()) {
                Ok(t) => t,
                Err(e) => {
                    return (0..batch_size)
                        .map(|_| Err(CoreError::Inference(e.to_string())))
                        .collect();
                }
            };
            ort::inputs![
                "input_ids" => input_ids_tensor,
                "attention_mask" => attention_mask_tensor,
                "token_type_ids" => tti_tensor,
            ]
        } else {
            ort::inputs![
                "input_ids" => input_ids_tensor,
                "attention_mask" => attention_mask_tensor,
            ]
        };

        // Match on variant to access session + postprocessing resources
        // within the same borrow scope.
        let mut results = match self {
            Self::Classifier(p) => {
                let outputs = match p.session.run(ort_inputs) {
                    Ok(o) => o,
                    Err(e) => {
                        let msg = e.to_string();
                        return (0..batch_size).map(|_| Err(CoreError::Inference(msg.clone()))).collect();
                    }
                };
                classifier::batch_postprocess_classifier(outputs, batch_size, &p.id2label)
            }
            Self::Embeddings(p) => {
                let outputs = match p.session.run(ort_inputs) {
                    Ok(o) => o,
                    Err(e) => {
                        let msg = e.to_string();
                        return (0..batch_size).map(|_| Err(CoreError::Inference(msg.clone()))).collect();
                    }
                };
                embeddings::batch_postprocess_embeddings(outputs, batch_size, max_seq_len, &attention_mask_array)
            }
            Self::Seq2Seq(p) => {
                let outputs = match p.session.run(ort_inputs) {
                    Ok(o) => o,
                    Err(e) => {
                        let msg = e.to_string();
                        return (0..batch_size).map(|_| Err(CoreError::Inference(msg.clone()))).collect();
                    }
                };
                seq2seq::batch_postprocess_seq2seq(outputs, batch_size, &p.tokenizer)
            }
            Self::TokenClassifier(p) => {
                let outputs = match p.session.run(ort_inputs) {
                    Ok(o) => o,
                    Err(e) => {
                        let msg = e.to_string();
                        return (0..batch_size).map(|_| Err(CoreError::Inference(msg.clone()))).collect();
                    }
                };
                token_classifier::batch_postprocess_token_classifier(outputs, batch, batch_size, max_seq_len, &p.id2label, &p.tokenizer)
            }
            Self::Asr(_) => {
                return (0..batch_size)
                    .map(|_| Err(CoreError::InvalidInput(
                        "ASR pipeline does not support batched execution".to_string(),
                    )))
                    .collect();
            }
        };

        // Replace results for audio indices with errors.
        for &idx in &audio_indices {
            if idx < results.len() {
                results[idx] = Err(CoreError::InvalidInput(
                    "text pipeline received audio prepared data in batch".to_string(),
                ));
            }
        }

        results
    }

    /// Execute single inference and return typed output (D-02, D-05).
    ///
    /// Accepts [`PreparedData`] and returns [`PipelineOutput`]. Callers
    /// convert to JSON via [`PipelineOutput::to_json()`] when needed.
    pub fn execute(
        &mut self,
        prepared: PreparedData,
    ) -> Result<PipelineOutput, CoreError> {
        match (self, prepared) {
            (Self::Classifier(p), PreparedData::Text(t)) => {
                let out = p.execute(t)?;
                Ok(PipelineOutput::Classifier {
                    label: out.label,
                    score: out.score,
                })
            }
            (Self::Embeddings(p), PreparedData::Text(t)) => {
                let out = p.execute(t)?;
                Ok(PipelineOutput::Embeddings(out))
            }
            (Self::Seq2Seq(p), PreparedData::Text(t)) => {
                let out = p.execute(t)?;
                Ok(PipelineOutput::Seq2Seq(out))
            }
            (Self::TokenClassifier(p), PreparedData::Text(t)) => {
                let out = p.execute(t)?;
                Ok(PipelineOutput::TokenClassifier(out))
            }
            (Self::Asr(p), PreparedData::Audio(a)) => {
                let out = p.execute(a)?;
                Ok(PipelineOutput::Asr(out))
            }
            (Self::Asr(_), PreparedData::Text(_)) => Err(CoreError::InvalidInput(
                "ASR pipeline requires audio prepared data".to_string(),
            )),
            (_, PreparedData::Audio(_)) => Err(CoreError::InvalidInput(
                "text pipeline received audio prepared data".to_string(),
            )),
        }
    }
}

// ---------------------------------------------------------------------------
// Batch post-processing free functions
// ---------------------------------------------------------------------------
//
// These are free functions (not methods on PipelineKind) to avoid borrow
// conflicts: `session.run()` borrows `self` mutably (via the inner pipeline),
// and the returned `SessionOutputs` holds that borrow. Post-processing needs
// immutable access to tokenizer/id2label from the same inner pipeline, so
// the variant match in `execute_batch` passes these references directly.

/// Pad and stack batch inputs into 2D tensors.
///
/// Pads each sample's `input_ids` and `attention_mask` with zeros to
/// `max_seq_len`, then stacks into `(batch_size, max_seq_len)` arrays.
fn pad_and_stack(
    batch: &[PreparedInput],
    batch_size: usize,
    max_seq_len: usize,
) -> Result<(Array2<i64>, Array2<i64>), CoreError> {
    let mut input_ids_flat = Vec::with_capacity(batch_size * max_seq_len);
    let mut attention_mask_flat = Vec::with_capacity(batch_size * max_seq_len);
    for prepared in batch {
        let pad_len = max_seq_len - prepared.sequence_length;
        input_ids_flat.extend_from_slice(&prepared.input_ids);
        input_ids_flat.extend(std::iter::repeat_n(0i64, pad_len));
        attention_mask_flat.extend_from_slice(&prepared.attention_mask);
        attention_mask_flat.extend(std::iter::repeat_n(0i64, pad_len));
    }

    let input_ids_array = Array2::from_shape_vec((batch_size, max_seq_len), input_ids_flat)
        .map_err(|e| CoreError::Inference(format!("batch tensor shape error: {e}")))?;
    let attention_mask_array =
        Array2::from_shape_vec((batch_size, max_seq_len), attention_mask_flat)
            .map_err(|e| CoreError::Inference(format!("batch tensor shape error: {e}")))?;

    Ok((input_ids_array, attention_mask_array))
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Extract `id2label` mapping from a parsed `config.json` value.
///
/// The `id2label` field is a JSON object with string keys ("0", "1", ...)
/// mapping to label strings. Returns a `Vec<String>` ordered by numeric key.
///
/// # Errors
///
/// Returns [`CoreError::ModelValidation`] if the `id2label` field is missing,
/// keys are non-numeric, values are non-string, or keys are not contiguous
/// from 0 (CR-02).
fn extract_id2label(config: &serde_json::Value) -> Result<Vec<String>, CoreError> {
    let id2label_obj = config
        .get("id2label")
        .and_then(|v| v.as_object())
        .ok_or_else(|| {
            CoreError::ModelValidation(
                "config.json missing 'id2label' object".to_string(),
            )
        })?;

    let mut entries: Vec<(usize, String)> = Vec::with_capacity(id2label_obj.len());
    for (key, value) in id2label_obj {
        let idx: usize = key.parse().map_err(|_| {
            CoreError::ModelValidation(format!(
                "id2label key '{key}' is not a valid numeric index",
            ))
        })?;
        let label = value.as_str().ok_or_else(|| {
            CoreError::ModelValidation(format!(
                "id2label value for key '{key}' is not a string",
            ))
        })?;
        entries.push((idx, label.to_string()));
    }

    entries.sort_by_key(|(idx, _)| *idx);

    // CR-02: Validate that keys are contiguous from 0..N.
    for (expected_index, (actual_key, _)) in entries.iter().enumerate() {
        if expected_index != *actual_key {
            return Err(CoreError::ModelValidation(format!(
                "id2label keys must be contiguous from 0; expected key {expected_index}, found {actual_key}",
            )));
        }
    }

    Ok(entries.into_iter().map(|(_, label)| label).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_id2label_parsing() {
        // Arrange
        let config: serde_json::Value = serde_json::json!({
            "id2label": {
                "0": "NEGATIVE",
                "1": "POSITIVE"
            }
        });

        // Act
        let labels = extract_id2label(&config).expect("should parse id2label");

        // Assert
        assert_eq!(labels, vec!["NEGATIVE", "POSITIVE"]);
    }

    #[test]
    fn test_id2label_ordering() {
        // Arrange -- keys intentionally out of order
        let config: serde_json::Value = serde_json::json!({
            "id2label": {
                "2": "NEUTRAL",
                "0": "NEGATIVE",
                "1": "POSITIVE"
            }
        });

        // Act
        let labels = extract_id2label(&config).expect("should parse id2label");

        // Assert -- must be sorted by numeric key
        assert_eq!(labels, vec!["NEGATIVE", "POSITIVE", "NEUTRAL"]);
    }

    #[test]
    fn test_id2label_missing() {
        // Arrange
        let config: serde_json::Value = serde_json::json!({
            "model_type": "distilbert"
        });

        // Act
        let result = extract_id2label(&config);

        // Assert
        assert!(result.is_err());
    }

    #[test]
    fn test_id2label_rejects_noncontiguous() {
        // Arrange -- gap at key 1
        let config: serde_json::Value = serde_json::json!({
            "id2label": {
                "0": "NEG",
                "2": "POS"
            }
        });

        // Act
        let result = extract_id2label(&config);

        // Assert
        assert!(result.is_err(), "non-contiguous keys should be rejected");
        let msg = format!("{}", result.unwrap_err());
        assert!(
            msg.contains("contiguous"),
            "error should mention 'contiguous': {msg}"
        );
    }

    #[test]
    fn test_id2label_accepts_contiguous() {
        // Arrange -- contiguous keys 0, 1, 2
        let config: serde_json::Value = serde_json::json!({
            "id2label": {
                "0": "A",
                "1": "B",
                "2": "C"
            }
        });

        // Act
        let result = extract_id2label(&config);

        // Assert
        assert!(result.is_ok(), "contiguous keys should be accepted");
        assert_eq!(result.unwrap(), vec!["A", "B", "C"]);
    }

    #[test]
    fn test_mock_pipeline_prepare_execute() {
        // Arrange
        let mut mock = MockPipeline::new();

        mock.expect_prepare()
            .returning(|_input: String| {
                Ok(PreparedInput {
                    input_ids: vec![101, 2023, 102],
                    attention_mask: vec![1, 1, 1],
                    sequence_length: 3,
                    encoding: None,
                })
            });

        mock.expect_execute()
            .returning(|_prepared: PreparedInput| {
                Ok(ClassifierOutput {
                    label: "POSITIVE".to_string(),
                    score: 0.95,
                })
            });

        // Act
        let prepared = mock
            .prepare("test input".to_string())
            .expect("prepare should succeed");
        let output = mock.execute(prepared).expect("execute should succeed");

        // Assert
        assert_eq!(output.label, "POSITIVE");
        assert!((output.score - 0.95).abs() < 1e-6);
    }

    #[test]
    fn test_token_type_ids_zeros_tensor_shape() {
        // Verify zeros tensor construction matches expected shape for
        // both single inference (1, seq_len) and batch (batch_size, seq_len).
        let single = Array2::<i64>::zeros((1, 7));
        assert_eq!(single.shape(), &[1, 7]);
        assert!(single.iter().all(|&v| v == 0));

        let batch = Array2::<i64>::zeros((4, 12));
        assert_eq!(batch.shape(), &[4, 12]);
        assert!(batch.iter().all(|&v| v == 0));
    }

    #[test]
    fn inference_input_from_string_produces_text() {
        let input = InferenceInput::from("hello".to_string());
        assert!(matches!(input, InferenceInput::Text(ref s) if s == "hello"));
    }

    #[test]
    fn prepared_data_into_text_returns_some_for_text() {
        let prepared = PreparedInput::new_for_test(vec![101], vec![1], 1);
        let data = PreparedData::Text(prepared);
        assert!(data.into_text().is_some());
    }

    #[test]
    fn prepared_data_into_text_returns_none_for_audio() {
        let features = Array2::<f32>::zeros((10, 80));
        let audio = PreparedAudio::new_for_test(features);
        let data = PreparedData::Audio(audio);
        assert!(data.into_text().is_none());
    }

    #[test]
    fn pipeline_output_classifier_to_json() {
        let output = PipelineOutput::Classifier {
            label: "POSITIVE".to_string(),
            score: 0.99,
        };
        let json = output.to_json();
        assert_eq!(json["label"], "POSITIVE");
        let score = json["score"].as_f64().unwrap();
        assert!((score - 0.99).abs() < 1e-4);
    }

    #[test]
    fn pipeline_output_embeddings_to_json() {
        let output = PipelineOutput::Embeddings(vec![0.1, 0.2, 0.3]);
        let json = output.to_json();
        let arr = json["embedding"].as_array().unwrap();
        assert_eq!(arr.len(), 3);
    }

    #[test]
    fn pipeline_output_seq2seq_to_json() {
        let output = PipelineOutput::Seq2Seq("translated text".to_string());
        let json = output.to_json();
        assert_eq!(json["generated_text"], "translated text");
    }

    #[test]
    fn pipeline_output_token_classifier_to_json() {
        let entities = vec![Entity {
            word: "John".to_string(),
            entity: "PER".to_string(),
            score: 0.98,
            start: 0,
            end: 4,
        }];
        let output = PipelineOutput::TokenClassifier(entities);
        let json = output.to_json();
        let arr = json["entities"].as_array().unwrap();
        assert_eq!(arr.len(), 1);
        assert_eq!(arr[0]["word"], "John");
        assert_eq!(arr[0]["entity"], "PER");
    }

    #[test]
    fn pipeline_output_asr_to_json() {
        let output = PipelineOutput::Asr("hello world".to_string());
        let json = output.to_json();
        assert_eq!(json["text"], "hello world");
    }

    #[test]
    fn test_pipeline_kind_variant_sizes() {
        // Verify PipelineKind doesn't trigger clippy large_enum_variant.
        // Both variants hold Session + Tokenizer + optional metadata,
        // so sizes should be comparable.
        let classifier_size = std::mem::size_of::<ClassifierPipeline>();
        let embeddings_size = std::mem::size_of::<EmbeddingsPipeline>();

        // Neither should be more than 3x the other.
        let ratio = if classifier_size > embeddings_size {
            classifier_size as f64 / embeddings_size as f64
        } else {
            embeddings_size as f64 / classifier_size as f64
        };

        assert!(
            ratio < 3.0,
            "variant size ratio {ratio:.1} exceeds 3x -- consider boxing the larger variant"
        );
    }
}
