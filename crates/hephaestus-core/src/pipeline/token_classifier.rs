//! Token classification pipeline (NER, POS tagging).

use std::path::Path;

use ort::session::Session;
use tokenizers::Tokenizer;

use crate::ep::ExecutionProvider;
use crate::error::CoreError;
use crate::postprocess;

use super::{
    check_outputs_nonempty, extract_id2label, load_session_and_tokenizer, run_onnx_inference,
    Entity, Pipeline, PipelineOutput, PreparedInput,
};

/// Token classification pipeline for NER, POS tagging, and similar tasks.
///
/// Loads an ONNX token classification model and its tokenizer, runs
/// inference to produce per-token logits, applies argmax to get
/// predicted labels, then merges subword tokens into word-level
/// entity spans using BIO tag conventions.
pub struct TokenClassifierPipeline {
    pub(crate) session: Session,
    pub(crate) tokenizer: Tokenizer,
    pub(crate) id2label: Vec<String>,
}

impl TokenClassifierPipeline {
    /// Construct a new token classification pipeline from a model directory.
    ///
    /// The directory must contain:
    /// - An ONNX model file (`onnx/model.onnx` or `model.onnx`)
    /// - `tokenizer.json`
    /// - `config.json` with an `id2label` mapping
    ///
    /// # Errors
    ///
    /// Returns [`CoreError`] if any required file is missing or invalid,
    /// or if the tokenizer outputs are incompatible with the model inputs.
    pub fn new(model_dir: &Path, ep: &ExecutionProvider) -> Result<Self, CoreError> {
        let (session, tokenizer) = load_session_and_tokenizer(model_dir, ep)?;

        // Load id2label from config.json.
        let config_path = model_dir.join("config.json");
        let config_text = std::fs::read_to_string(&config_path)?;
        let config: serde_json::Value = serde_json::from_str(&config_text)?;
        let id2label = extract_id2label(&config)?;

        Ok(Self {
            session,
            tokenizer,
            id2label,
        })
    }
}

/// Tokenize text for token classification: like every text pipeline, but
/// the encoding is kept for merging subword tokens into words.
pub(super) fn tokenize_keeping_encoding(
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
        encoding: Some(encoding),
    })
}

impl Pipeline for TokenClassifierPipeline {
    type Input = String;
    type Prepared = PreparedInput;
    type Output = Vec<Entity>;

    fn prepare(&self, input: String) -> Result<PreparedInput, CoreError> {
        tokenize_keeping_encoding(&self.tokenizer, input)
    }

    fn execute(
        &mut self,
        prepared: PreparedInput,
    ) -> Result<Vec<Entity>, CoreError> {
        let encoding = prepared.encoding.as_ref().ok_or_else(|| {
            CoreError::Inference(
                "TokenClassifierPipeline requires encoding in PreparedInput".to_string(),
            )
        })?;

        let num_tokens = prepared.sequence_length;
        let outputs = run_onnx_inference(&mut self.session, &prepared)?;

        // WR-05: Guard against models with zero output tensors.
        check_outputs_nonempty(&outputs)?;

        // Extract logits tensor -- shape (1, num_tokens, num_labels).
        let tensor = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| CoreError::Inference(e.to_string()))?;
        let (shape, data) = tensor;

        let num_labels = if shape.len() == 3 {
            shape[2] as usize
        } else {
            return Err(CoreError::Inference(format!(
                "expected 3D output tensor (batch, seq_len, num_labels), got {}-D shape",
                shape.len()
            )));
        };

        // Per-token softmax + argmax to get predicted label indices and probability scores.
        let predictions = postprocess::softmax_argmax_per_token(data, num_tokens, num_labels)?;

        // Merge subword tokens into word-level entity spans.
        let mut entities =
            postprocess::merge_subword_entities(&predictions, encoding, &self.id2label);

        for entity in &mut entities {
            // Decode the token IDs corresponding to this entity's char span.
            // Since we have the full encoding, find tokens whose offsets overlap.
            let token_ids: Vec<u32> = encoding
                .get_ids()
                .iter()
                .zip(encoding.get_offsets())
                .filter(|(_, (start, end))| *start < entity.end && *end > entity.start)
                .map(|(&id, _)| id)
                .collect();

            entity.word = self
                .tokenizer
                .decode(&token_ids, true)
                .unwrap_or_default();
        }

        Ok(entities)
    }
}

/// Token classifier batch post-processing: per-token argmax + subword merging per sample.
pub(super) fn batch_postprocess_token_classifier(
    outputs: ort::session::SessionOutputs<'_>,
    batch: Vec<PreparedInput>,
    batch_size: usize,
    max_seq_len: usize,
    id2label: &[String],
    tokenizer: &Tokenizer,
) -> Vec<Result<PipelineOutput, CoreError>> {
    // WR-05: Guard against models with zero output tensors.
    if let Err(e) = check_outputs_nonempty(&outputs) {
        return (0..batch_size)
            .map(|_| Err(CoreError::Inference(e.to_string())))
            .collect();
    }
    let tensor = match outputs[0].try_extract_tensor::<f32>() {
        Ok(t) => t,
        Err(e) => {
            let msg = e.to_string();
            return (0..batch_size)
                .map(|_| Err(CoreError::Inference(msg.clone())))
                .collect();
        }
    };
    let (shape, data) = tensor;
    let num_labels = if shape.len() == 3 { shape[2] as usize } else { 0 };
    if num_labels == 0 {
        return (0..batch_size)
            .map(|_| Err(CoreError::Inference("unexpected token classifier output shape".into())))
            .collect();
    }

    batch
        .into_iter()
        .enumerate()
        .map(|(i, prepared)| {
            let num_tokens = prepared.sequence_length;
            let sample_start = i * max_seq_len * num_labels;
            let sample_data = &data[sample_start..sample_start + num_tokens * num_labels];
            let predictions = postprocess::softmax_argmax_per_token(sample_data, num_tokens, num_labels)?;

            let encoding = match &prepared.encoding {
                Some(enc) => enc,
                None => {
                    return Err(CoreError::Inference(
                        "TokenClassifier batch requires encoding in PreparedInput".into(),
                    ))
                }
            };

            let mut entities =
                postprocess::merge_subword_entities(&predictions, encoding, id2label);

            for entity in &mut entities {
                let token_ids: Vec<u32> = encoding
                    .get_ids()
                    .iter()
                    .zip(encoding.get_offsets())
                    .filter(|(_, (start, end))| *start < entity.end && *end > entity.start)
                    .map(|(&id, _)| id)
                    .collect();
                entity.word = tokenizer.decode(&token_ids, true).unwrap_or_default();
            }

            Ok(PipelineOutput::TokenClassifier(entities))
        })
        .collect()
}
