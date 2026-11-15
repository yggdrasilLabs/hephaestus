//! Seq2seq pipeline: fused single-pass models and encoder-decoder exports.

use std::path::Path;
use std::time::Duration;

use ort::session::Session;
use tokenizers::Tokenizer;

use crate::ep::ExecutionProvider;
use crate::error::CoreError;

use super::encoder_decoder::{EncoderDecoder, EncoderDecoderFiles, SEQ2SEQ_MIN_INPUT_TOKENS};
use super::{
    check_outputs_nonempty, load_session_and_tokenizer, resolve_onnx_path, run_onnx_inference,
    tokenize_text, Pipeline, PipelineOutput, PreparedData, PreparedInput,
};

/// Optional settings for [`Seq2SeqPipeline`].
///
/// The default value changes nothing: [`Seq2SeqPipeline::with_options`]
/// with default options is exactly [`Seq2SeqPipeline::new`].
///
/// ```
/// use std::time::Duration;
///
/// use hephaestus_core::Seq2SeqOptions;
///
/// let options = Seq2SeqOptions::new()
///     .with_max_input_tokens(1024)
///     .with_max_new_tokens(200)
///     .with_timeout(Duration::from_secs(30));
/// assert_ne!(options, Seq2SeqOptions::default());
/// ```
#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[must_use = "options do nothing until passed to Seq2SeqPipeline::with_options"]
pub struct Seq2SeqOptions {
    max_input_tokens: Option<usize>,
    max_new_tokens: Option<usize>,
    timeout: Option<Duration>,
}

impl Seq2SeqOptions {
    /// Options that leave every setting at its default.
    pub fn new() -> Self {
        Self::default()
    }

    /// Truncate inputs to at most `tokens` tokens, special tokens included.
    ///
    /// Unset, inputs are truncated to 512 tokens like every other text
    /// pipeline. Applies to both seq2seq modes.
    pub fn with_max_input_tokens(mut self, tokens: usize) -> Self {
        self.max_input_tokens = Some(tokens);
        self
    }

    /// Generate at most `tokens` tokens per input in encoder-decoder mode.
    ///
    /// Unset, the limit comes from the model's own configuration. A fused
    /// model has its output length baked into the graph and ignores this.
    pub fn with_max_new_tokens(mut self, tokens: usize) -> Self {
        self.max_new_tokens = Some(tokens);
        self
    }

    /// Abort encoder-decoder generation that runs longer than `timeout`.
    ///
    /// The budget covers one input, or one batch when inputs arrive
    /// batched. Unset, generation is bounded by the token limit alone. A
    /// fused model runs a single forward pass and ignores this.
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    pub(super) fn max_input_tokens(&self) -> Option<usize> {
        self.max_input_tokens
    }

    pub(super) fn max_new_tokens(&self) -> Option<usize> {
        self.max_new_tokens
    }

    pub(super) fn timeout(&self) -> Option<Duration> {
        self.timeout
    }
}

/// Seq2seq pipeline backed by ONNX models (D-10).
///
/// Runs in one of two modes, chosen only by the files in the model
/// directory:
///
/// - **Fused**, when `onnx/model.onnx` or `model.onnx` exists: a single
///   graph (e.g., exported with beam search baked in) returns finished
///   token IDs in one forward pass. No auto-regressive decode loop.
/// - **Encoder-decoder**, otherwise, when `encoder_model.onnx` and a
///   decoder graph exist (the standard Optimum export): the encoder runs
///   once, then a greedy decode loop runs the decoder once per token.
///
/// Both modes decode the token IDs back to text via the tokenizer.
pub struct Seq2SeqPipeline {
    /// The fused model, or the encoder when `decoder` is `Some`.
    pub(crate) session: Session,
    pub(crate) tokenizer: Tokenizer,
    /// Decoder state; `Some` only in encoder-decoder mode.
    decoder: Option<Box<EncoderDecoder>>,
}

impl Seq2SeqPipeline {
    /// Construct a new seq2seq pipeline from a model directory.
    ///
    /// The directory must contain:
    /// - An ONNX model file (`onnx/model.onnx` or `model.onnx`), or an
    ///   encoder-decoder export (`encoder_model.onnx` plus
    ///   `decoder_model.onnx` or `decoder_model_merged.onnx`, together in
    ///   `onnx/` or in the directory itself)
    /// - `tokenizer.json`
    /// - `config.json` (encoder-decoder export only; `generation_config.json`
    ///   is read when present)
    ///
    /// # Errors
    ///
    /// Returns [`CoreError`] if any required file is missing or invalid,
    /// or if the tokenizer outputs are incompatible with the model inputs.
    pub fn new(model_dir: &Path, ep: &ExecutionProvider) -> Result<Self, CoreError> {
        Self::with_options(model_dir, ep, &Seq2SeqOptions::default())
    }

    /// Construct a new seq2seq pipeline with explicit [`Seq2SeqOptions`].
    ///
    /// See [`Seq2SeqPipeline::new`] for the files the directory must hold.
    ///
    /// # Errors
    ///
    /// Returns [`CoreError`] if any required file is missing or invalid,
    /// if a graph does not have the inputs and outputs its mode needs, or
    /// if an option is out of range for the model.
    pub fn with_options(
        model_dir: &Path,
        ep: &ExecutionProvider,
        options: &Seq2SeqOptions,
    ) -> Result<Self, CoreError> {
        // A fused model always wins: the encoder-decoder export is only
        // considered when no single-file model exists.
        if resolve_onnx_path(model_dir, "model.onnx").is_err()
            && let Some(files) = EncoderDecoderFiles::locate(model_dir)?
        {
            let (session, tokenizer, decoder) =
                EncoderDecoder::load(&files, model_dir, ep, options)?;
            return Ok(Self {
                session,
                tokenizer,
                decoder: Some(Box::new(decoder)),
            });
        }

        let (session, mut tokenizer) = load_session_and_tokenizer(model_dir, ep)?;
        if let Some(max_length) = options.max_input_tokens() {
            if max_length < SEQ2SEQ_MIN_INPUT_TOKENS {
                return Err(CoreError::Config(format!(
                    "seq2seq max input tokens {max_length} is below the minimum of {SEQ2SEQ_MIN_INPUT_TOKENS}",
                )));
            }
            tokenizer
                .with_truncation(Some(tokenizers::TruncationParams {
                    max_length,
                    ..Default::default()
                }))
                .map_err(|e| CoreError::Tokenization(e.to_string()))?;
        }
        Ok(Self {
            session,
            tokenizer,
            decoder: None,
        })
    }

    /// Whether this pipeline runs the encoder-decoder decode loop.
    pub(super) fn is_encoder_decoder(&self) -> bool {
        self.decoder.is_some()
    }

    /// Run encoder-decoder generation for a batch, one input after another.
    ///
    /// Generation is token by token, so inputs cannot share one padded
    /// forward pass the way the other profiles do. Returns one result per
    /// input, in input order. The time budget covers the whole batch: its
    /// requests arrived together, so they share a deadline.
    pub(super) fn execute_batch_sequential(
        &mut self,
        batch: Vec<PreparedData>,
    ) -> Vec<Result<PipelineOutput, CoreError>> {
        let Some(decoder) = self.decoder.as_mut() else {
            return batch
                .iter()
                .map(|_| {
                    Err(CoreError::Inference(
                        "seq2seq pipeline has no decoder for sequential batch execution"
                            .to_string(),
                    ))
                })
                .collect();
        };
        let deadline = decoder.deadline_from_now();

        batch
            .into_iter()
            .map(|item| match item {
                PreparedData::Text(prepared) => {
                    let ids = decoder.generate(&mut self.session, &prepared, deadline)?;
                    self.tokenizer
                        .decode(&ids, true)
                        .map(PipelineOutput::Seq2Seq)
                        .map_err(|e| CoreError::Inference(e.to_string()))
                }
                PreparedData::Audio(_) => Err(CoreError::InvalidInput(
                    "text pipeline received audio prepared data in batch".to_string(),
                )),
            })
            .collect()
    }
}

impl Pipeline for Seq2SeqPipeline {
    type Input = String;
    type Prepared = PreparedInput;
    type Output = String;

    fn prepare(&self, input: String) -> Result<PreparedInput, CoreError> {
        tokenize_text(&self.tokenizer, input)
    }

    fn execute(&mut self, prepared: PreparedInput) -> Result<String, CoreError> {
        if let Some(decoder) = self.decoder.as_mut() {
            let deadline = decoder.deadline_from_now();
            let ids = decoder.generate(&mut self.session, &prepared, deadline)?;
            return self
                .tokenizer
                .decode(&ids, true)
                .map_err(|e| CoreError::Inference(e.to_string()));
        }

        let outputs = run_onnx_inference(&mut self.session, &prepared)?;

        // WR-05: Guard against models with zero output tensors.
        check_outputs_nonempty(&outputs)?;

        // Fused seq2seq models output generated token IDs.
        // Try extracting as i64 first (most common); fall back to f32 and round.
        // WR-02: Use checked u32 conversion instead of bare `as u32` casts.
        let output_ids: Vec<u32> =
            if let Ok(tensor) = outputs[0].try_extract_tensor::<i64>() {
                let (_, data) = tensor;
                data.iter()
                    .map(|&id| {
                        u32::try_from(id).map_err(|_| {
                            CoreError::Inference(format!("negative token ID {id} in seq2seq output"))
                        })
                    })
                    .collect::<Result<Vec<u32>, CoreError>>()?
            } else if let Ok(tensor) = outputs[0].try_extract_tensor::<f32>() {
                let (_, data) = tensor;
                data.iter()
                    .map(|&v| {
                        let rounded = v.round();
                        if rounded < 0.0 || rounded > u32::MAX as f32 {
                            return Err(CoreError::Inference(format!(
                                "token ID {v} out of u32 range in seq2seq output"
                            )));
                        }
                        Ok(rounded as u32)
                    })
                    .collect::<Result<Vec<u32>, CoreError>>()?
            } else {
                return Err(CoreError::Inference(
                    "seq2seq output tensor is neither i64 nor f32".to_string(),
                ));
            };

        // Decode token IDs back to text, skipping special tokens.
        self.tokenizer
            .decode(&output_ids, true)
            .map_err(|e| CoreError::Inference(e.to_string()))
    }
}

/// Seq2Seq batch post-processing: extract token IDs and decode per sample.
/// WR-02: Uses checked u32 conversion instead of bare `as u32` casts.
pub(super) fn batch_postprocess_seq2seq(
    outputs: ort::session::SessionOutputs<'_>,
    batch_size: usize,
    tokenizer: &Tokenizer,
) -> Vec<Result<PipelineOutput, CoreError>> {
    // WR-05: Guard against models with zero output tensors.
    if let Err(e) = check_outputs_nonempty(&outputs) {
        return (0..batch_size)
            .map(|_| Err(CoreError::Inference(e.to_string())))
            .collect();
    }

    // Try i64 first, fall back to f32.
    if let Ok(tensor) = outputs[0].try_extract_tensor::<i64>() {
        let (shape, data) = tensor;
        let seq_len = if shape.len() >= 2 {
            shape[shape.len() - 1] as usize
        } else {
            data.len() / batch_size
        };
        return (0..batch_size)
            .map(|i| {
                let sample = &data[i * seq_len..(i + 1) * seq_len];
                let ids: Vec<u32> = sample
                    .iter()
                    .map(|&v| {
                        u32::try_from(v).map_err(|_| {
                            CoreError::Inference(format!(
                                "negative token ID {v} in seq2seq output"
                            ))
                        })
                    })
                    .collect::<Result<Vec<u32>, CoreError>>()?;
                match tokenizer.decode(&ids, true) {
                    Ok(text) => Ok(PipelineOutput::Seq2Seq(text)),
                    Err(e) => Err(CoreError::Inference(e.to_string())),
                }
            })
            .collect();
    }

    if let Ok(tensor) = outputs[0].try_extract_tensor::<f32>() {
        let (shape, data) = tensor;
        let seq_len = if shape.len() >= 2 {
            shape[shape.len() - 1] as usize
        } else {
            data.len() / batch_size
        };
        return (0..batch_size)
            .map(|i| {
                let sample = &data[i * seq_len..(i + 1) * seq_len];
                let ids: Vec<u32> = sample
                    .iter()
                    .map(|&v| {
                        let rounded = v.round();
                        if rounded < 0.0 || rounded > u32::MAX as f32 {
                            return Err(CoreError::Inference(format!(
                                "token ID {v} out of u32 range in seq2seq output"
                            )));
                        }
                        Ok(rounded as u32)
                    })
                    .collect::<Result<Vec<u32>, CoreError>>()?;
                match tokenizer.decode(&ids, true) {
                    Ok(text) => Ok(PipelineOutput::Seq2Seq(text)),
                    Err(e) => Err(CoreError::Inference(e.to_string())),
                }
            })
            .collect();
    }

    (0..batch_size)
        .map(|_| Err(CoreError::Inference("seq2seq output tensor is neither i64 nor f32".into())))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::path::PathBuf;

    use crate::pipeline::{ClassifierPipeline, EmbeddingsPipeline, TokenClassifierPipeline};

    /// Token cap every text profile applies today.
    const TODAYS_INPUT_CAP: usize = 512;

    fn fixture(name: &str) -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/seq2seq")
            .join(name)
    }

    /// A model directory around the given fixture graphs, with a
    /// configuration every profile can load.
    fn model_dir(graphs: &[&str]) -> tempfile::TempDir {
        let dir = tempfile::tempdir().expect("temp dir");
        let config = serde_json::json!({
            "decoder_start_token_id": 2,
            "eos_token_id": 2,
            "id2label": { "0": "O", "1": "B-PER" },
            "max_length": 8,
            "vocab_size": 16
        });
        std::fs::write(dir.path().join("config.json"), config.to_string()).expect("write config");
        for name in graphs.iter().copied().chain(["tokenizer.json"]) {
            std::fs::copy(fixture(name), dir.path().join(name)).expect("copy fixture");
        }
        dir
    }

    /// 600 words: more than any cap under test.
    fn long_text() -> String {
        vec!["a"; 600].join(" ")
    }

    fn cap_of(tokenizer: &Tokenizer) -> Option<usize> {
        tokenizer.get_truncation().map(|params| params.max_length)
    }

    #[test]
    fn shared_loader_still_caps_input_at_512_tokens() {
        // Arrange
        let dir = model_dir(&["model.onnx"]);

        // Act
        let (_session, tokenizer) =
            load_session_and_tokenizer(dir.path(), &ExecutionProvider::Cpu).expect("load");

        // Assert
        assert_eq!(cap_of(&tokenizer), Some(TODAYS_INPUT_CAP));
    }

    #[test]
    fn classifier_still_truncates_input_to_512_tokens() {
        // Arrange
        let dir = model_dir(&["model.onnx"]);
        let pipeline = ClassifierPipeline::new(dir.path(), &ExecutionProvider::Cpu).expect("load");

        // Act
        let prepared = pipeline.prepare(long_text()).expect("prepare");

        // Assert
        assert_eq!(prepared.sequence_length, TODAYS_INPUT_CAP);
    }

    #[test]
    fn embeddings_still_truncate_input_to_512_tokens() {
        // Arrange
        let dir = model_dir(&["model.onnx"]);
        let pipeline = EmbeddingsPipeline::new(dir.path(), &ExecutionProvider::Cpu).expect("load");

        // Act
        let prepared = pipeline.prepare(long_text()).expect("prepare");

        // Assert
        assert_eq!(prepared.sequence_length, TODAYS_INPUT_CAP);
    }

    #[test]
    fn token_classifier_still_truncates_input_to_512_tokens() {
        // Arrange
        let dir = model_dir(&["model.onnx"]);
        let pipeline =
            TokenClassifierPipeline::new(dir.path(), &ExecutionProvider::Cpu).expect("load");

        // Act
        let prepared = pipeline.prepare(long_text()).expect("prepare");

        // Assert
        assert_eq!(prepared.sequence_length, TODAYS_INPUT_CAP);
    }

    #[test]
    fn fused_seq2seq_keeps_512_token_cap_without_options() {
        // Arrange
        let dir = model_dir(&["model.onnx"]);

        // Act
        let pipeline = Seq2SeqPipeline::new(dir.path(), &ExecutionProvider::Cpu).expect("load");

        // Assert
        assert!(!pipeline.is_encoder_decoder());
        assert_eq!(cap_of(&pipeline.tokenizer), Some(TODAYS_INPUT_CAP));
        let prepared = pipeline.prepare(long_text()).expect("prepare");
        assert_eq!(prepared.sequence_length, TODAYS_INPUT_CAP);
    }

    #[test]
    fn fused_seq2seq_uses_explicit_input_cap() {
        // Arrange
        let dir = model_dir(&["model.onnx"]);
        let options = Seq2SeqOptions::new().with_max_input_tokens(64);

        // Act
        let pipeline = Seq2SeqPipeline::with_options(dir.path(), &ExecutionProvider::Cpu, &options)
            .expect("load");

        // Assert
        assert_eq!(cap_of(&pipeline.tokenizer), Some(64));
    }

    #[test]
    fn fused_seq2seq_rejects_input_cap_below_minimum() {
        // Arrange
        let dir = model_dir(&["model.onnx"]);
        let options = Seq2SeqOptions::new().with_max_input_tokens(1);

        // Act
        let result = Seq2SeqPipeline::with_options(dir.path(), &ExecutionProvider::Cpu, &options);

        // Assert
        assert!(matches!(result, Err(CoreError::Config(_))));
    }

    #[test]
    fn encoder_decoder_seq2seq_defaults_to_512_token_cap() {
        // Arrange
        let dir = model_dir(&["encoder_model.onnx", "decoder_model.onnx"]);

        // Act
        let pipeline = Seq2SeqPipeline::new(dir.path(), &ExecutionProvider::Cpu).expect("load");

        // Assert
        assert!(pipeline.is_encoder_decoder());
        assert_eq!(cap_of(&pipeline.tokenizer), Some(TODAYS_INPUT_CAP));
    }

    #[test]
    fn options_default_equals_new() {
        // Arrange / Act
        let options = Seq2SeqOptions::default();

        // Assert
        assert_eq!(options, Seq2SeqOptions::new());
        assert_eq!(options.max_input_tokens(), None);
        assert_eq!(options.max_new_tokens(), None);
        assert_eq!(options.timeout(), None);
    }

    /// Token-for-token parity with a Hugging Face reference on a real model.
    ///
    /// Needs an exported model and a cases file, so it never runs by
    /// default. Point `HEPHAESTUS_SEQ2SEQ_PARITY_MODEL_DIR` at an Optimum
    /// export and `HEPHAESTUS_SEQ2SEQ_PARITY_CASES` at a JSON file of the
    /// form `{"max_input_tokens": N, "cases": [{"name", "text",
    /// "input_ids", "token_ids"}]}`, where `token_ids` are the reference's
    /// greedy output without the decoder start token, then run:
    ///
    /// `cargo test -p hephaestus-core --release --lib token_ids_match_reference -- --ignored --nocapture`
    #[test]
    #[ignore = "requires an exported model and reference cases (see doc comment)"]
    fn encoder_decoder_token_ids_match_reference_cases() {
        // Arrange
        let model_dir = std::env::var("HEPHAESTUS_SEQ2SEQ_PARITY_MODEL_DIR")
            .expect("HEPHAESTUS_SEQ2SEQ_PARITY_MODEL_DIR must be set");
        let cases_path = std::env::var("HEPHAESTUS_SEQ2SEQ_PARITY_CASES")
            .expect("HEPHAESTUS_SEQ2SEQ_PARITY_CASES must be set");
        let cases: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(cases_path).expect("read cases"))
                .expect("parse cases");
        let ids = |value: &serde_json::Value| -> Vec<i64> {
            value
                .as_array()
                .expect("id list")
                .iter()
                .map(|id| id.as_i64().expect("integer id"))
                .collect()
        };
        let max_input_tokens = usize::try_from(
            cases["max_input_tokens"]
                .as_u64()
                .expect("max_input_tokens"),
        )
        .expect("cap fits usize");
        let options = Seq2SeqOptions::new().with_max_input_tokens(max_input_tokens);
        let mut pipeline =
            Seq2SeqPipeline::with_options(Path::new(&model_dir), &ExecutionProvider::Cpu, &options)
                .expect("load");
        assert!(
            pipeline.is_encoder_decoder(),
            "model directory must hold an encoder-decoder export"
        );

        for case in cases["cases"].as_array().expect("cases array") {
            let name = case["name"].as_str().expect("case name");
            let text = case["text"].as_str().expect("case text").to_string();

            // Act
            let prepared = pipeline.prepare(text).expect("prepare");
            let input_ids = prepared.input_ids.clone();
            let decoder = pipeline.decoder.as_mut().expect("encoder-decoder mode");
            let generated = decoder
                .generate(&mut pipeline.session, &prepared, None)
                .expect("generate");
            let text = pipeline.tokenizer.decode(&generated, true).expect("decode");
            println!(
                "{name}: {} input tokens, {} generated tokens",
                input_ids.len(),
                generated.len()
            );
            println!("{name}: {text}");

            // Assert
            assert_eq!(
                input_ids,
                ids(&case["input_ids"]),
                "{name}: input token IDs differ"
            );
            let generated: Vec<i64> = generated.into_iter().map(i64::from).collect();
            assert_eq!(
                generated,
                ids(&case["token_ids"]),
                "{name}: generated token IDs differ"
            );
            if let Some(reference_text) = case["reference_text"].as_str() {
                assert_eq!(text, reference_text, "{name}: decoded text differs");
            }
        }
    }
}
