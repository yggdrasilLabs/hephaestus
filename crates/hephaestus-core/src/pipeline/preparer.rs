//! Input preparation without the model.
//!
//! [`PipelineKind::prepare`] needs the pipeline, which is busy for as long
//! as the model runs. A [`Preparer`] holds a copy of the pipeline's
//! tokenizer (or feature extraction settings) and nothing else, so a
//! server can prepare the next requests while a generation is running.

use tokenizers::Tokenizer;

use crate::error::CoreError;

use super::asr::AudioFeatures;
use super::text_generation::PromptBuilder;
use super::token_classifier::tokenize_keeping_encoding;
use super::{
    InferenceInput, PipelineKind, PreparedData, TextGenerationInput, tokenize_text,
};

/// How one profile prepares its input.
enum Kind {
    /// Classifier, embeddings and seq2seq: plain tokenization.
    Text(Tokenizer),
    /// Token classification: tokenization that keeps the encoding.
    TokenClassifier(Tokenizer),
    /// ASR: feature extraction from audio samples.
    Asr(AudioFeatures),
    /// Text generation: chat template, then tokenization.
    TextGeneration(Box<PromptBuilder>),
}

/// Prepares inputs for a pipeline without borrowing it.
///
/// Obtained from [`PipelineKind::preparer`]. [`Preparer::prepare`] returns
/// exactly what [`PipelineKind::prepare`] of that pipeline returns, for
/// every input. The preparer is a snapshot: it does not follow later
/// changes to the pipeline (there are none once a pipeline is built).
pub struct Preparer {
    kind: Kind,
}

impl PipelineKind {
    /// A [`Preparer`] for this pipeline.
    ///
    /// Copies the tokenizer, so call it once and share the result.
    pub fn preparer(&self) -> Preparer {
        let kind = match self {
            Self::Classifier(p) => Kind::Text(p.tokenizer.clone()),
            Self::Embeddings(p) => Kind::Text(p.tokenizer.clone()),
            Self::Seq2Seq(p) => Kind::Text(p.tokenizer.clone()),
            Self::TokenClassifier(p) => Kind::TokenClassifier(p.tokenizer.clone()),
            Self::Asr(p) => Kind::Asr(p.features()),
            Self::TextGeneration(p) => Kind::TextGeneration(Box::new(p.prompt_builder())),
        };
        Preparer { kind }
    }
}

impl Preparer {
    /// Whether requests to this pipeline may carry a system message; see
    /// [`PipelineKind::accepts_system_prompt`].
    pub fn accepts_system_prompt(&self) -> bool {
        matches!(self.kind, Kind::TextGeneration(_))
    }

    /// Prepare input for the pipeline this preparer was taken from; see
    /// [`PipelineKind::prepare`].
    ///
    /// # Errors
    ///
    /// Returns [`CoreError`] if tokenization or feature extraction fails,
    /// or if the input is of the wrong kind for the profile.
    pub fn prepare(&self, input: impl Into<InferenceInput>) -> Result<PreparedData, CoreError> {
        match (&self.kind, input.into()) {
            (Kind::TextGeneration(prompt), InferenceInput::Text(text)) => Ok(PreparedData::Text(
                prompt.prepare(TextGenerationInput { text, system: None })?,
            )),
            (Kind::TextGeneration(prompt), InferenceInput::TextWithSystem { text, system }) => {
                Ok(PreparedData::Text(prompt.prepare(TextGenerationInput {
                    text,
                    system: Some(system),
                })?))
            }
            // No other profile has a system message: the text is handled
            // exactly as if it had arrived alone.
            (_, InferenceInput::TextWithSystem { text, .. }) => {
                self.prepare(InferenceInput::Text(text))
            }
            (Kind::Text(tokenizer), InferenceInput::Text(text)) => {
                Ok(PreparedData::Text(tokenize_text(tokenizer, text)?))
            }
            (Kind::TokenClassifier(tokenizer), InferenceInput::Text(text)) => Ok(
                PreparedData::Text(tokenize_keeping_encoding(tokenizer, text)?),
            ),
            (Kind::Asr(features), InferenceInput::Audio(audio)) => {
                Ok(PreparedData::Audio(features.extract(audio)?))
            }
            (Kind::Asr(_), InferenceInput::Text(_)) => Err(CoreError::InvalidInput(
                "ASR pipeline requires audio input".to_string(),
            )),
            (_, InferenceInput::Audio(_)) => Err(CoreError::InvalidInput(
                "text pipeline requires text input, got audio".to_string(),
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::*;
    use crate::ep::ExecutionProvider;
    use crate::pipeline::{
        ClassifierPipeline, EmbeddingsPipeline, Seq2SeqOptions, Seq2SeqPipeline,
        TextGenerationOptions, TextGenerationPipeline, TokenClassifierPipeline,
    };

    fn fixture(dir: &str, name: &str) -> std::path::PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures")
            .join(dir)
            .join(name)
    }

    /// A model directory every text profile can load.
    fn model_dir() -> tempfile::TempDir {
        let dir = tempfile::tempdir().expect("temp dir");
        let config = serde_json::json!({
            "bos_token_id": 0,
            "decoder_start_token_id": 2,
            "eos_token_id": 2,
            "id2label": { "0": "O", "1": "B-PER" },
            "max_length": 8,
            "max_position_embeddings": 64,
            "vocab_size": 16
        });
        std::fs::write(dir.path().join("config.json"), config.to_string()).expect("write config");
        for name in ["model.onnx", "tokenizer.json"] {
            std::fs::copy(fixture("seq2seq", name), dir.path().join(name)).expect("copy fixture");
        }
        dir
    }

    /// The fields of a prepared text input.
    fn fields(prepared: PreparedData) -> (Vec<i64>, Vec<i64>, usize, bool) {
        let text = prepared.into_text().expect("text input");
        (
            text.input_ids,
            text.attention_mask,
            text.sequence_length,
            text.encoding.is_some(),
        )
    }

    /// Inputs of every kind a request can carry, one above every cap.
    fn inputs() -> Vec<InferenceInput> {
        vec![
            InferenceInput::Text("a b c".to_string()),
            InferenceInput::Text(vec!["d"; 600].join(" ")),
            InferenceInput::TextWithSystem {
                text: "e f".to_string(),
                system: "g h".to_string(),
            },
            InferenceInput::TextWithSystem {
                text: "i".to_string(),
                system: String::new(),
            },
            InferenceInput::Audio(vec![0.0; 8]),
        ]
    }

    fn assert_prepares_like_its_pipeline(name: &str, pipeline: &PipelineKind) {
        let preparer = pipeline.preparer();
        assert_eq!(
            preparer.accepts_system_prompt(),
            pipeline.accepts_system_prompt(),
            "{name}"
        );
        for (from_pipeline, from_preparer) in inputs().into_iter().zip(inputs()) {
            match (pipeline.prepare(from_pipeline), preparer.prepare(from_preparer)) {
                (Ok(expected), Ok(actual)) => assert_eq!(fields(actual), fields(expected), "{name}"),
                (Err(expected), Err(actual)) => {
                    assert_eq!(actual.to_string(), expected.to_string(), "{name}");
                }
                (expected, actual) => panic!(
                    "{name}: pipeline ok={}, preparer ok={}",
                    expected.is_ok(),
                    actual.is_ok()
                ),
            }
        }
    }

    #[test]
    fn preparer_prepares_exactly_what_its_pipeline_prepares() {
        // Arrange -- one pipeline of every text profile.
        let ep = ExecutionProvider::Cpu;
        let dir = model_dir();
        let encoder_decoder = tempfile::tempdir().expect("temp dir");
        for name in ["encoder_model.onnx", "decoder_model.onnx", "tokenizer.json"] {
            std::fs::copy(fixture("seq2seq", name), encoder_decoder.path().join(name))
                .expect("copy fixture");
        }
        std::fs::copy(dir.path().join("config.json"), encoder_decoder.path().join("config.json"))
            .expect("copy config");
        let text_generation = tempfile::tempdir().expect("temp dir");
        for (from, name) in [
            ("text_generation", "model.onnx"),
            ("text_generation", "tokenizer_config.json"),
            ("seq2seq", "tokenizer.json"),
        ] {
            std::fs::copy(fixture(from, name), text_generation.path().join(name))
                .expect("copy fixture");
        }
        std::fs::copy(dir.path().join("config.json"), text_generation.path().join("config.json"))
            .expect("copy config");

        let pipelines = [
            (
                "classifier",
                PipelineKind::Classifier(ClassifierPipeline::new(dir.path(), &ep).expect("load")),
            ),
            (
                "embeddings",
                PipelineKind::Embeddings(EmbeddingsPipeline::new(dir.path(), &ep).expect("load")),
            ),
            (
                "token classifier",
                PipelineKind::TokenClassifier(
                    TokenClassifierPipeline::new(dir.path(), &ep).expect("load"),
                ),
            ),
            (
                "fused seq2seq",
                PipelineKind::Seq2Seq(Seq2SeqPipeline::new(dir.path(), &ep).expect("load")),
            ),
            (
                "encoder-decoder seq2seq",
                PipelineKind::Seq2Seq(
                    Seq2SeqPipeline::with_options(
                        encoder_decoder.path(),
                        &ep,
                        &Seq2SeqOptions::new().with_max_input_tokens(16),
                    )
                    .expect("load"),
                ),
            ),
            (
                "text generation",
                PipelineKind::TextGeneration(
                    TextGenerationPipeline::with_options(
                        text_generation.path(),
                        &ep,
                        &TextGenerationOptions::new()
                            .with_max_input_tokens(16)
                            .with_max_new_tokens(8)
                            .with_system_prompt("j k"),
                    )
                    .expect("load"),
                ),
            ),
        ];

        // Act + Assert
        for (name, pipeline) in &pipelines {
            assert_prepares_like_its_pipeline(name, pipeline);
        }
    }

    #[test]
    fn raw_waveform_features_pass_the_samples_through() {
        // No ASR fixture graph exists, so the settings are checked alone.
        let features = AudioFeatures::for_test("none");

        let prepared = features.extract(vec![0.5, -0.5, 0.25]).expect("extract");

        assert_eq!(prepared.features.shape(), [1, 3]);
    }
}
