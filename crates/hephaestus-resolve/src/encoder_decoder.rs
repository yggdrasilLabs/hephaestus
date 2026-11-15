//! File layout of an encoder-decoder (seq2seq) ONNX export.
//!
//! The standard Optimum export of a seq2seq model has no `model.onnx`.
//! It is `encoder_model.onnx` plus a decoder graph, with generation
//! settings in an optional `generation_config.json`. The storage and
//! Hugging Face tiers look for this layout only after the single-file
//! lookup has missed, so a model that resolves through `model.onnx`
//! never reaches the code in this module.

/// Encoder graph of an encoder-decoder export.
pub(crate) const ENCODER_ONNX: &str = "encoder_model.onnx";

/// Decoder graphs in order of preference.
///
/// The runtime decodes without a key/value cache, so the plain decoder is
/// the graph it needs; the merged decoder is used when it is the only one
/// published. Exactly one of them is fetched.
pub(crate) const DECODER_ONNX_CANDIDATES: [&str; 2] =
    ["decoder_model.onnx", "decoder_model_merged.onnx"];

/// Generation settings file. Optional: the settings may live in `config.json`.
pub(crate) const GENERATION_CONFIG_JSON: &str = "generation_config.json";

/// Subdirectory some repositories keep their ONNX files in, with the
/// trailing separator used in storage and repository paths.
pub(crate) const ONNX_SUBDIR_PREFIX: &str = "onnx/";

/// Architecture name of Whisper models in `config.json`.
const WHISPER_ARCHITECTURE: &str = "WhisperForConditionalGeneration";

/// Whether a `config.json` describes a Whisper speech model.
///
/// Whisper is published in the same encoder/decoder file layout as text
/// seq2seq models, but it is served by the ASR pipeline, and this resolver
/// has never fetched its files. The encoder-decoder lookup skips Whisper
/// so that its resolution stays exactly as it was: no ONNX export found.
/// Unparseable input is not Whisper.
pub(crate) fn is_whisper_config(config_json: &[u8]) -> bool {
    serde_json::from_slice::<serde_json::Value>(config_json)
        .ok()
        .and_then(|config| {
            config
                .get("architectures")?
                .as_array()
                .map(|architectures| {
                    architectures
                        .iter()
                        .any(|name| name.as_str() == Some(WHISPER_ARCHITECTURE))
                })
        })
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn whisper_config_is_recognised() {
        // Arrange
        let config = br#"{"architectures": ["WhisperForConditionalGeneration"]}"#;

        // Act / Assert
        assert!(is_whisper_config(config));
    }

    #[test]
    fn text_seq2seq_config_is_not_whisper() {
        // Arrange
        let config = br#"{"architectures": ["BartForConditionalGeneration"]}"#;

        // Act / Assert
        assert!(!is_whisper_config(config));
    }

    #[test]
    fn config_without_architectures_is_not_whisper() {
        // Arrange
        let config = br#"{"model_type": "t5"}"#;

        // Act / Assert
        assert!(!is_whisper_config(config));
    }

    #[test]
    fn invalid_json_is_not_whisper() {
        // Arrange
        let config = b"not json {{{";

        // Act / Assert
        assert!(!is_whisper_config(config));
    }

    #[test]
    fn plain_decoder_is_preferred_over_merged() {
        assert_eq!(
            DECODER_ONNX_CANDIDATES,
            ["decoder_model.onnx", "decoder_model_merged.onnx"]
        );
    }
}
