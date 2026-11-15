//! Extra files of a text generation (decoder-only) model.
//!
//! A decoder-only instruction model is a single-file model as far as its
//! graph goes (`model.onnx`, possibly with external data), so it resolves
//! through the ordinary single-file lookup. Serving it needs three more
//! small files, which that lookup fetches only when `config.json` says the
//! model is a causal language model. For every other model the lookup
//! requests exactly the files it always did.

/// Files fetched in addition to the graph, `tokenizer.json` and
/// `config.json`. All optional here: the pipeline decides what it cannot
/// do without (the chat template lives in one of two of these files).
pub(crate) const TEXT_GENERATION_FILES: [&str; 3] = [
    "tokenizer_config.json",
    "generation_config.json",
    "chat_template.jinja",
];

/// Suffix of the architecture name of decoder-only causal language models.
const CAUSAL_LM_SUFFIX: &str = "ForCausalLM";

/// Pipeline tag of text generation models.
const TEXT_GENERATION_TAG: &str = "text-generation";

/// Whether a `config.json` describes a text generation model.
///
/// Mirrors the runtime's profile detection: an architecture named
/// `...ForCausalLM`, or the `text-generation` pipeline tag. Unparseable
/// input is not a text generation model.
pub(crate) fn is_text_generation_config(config_json: &[u8]) -> bool {
    let Ok(config) = serde_json::from_slice::<serde_json::Value>(config_json) else {
        return false;
    };
    let causal_lm = config
        .get("architectures")
        .and_then(|value| value.as_array())
        .is_some_and(|architectures| {
            architectures
                .iter()
                .filter_map(|name| name.as_str())
                .any(|name| name.ends_with(CAUSAL_LM_SUFFIX))
        });
    causal_lm || config.get("pipeline_tag").and_then(|tag| tag.as_str()) == Some(TEXT_GENERATION_TAG)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn causal_lm_config_is_text_generation() {
        assert!(is_text_generation_config(
            br#"{"architectures": ["LlamaForCausalLM"], "model_type": "llama"}"#
        ));
        assert!(is_text_generation_config(br#"{"pipeline_tag": "text-generation"}"#));
    }

    #[test]
    fn other_configs_are_not_text_generation() {
        for config in [
            &br#"{"architectures": ["DistilBertForSequenceClassification"]}"#[..],
            br#"{"architectures": ["BertForTokenClassification"]}"#,
            br#"{"architectures": ["BertModel"]}"#,
            br#"{"architectures": ["T5ForConditionalGeneration"]}"#,
            br#"{"architectures": ["WhisperForConditionalGeneration"]}"#,
            br#"{"architectures": ["Wav2Vec2ForCTC"]}"#,
            br#"{"pipeline_tag": "text2text-generation"}"#,
            br#"{}"#,
            b"not json {{{",
        ] {
            assert!(
                !is_text_generation_config(config),
                "{}",
                String::from_utf8_lossy(config)
            );
        }
    }

    #[test]
    fn extra_files_are_the_template_and_generation_settings() {
        assert_eq!(
            TEXT_GENERATION_FILES,
            ["tokenizer_config.json", "generation_config.json", "chat_template.jinja"]
        );
    }
}
