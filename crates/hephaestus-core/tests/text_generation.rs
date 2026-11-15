//! End-to-end tests for the text generation pipeline.
//!
//! Runs real ONNX Runtime sessions over the tiny graphs in
//! `tests/fixtures/text_generation` (see `make_fixtures.py` there). The
//! next token of those graphs depends on every token so far, and the
//! cached graph only gets it right when the prompt is fed once, each later
//! step feeds the newest token with the previous step's cache, and
//! `attention_mask` and `position_ids` cover the whole sequence. The
//! expected text of every test is worked out by [`simulate`].

use std::path::{Path, PathBuf};
use std::time::Duration;

use hephaestus_core::{
    CoreError, ExecutionProvider, InferenceInput, Pipeline, PipelineKind, PipelineOutput,
    Seq2SeqPipeline, TextGenerationInput, TextGenerationOptions, TextGenerationPipeline,
};
use tempfile::TempDir;

/// Fixture graph with a key/value cache.
const CACHED: &str = "model.onnx";
/// Fixture graph without one.
const UNCACHED: &str = "model_no_cache.onnx";
const EOS: i64 = 2;
const VOCAB: i64 = 16;

fn fixture(dir: &str, name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(dir)
        .join(name)
}

fn base_config() -> serde_json::Value {
    serde_json::json!({
        "architectures": ["LlamaForCausalLM"],
        "bos_token_id": 0,
        "eos_token_id": 2,
        "max_position_embeddings": 512,
        "vocab_size": 16
    })
}

/// A model directory around one fixture graph, stored as `graph_path`.
fn model_dir_at(
    graph: &str,
    graph_path: &str,
    config: &serde_json::Value,
    generation_config: Option<&serde_json::Value>,
) -> TempDir {
    let dir = tempfile::tempdir().expect("temp dir");
    let target = dir.path().join(graph_path);
    std::fs::create_dir_all(target.parent().expect("graph has a parent")).expect("mkdir");
    std::fs::copy(fixture("text_generation", graph), target).expect("copy graph");
    std::fs::copy(
        fixture("text_generation", "tokenizer_config.json"),
        dir.path().join("tokenizer_config.json"),
    )
    .expect("copy tokenizer config");
    std::fs::copy(
        fixture("seq2seq", "tokenizer.json"),
        dir.path().join("tokenizer.json"),
    )
    .expect("copy tokenizer");
    std::fs::write(dir.path().join("config.json"), config.to_string()).expect("write config");
    if let Some(generation_config) = generation_config {
        std::fs::write(
            dir.path().join("generation_config.json"),
            generation_config.to_string(),
        )
        .expect("write generation config");
    }
    dir
}

fn model_dir(graph: &str) -> TempDir {
    model_dir_at(graph, "model.onnx", &base_config(), None)
}

fn load(dir: &TempDir, options: &TextGenerationOptions) -> TextGenerationPipeline {
    TextGenerationPipeline::with_options(dir.path(), &ExecutionProvider::Cpu, options)
        .expect("pipeline should load")
}

/// Token IDs of a string of fixture words (`a` = 4 ... `l` = 15).
fn ids(words: &str) -> Vec<i64> {
    words
        .split_whitespace()
        .map(|word| i64::from(word.as_bytes()[0] - b'a') + 4)
        .collect()
}

/// The fixture words of a token sequence, special tokens left out.
fn words(tokens: &[i64]) -> String {
    tokens
        .iter()
        .filter(|&&token| token >= 4)
        .map(|&token| char::from(b'a' + u8::try_from(token - 4).expect("word token")).to_string())
        .collect::<Vec<_>>()
        .join(" ")
}

/// Prompt tokens the fixture chat template produces.
fn prompt(system: Option<&str>, user: &str) -> Vec<i64> {
    let rendered = match system {
        Some(system) => format!("a {system} b c {user} d e"),
        None => format!("c {user} d e"),
    };
    ids(&rendered)
}

/// What greedy decoding of the fixture graphs must generate.
///
/// Mirrors `make_fixtures.py` (logits `8 - |v - target|`) and the Hugging
/// Face rules for `repetition_penalty` and `no_repeat_ngram_size`.
fn simulate(prompt: &[i64], max_new_tokens: usize, penalty: f32, ngram: usize) -> Vec<i64> {
    let mut tokens = prompt.to_vec();
    for _ in 0..max_new_tokens {
        let sum: i64 = tokens.iter().sum();
        let target = if sum % 13 == 0 { EOS } else { sum % 12 + 4 };
        let mut logits: Vec<f32> = (0..VOCAB).map(|v| (8 - (v - target).abs()) as f32).collect();
        for token in 0..VOCAB {
            if tokens.contains(&token) {
                let logit = &mut logits[usize::try_from(token).expect("index")];
                *logit = if *logit > 0.0 { *logit / penalty } else { *logit * penalty };
            }
        }
        if ngram > 0 && tokens.len() + 1 >= ngram {
            let prefix = &tokens[tokens.len() - (ngram - 1)..];
            for window in tokens.windows(ngram) {
                if &window[..ngram - 1] == prefix {
                    logits[usize::try_from(window[ngram - 1]).expect("index")] = f32::NEG_INFINITY;
                }
            }
        }
        let mut best = 0;
        for (index, logit) in logits.iter().enumerate() {
            if *logit > logits[best] {
                best = index;
            }
        }
        let next = i64::try_from(best).expect("token");
        tokens.push(next);
        if next == EOS {
            break;
        }
    }
    tokens[prompt.len()..].to_vec()
}

fn generate(pipeline: &mut TextGenerationPipeline, system: Option<&str>, user: &str) -> String {
    let prepared = pipeline
        .prepare(TextGenerationInput {
            text: user.to_string(),
            system: system.map(str::to_string),
        })
        .expect("prepare");
    pipeline.execute(prepared).expect("execute")
}

#[test]
fn generates_the_expected_text_with_the_cache() {
    // Arrange
    let dir = model_dir(CACHED);
    let mut pipeline = load(&dir, &TextGenerationOptions::new().with_max_new_tokens(40));
    assert!(pipeline.uses_kv_cache());

    for user in ["h i", "a", "l k j i h g", "a b c d e f g h i j k l a b c"] {
        // Act
        let text = generate(&mut pipeline, None, user);

        // Assert
        let expected = simulate(&prompt(None, user), 40, 1.0, 0);
        assert_eq!(text, words(&expected), "user text '{user}'");
    }
}

#[test]
fn cached_and_uncached_loops_generate_the_same_text() {
    // Arrange -- the same graph with the cache used and switched off, and
    // a graph exported without one.
    let cached_dir = model_dir(CACHED);
    let uncached_dir = model_dir(UNCACHED);
    let options = TextGenerationOptions::new().with_max_new_tokens(30);
    let mut cached = load(&cached_dir, &options);
    let mut cache_off = load(&cached_dir, &options.clone().with_kv_cache(false));
    let mut no_cache_graph = load(&uncached_dir, &options);
    assert!(cached.uses_kv_cache());
    assert!(!cache_off.uses_kv_cache());
    assert!(!no_cache_graph.uses_kv_cache());

    for (system, user) in [
        (None, "h i"),
        (Some("f g"), "h i"),
        (Some("l"), "a b c d e f"),
        (None, "k"),
    ] {
        // Act
        let with_cache = generate(&mut cached, system, user);
        let without_cache = generate(&mut cache_off, system, user);
        let plain_graph = generate(&mut no_cache_graph, system, user);

        // Assert
        let expected = words(&simulate(&prompt(system, user), 30, 1.0, 0));
        assert_eq!(with_cache, expected, "cached, {system:?} / '{user}'");
        assert_eq!(without_cache, expected, "cache off, {system:?} / '{user}'");
        assert_eq!(plain_graph, expected, "graph without cache, {system:?} / '{user}'");
    }
}

#[test]
fn graph_in_onnx_subdirectory_is_found() {
    // Arrange -- the layout of Hugging Face repositories.
    let dir = model_dir_at(CACHED, "onnx/model.onnx", &base_config(), None);
    let mut pipeline = load(&dir, &TextGenerationOptions::new().with_max_new_tokens(10));

    // Act
    let text = generate(&mut pipeline, None, "h i");

    // Assert
    assert_eq!(text, words(&simulate(&prompt(None, "h i"), 10, 1.0, 0)));
}

#[test]
fn generation_stops_at_the_end_token_and_at_the_token_limit() {
    // Arrange -- "c h i d e" reaches </s> after 5 words; see `simulate`.
    let dir = model_dir(CACHED);
    let full = simulate(&prompt(None, "h i"), 40, 1.0, 0);
    assert_eq!(full.last(), Some(&EOS), "fixture prompt should end by itself");
    assert!(full.len() > 3);

    // Act
    let mut unlimited = load(&dir, &TextGenerationOptions::new().with_max_new_tokens(40));
    let mut limited = load(&dir, &TextGenerationOptions::new().with_max_new_tokens(3));
    let complete = generate(&mut unlimited, None, "h i");
    let cut = generate(&mut limited, None, "h i");

    // Assert -- the end token is not part of the text.
    assert_eq!(complete, words(&full));
    assert_eq!(cut, words(&full[..3]));
}

#[test]
fn system_message_comes_from_the_request_or_the_default() {
    // Arrange
    let dir = model_dir(CACHED);
    let options = TextGenerationOptions::new().with_max_new_tokens(20);
    let mut without_default = load(&dir, &options);
    let mut with_default = load(&dir, &options.clone().with_system_prompt("f g"));
    let expect = |system: Option<&str>| words(&simulate(&prompt(system, "h i"), 20, 1.0, 0));
    assert_ne!(expect(None), expect(Some("f g")), "fixture must tell the prompts apart");
    assert_ne!(expect(Some("f g")), expect(Some("j")));

    // Act / Assert -- no default: the request decides.
    assert_eq!(generate(&mut without_default, None, "h i"), expect(None));
    assert_eq!(generate(&mut without_default, Some("f g"), "h i"), expect(Some("f g")));
    // Default set: used when the request brings none, replaced by the
    // request's own, and switched off by an empty string.
    assert_eq!(generate(&mut with_default, None, "h i"), expect(Some("f g")));
    assert_eq!(generate(&mut with_default, Some("j"), "h i"), expect(Some("j")));
    assert_eq!(generate(&mut with_default, Some(""), "h i"), expect(None));
}

#[test]
fn pipeline_kind_routes_text_and_system_message() {
    // Arrange
    let dir = model_dir(CACHED);
    let options = TextGenerationOptions::new().with_max_new_tokens(20);
    let mut kind = PipelineKind::TextGeneration(load(&dir, &options));
    assert!(kind.accepts_system_prompt());
    let mut run = |input: InferenceInput| {
        let prepared = kind.prepare(input).expect("prepare");
        match kind.execute(prepared).expect("execute") {
            PipelineOutput::TextGeneration(text) => text,
            _ => panic!("expected text generation output"),
        }
    };

    // Act
    let plain = run(InferenceInput::Text("h i".to_string()));
    let with_system = run(InferenceInput::TextWithSystem {
        text: "h i".to_string(),
        system: "f g".to_string(),
    });

    // Assert
    assert_eq!(plain, words(&simulate(&prompt(None, "h i"), 20, 1.0, 0)));
    assert_eq!(with_system, words(&simulate(&prompt(Some("f g"), "h i"), 20, 1.0, 0)));
}

#[test]
fn output_has_the_seq2seq_response_shape() {
    // Arrange
    let output = PipelineOutput::TextGeneration("some text".to_string());

    // Act
    let json = output.to_json();

    // Assert
    assert_eq!(json, serde_json::json!({ "generated_text": "some text" }));
}

#[test]
fn audio_input_is_rejected() {
    // Arrange
    let dir = model_dir(CACHED);
    let kind = PipelineKind::TextGeneration(load(&dir, &TextGenerationOptions::new()));

    // Act
    let result = kind.prepare(InferenceInput::Audio(vec![0.0; 16]));

    // Assert
    assert!(matches!(result, Err(CoreError::InvalidInput(_))));
}

#[test]
fn batch_runs_every_request_in_order() {
    // Arrange
    let dir = model_dir(CACHED);
    let options = TextGenerationOptions::new().with_max_new_tokens(20);
    let mut kind = PipelineKind::TextGeneration(load(&dir, &options));
    let users = ["h i", "a", "l k j"];
    let batch = users
        .iter()
        .map(|user| kind.prepare(user.to_string()).expect("prepare"))
        .collect();

    // Act
    let results = kind.execute_batch(batch);

    // Assert
    assert_eq!(results.len(), users.len());
    for (user, result) in users.iter().zip(results) {
        match result.expect("batch item") {
            PipelineOutput::TextGeneration(text) => {
                assert_eq!(text, words(&simulate(&prompt(None, user), 20, 1.0, 0)));
            }
            _ => panic!("expected text generation output"),
        }
    }
}

#[test]
fn long_user_text_is_cut_to_the_input_cap() {
    // Arrange -- 40 words against a cap of 16 prompt tokens: the template
    // takes 3 ("c", "d", "e"), so 13 words of the user text remain.
    let dir = model_dir(CACHED);
    let options = TextGenerationOptions::new()
        .with_max_input_tokens(16)
        .with_max_new_tokens(20);
    let mut pipeline = load(&dir, &options);
    let user = ["a b c d e f g h i j"; 4].join(" ");
    let kept = user.split(' ').take(13).collect::<Vec<_>>().join(" ");

    // Act
    let text = generate(&mut pipeline, None, &user);

    // Assert -- same as sending the first 13 words: the end of the
    // template, where the reply starts, is intact.
    assert_eq!(text, words(&simulate(&prompt(None, &kept), 20, 1.0, 0)));
}

#[test]
fn system_message_is_never_cut() {
    // Arrange -- system message of 6 words; the template takes 5 with it,
    // leaving 5 words of user text under a cap of 16.
    let dir = model_dir(CACHED);
    let options = TextGenerationOptions::new()
        .with_max_input_tokens(16)
        .with_max_new_tokens(20);
    let mut pipeline = load(&dir, &options);
    let system = "l k j i h g";
    let user = "a b c d e f g h i j";

    // Act
    let text = generate(&mut pipeline, Some(system), user);

    // Assert
    let expected = simulate(&prompt(Some(system), "a b c d e"), 20, 1.0, 0);
    assert_eq!(text, words(&expected));
}

#[test]
fn prompt_that_cannot_fit_is_rejected() {
    // Arrange -- the system message alone fills the cap.
    let dir = model_dir(CACHED);
    let options = TextGenerationOptions::new().with_max_input_tokens(16);
    let pipeline = load(&dir, &options);
    let system = ["a b c d"; 5].join(" ");

    // Act
    let result = pipeline.prepare(TextGenerationInput {
        text: "h i".to_string(),
        system: Some(system),
    });

    // Assert
    let err = result.err().expect("oversized prompt should fail");
    assert!(matches!(err, CoreError::InvalidInput(_)), "{err:?}");
}

#[test]
fn default_system_message_that_cannot_fit_fails_at_startup() {
    // Arrange
    let dir = model_dir(CACHED);
    let options = TextGenerationOptions::new()
        .with_max_input_tokens(16)
        .with_system_prompt(["a b c d"; 5].join(" "));

    // Act
    let result = TextGenerationPipeline::with_options(dir.path(), &ExecutionProvider::Cpu, &options);

    // Assert
    assert!(matches!(result, Err(CoreError::InvalidInput(_))));
}

#[test]
fn repetition_penalty_and_ngram_rule_come_from_the_generation_config() {
    for (penalty, ngram) in [(1.5_f32, 0_usize), (1.0, 2), (4.0, 3)] {
        // Arrange
        let generation_config = serde_json::json!({
            "eos_token_id": 2,
            "repetition_penalty": penalty,
            "no_repeat_ngram_size": ngram,
        });
        let dir = model_dir_at(CACHED, "model.onnx", &base_config(), Some(&generation_config));
        let mut pipeline = load(&dir, &TextGenerationOptions::new().with_max_new_tokens(24));
        let plain = simulate(&prompt(None, "h i"), 24, 1.0, 0);
        let expected = simulate(&prompt(None, "h i"), 24, penalty, ngram);
        assert_ne!(plain, expected, "settings {penalty}/{ngram} must change the fixture output");

        // Act
        let text = generate(&mut pipeline, None, "h i");

        // Assert
        assert_eq!(text, words(&expected), "penalty {penalty}, ngram {ngram}");
    }
}

#[test]
fn extra_stop_tokens_from_the_generation_config_end_generation() {
    // Arrange -- the word "e" (8) declared as a second end token. The
    // prompt "c b d d e" continues "j k a e a e ..." on its own.
    let generation_config = serde_json::json!({ "eos_token_id": [2, 8], "max_new_tokens": 12 });
    let dir = model_dir_at(CACHED, "model.onnx", &base_config(), Some(&generation_config));
    let mut pipeline = load(&dir, &TextGenerationOptions::new());
    let plain = simulate(&prompt(None, "b d"), 12, 1.0, 0);
    let stop = plain.iter().position(|&token| token == 8).expect("fixture output holds 'e'");
    assert!(stop > 0 && stop + 1 < plain.len(), "stop token must cut the output short");

    // Act
    let text = generate(&mut pipeline, None, "b d");

    // Assert -- nothing after the stop token; the stop token itself is a
    // plain word here, so it stays in the text.
    assert_eq!(text, words(&plain[..=stop]));
}

#[test]
fn model_token_limit_is_used_when_the_operator_sets_none() {
    // Arrange -- a prompt that never reaches </s> within the limits.
    let user = "b d";
    let unbounded = simulate(&prompt(None, user), 300, 1.0, 0);
    assert!(unbounded.len() > 7, "fixture prompt must run long enough");
    let generation_config = serde_json::json!({ "eos_token_id": 2, "max_new_tokens": 7 });
    let dir = model_dir_at(CACHED, "model.onnx", &base_config(), Some(&generation_config));

    // Act
    let from_model = generate(&mut load(&dir, &TextGenerationOptions::new()), None, user);
    let from_operator = generate(
        &mut load(&dir, &TextGenerationOptions::new().with_max_new_tokens(4)),
        None,
        user,
    );

    // Assert
    assert_eq!(from_model, words(&unbounded[..7]));
    assert_eq!(from_operator, words(&unbounded[..4]));
}

#[test]
fn expired_time_budget_stops_generation() {
    // Arrange
    let dir = model_dir(CACHED);
    let options = TextGenerationOptions::new().with_timeout(Duration::ZERO);
    let mut pipeline = load(&dir, &options);
    let prepared = pipeline
        .prepare(TextGenerationInput::from("h i".to_string()))
        .expect("prepare");

    // Act
    let result = pipeline.execute(prepared);

    // Assert
    assert!(matches!(result, Err(CoreError::Timeout(_))));
}

#[test]
fn warm_up_runs_without_a_request() {
    // Arrange
    let dir = model_dir(CACHED);
    let mut pipeline = load(&dir, &TextGenerationOptions::new());

    // Act / Assert
    pipeline.warm_up().expect("warm-up should succeed");
}

#[test]
fn model_without_chat_template_fails_at_startup() {
    // Arrange
    let dir = model_dir(CACHED);
    std::fs::write(dir.path().join("tokenizer_config.json"), "{}").expect("write");

    // Act
    let result = TextGenerationPipeline::new(dir.path(), &ExecutionProvider::Cpu);

    // Assert
    let err = result.err().expect("missing template should fail");
    assert!(matches!(err, CoreError::ModelValidation(_)), "{err:?}");
    assert!(err.to_string().contains("chat template"), "{err}");
}

#[test]
fn chat_template_file_is_used_when_the_tokenizer_config_has_none() {
    // Arrange -- the layout current `transformers` releases save.
    let dir = model_dir(CACHED);
    std::fs::write(dir.path().join("tokenizer_config.json"), "{}").expect("write");
    std::fs::write(
        dir.path().join("chat_template.jinja"),
        "{% for message in messages %}{{ message['content'] }} {% endfor %}e",
    )
    .expect("write");
    let mut pipeline = load(&dir, &TextGenerationOptions::new().with_max_new_tokens(10));

    // Act
    let text = generate(&mut pipeline, None, "h i");

    // Assert
    assert_eq!(text, words(&simulate(&ids("h i e"), 10, 1.0, 0)));
}

#[test]
fn seq2seq_graph_is_rejected_with_its_inputs_named() {
    // Arrange -- a decoder of an encoder-decoder export under the name of
    // a decoder-only graph.
    let dir = model_dir(CACHED);
    std::fs::copy(
        fixture("seq2seq", "decoder_model.onnx"),
        dir.path().join("model.onnx"),
    )
    .expect("copy graph");

    // Act
    let result = TextGenerationPipeline::new(dir.path(), &ExecutionProvider::Cpu);

    // Assert
    let err = result.err().expect("wrong graph should fail");
    assert!(matches!(err, CoreError::ModelValidation(_)), "{err:?}");
    assert!(err.to_string().contains("encoder_hidden_states"), "{err}");
}

#[test]
fn limits_that_exceed_the_model_context_fail_at_startup() {
    // Arrange -- context of 512 tokens (see `base_config`).
    let dir = model_dir(CACHED);
    let options = TextGenerationOptions::new()
        .with_max_input_tokens(500)
        .with_max_new_tokens(100);

    // Act
    let result = TextGenerationPipeline::with_options(dir.path(), &ExecutionProvider::Cpu, &options);

    // Assert
    assert!(matches!(result, Err(CoreError::Config(_))));
}

#[test]
fn other_text_profiles_ignore_a_system_message() {
    // Arrange -- the fused seq2seq fixture stands for every profile that
    // has no system message.
    let dir = tempfile::tempdir().expect("temp dir");
    for name in ["model.onnx", "tokenizer.json"] {
        std::fs::copy(fixture("seq2seq", name), dir.path().join(name)).expect("copy fixture");
    }
    let pipeline = Seq2SeqPipeline::new(dir.path(), &ExecutionProvider::Cpu).expect("load");
    let mut kind = PipelineKind::Seq2Seq(pipeline);
    assert!(!kind.accepts_system_prompt());
    let mut run = |input: InferenceInput| {
        let prepared = kind.prepare(input).expect("prepare");
        kind.execute(prepared).expect("execute").to_json()
    };

    // Act
    let plain = run(InferenceInput::Text("a b c".to_string()));
    let with_system = run(InferenceInput::TextWithSystem {
        text: "a b c".to_string(),
        system: "d e f".to_string(),
    });

    // Assert
    assert_eq!(plain, with_system);
}
