//! End-to-end tests for the key/value cache of the encoder-decoder loop.
//!
//! Runs real ONNX Runtime sessions over the tiny graphs in
//! `tests/fixtures/seq2seq` (see `make_fixtures.py` there).
//! `decoder_model_merged_cached.onnx` follows the Optimum merged-decoder
//! convention and only echoes its input when the cache is driven
//! correctly: uncached first step with a zero dummy cache, then the newest
//! token alone with the grown decoder cache and the *first* step's encoder
//! cache (later steps return that one empty).
//!
//! The cached and the uncached loop must generate the same text for every
//! input and every generation setting.

use std::path::{Path, PathBuf};

use hephaestus_core::{
    CoreError, ExecutionProvider, Pipeline, PipelineKind, PipelineOutput, Seq2SeqOptions,
    Seq2SeqPipeline,
};
use tempfile::TempDir;

const ENCODER: &str = "encoder_model.onnx";
const DECODER: &str = "decoder_model.onnx";
const MERGED_DECODER: &str = "decoder_model_merged.onnx";
/// Fixture: a merged decoder whose cache can be fed back.
const CACHED_MERGED_FIXTURE: &str = "decoder_model_merged_cached.onnx";
/// Fixture: a merged decoder that returns nothing for its cache input.
const UNCACHEABLE_MERGED_FIXTURE: &str = "decoder_model_merged.onnx";

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/seq2seq")
        .join(name)
}

fn base_config() -> serde_json::Value {
    serde_json::json!({
        "architectures": ["BartForConditionalGeneration"],
        "bos_token_id": 0,
        "decoder_start_token_id": 2,
        "eos_token_id": 2,
        "max_length": 32,
        "max_position_embeddings": 64,
        "pad_token_id": 1,
        "vocab_size": 16
    })
}

/// Build a model directory holding the tokenizer, `config`, and the given
/// fixture graphs. Each entry is `(fixture file, path inside the directory)`.
fn model_dir(config: &serde_json::Value, graphs: &[(&str, &str)]) -> TempDir {
    let dir = tempfile::tempdir().expect("temp dir");
    std::fs::copy(fixture("tokenizer.json"), dir.path().join("tokenizer.json"))
        .expect("copy tokenizer");
    std::fs::write(dir.path().join("config.json"), config.to_string()).expect("write config");
    for (source, target) in graphs {
        let target = dir.path().join(target);
        std::fs::create_dir_all(target.parent().expect("target has a parent")).expect("mkdir");
        std::fs::copy(fixture(source), &target).expect("copy graph");
    }
    dir
}

/// The full Optimum file set: encoder, plain decoder, cacheable merged decoder.
fn full_export_dir(config: &serde_json::Value) -> TempDir {
    model_dir(
        config,
        &[
            (ENCODER, ENCODER),
            (DECODER, DECODER),
            (CACHED_MERGED_FIXTURE, MERGED_DECODER),
        ],
    )
}

fn load(dir: &TempDir) -> Seq2SeqPipeline {
    Seq2SeqPipeline::new(dir.path(), &ExecutionProvider::Cpu).expect("pipeline should load")
}

fn load_with(dir: &TempDir, options: &Seq2SeqOptions) -> Seq2SeqPipeline {
    Seq2SeqPipeline::with_options(dir.path(), &ExecutionProvider::Cpu, options)
        .expect("pipeline should load")
}

fn generate(pipeline: &mut Seq2SeqPipeline, text: &str) -> Result<String, CoreError> {
    let prepared = pipeline.prepare(text.to_string())?;
    pipeline.execute(prepared)
}

// --- Which loop runs ---

#[test]
fn merged_decoder_is_preferred_and_decodes_with_cache() {
    // Arrange
    let dir = full_export_dir(&base_config());
    let mut pipeline = load(&dir);

    // Act
    let text = generate(&mut pipeline, "a b c d e").expect("generation should succeed");

    // Assert
    assert!(pipeline.uses_kv_cache());
    assert_eq!(text, "a b c d e");
}

#[test]
fn merged_decoder_alone_decodes_with_cache() {
    // Arrange
    let dir = model_dir(
        &base_config(),
        &[(ENCODER, ENCODER), (CACHED_MERGED_FIXTURE, MERGED_DECODER)],
    );
    let mut pipeline = load(&dir);

    // Act
    let text = generate(&mut pipeline, "d e f").expect("generation should succeed");

    // Assert
    assert!(pipeline.uses_kv_cache());
    assert_eq!(text, "d e f");
}

#[test]
fn merged_decoder_in_onnx_subdirectory_decodes_with_cache() {
    // Arrange
    let dir = model_dir(
        &base_config(),
        &[
            (ENCODER, "onnx/encoder_model.onnx"),
            (DECODER, "onnx/decoder_model.onnx"),
            (CACHED_MERGED_FIXTURE, "onnx/decoder_model_merged.onnx"),
        ],
    );
    let mut pipeline = load(&dir);

    // Act
    let text = generate(&mut pipeline, "a b c").expect("generation should succeed");

    // Assert
    assert!(pipeline.uses_kv_cache());
    assert_eq!(text, "a b c");
}

#[test]
fn plain_decoder_alone_decodes_without_cache() {
    // Arrange -- an export without a merged decoder.
    let dir = model_dir(&base_config(), &[(ENCODER, ENCODER), (DECODER, DECODER)]);
    let mut pipeline = load(&dir);

    // Act
    let text = generate(&mut pipeline, "a b c").expect("generation should succeed");

    // Assert
    assert!(!pipeline.uses_kv_cache());
    assert_eq!(text, "a b c");
}

#[test]
fn cache_switched_off_runs_the_plain_decoder() {
    // Arrange -- the cacheable merged decoder is present but not wanted.
    let dir = full_export_dir(&base_config());
    let options = Seq2SeqOptions::new().with_kv_cache(false);
    let mut pipeline = load_with(&dir, &options);

    // Act
    let text = generate(&mut pipeline, "a b c").expect("generation should succeed");

    // Assert
    assert!(!pipeline.uses_kv_cache());
    assert_eq!(text, "a b c");
}

#[test]
fn cache_switched_off_runs_merged_decoder_uncached_when_it_is_the_only_decoder() {
    // Arrange
    let dir = model_dir(
        &base_config(),
        &[(ENCODER, ENCODER), (CACHED_MERGED_FIXTURE, MERGED_DECODER)],
    );
    let options = Seq2SeqOptions::new().with_kv_cache(false);
    let mut pipeline = load_with(&dir, &options);

    // Act
    let text = generate(&mut pipeline, "a b c d").expect("generation should succeed");

    // Assert
    assert!(!pipeline.uses_kv_cache());
    assert_eq!(text, "a b c d");
}

#[test]
fn merged_decoder_without_cache_outputs_falls_back_to_plain_decoder() {
    // Arrange -- this merged fixture has a cache input but no output to
    // feed it from.
    let dir = model_dir(
        &base_config(),
        &[
            (ENCODER, ENCODER),
            (DECODER, DECODER),
            (UNCACHEABLE_MERGED_FIXTURE, MERGED_DECODER),
        ],
    );
    let mut pipeline = load(&dir);

    // Act
    let text = generate(&mut pipeline, "a b c").expect("generation should succeed");

    // Assert
    assert!(!pipeline.uses_kv_cache());
    assert_eq!(text, "a b c");
}

#[test]
fn unreadable_merged_decoder_falls_back_to_plain_decoder() {
    // Arrange -- a broken merged file must not take down an export that
    // loads today.
    let dir = model_dir(&base_config(), &[(ENCODER, ENCODER), (DECODER, DECODER)]);
    std::fs::write(dir.path().join(MERGED_DECODER), b"not an onnx graph").expect("write");
    let mut pipeline = load(&dir);

    // Act
    let text = generate(&mut pipeline, "a b c").expect("generation should succeed");

    // Assert
    assert!(!pipeline.uses_kv_cache());
    assert_eq!(text, "a b c");
}

#[test]
fn fused_model_still_wins_over_merged_decoder() {
    // Arrange
    let dir = model_dir(
        &base_config(),
        &[
            ("model.onnx", "model.onnx"),
            (ENCODER, ENCODER),
            (DECODER, DECODER),
            (CACHED_MERGED_FIXTURE, MERGED_DECODER),
        ],
    );
    let mut pipeline = load(&dir);

    // Act
    let text = generate(&mut pipeline, "a b c").expect("generation should succeed");

    // Assert -- the fused output (every word shifted by one), not the echo.
    assert!(!pipeline.uses_kv_cache());
    assert_eq!(text, "b c d");
}

// --- Cached and uncached loops agree ---

/// Generation settings that exercise every rule of the decode loop.
fn configs() -> Vec<(&'static str, serde_json::Value)> {
    let with = |edits: &[(&str, serde_json::Value)]| {
        let mut config = base_config();
        for (key, value) in edits {
            config[*key] = value.clone();
        }
        config
    };
    vec![
        ("base", base_config()),
        (
            "forced_bos",
            with(&[("forced_bos_token_id", serde_json::json!(0))]),
        ),
        (
            "forced_bos_and_eos",
            with(&[
                ("forced_bos_token_id", serde_json::json!(0)),
                ("forced_eos_token_id", serde_json::json!(2)),
                ("max_length", serde_json::json!(6)),
            ]),
        ),
        (
            "no_repeat_2",
            with(&[("no_repeat_ngram_size", serde_json::json!(2))]),
        ),
        (
            "no_repeat_3_forced_bos",
            with(&[
                ("no_repeat_ngram_size", serde_json::json!(3)),
                ("forced_bos_token_id", serde_json::json!(0)),
            ]),
        ),
        (
            "never_eos_short_limit",
            with(&[
                ("eos_token_id", serde_json::json!(15)),
                ("max_length", serde_json::json!(8)),
            ]),
        ),
        (
            "limit_of_one",
            with(&[("max_length", serde_json::json!(2))]),
        ),
        (
            "limit_of_two",
            with(&[("max_length", serde_json::json!(3))]),
        ),
    ]
}

const INPUTS: [&str; 8] = [
    "a",
    "a b",
    "a b c",
    "a a a b",
    "a b a b a b a b",
    "d e f g h i j k l a b c",
    "a a a a a a a a a a a a",
    "l k j i h g f e d c b a l k j i h g f e d c b a l k j i h g f e d c b a",
];

#[test]
fn cached_and_uncached_loops_generate_the_same_text() {
    for (name, config) in configs() {
        // Arrange -- the same export, loaded once per loop.
        let dir = full_export_dir(&config);
        let mut cached = load(&dir);
        let mut uncached = load_with(&dir, &Seq2SeqOptions::new().with_kv_cache(false));
        assert!(cached.uses_kv_cache(), "{name}: cache should be on");
        assert!(!uncached.uses_kv_cache(), "{name}: cache should be off");

        for input in INPUTS {
            // Act
            let with_cache = generate(&mut cached, input).expect("cached generation");
            let without_cache = generate(&mut uncached, input).expect("uncached generation");

            // Assert
            assert_eq!(
                with_cache, without_cache,
                "{name}: cached and uncached text differ for input '{input}'"
            );
        }
    }
}

#[test]
fn cached_loop_matches_the_results_the_uncached_loop_is_tested_for() {
    // Arrange / Act / Assert -- the expectations of the uncached tests in
    // `seq2seq_encoder_decoder.rs`, on the cached loop.
    let mut forced = base_config();
    forced["forced_bos_token_id"] = serde_json::json!(0);
    let dir = full_export_dir(&forced);
    assert_eq!(generate(&mut load(&dir), "a b c").expect("generate"), "b c");

    let mut no_repeat = base_config();
    no_repeat["no_repeat_ngram_size"] = serde_json::json!(2);
    let dir = full_export_dir(&no_repeat);
    assert_eq!(
        generate(&mut load(&dir), "a a a b").expect("generate"),
        "a a b"
    );

    let mut never_eos = base_config();
    never_eos["eos_token_id"] = serde_json::json!(15);
    never_eos["max_length"] = serde_json::json!(8);
    let dir = full_export_dir(&never_eos);
    assert_eq!(
        generate(&mut load(&dir), "a b c d e f g h i j").expect("generate"),
        "a b c d e f g"
    );
    assert_eq!(generate(&mut load(&dir), "a b").expect("generate"), "a b");

    let dir = full_export_dir(&base_config());
    let options = Seq2SeqOptions::new().with_max_new_tokens(3);
    assert_eq!(
        generate(&mut load_with(&dir, &options), "a b c d e f").expect("generate"),
        "a b c"
    );
}

#[test]
fn consecutive_inputs_do_not_share_a_cache() {
    // Arrange
    let dir = full_export_dir(&base_config());
    let mut pipeline = load(&dir);

    // Act -- a long input, then a short one, then the long one again.
    let first = generate(&mut pipeline, "a b c d e").expect("first generation");
    let second = generate(&mut pipeline, "f").expect("second generation");
    let third = generate(&mut pipeline, "a b c d e").expect("third generation");

    // Assert
    assert_eq!(first, "a b c d e");
    assert_eq!(second, "f");
    assert_eq!(third, "a b c d e");
}

#[test]
fn batch_with_cache_returns_one_result_per_input_in_order() {
    // Arrange
    let dir = full_export_dir(&base_config());
    let mut pipeline = PipelineKind::Seq2Seq(load(&dir));
    let texts = ["a b c d", "e", "f g"];
    let batch = texts
        .iter()
        .map(|text| pipeline.prepare((*text).to_string()).expect("prepare"))
        .collect();

    // Act
    let results = pipeline.execute_batch(batch);

    // Assert
    let generated: Vec<String> = results
        .into_iter()
        .map(|result| match result.expect("batch item should succeed") {
            PipelineOutput::Seq2Seq(text) => text,
            _ => panic!("expected seq2seq output"),
        })
        .collect();
    assert_eq!(generated, texts);
}

#[test]
fn response_shape_is_unchanged_with_cache() {
    // Arrange
    let dir = full_export_dir(&base_config());
    let mut pipeline = PipelineKind::Seq2Seq(load(&dir));

    // Act
    let prepared = pipeline
        .prepare("a b c".to_string())
        .expect("prepare should succeed");
    let output = pipeline.execute(prepared).expect("execute should succeed");

    // Assert
    assert_eq!(
        output.to_json(),
        serde_json::json!({ "generated_text": "a b c" })
    );
}

#[test]
fn spent_time_budget_fails_with_timeout_with_cache() {
    // Arrange
    let dir = full_export_dir(&base_config());
    let options = Seq2SeqOptions::new().with_timeout(std::time::Duration::ZERO);
    let mut pipeline = load_with(&dir, &options);

    // Act
    let result = generate(&mut pipeline, "a b c");

    // Assert
    assert!(
        matches!(result, Err(CoreError::Timeout(_))),
        "expected a timeout, got: {result:?}"
    );
}
