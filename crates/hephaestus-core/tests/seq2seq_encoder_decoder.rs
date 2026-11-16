//! End-to-end tests for the seq2seq pipeline's two modes.
//!
//! Runs real ONNX Runtime sessions over the tiny graphs in
//! `tests/fixtures/seq2seq` (see `make_fixtures.py` there). The
//! encoder-decoder fixture echoes its input, and the fused fixture shifts
//! every word by one, so the text that comes back tells which mode ran:
//!
//! - encoder-decoder: `"a b c"` -> `"a b c"`
//! - fused:           `"a b c"` -> `"b c d"`

use std::path::{Path, PathBuf};
use std::time::Duration;

use hephaestus_core::{
    CoreError, ExecutionProvider, GenerationControl, Pipeline, PipelineKind, PipelineOutput,
    PreparedAudio, PreparedData, Seq2SeqOptions, Seq2SeqPipeline, TokenUsage,
};
use tempfile::TempDir;

const ENCODER: &str = "encoder_model.onnx";
const DECODER: &str = "decoder_model.onnx";
const MERGED_DECODER: &str = "decoder_model_merged.onnx";
const FUSED: &str = "model.onnx";

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/seq2seq")
        .join(name)
}

/// Generation settings of a BART-like model over the fixture vocabulary.
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

fn encoder_decoder_dir(config: &serde_json::Value) -> TempDir {
    model_dir(config, &[(ENCODER, ENCODER), (DECODER, DECODER)])
}

fn load(dir: &TempDir) -> Seq2SeqPipeline {
    Seq2SeqPipeline::new(dir.path(), &ExecutionProvider::Cpu).expect("pipeline should load")
}

fn load_with(dir: &TempDir, options: &Seq2SeqOptions) -> Result<Seq2SeqPipeline, CoreError> {
    Seq2SeqPipeline::with_options(dir.path(), &ExecutionProvider::Cpu, options)
}

fn generate(pipeline: &mut Seq2SeqPipeline, text: &str) -> Result<String, CoreError> {
    let prepared = pipeline.prepare(text.to_string())?;
    pipeline.execute(prepared)
}

fn words(count: usize) -> String {
    vec!["a"; count].join(" ")
}

// --- Encoder-decoder mode, end to end ---

#[test]
fn encoder_decoder_export_generates_text_through_pipeline_kind() {
    // Arrange
    let dir = encoder_decoder_dir(&base_config());
    let mut pipeline = PipelineKind::Seq2Seq(load(&dir));

    // Act
    let prepared = pipeline
        .prepare("a b c".to_string())
        .expect("prepare should succeed");
    let output = pipeline.execute(prepared).expect("execute should succeed");

    // Assert -- same response shape as the fused mode.
    assert_eq!(
        output.to_json(),
        serde_json::json!({ "generated_text": "a b c" })
    );
}

#[test]
fn encoder_decoder_export_is_found_in_onnx_subdirectory() {
    // Arrange
    let dir = model_dir(
        &base_config(),
        &[
            (ENCODER, "onnx/encoder_model.onnx"),
            (DECODER, "onnx/decoder_model.onnx"),
        ],
    );
    let mut pipeline = load(&dir);

    // Act
    let text = generate(&mut pipeline, "d e f").expect("generation should succeed");

    // Assert
    assert_eq!(text, "d e f");
}

#[test]
fn merged_decoder_runs_when_it_is_the_only_decoder() {
    // Arrange -- the fixture shifts every token if `use_cache_branch` is
    // not false or the dummy cache is not all zeros.
    let dir = model_dir(
        &base_config(),
        &[(ENCODER, ENCODER), (MERGED_DECODER, MERGED_DECODER)],
    );
    let mut pipeline = load(&dir);

    // Act
    let text = generate(&mut pipeline, "a b c").expect("generation should succeed");

    // Assert
    assert_eq!(text, "a b c");
}

#[test]
fn consecutive_inputs_do_not_leak_state() {
    // Arrange
    let dir = encoder_decoder_dir(&base_config());
    let mut pipeline = load(&dir);

    // Act
    let first = generate(&mut pipeline, "a b c d e").expect("first generation");
    let second = generate(&mut pipeline, "f").expect("second generation");

    // Assert
    assert_eq!(first, "a b c d e");
    assert_eq!(second, "f");
}

// --- Mode selection by files ---

#[test]
fn fused_model_runs_as_before_when_it_is_the_only_model() {
    // Arrange
    let dir = model_dir(&base_config(), &[(FUSED, FUSED)]);
    let mut pipeline = load(&dir);

    // Act
    let text = generate(&mut pipeline, "a b c").expect("generation should succeed");

    // Assert
    assert_eq!(text, "b c d");
}

#[test]
fn fused_model_wins_when_encoder_decoder_files_are_also_present() {
    // Arrange -- model.onnx next to a complete encoder-decoder export.
    let dir = model_dir(
        &base_config(),
        &[(FUSED, FUSED), (ENCODER, ENCODER), (DECODER, DECODER)],
    );
    let mut pipeline = load(&dir);

    // Act
    let text = generate(&mut pipeline, "a b c").expect("generation should succeed");

    // Assert -- the fused output, not the echo.
    assert_eq!(text, "b c d");
}

#[test]
fn fused_model_in_onnx_subdirectory_wins_over_flat_encoder_decoder_files() {
    // Arrange
    let dir = model_dir(
        &base_config(),
        &[
            (FUSED, "onnx/model.onnx"),
            (ENCODER, ENCODER),
            (DECODER, DECODER),
        ],
    );
    let mut pipeline = load(&dir);

    // Act
    let text = generate(&mut pipeline, "a b c").expect("generation should succeed");

    // Assert
    assert_eq!(text, "b c d");
}

#[test]
fn fused_model_batches_as_before() {
    // Arrange -- equal-length inputs, so padding does not enter the result.
    let dir = model_dir(
        &base_config(),
        &[(FUSED, FUSED), (ENCODER, ENCODER), (DECODER, DECODER)],
    );
    let mut pipeline = PipelineKind::Seq2Seq(load(&dir));
    let batch = ["a b", "c d"]
        .into_iter()
        .map(|text| pipeline.prepare(text.to_string()).expect("prepare"))
        .collect();

    // Act
    let results = pipeline.execute_batch(batch);

    // Assert
    let texts: Vec<serde_json::Value> = results
        .into_iter()
        .map(|result| result.expect("batch item should succeed").to_json())
        .collect();
    assert_eq!(
        texts,
        vec![
            serde_json::json!({ "generated_text": "b c" }),
            serde_json::json!({ "generated_text": "d e" }),
        ]
    );
}

#[test]
fn directory_without_any_model_reports_the_missing_single_file_model() {
    // Arrange
    let dir = model_dir(&base_config(), &[]);

    // Act
    let result = load_with(&dir, &Seq2SeqOptions::new());

    // Assert -- the error existing deployments would see today.
    let message = result.err().expect("load should fail").to_string();
    assert!(
        message.contains("ONNX model not found"),
        "unexpected error: {message}"
    );
    assert!(
        message.contains("model.onnx"),
        "unexpected error: {message}"
    );
}

// --- Generation settings from the model configuration ---

#[test]
fn forced_bos_token_replaces_the_first_generated_token() {
    // Arrange -- with the first token forced, the echo resumes at the
    // second word.
    let mut config = base_config();
    config["forced_bos_token_id"] = serde_json::json!(0);
    let dir = encoder_decoder_dir(&config);
    let mut pipeline = load(&dir);

    // Act
    let text = generate(&mut pipeline, "a b c").expect("generation should succeed");

    // Assert
    assert_eq!(text, "b c");
}

#[test]
fn generation_config_file_overrides_model_config() {
    // Arrange -- config.json allows 31 new tokens, generation_config.json 2.
    let dir = encoder_decoder_dir(&base_config());
    std::fs::write(
        dir.path().join("generation_config.json"),
        serde_json::json!({ "max_length": 3 }).to_string(),
    )
    .expect("write generation config");
    let mut pipeline = load(&dir);

    // Act
    let text = generate(&mut pipeline, "a b c d").expect("generation should succeed");

    // Assert
    assert_eq!(text, "a b");
}

#[test]
fn no_repeat_ngram_size_blocks_a_repeated_pair() {
    // Arrange -- echoing "a a a b" would repeat the pair "a a"; with the
    // rule the third "a" is banned and the next-best token (id 0, dropped
    // by decoding as special) takes its place.
    let mut config = base_config();
    config["no_repeat_ngram_size"] = serde_json::json!(2);
    let dir = encoder_decoder_dir(&config);
    let mut pipeline = load(&dir);

    // Act
    let text = generate(&mut pipeline, "a a a b").expect("generation should succeed");

    // Assert
    assert_eq!(text, "a a b");
}

// --- Runaway bound ---

#[test]
fn model_that_never_emits_eos_stops_at_configured_max_length() {
    // Arrange -- token 15 never appears, so nothing ends generation.
    let mut config = base_config();
    config["eos_token_id"] = serde_json::json!(15);
    config["max_length"] = serde_json::json!(8);
    let dir = encoder_decoder_dir(&config);
    let mut pipeline = load(&dir);

    // Act
    let text = generate(&mut pipeline, "a b c d e f g h i j").expect("generation should succeed");

    // Assert -- max_length 8 counts the start token, leaving exactly
    // seven generated tokens of the ten the echo could produce.
    assert_eq!(text, "a b c d e f g");
}

#[test]
fn model_that_never_emits_eos_returns_within_the_limit_on_short_input() {
    // Arrange -- past the end of the input the echo repeats the closing
    // token forever; token 15 still never appears.
    let mut config = base_config();
    config["eos_token_id"] = serde_json::json!(15);
    config["max_length"] = serde_json::json!(8);
    let dir = encoder_decoder_dir(&config);
    let mut pipeline = load(&dir);

    // Act
    let text = generate(&mut pipeline, "a b").expect("generation should succeed");

    // Assert -- "a", "b", then five closing tokens that decoding drops.
    assert_eq!(text, "a b");
}

#[test]
fn explicit_token_limit_cuts_generation_short() {
    // Arrange
    let dir = encoder_decoder_dir(&base_config());
    let options = Seq2SeqOptions::new().with_max_new_tokens(3);
    let mut pipeline = load_with(&dir, &options).expect("pipeline should load");

    // Act
    let text = generate(&mut pipeline, "a b c d e f").expect("generation should succeed");

    // Assert -- exactly three tokens, although the input has more to echo.
    assert_eq!(text, "a b c");
}

#[test]
fn generation_past_its_time_budget_fails_with_timeout() {
    // Arrange -- a budget that is spent before generation starts.
    let dir = encoder_decoder_dir(&base_config());
    let options = Seq2SeqOptions::new().with_timeout(Duration::ZERO);
    let mut pipeline = load_with(&dir, &options).expect("pipeline should load");

    // Act
    let result = generate(&mut pipeline, "a b c");

    // Assert
    assert!(
        matches!(result, Err(CoreError::Timeout(_))),
        "expected a timeout, got: {result:?}"
    );
}

// --- Input token cap ---

#[test]
fn input_cap_defaults_to_512_tokens() {
    // Arrange -- no position limit in the configuration, 600 input words.
    let mut config = base_config();
    config
        .as_object_mut()
        .expect("config is an object")
        .remove("max_position_embeddings");
    let dir = encoder_decoder_dir(&config);
    let options = Seq2SeqOptions::new().with_max_new_tokens(700);
    let mut pipeline = load_with(&dir, &options).expect("pipeline should load");

    // Act
    let text = generate(&mut pipeline, &words(600)).expect("generation should succeed");

    // Assert -- 512 tokens are <s>, 510 words and </s>.
    assert_eq!(text.split(' ').count(), 510);
}

#[test]
fn explicit_input_cap_truncates_input() {
    // Arrange
    let dir = encoder_decoder_dir(&base_config());
    let options = Seq2SeqOptions::new().with_max_input_tokens(8);
    let mut pipeline = load_with(&dir, &options).expect("pipeline should load");

    // Act
    let text = generate(&mut pipeline, "a b c d e f g h i j").expect("generation should succeed");

    // Assert -- 8 tokens are <s>, six words and </s>.
    assert_eq!(text, "a b c d e f");
}

#[test]
fn explicit_input_cap_applies_to_fused_model_too() {
    // Arrange
    let dir = model_dir(&base_config(), &[(FUSED, FUSED)]);
    let options = Seq2SeqOptions::new().with_max_input_tokens(8);
    let mut pipeline = load_with(&dir, &options).expect("pipeline should load");

    // Act
    let text = generate(&mut pipeline, "a b c d e f g h i j").expect("generation should succeed");

    // Assert -- six words survive truncation and are shifted by one.
    assert_eq!(text, "b c d e f g");
}

// --- Load-time validation ---

#[test]
fn encoder_without_decoder_fails_at_load() {
    // Arrange
    let dir = model_dir(&base_config(), &[(ENCODER, ENCODER)]);

    // Act
    let result = load_with(&dir, &Seq2SeqOptions::new());

    // Assert
    let message = result.err().expect("load should fail").to_string();
    assert!(message.contains(DECODER), "unexpected error: {message}");
}

#[test]
fn decoder_with_unusable_inputs_fails_at_load() {
    // Arrange -- the fused graph has no encoder-hidden-state input.
    let dir = model_dir(&base_config(), &[(ENCODER, ENCODER), (FUSED, DECODER)]);

    // Act
    let result = load_with(&dir, &Seq2SeqOptions::new());

    // Assert
    let message = result.err().expect("load should fail").to_string();
    assert!(
        message.contains("encoder_hidden_states"),
        "unexpected error: {message}"
    );
}

#[test]
fn configuration_without_start_token_fails_at_load() {
    // Arrange
    let config = serde_json::json!({ "eos_token_id": 2, "vocab_size": 16 });
    let dir = encoder_decoder_dir(&config);

    // Act
    let result = load_with(&dir, &Seq2SeqOptions::new());

    // Assert
    let message = result.err().expect("load should fail").to_string();
    assert!(
        message.contains("decoder_start_token_id"),
        "unexpected error: {message}"
    );
}

#[test]
fn input_cap_beyond_model_position_limit_fails_at_load() {
    // Arrange -- the configuration declares 64 positions.
    let dir = encoder_decoder_dir(&base_config());
    let options = Seq2SeqOptions::new().with_max_input_tokens(65);

    // Act
    let result = load_with(&dir, &options);

    // Assert
    assert!(
        matches!(result, Err(CoreError::Config(_))),
        "expected a config error"
    );
}

// --- Batcher contract ---

#[test]
fn batch_returns_one_result_per_input_in_order() {
    // Arrange -- inputs of different lengths, as the batcher would send.
    let dir = encoder_decoder_dir(&base_config());
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
fn batch_reports_audio_item_as_error_without_failing_its_neighbours() {
    // Arrange
    let dir = encoder_decoder_dir(&base_config());
    let mut pipeline = PipelineKind::Seq2Seq(load(&dir));
    let audio = PreparedAudio::new_for_test(ndarray::Array2::<f32>::zeros((1, 4)));
    let batch = vec![
        pipeline.prepare("a b".to_string()).expect("prepare"),
        PreparedData::Audio(audio),
        pipeline.prepare("c".to_string()).expect("prepare"),
    ];

    // Act
    let results = pipeline.execute_batch(batch);

    // Assert
    assert_eq!(results.len(), 3);
    assert!(matches!(&results[0], Ok(PipelineOutput::Seq2Seq(text)) if text == "a b"));
    assert!(matches!(&results[1], Err(CoreError::InvalidInput(_))));
    assert!(matches!(&results[2], Ok(PipelineOutput::Seq2Seq(text)) if text == "c"));
}

#[test]
fn empty_batch_returns_no_results() {
    // Arrange
    let dir = encoder_decoder_dir(&base_config());
    let mut pipeline = PipelineKind::Seq2Seq(load(&dir));

    // Act
    let results = pipeline.execute_batch(Vec::new());

    // Assert
    assert!(results.is_empty());
}

#[test]
fn batch_shares_one_time_budget() {
    // Arrange -- a spent budget fails every item, and still yields one
    // result per input.
    let dir = encoder_decoder_dir(&base_config());
    let options = Seq2SeqOptions::new().with_timeout(Duration::ZERO);
    let mut pipeline =
        PipelineKind::Seq2Seq(load_with(&dir, &options).expect("pipeline should load"));
    let batch = ["a", "b"]
        .into_iter()
        .map(|text| pipeline.prepare(text.to_string()).expect("prepare"))
        .collect();

    // Act
    let results = pipeline.execute_batch(batch);

    // Assert
    assert_eq!(results.len(), 2);
    assert!(
        results
            .iter()
            .all(|result| matches!(result, Err(CoreError::Timeout(_))))
    );
}

// --- Per-request limits and token counts ---

#[test]
fn cancelled_generation_stops_between_decoder_steps() {
    // Arrange -- cancelled on the fourth time the loop asks: once before
    // the encoder pass, then once per generated token.
    let dir = encoder_decoder_dir(&base_config());
    let mut pipeline = PipelineKind::Seq2Seq(load(&dir));
    let prepared = pipeline
        .prepare("a b c d e f g h".to_string())
        .expect("prepare");
    let asked = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let counter = std::sync::Arc::clone(&asked);
    let control = GenerationControl::new().with_cancel_check(move || {
        counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst) >= 3
    });

    // Act
    let result = pipeline.execute_with(prepared, &control);

    // Assert -- two tokens were generated, the third was never started.
    let err = result.err().expect("cancelled generation should fail");
    assert!(matches!(err, CoreError::Cancelled(_)), "{err:?}");
    assert!(err.to_string().contains("after 2 tokens"), "{err}");
    assert_eq!(asked.load(std::sync::atomic::Ordering::SeqCst), 4);
}

#[test]
fn encoder_decoder_execution_reports_token_counts() {
    // Arrange -- the fixture echoes its input: <s> a b c </s> is five
    // encoder tokens, and "a b c" plus the end token four generated ones.
    let dir = encoder_decoder_dir(&base_config());
    let mut pipeline = PipelineKind::Seq2Seq(load(&dir));
    let prepared = pipeline.prepare("a b c".to_string()).expect("prepare");

    // Act
    let execution = pipeline
        .execute_with(prepared, &GenerationControl::default())
        .expect("execute");

    // Assert
    let usage = execution.usage.expect("encoder-decoder mode counts tokens");
    assert_eq!(usage.prompt_tokens, 5);
    assert_eq!(usage.completion_tokens, 4);
    assert_eq!(
        execution.to_json(),
        serde_json::json!({
            "generated_text": "a b c",
            "prompt_tokens": 5,
            "completion_tokens": 4,
        })
    );
}

#[test]
fn fused_model_execution_has_no_token_counts_and_the_same_json() {
    // Arrange
    let dir = model_dir(&base_config(), &[(FUSED, FUSED)]);
    let mut pipeline = PipelineKind::Seq2Seq(load(&dir));
    let prepared = pipeline.prepare("a b c".to_string()).expect("prepare");
    let again = pipeline.prepare("a b c".to_string()).expect("prepare");

    // Act -- an expired deadline and a cancelled caller: a single forward
    // pass ignores both.
    let control = GenerationControl::new()
        .with_deadline(std::time::Instant::now())
        .with_cancel_check(|| true);
    let execution = pipeline.execute_with(prepared, &control).expect("execute");
    let plain = pipeline.execute(again).expect("execute");

    // Assert
    assert_eq!(execution.usage, None::<TokenUsage>);
    assert_eq!(execution.to_json().to_string(), plain.to_json().to_string());
}
