//! End-to-end tests for batched text generation.
//!
//! Runs real ONNX Runtime sessions over `model_batch.onnx` and
//! `model_batch_positions.onnx` in `tests/fixtures/text_generation` (see
//! `make_fixtures.py` there). In those graphs a row's next token depends
//! on every token the row is allowed to see and on how far back each one
//! is, with positions counted by column (first graph) or fed by the caller
//! (second). A row therefore gives the tokens it gives alone only if its
//! padding is on the left and masked, its real tokens are not masked, its
//! cache stays in the columns it was computed at, and it never gets
//! another row's cache.
//!
//! Every test compares against the same request run alone through the
//! single-request loop, and that loop against [`simulate`].
//!
//! Those graphs compute in whole numbers. `tiny_qwen3/` is a real
//! transformer with random weights (see `make_tiny_qwen3.py`): rotary
//! positions counted by column, grouped-query attention, a float32 mask.
//! The tests at the end run it batched against alone.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use hephaestus_core::{
    BatchProgress, CoreError, ExecutionProvider, GenerationBatch, GenerationControl,
    GenerationRequest, PipelineKind, PipelineOutput, PreparedData, TextGenerationOptions,
    TextGenerationPipeline, TokenUsage,
};
use tempfile::TempDir;

/// Positions counted by column, like an export without `position_ids`.
const BY_COLUMN: &str = "model_batch.onnx";
/// Positions fed through `position_ids`.
const BY_POSITION_IDS: &str = "model_batch_positions.onnx";
const GRAPHS: [&str; 2] = [BY_COLUMN, BY_POSITION_IDS];
const EOS: i64 = 2;
const MAX_NEW_TOKENS: usize = 40;

/// User messages whose prompts have 4 to 18 tokens and whose replies, on
/// both graphs, have 1 to 27.
const USERS: [&str; 8] = [
    "h i",
    "a",
    "l k j i h g",
    "a b c d e f g h i j k l a b c",
    "k",
    "b b",
    "e f g h i j k l",
    "j k l a",
];

fn fixture(dir: &str, name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(dir)
        .join(name)
}

fn model_dir(graph: &str, generation_config: Option<&serde_json::Value>) -> TempDir {
    let dir = tempfile::tempdir().expect("temp dir");
    std::fs::copy(
        fixture("text_generation", graph),
        dir.path().join("model.onnx"),
    )
    .expect("copy graph");
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
    let config = serde_json::json!({
        "architectures": ["LlamaForCausalLM"],
        "bos_token_id": 0,
        "eos_token_id": 2,
        "max_position_embeddings": 512,
        "vocab_size": 16
    });
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

/// A model directory around the tiny random Qwen3 export.
fn qwen3_dir() -> TempDir {
    let dir = tempfile::tempdir().expect("temp dir");
    for (from, file) in [
        ("text_generation/tiny_qwen3", "model.onnx"),
        ("text_generation/tiny_qwen3", "config.json"),
        ("text_generation", "tokenizer_config.json"),
        ("seq2seq", "tokenizer.json"),
    ] {
        std::fs::copy(fixture(from, file), dir.path().join(file)).expect("copy fixture");
    }
    dir
}

fn options(width: usize) -> TextGenerationOptions {
    TextGenerationOptions::new()
        .with_max_new_tokens(MAX_NEW_TOKENS)
        .with_max_batch(width)
}

fn load(dir: &TempDir, options: &TextGenerationOptions) -> PipelineKind {
    PipelineKind::TextGeneration(
        TextGenerationPipeline::with_options(dir.path(), &ExecutionProvider::Cpu, options)
            .expect("pipeline should load"),
    )
}

fn generation(kind: &mut PipelineKind) -> &mut TextGenerationPipeline {
    match kind {
        PipelineKind::TextGeneration(pipeline) => pipeline,
        _ => panic!("expected a text generation pipeline"),
    }
}

/// Prompt tokens of a user message: the fixture template renders
/// `c <user> d e`, and the words `a` .. `l` are tokens 4 .. 15.
fn prompt(user: &str) -> Vec<i64> {
    format!("c {user} d e")
        .split_whitespace()
        .map(|word| i64::from(word.as_bytes()[0] - b'a') + 4)
        .collect()
}

fn words(tokens: &[i64]) -> String {
    tokens
        .iter()
        .filter(|&&token| token >= 4)
        .map(|&token| char::from(b'a' + u8::try_from(token - 4).expect("word token")).to_string())
        .collect::<Vec<_>>()
        .join(" ")
}

/// What greedy decoding of the batch fixture graphs generates for a
/// sequence alone; mirrors `make_batched` in `make_fixtures.py`.
fn simulate(graph: &str, prompt: &[i64], max_new_tokens: usize) -> Vec<i64> {
    let mut tokens = prompt.to_vec();
    for _ in 0..max_new_tokens {
        let newest = i64::try_from(tokens.len()).expect("length") - 1;
        let mut total: i64 = tokens
            .iter()
            .zip(0_i64..)
            .map(|(token, position)| (token + 1) * (newest - position + 1))
            .sum();
        if graph == BY_POSITION_IDS {
            total += newest;
        }
        let next = if total % 13 == 0 { EOS } else { total % 12 + 4 };
        tokens.push(next);
        if next == EOS {
            break;
        }
    }
    tokens[prompt.len()..].to_vec()
}

/// Text and token counts of a finished request.
type Reply = (String, TokenUsage);

fn reply(execution: hephaestus_core::Execution) -> Reply {
    let text = match execution.output {
        PipelineOutput::TextGeneration(text) => text,
        _ => panic!("expected text generation output"),
    };
    (text, execution.usage.expect("token counts"))
}

/// The reply to `user` from the single-request loop.
fn alone(kind: &mut PipelineKind, user: &str) -> Reply {
    let prepared = kind.prepare(user.to_string()).expect("prepare");
    reply(
        kind.execute_with(prepared, &GenerationControl::default())
            .expect("execute"),
    )
}

fn request(
    kind: &PipelineKind,
    id: u64,
    user: &str,
    control: GenerationControl,
) -> GenerationRequest {
    GenerationRequest {
        id,
        prepared: kind.prepare(user.to_string()).expect("prepare"),
        control,
    }
}

fn prompt_tokens(prepared: &PreparedData) -> usize {
    prepared.token_count().expect("text prompt")
}

/// Collects the requests that left a batch, by ID.
#[derive(Default)]
struct Finished(Vec<(u64, Result<Reply, CoreError>)>);

impl Finished {
    fn record(&mut self, progress: BatchProgress) -> Vec<u64> {
        let mut ids = Vec::new();
        for done in progress.finished {
            ids.push(done.id);
            self.0.push((done.id, done.result.map(reply)));
        }
        ids
    }

    fn get(&self, id: u64) -> &Result<Reply, CoreError> {
        let mut found = self.0.iter().filter(|(done, _)| *done == id);
        let (_, result) = found
            .next()
            .unwrap_or_else(|| panic!("request {id} did not finish"));
        assert!(found.next().is_none(), "request {id} finished twice");
        result
    }

    fn reply(&self, id: u64) -> &Reply {
        self.get(id)
            .as_ref()
            .unwrap_or_else(|error| panic!("request {id} failed: {error}"))
    }
}

/// Step `batch` until nothing is running.
fn drain(
    pipeline: &mut TextGenerationPipeline,
    batch: &mut GenerationBatch,
    finished: &mut Finished,
) {
    while !batch.is_empty() {
        finished.record(pipeline.step(batch));
    }
}

#[test]
fn fixture_graphs_alone_generate_what_the_simulation_says() {
    for graph in GRAPHS {
        // Arrange
        let dir = model_dir(graph, None);
        let mut kind = load(
            &dir,
            &TextGenerationOptions::new().with_max_new_tokens(MAX_NEW_TOKENS),
        );
        assert_eq!(kind.generation_batch_width(), 1);
        let mut lengths = Vec::new();

        for user in USERS {
            // Act
            let (text, usage) = alone(&mut kind, user);

            // Assert
            let expected = simulate(graph, &prompt(user), MAX_NEW_TOKENS);
            assert_eq!(text, words(&expected), "{graph}: '{user}'");
            assert_eq!(usage.prompt_tokens, prompt(user).len());
            assert_eq!(usage.completion_tokens, expected.len());
            lengths.push(expected.len());
        }
        // The requests must differ in how long they run, or leaving
        // early and joining late would not be exercised.
        lengths.sort_unstable();
        lengths.dedup();
        assert!(lengths.len() >= 5, "{graph}: reply lengths {lengths:?}");
    }
}

#[test]
fn batched_replies_equal_solo_replies_for_every_width_and_order() {
    for graph in GRAPHS {
        // Arrange
        let dir = model_dir(graph, None);
        let mut solo = load(&dir, &options(1));
        let expected: Vec<Reply> = USERS.iter().map(|user| alone(&mut solo, user)).collect();
        let by_length = |descending: bool| {
            let mut order: Vec<usize> = (0..USERS.len()).collect();
            order.sort_by_key(|&index| prompt(USERS[index]).len());
            if descending {
                order.reverse();
            }
            order
        };
        let orders: Vec<Vec<usize>> = vec![
            (0..USERS.len()).collect(),
            (0..USERS.len()).rev().collect(),
            vec![3, 0, 6, 1, 7, 2, 5, 4],
            by_length(false),
            by_length(true),
        ];

        for width in [2, 3, 4, 8] {
            let mut kind = load(&dir, &options(width));
            assert_eq!(kind.generation_batch_width(), width);
            for order in &orders {
                let batch = order
                    .iter()
                    .map(|&index| kind.prepare(USERS[index].to_string()).expect("prepare"))
                    .collect();

                // Act
                let results = kind.execute_batch_with(batch, &[]);

                // Assert -- text and token counts, request by request.
                assert_eq!(results.len(), order.len());
                for (result, &index) in results.into_iter().zip(order) {
                    let got = reply(result.expect("generation should succeed"));
                    assert_eq!(
                        got, expected[index],
                        "{graph}, width {width}, order {order:?}: '{}'",
                        USERS[index]
                    );
                }
            }
        }
    }
}

#[test]
fn width_one_is_the_single_request_loop() {
    // Arrange -- no width, width 1, and a width-1 pipeline driven through
    // the batch calls.
    let dir = model_dir(BY_COLUMN, None);
    let mut unset = load(
        &dir,
        &TextGenerationOptions::new().with_max_new_tokens(MAX_NEW_TOKENS),
    );
    let mut one = load(&dir, &options(1));
    let prepare_all = |kind: &PipelineKind| -> Vec<PreparedData> {
        USERS
            .iter()
            .map(|user| kind.prepare((*user).to_string()).expect("prepare"))
            .collect()
    };

    // Act
    let from_unset: Vec<Reply> = unset
        .execute_batch_with(prepare_all(&unset), &[])
        .into_iter()
        .map(|result| reply(result.expect("unset width")))
        .collect();
    let from_one: Vec<Reply> = one
        .execute_batch_with(prepare_all(&one), &[])
        .into_iter()
        .map(|result| reply(result.expect("width one")))
        .collect();
    let mut driven = Finished::default();
    let mut batch = GenerationBatch::new();
    for (id, user) in (0_u64..).zip(USERS) {
        let next = request(&one, id, user, GenerationControl::default());
        let pipeline = generation(&mut one);
        assert!(pipeline.admits(&batch, &[], prompt_tokens(&next.prepared)));
        driven.record(pipeline.join(&mut batch, vec![next]));
        // Nothing else gets in while one request runs.
        assert!(batch.is_empty() || !pipeline.admits(&batch, &[], 4));
        assert!(batch.len() <= 1);
        drain(pipeline, &mut batch, &mut driven);
    }

    // Assert
    for (id, user) in (0_u64..).zip(USERS) {
        let expected = alone(&mut unset, user);
        let index = usize::try_from(id).expect("index");
        assert_eq!(from_unset[index], expected, "unset: '{user}'");
        assert_eq!(from_one[index], expected, "width one: '{user}'");
        assert_eq!(driven.reply(id), &expected, "driven: '{user}'");
    }
}

#[test]
fn request_joining_a_running_batch_gets_its_solo_reply_and_changes_no_other() {
    for graph in GRAPHS {
        // Arrange -- "b b" (5 prompt tokens) runs long on both graphs.
        let dir = model_dir(graph, None);
        let mut solo = load(&dir, &options(1));
        let mut kind = load(&dir, &options(4));
        let mut batch = GenerationBatch::new();
        let mut finished = Finished::default();
        let first = request(&kind, 0, "b b", GenerationControl::default());
        let second = request(&kind, 1, "a", GenerationControl::default());
        let long = request(&kind, 2, "e f g h i j k l", GenerationControl::default());
        let third = request(&kind, 3, "h i", GenerationControl::default());
        let long_prompt = prompt_tokens(&long.prepared);
        assert_eq!(long_prompt, 11);
        let pipeline = generation(&mut kind);

        // Act -- one request starts alone; others join between its steps.
        finished.record(pipeline.join(&mut batch, vec![first]));
        assert_eq!((batch.len(), batch.columns()), (1, 5));
        for _ in 0..2 {
            finished.record(pipeline.step(&mut batch));
        }
        // A 4-token prompt joins a batch 7 columns wide, 3 of them padding.
        assert!(pipeline.admits(&batch, &[], prompt_tokens(&second.prepared)));
        let progress = pipeline.join(&mut batch, vec![second]);
        assert_eq!(progress.rows, 1, "only the new prompt is read");
        finished.record(progress);
        assert_eq!((batch.len(), batch.columns()), (2, 7));
        // An 11-token prompt has to wait until the batch is 11 wide.
        let mut waited = 0;
        while !pipeline.admits(&batch, &[], long_prompt) {
            assert!(!batch.is_empty() && batch.columns() < long_prompt);
            finished.record(pipeline.step(&mut batch));
            waited += 1;
        }
        assert_eq!((waited, batch.columns()), (4, 11));
        finished.record(pipeline.join(&mut batch, vec![long, third]));
        drain(pipeline, &mut batch, &mut finished);

        // Assert
        for (id, user) in [(0, "b b"), (1, "a"), (2, "e f g h i j k l"), (3, "h i")] {
            assert_eq!(
                finished.reply(id),
                &alone(&mut solo, user),
                "{graph}: '{user}'"
            );
        }
        assert_eq!(
            batch.columns(),
            0,
            "an emptied batch starts again from nothing"
        );
    }
}

#[test]
fn request_that_ends_early_leaves_at_once_and_the_rest_go_on() {
    for graph in GRAPHS {
        // Arrange -- replies of very different lengths, started together.
        let dir = model_dir(graph, None);
        let mut solo = load(&dir, &options(1));
        let mut kind = load(&dir, &options(8));
        let expected: Vec<Reply> = USERS.iter().map(|user| alone(&mut solo, user)).collect();
        let requests: Vec<GenerationRequest> = (0_u64..)
            .zip(USERS)
            .map(|(id, user)| request(&kind, id, user, GenerationControl::default()))
            .collect();
        let pipeline = generation(&mut kind);
        let mut batch = GenerationBatch::new();
        let mut finished = Finished::default();
        let mut left_after: Vec<(u64, usize)> = Vec::new();

        // Act -- the first token comes from `join`, one more per step.
        let mut generated = 1;
        let progress = pipeline.join(&mut batch, requests);
        assert_eq!(progress.rows, USERS.len());
        for id in finished.record(progress) {
            left_after.push((id, generated));
        }
        while !batch.is_empty() {
            let running = batch.len();
            let progress = pipeline.step(&mut batch);
            generated += 1;
            assert_eq!(progress.rows, running);
            let left = finished.record(progress);
            assert_eq!(
                batch.len(),
                running - left.len(),
                "{graph}: rows leave in their step"
            );
            left_after.extend(left.into_iter().map(|id| (id, generated)));
        }

        // Assert -- each left in the step that produced its last token.
        assert_eq!(left_after.len(), USERS.len());
        for (id, tokens) in left_after {
            let index = usize::try_from(id).expect("index");
            assert_eq!(
                finished.reply(id),
                &expected[index],
                "{graph}: '{}'",
                USERS[index]
            );
            assert_eq!(
                tokens, expected[index].1.completion_tokens,
                "{graph}: '{}'",
                USERS[index]
            );
        }
    }
}

#[test]
fn cancelled_and_timed_out_requests_leave_alone_and_free_their_place() {
    for graph in GRAPHS {
        // Arrange -- a full batch of three; one caller goes away and one
        // deadline passes while it runs. A fourth request is waiting.
        let dir = model_dir(graph, None);
        let mut solo = load(&dir, &options(1));
        let mut kind = load(&dir, &options(3));
        let gone = Arc::new(AtomicBool::new(false));
        let flag = Arc::clone(&gone);
        let cancellable =
            GenerationControl::new().with_cancel_check(move || flag.load(Ordering::SeqCst));
        let deadline = Instant::now() + Duration::from_millis(400);
        // Requests that run for at least 9 tokens on the graph in hand.
        let (stays, cancelled, timed_out) = if graph == BY_COLUMN {
            ("b b", "a", "l k j i h g")
        } else {
            ("b b", "k", "a b c d e f g h i j k l a b c")
        };
        let requests = vec![
            request(&kind, 0, stays, GenerationControl::default()),
            request(&kind, 1, cancelled, cancellable),
            request(
                &kind,
                2,
                timed_out,
                GenerationControl::new().with_deadline(deadline),
            ),
        ];
        let waiting = request(&kind, 3, "h i", GenerationControl::default());
        let waiting_prompt = prompt_tokens(&waiting.prepared);
        let pipeline = generation(&mut kind);
        let mut batch = GenerationBatch::new();
        let mut finished = Finished::default();

        // Act
        finished.record(pipeline.join(&mut batch, requests));
        finished.record(pipeline.step(&mut batch));
        assert_eq!(batch.len(), 3);
        assert!(
            !pipeline.admits(&batch, &[], waiting_prompt),
            "the batch is full"
        );
        gone.store(true, Ordering::SeqCst);
        let left = finished.record(pipeline.step(&mut batch));
        assert_eq!(left, vec![1], "{graph}: only the cancelled request leaves");
        assert_eq!(batch.len(), 2);
        std::thread::sleep(deadline.saturating_duration_since(Instant::now()));
        let left = finished.record(pipeline.step(&mut batch));
        assert_eq!(left, vec![2], "{graph}: only the timed-out request leaves");
        assert_eq!(batch.len(), 1);
        // The places are free again: the waiting request joins.
        assert!(pipeline.admits(&batch, &[], waiting_prompt));
        finished.record(pipeline.join(&mut batch, vec![waiting]));
        drain(pipeline, &mut batch, &mut finished);

        // Assert
        let cancelled_error = finished.get(1).as_ref().expect_err("cancelled");
        assert!(
            matches!(cancelled_error, CoreError::Cancelled(_)),
            "{cancelled_error:?}"
        );
        assert!(
            cancelled_error.to_string().contains("after 2 tokens"),
            "{cancelled_error}"
        );
        let timeout_error = finished.get(2).as_ref().expect_err("timed out");
        assert!(
            matches!(timeout_error, CoreError::Timeout(_)),
            "{timeout_error:?}"
        );
        assert!(
            timeout_error.to_string().contains("after 3 tokens"),
            "{timeout_error}"
        );
        assert_eq!(
            finished.reply(0),
            &alone(&mut solo, stays),
            "{graph}: the one that stayed"
        );
        assert_eq!(
            finished.reply(3),
            &alone(&mut solo, "h i"),
            "{graph}: the one that joined"
        );
    }
}

#[test]
fn request_that_is_already_cancelled_or_ends_with_its_first_token_never_enters() {
    // Arrange -- "j k l a" answers with the end token at once.
    let dir = model_dir(BY_COLUMN, None);
    let mut solo = load(&dir, &options(1));
    let mut kind = load(&dir, &options(4));
    let requests = vec![
        request(&kind, 0, "j k l a", GenerationControl::default()),
        request(
            &kind,
            1,
            "a",
            GenerationControl::new().with_cancel_check(|| true),
        ),
        request(&kind, 2, "b b", GenerationControl::default()),
        request(
            &kind,
            3,
            "a",
            GenerationControl::new().with_deadline(Instant::now()),
        ),
    ];
    let pipeline = generation(&mut kind);
    let mut batch = GenerationBatch::new();
    let mut finished = Finished::default();

    // Act
    let progress = pipeline.join(&mut batch, requests);
    let rows = progress.rows;
    let left = finished.record(progress);
    let running = batch.len();
    drain(pipeline, &mut batch, &mut finished);

    // Assert -- two prompts were read; one of them stayed.
    assert_eq!(rows, 2);
    assert_eq!(left, vec![1, 3, 0]);
    assert_eq!(running, 1);
    let ended = alone(&mut solo, "j k l a");
    assert_eq!(ended.1.completion_tokens, 1);
    assert_eq!(finished.reply(0), &ended);
    assert!(matches!(finished.get(1), Err(CoreError::Cancelled(_))));
    assert!(matches!(finished.get(3), Err(CoreError::Timeout(_))));
    assert_eq!(finished.reply(2), &alone(&mut solo, "b b"));
}

#[test]
fn batch_never_holds_more_requests_than_its_width() {
    // Arrange -- more requests than places, admitted as places free up.
    let dir = model_dir(BY_COLUMN, None);
    let width = 3;
    let mut solo = load(&dir, &options(1));
    let mut kind = load(&dir, &options(width));
    let users: Vec<&str> = USERS.iter().chain(&USERS).copied().collect();
    let mut waiting: std::collections::VecDeque<GenerationRequest> = (0_u64..)
        .zip(&users)
        .map(|(id, user)| request(&kind, id, user, GenerationControl::default()))
        .collect();
    let pipeline = generation(&mut kind);
    let mut batch = GenerationBatch::new();
    let mut finished = Finished::default();
    let mut widest = 0;
    let mut joined_running_batch = 0;

    // Act
    while !(waiting.is_empty() && batch.is_empty()) {
        let mut joining = Vec::new();
        let mut prompts = Vec::new();
        while let Some(next) = waiting.front() {
            let tokens = prompt_tokens(&next.prepared);
            if !pipeline.admits(&batch, &prompts, tokens) {
                break;
            }
            prompts.push(tokens);
            joining.extend(waiting.pop_front());
        }
        if !joining.is_empty() {
            joined_running_batch += usize::from(!batch.is_empty());
            finished.record(pipeline.join(&mut batch, joining));
        }
        widest = widest.max(batch.len());
        assert!(batch.len() <= width);
        if !batch.is_empty() {
            let progress = pipeline.step(&mut batch);
            assert!(progress.rows <= width);
            finished.record(progress);
        }
    }

    // Assert
    assert_eq!(widest, width, "the batch should fill up");
    assert!(
        joined_running_batch > 0,
        "requests should join a running batch"
    );
    for (id, user) in (0_u64..).zip(&users) {
        assert_eq!(finished.reply(id), &alone(&mut solo, user), "'{user}'");
    }
}

#[test]
fn token_limit_and_repetition_rules_apply_to_each_row() {
    // Arrange -- a penalty and an n-gram rule, which read each row's own
    // tokens, and a limit that cuts most replies short.
    let generation_config =
        serde_json::json!({ "repetition_penalty": 1.3, "no_repeat_ngram_size": 2 });
    let limited = |width: usize| {
        TextGenerationOptions::new()
            .with_max_new_tokens(6)
            .with_max_batch(width)
    };
    for graph in GRAPHS {
        let dir = model_dir(graph, Some(&generation_config));
        let plain_dir = model_dir(graph, None);
        let mut solo = load(&dir, &limited(1));
        let mut plain = load(&plain_dir, &limited(1));
        let mut kind = load(&dir, &limited(4));
        let batch = USERS
            .iter()
            .map(|user| kind.prepare((*user).to_string()).expect("prepare"))
            .collect();

        // Act
        let results = kind.execute_batch_with(batch, &[]);

        // Assert
        let mut at_limit = 0;
        let mut changed_by_rules = 0;
        for (result, user) in results.into_iter().zip(USERS) {
            let got = reply(result.expect("generation should succeed"));
            let expected = alone(&mut solo, user);
            assert_eq!(got, expected, "{graph}: '{user}'");
            assert!(got.1.completion_tokens <= 6);
            at_limit += usize::from(got.1.completion_tokens == 6);
            changed_by_rules += usize::from(alone(&mut plain, user) != expected);
        }
        assert!(at_limit >= 3, "{graph}: the limit should cut replies short");
        assert!(
            changed_by_rules >= 3,
            "{graph}: the rules should change replies"
        );
    }
}

#[test]
fn prompt_is_not_admitted_where_most_of_its_row_would_be_padding() {
    // Arrange -- at most 3 new tokens: a 4-token prompt may be padded by
    // 4 + 3 columns, so it fits a batch 11 wide and not one 12 wide.
    let dir = model_dir(BY_COLUMN, None);
    let limited = TextGenerationOptions::new()
        .with_max_new_tokens(3)
        .with_max_batch(4);
    let mut kind = load(&dir, &limited);
    let eleven_wide = request(&kind, 0, "e f g h i j k l", GenerationControl::default());
    let too_long = request(
        &kind,
        1,
        "a b c d e f g h i j k l a b c",
        GenerationControl::default(),
    );
    assert_eq!(prompt_tokens(&eleven_wide.prepared), 11);
    let pipeline = generation(&mut kind);
    let mut batch = GenerationBatch::new();
    let mut finished = Finished::default();

    // Act / Assert -- as a group that starts together...
    assert!(pipeline.admits(&batch, &[11], 4));
    assert!(!pipeline.admits(&batch, &[12], 4));
    assert!(!pipeline.admits(&batch, &[4], 12));
    // ...and against a batch that is running.
    finished.record(pipeline.join(&mut batch, vec![eleven_wide]));
    assert_eq!((batch.len(), batch.columns()), (1, 11));
    assert!(pipeline.admits(&batch, &[], 4));
    assert!(
        !pipeline.admits(&batch, &[], 12),
        "longer than the batch is wide"
    );
    finished.record(pipeline.step(&mut batch));
    assert_eq!((batch.len(), batch.columns()), (1, 12));
    assert!(!pipeline.admits(&batch, &[], 4), "more padding than row");
    assert!(pipeline.admits(&batch, &[], 12));
    // A request that does not fit and is joined anyway fails alone.
    finished.record(pipeline.join(&mut batch, vec![too_long]));
    assert!(matches!(finished.get(1), Err(CoreError::Inference(_))));
    assert_eq!(batch.len(), 1);
    drain(pipeline, &mut batch, &mut finished);
    assert_eq!(finished.reply(0).1.completion_tokens, 3);
}

#[test]
fn batch_wider_than_the_input_cap_takes_no_more_requests() {
    // Arrange -- prompts of at most 16 tokens; "b b" runs for 27 steps.
    let dir = model_dir(BY_COLUMN, None);
    let capped = options(4).with_max_input_tokens(16);
    let mut solo = load(&dir, &options(1));
    let mut kind = load(&dir, &capped);
    let first = request(&kind, 0, "b b", GenerationControl::default());
    let pipeline = generation(&mut kind);
    let mut batch = GenerationBatch::new();
    let mut finished = Finished::default();

    // Act
    finished.record(pipeline.join(&mut batch, vec![first]));
    while batch.columns() < 16 {
        assert!(
            !batch.is_empty(),
            "the first request should still be running"
        );
        assert!(pipeline.admits(&batch, &[], 5));
        finished.record(pipeline.step(&mut batch));
    }
    assert!(pipeline.admits(&batch, &[], 16));
    finished.record(pipeline.step(&mut batch));

    // Assert -- 17 columns: no row may start that far out.
    assert!(!batch.is_empty());
    assert!(!pipeline.admits(&batch, &[], 16));
    assert!(!pipeline.admits(&batch, &[], 5));
    drain(pipeline, &mut batch, &mut finished);
    assert!(
        pipeline.admits(&batch, &[], 5),
        "an empty batch takes anyone"
    );
    assert_eq!(finished.reply(0), &alone(&mut solo, "b b"));
}

#[test]
fn graph_that_cannot_batch_keeps_generating_one_request_at_a_time() {
    // Arrange -- no cache in the graph, and the cache switched off.
    let no_cache = model_dir("model_no_cache.onnx", None);
    let cached = model_dir(BY_COLUMN, None);
    let mut without_cache = load(&no_cache, &options(4));
    let mut cache_off = load(&cached, &options(4).with_kv_cache(false));
    let mut solo = load(&cached, &options(1));

    // Act / Assert
    assert_eq!(without_cache.generation_batch_width(), 1);
    assert_eq!(cache_off.generation_batch_width(), 1);
    let batch = ["h i", "a"]
        .iter()
        .map(|user| cache_off.prepare((*user).to_string()).expect("prepare"))
        .collect();
    let results = cache_off.execute_batch_with(batch, &[]);
    for (result, user) in results.into_iter().zip(["h i", "a"]) {
        assert_eq!(reply(result.expect("generation")), alone(&mut solo, user));
    }
    let prepared = without_cache.prepare("h i".to_string()).expect("prepare");
    assert!(without_cache.execute(prepared).is_ok());
}

#[test]
fn batch_width_out_of_range_fails_at_startup() {
    let dir = model_dir(BY_COLUMN, None);
    for width in [0, hephaestus_core::TEXT_GENERATION_MAX_BATCH_LIMIT + 1] {
        // Act
        let result = TextGenerationPipeline::with_options(
            dir.path(),
            &ExecutionProvider::Cpu,
            &options(width),
        );

        // Assert
        let error = result
            .err()
            .unwrap_or_else(|| panic!("width {width} should be refused"));
        assert!(matches!(error, CoreError::Config(_)), "{error:?}");
        assert!(error.to_string().contains("max batch"), "{error}");
    }
}

#[test]
fn cancelled_item_of_a_batch_does_not_stop_its_neighbours() {
    // Arrange -- the dynamic batcher's call, on a pipeline that batches.
    let dir = model_dir(BY_COLUMN, None);
    let mut solo = load(&dir, &options(1));
    let mut kind = load(&dir, &options(3));
    let batch = ["b b", "a", "h i"]
        .iter()
        .map(|user| kind.prepare((*user).to_string()).expect("prepare"))
        .collect();
    let asked = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let counter = Arc::clone(&asked);
    let controls = [
        GenerationControl::default(),
        GenerationControl::new()
            .with_cancel_check(move || counter.fetch_add(1, Ordering::SeqCst) >= 4),
        GenerationControl::default(),
    ];

    // Act
    let results = kind.execute_batch_with(batch, &controls);

    // Assert
    assert_eq!(results.len(), 3);
    let mut results = results.into_iter();
    assert_eq!(
        reply(results.next().expect("first").expect("first")),
        alone(&mut solo, "b b")
    );
    let error = results.next().expect("second").err().expect("cancelled");
    assert!(matches!(error, CoreError::Cancelled(_)), "{error:?}");
    assert!(error.to_string().contains("after 4 tokens"), "{error}");
    assert_eq!(
        reply(results.next().expect("third").expect("third")),
        alone(&mut solo, "h i")
    );
}

/// User messages for the tiny Qwen3: prompts of 4 to 31 tokens.
const QWEN3_USERS: [&str; 8] = [
    "b b",
    "f g a",
    "l k j i h g",
    "a b c d e f g h i j k l a b c",
    "h i",
    "f g h",
    "k k k k k k k k k k k k k k k k k k k k k k k k k k k k",
    "d",
];

fn qwen3_options(width: usize) -> TextGenerationOptions {
    TextGenerationOptions::new()
        .with_max_new_tokens(48)
        .with_max_batch(width)
}

#[test]
fn real_transformer_gives_batched_requests_their_solo_tokens() {
    // Arrange
    let dir = qwen3_dir();
    let mut solo = load(&dir, &qwen3_options(1));
    let expected: Vec<Reply> = QWEN3_USERS
        .iter()
        .map(|user| alone(&mut solo, user))
        .collect();
    let mut distinct: Vec<&str> = expected.iter().map(|(text, _)| text.as_str()).collect();
    distinct.sort_unstable();
    distinct.dedup();
    assert!(
        distinct.len() >= 6,
        "replies should depend on the prompt: {distinct:?}"
    );
    let mut by_length: Vec<usize> = (0..QWEN3_USERS.len()).collect();
    by_length.sort_by_key(|&index| prompt(QWEN3_USERS[index]).len());
    let orders: [Vec<usize>; 4] = [
        (0..QWEN3_USERS.len()).collect(),
        (0..QWEN3_USERS.len()).rev().collect(),
        by_length.clone(),
        by_length.into_iter().rev().collect(),
    ];

    for width in [2, 4, 8] {
        let mut kind = load(&dir, &qwen3_options(width));
        assert_eq!(
            kind.generation_batch_width(),
            width,
            "no position_ids, but a cache and a mask"
        );
        for order in &orders {
            let batch = order
                .iter()
                .map(|&index| {
                    kind.prepare(QWEN3_USERS[index].to_string())
                        .expect("prepare")
                })
                .collect();

            // Act
            let results = kind.execute_batch_with(batch, &[]);

            // Assert
            for (result, &index) in results.into_iter().zip(order) {
                let got = reply(result.expect("generation should succeed"));
                assert_eq!(
                    got, expected[index],
                    "width {width}, order {order:?}: '{}'",
                    QWEN3_USERS[index]
                );
            }
        }
    }
}

#[test]
fn real_transformer_requests_join_and_leave_a_running_batch() {
    // Arrange -- one request joins every three steps, as soon as it fits;
    // one caller goes away after ten of its tokens.
    let dir = qwen3_dir();
    let mut solo = load(&dir, &qwen3_options(1));
    let mut kind = load(&dir, &qwen3_options(4));
    let asked = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let counter = Arc::clone(&asked);
    let leaves_early = GenerationControl::new()
        .with_cancel_check(move || counter.fetch_add(1, Ordering::SeqCst) >= 10);
    let cancelled: u64 = 2;
    let mut waiting: std::collections::VecDeque<GenerationRequest> = (0_u64..)
        .zip(QWEN3_USERS)
        .map(|(id, user)| {
            let control = if id == cancelled {
                leaves_early.clone()
            } else {
                GenerationControl::default()
            };
            request(&kind, id, user, control)
        })
        .collect();
    let pipeline = generation(&mut kind);
    let mut batch = GenerationBatch::new();
    let mut finished = Finished::default();
    let mut joined_running_batch = 0;
    let mut steps_since_join = 3;

    // Act
    while !(waiting.is_empty() && batch.is_empty()) {
        let due = batch.is_empty() || steps_since_join >= 3;
        let fits = waiting
            .front()
            .is_some_and(|next| pipeline.admits(&batch, &[], prompt_tokens(&next.prepared)));
        if due && fits {
            joined_running_batch += usize::from(!batch.is_empty());
            let next = waiting.pop_front().expect("a waiting request");
            finished.record(pipeline.join(&mut batch, vec![next]));
            steps_since_join = 0;
        }
        if !batch.is_empty() {
            finished.record(pipeline.step(&mut batch));
            steps_since_join += 1;
        }
    }

    // Assert
    assert!(
        joined_running_batch >= 4,
        "{joined_running_batch} requests joined a running batch"
    );
    for (id, user) in (0_u64..).zip(QWEN3_USERS) {
        if id == cancelled {
            let error = finished.get(id).as_ref().expect_err("cancelled");
            assert!(matches!(error, CoreError::Cancelled(_)), "{error:?}");
            assert!(error.to_string().contains("after 10 tokens"), "{error}");
        } else {
            assert_eq!(finished.reply(id), &alone(&mut solo, user), "'{user}'");
        }
    }
}
