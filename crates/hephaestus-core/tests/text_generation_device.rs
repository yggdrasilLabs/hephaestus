//! Half-precision graphs, and the key/value cache kept on the device.
//!
//! Two things are checked here, both on the CPU provider:
//!
//! - **float16 graphs** (`tiny_qwen3/model_fp16*.onnx`, see
//!   `make_tiny_qwen3_fp16.py`): they load, generate, and give batched
//!   requests the tokens they get alone, whichever of the cache and the
//!   logits is `float16`;
//! - **the device cache** (`TextGenerationOptions::with_device_cache`):
//!   forward passes through an I/O binding, the cache fed back where the
//!   session left it, rows moved by the cache-rows graph. With the CPU
//!   provider the device is host memory, so the whole path runs here and
//!   must pick the tokens of the host path, for every graph, at every
//!   width, with requests joining and leaving.
//!
//! What this cannot show is that the same path keeps the cache in GPU
//! memory under the CUDA provider; that needs a GPU (see
//! `docs/gpu-verification.md`).

use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use hephaestus_core::{
    BatchProgress, CoreError, ExecutionProvider, GenerationBatch, GenerationControl,
    GenerationRequest, PipelineKind, PipelineOutput, TextGenerationOptions, TextGenerationPipeline,
    TokenUsage,
};
use tempfile::TempDir;

/// The tiny Qwen3 in full precision.
const FLOAT32: &str = "tiny_qwen3/model.onnx";
/// float16 cache and logits.
const FLOAT16: &str = "tiny_qwen3/model_fp16.onnx";
/// float16 cache, float32 logits.
const FLOAT16_CACHE: &str = "tiny_qwen3/model_fp16_logits32.onnx";
/// float32 cache, float16 logits.
const FLOAT16_LOGITS: &str = "tiny_qwen3/model_logits16.onnx";
const HALF_PRECISION: [&str; 3] = [FLOAT16, FLOAT16_CACHE, FLOAT16_LOGITS];
/// The whole-number graphs of `make_fixtures.py`: positions by column,
/// and through `position_ids`. A wrong cache row changes their tokens.
const WHOLE_NUMBERS: [&str; 2] = ["model_batch.onnx", "model_batch_positions.onnx"];

/// User messages: prompts of 4 to 31 tokens.
const USERS: [&str; 8] = [
    "b b",
    "f g a",
    "l k j i h g",
    "a b c d e f g h i j k l a b c",
    "h i",
    "f g h",
    "k k k k k k k k k k k k k k k k k k k k k k k k k k k k",
    "d",
];

/// The user messages whose batched tokens are compared with their solo
/// tokens on `graph`.
///
/// On the float32 and whole-number graphs that is all of them. In half
/// precision the logits of a padded row differ from those of the same
/// request alone by up to about 1e-2 (float16 has 11 bits of mantissa),
/// and `USERS[2]` reaches, seventeen or forty-two tokens in depending on
/// the graph, two candidates closer together than that (margins of 0
/// and 8e-5), so its batched reply may part from its solo one there.
/// That is rounding, not a wrong cache row: the unit test
/// `half_precision_batches_part_from_solo_only_at_near_ties` in
/// `src/pipeline/provider_tests.rs` checks every prompt, with the margin
/// at each point where tokens part. Here that one prompt still runs in every
/// batch; only its reply is not compared with the solo one.
fn compared_with_solo(graph: &str) -> Vec<usize> {
    (0..USERS.len())
        .filter(|&index| !(HALF_PRECISION.contains(&graph) && index == 2))
        .collect()
}

fn fixture(path: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(path)
}

/// A model directory around one graph of `tests/fixtures/text_generation`.
fn model_dir(graph: &str) -> TempDir {
    let dir = tempfile::tempdir().expect("temp dir");
    let copy = |from: &str, to: &str| {
        std::fs::copy(fixture(from), dir.path().join(to)).expect("copy fixture");
    };
    copy(&format!("text_generation/{graph}"), "model.onnx");
    copy(
        "text_generation/tokenizer_config.json",
        "tokenizer_config.json",
    );
    copy("seq2seq/tokenizer.json", "tokenizer.json");
    if graph.starts_with("tiny_qwen3/") {
        copy("text_generation/tiny_qwen3/config.json", "config.json");
    } else {
        let config = serde_json::json!({
            "architectures": ["LlamaForCausalLM"],
            "bos_token_id": 0,
            "eos_token_id": 2,
            "max_position_embeddings": 512,
            "vocab_size": 16
        });
        std::fs::write(dir.path().join("config.json"), config.to_string()).expect("config");
    }
    dir
}

fn options(width: usize, device_cache: bool) -> TextGenerationOptions {
    TextGenerationOptions::new()
        .with_max_new_tokens(48)
        .with_max_batch(width)
        .with_device_cache(device_cache)
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

fn everyone_alone(kind: &mut PipelineKind) -> Vec<Reply> {
    USERS.iter().map(|user| alone(kind, user)).collect()
}

fn distinct(replies: &[Reply]) -> usize {
    let mut texts: Vec<&str> = replies.iter().map(|(text, _)| text.as_str()).collect();
    texts.sort_unstable();
    texts.dedup();
    texts.len()
}

/// Every user message through `execute_batch_with`, in `order`; replies
/// by user index.
fn together(kind: &mut PipelineKind, order: &[usize]) -> Vec<Reply> {
    let batch = order
        .iter()
        .map(|&index| kind.prepare(USERS[index].to_string()).expect("prepare"))
        .collect();
    let mut replies: Vec<Option<Reply>> = USERS.iter().map(|_| None).collect();
    for (result, &index) in kind.execute_batch_with(batch, &[]).into_iter().zip(order) {
        replies[index] = Some(reply(result.expect("generation should succeed")));
    }
    replies
        .into_iter()
        .map(|reply| reply.expect("a reply"))
        .collect()
}

fn orders() -> [Vec<usize>; 3] {
    let mut by_length: Vec<usize> = (0..USERS.len()).collect();
    by_length.sort_by_key(|&index| USERS[index].len());
    [
        (0..USERS.len()).collect(),
        (0..USERS.len()).rev().collect(),
        by_length,
    ]
}

/// What became of each request of [`join_and_leave`]: its reply, or the
/// error it left with.
type Outcome = Result<Reply, String>;

/// Requests joining a running batch one by one, three steps apart or as
/// soon as they fit, while request 2 is cancelled after ten tokens: rows
/// join next to running ones (with padding to zero), leave from the
/// middle, and leave last.
///
/// Returns the outcomes by request and how many joined a running batch.
fn join_and_leave(kind: &mut PipelineKind, width: usize) -> (Vec<Outcome>, usize) {
    let asked = Arc::new(AtomicUsize::new(0));
    let counter = Arc::clone(&asked);
    let leaves_early = GenerationControl::new()
        .with_cancel_check(move || counter.fetch_add(1, Ordering::SeqCst) >= 10);
    let mut waiting: VecDeque<GenerationRequest> = (0_u64..)
        .zip(USERS)
        .map(|(id, user)| GenerationRequest {
            id,
            prepared: kind.prepare(user.to_string()).expect("prepare"),
            control: if id == 2 {
                leaves_early.clone()
            } else {
                GenerationControl::default()
            },
        })
        .collect();
    let pipeline = generation(kind);
    assert_eq!(pipeline.max_batch(), width);
    let mut batch = GenerationBatch::new();
    let mut outcomes: Vec<Option<Outcome>> = USERS.iter().map(|_| None).collect();
    let mut record = |progress: BatchProgress| {
        for done in progress.finished {
            let index = usize::try_from(done.id).expect("index");
            assert!(outcomes[index].is_none(), "request {index} finished twice");
            outcomes[index] = Some(done.result.map(reply).map_err(|error| error.to_string()));
        }
    };
    let mut joined_running_batch = 0;
    let mut steps_since_join = 3;
    while !(waiting.is_empty() && batch.is_empty()) {
        let due = batch.is_empty() || steps_since_join >= 3;
        let fits = waiting.front().is_some_and(|next| {
            let tokens = next.prepared.token_count().expect("text prompt");
            pipeline.admits(&batch, &[], tokens)
        });
        if due && fits {
            joined_running_batch += usize::from(!batch.is_empty());
            let next = waiting.pop_front().expect("a waiting request");
            record(pipeline.join(&mut batch, vec![next]));
            steps_since_join = 0;
        }
        if !batch.is_empty() {
            record(pipeline.step(&mut batch));
            steps_since_join += 1;
        }
    }
    let outcomes = outcomes
        .into_iter()
        .map(|outcome| outcome.expect("every request finished"))
        .collect();
    (outcomes, joined_running_batch)
}

#[test]
fn half_precision_graphs_load_and_generate() {
    // Arrange
    let full = everyone_alone(&mut load(&model_dir(FLOAT32), &options(1, false)));
    let mut replies = Vec::new();

    for graph in HALF_PRECISION {
        // Act
        let mut kind = load(&model_dir(graph), &options(1, false));
        let alone = everyone_alone(&mut kind);

        // Assert -- real replies that depend on the prompt.
        assert!(generation(&mut kind).uses_kv_cache(), "{graph}");
        assert!(distinct(&alone) >= 5, "{graph}: {alone:?}");
        for ((text, usage), user) in alone.iter().zip(USERS) {
            assert!(usage.completion_tokens >= 1, "{graph}: '{user}'");
            assert!(usage.completion_tokens <= 48, "{graph}: '{user}'");
            assert_eq!(
                text.is_empty(),
                usage.completion_tokens == 1,
                "{graph}: '{user}'"
            );
        }
        replies.push(alone);
    }

    // Half precision is the same model: most replies survive it.
    for (graph, half) in HALF_PRECISION.iter().zip(&replies) {
        let same = half.iter().zip(&full).filter(|(a, b)| a == b).count();
        assert!(same >= 4, "{graph}: only {same} of 8 replies as in float32");
    }
}

#[test]
fn half_precision_graph_without_cache_reuse_gives_the_same_tokens() {
    for graph in HALF_PRECISION {
        // Arrange -- the loop that re-reads everything, fed empty float16
        // or float32 cache tensors at every step.
        let dir = model_dir(graph);
        let mut cached = load(&dir, &options(1, false));
        let mut uncached = load(&dir, &options(1, false).with_kv_cache(false));
        assert!(!generation(&mut uncached).uses_kv_cache());

        // Act / Assert
        for user in &USERS[..4] {
            assert_eq!(
                alone(&mut uncached, user),
                alone(&mut cached, user),
                "{graph}: '{user}'"
            );
        }
    }
}

#[test]
fn half_precision_batches_give_requests_their_solo_tokens() {
    for graph in HALF_PRECISION {
        // Arrange
        let dir = model_dir(graph);
        let expected = everyone_alone(&mut load(&dir, &options(1, false)));

        for width in [2, 4, 8] {
            let mut kind = load(&dir, &options(width, false));
            assert_eq!(kind.generation_batch_width(), width, "{graph}");
            for order in orders() {
                // Act
                let replies = together(&mut kind, &order);

                // Assert
                for index in compared_with_solo(graph) {
                    assert_eq!(
                        replies[index], expected[index],
                        "{graph}, width {width}, order {order:?}: '{}'",
                        USERS[index]
                    );
                }
            }
        }
    }
}

#[test]
fn half_precision_requests_join_and_leave_a_running_batch() {
    for graph in HALF_PRECISION {
        // Arrange
        let dir = model_dir(graph);
        let expected = everyone_alone(&mut load(&dir, &options(1, false)));

        // Act
        let (outcomes, joined_running_batch) =
            join_and_leave(&mut load(&dir, &options(4, false)), 4);

        // Assert
        assert!(
            joined_running_batch >= 4,
            "{graph}: {joined_running_batch} joined"
        );
        let error = outcomes[2].as_ref().expect_err("cancelled");
        assert!(error.contains("after 10 tokens"), "{graph}: {error}");
        for index in compared_with_solo(graph) {
            assert_eq!(
                outcomes[index].as_ref(),
                Ok(&expected[index]),
                "{graph}: '{}'",
                USERS[index]
            );
        }
    }
}

#[test]
fn device_cache_is_off_on_the_cpu_unless_asked_for() {
    let dir = model_dir(FLOAT32);
    let default = TextGenerationOptions::new().with_max_new_tokens(8);

    let mut unset = load(&dir, &default);
    let mut asked = load(&dir, &default.clone().with_device_cache(true));

    assert!(!generation(&mut unset).keeps_cache_on_device());
    assert!(generation(&mut asked).keeps_cache_on_device());
}

#[test]
fn device_cache_needs_a_key_value_cache() {
    // Arrange -- a graph without cache tensors, and one told not to use
    // them.
    let without = model_dir("model_no_cache.onnx");
    let unused = model_dir(FLOAT32);
    let cases = [
        (&without, options(1, true)),
        (&unused, options(1, true).with_kv_cache(false)),
    ];

    for (dir, options) in cases {
        // Act
        let result =
            TextGenerationPipeline::with_options(dir.path(), &ExecutionProvider::Cpu, &options);

        // Assert
        let error = result.err().expect("should fail at startup");
        assert!(matches!(error, CoreError::Config(_)), "{error:?}");
        assert!(error.to_string().contains("key/value cache"), "{error}");
    }
}

#[test]
fn device_cache_picks_the_tokens_of_the_host_cache_for_one_request() {
    for graph in WHOLE_NUMBERS
        .into_iter()
        .chain([FLOAT32])
        .chain(HALF_PRECISION)
    {
        // Arrange -- width 1: the single-request loop, both ways.
        let dir = model_dir(graph);
        let mut host = load(&dir, &options(1, false));
        let mut device = load(&dir, &options(1, true));
        assert!(!generation(&mut host).keeps_cache_on_device());
        assert!(generation(&mut device).keeps_cache_on_device());

        // Act
        let expected = everyone_alone(&mut host);
        let replies = everyone_alone(&mut device);

        // Assert
        assert!(distinct(&expected) >= 5, "{graph}: {expected:?}");
        assert_eq!(replies, expected, "{graph}");
    }
}

#[test]
fn device_cache_picks_the_tokens_of_the_host_cache_in_a_batch() {
    for graph in WHOLE_NUMBERS
        .into_iter()
        .chain([FLOAT32])
        .chain(HALF_PRECISION)
    {
        let dir = model_dir(graph);
        let expected = everyone_alone(&mut load(&dir, &options(1, false)));

        for width in [2, 4, 8] {
            // Arrange
            let mut host = load(&dir, &options(width, false));
            let mut device = load(&dir, &options(width, true));
            assert_eq!(device.generation_batch_width(), width, "{graph}");

            for order in orders() {
                // Act
                let from_host = together(&mut host, &order);
                let from_device = together(&mut device, &order);

                // Assert -- exactly what the host cache gives, which is
                // what every request gets alone.
                assert_eq!(
                    from_device, from_host,
                    "{graph}, width {width}, order {order:?}"
                );
                for index in compared_with_solo(graph) {
                    assert_eq!(
                        from_device[index], expected[index],
                        "{graph}, width {width}, order {order:?}: '{}'",
                        USERS[index]
                    );
                }
            }
        }
    }
}

#[test]
fn device_cache_requests_join_and_leave_as_with_the_host_cache() {
    for graph in WHOLE_NUMBERS
        .into_iter()
        .chain([FLOAT32])
        .chain(HALF_PRECISION)
    {
        let dir = model_dir(graph);
        let expected = everyone_alone(&mut load(&dir, &options(1, false)));

        for width in [2, 4] {
            // Act
            let (from_host, joined_host) =
                join_and_leave(&mut load(&dir, &options(width, false)), width);
            let (from_device, joined_device) =
                join_and_leave(&mut load(&dir, &options(width, true)), width);

            // Assert -- the same batches formed, and the same outcomes.
            assert_eq!(joined_device, joined_host, "{graph}, width {width}");
            assert!(
                joined_device >= 2,
                "{graph}, width {width}: {joined_device} joined"
            );
            assert_eq!(from_device, from_host, "{graph}, width {width}");
            // Request 2 is cancelled after ten tokens, unless its reply
            // is shorter than that (the whole-number graphs).
            let cancelled = from_device[2].is_err();
            assert_eq!(cancelled, expected[2].1.completion_tokens > 10, "{graph}");
            for index in compared_with_solo(graph) {
                match &from_device[index] {
                    Err(error) => {
                        assert_eq!(index, 2, "{graph}: {error}");
                        assert!(error.contains("after 10 tokens"), "{graph}: {error}");
                    }
                    Ok(reply) => assert_eq!(
                        reply, &expected[index],
                        "{graph}, width {width}: '{}'",
                        USERS[index]
                    ),
                }
            }
        }
    }
}

#[test]
fn device_cache_warm_up_and_token_limit_work() {
    // Arrange -- a limit of one token ends a request in the pass that
    // reads its prompt, before any cache is fed back.
    let dir = model_dir(FLOAT16);
    let one = TextGenerationOptions::new()
        .with_max_new_tokens(1)
        .with_max_batch(4)
        .with_device_cache(true);
    let mut kind = load(&dir, &one);

    // Act
    generation(&mut kind).warm_up().expect("warm up");
    let replies = together(&mut kind, &[0, 1, 2, 3, 4, 5, 6, 7]);

    // Assert
    for (_, usage) in replies {
        assert_eq!(usage.completion_tokens, 1);
    }
}

#[test]
fn asking_for_cuda_without_a_usable_provider_fails_at_startup() {
    // Arrange -- this machine has no CUDA: either the binary was built
    // without the `cuda` feature, or it was and ONNX Runtime cannot set
    // the provider up. Neither may end in a session on the CPU.
    let dir = model_dir(FLOAT32);

    // Act
    let result = TextGenerationPipeline::with_options(
        dir.path(),
        &ExecutionProvider::Cuda,
        &TextGenerationOptions::new(),
    );

    // Assert
    let error = result.err().expect("cuda must not fall back to the CPU");
    let message = error.to_string();
    println!("{message}");
    assert!(
        matches!(error, CoreError::Config(_) | CoreError::ModelLoad(_)),
        "{error:?}"
    );
    assert!(message.contains("cuda"), "{message}");
    assert!(message.contains("fall"), "{message}");
    assert!(message.contains("EXECUTION_PROVIDER=cpu"), "{message}");
}
