//! Text generation compared across cache homes, float types and
//! execution providers, logit by logit.
//!
//! The integration tests compare replies. These tests look at the
//! next-token logits of every step, which only this crate can see, and so
//! can tell a wrong computation from a near tie that rounding decides the
//! other way:
//!
//! - the cache on the device against the cache in host memory, on the CPU
//!   provider: the same graph on the same tensors, so the logits must be
//!   equal bit for bit;
//! - batched against solo generation in half precision: tokens may part
//!   only where the two best candidates are as close as the logits of the
//!   two runs are to each other;
//! - a GPU against the CPU on a real model
//!   ([`gpu_generation_matches_cpu_on_a_real_model`], ignored: it needs a
//!   GPU and a model; `docs/gpu-verification.md` has the command).

use std::cell::RefCell;
use std::collections::{HashMap, VecDeque};
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::time::{Duration, Instant};

use tempfile::TempDir;

use crate::ep::ExecutionProvider;
use crate::error::CoreError;

use super::control::{Execution, GenerationControl};
use super::generation_batch::{BatchProgress, GenerationBatch, GenerationRequest};
use super::{
    Pipeline, PreparedData, PreparedInput, TextGenerationInput, TextGenerationOptions,
    TextGenerationPipeline,
};

/// A near tie is a step whose two best logits are no further apart than
/// this many times the largest difference seen between the two runs.
const NEAR_TIE_FACTOR: f32 = 4.0;

/// Logit differences above this are not rounding, whatever the margin:
/// logits of a language model span tens, and two correct runs of one
/// graph agree to a small fraction of one. Tokens cannot part unless the
/// logits differ by at least half the margin, so with every logit kept
/// this limit is what separates a near tie from a wrong computation.
const ROUNDING_LIMIT: f32 = 1.0;

/// What one step of a generation looked like.
struct Step {
    /// The most likely token, and the next most likely one.
    best: usize,
    runner_up: usize,
    best_logit: f32,
    runner_up_logit: f32,
    /// Every logit, for the steps that keep them.
    logits: Option<Vec<f32>>,
}

impl Step {
    fn of(logits: &[f32], keep: bool) -> Self {
        let mut best = 0;
        let mut runner_up = usize::MAX;
        for (index, &logit) in logits.iter().enumerate().skip(1) {
            if logit > logits[best] {
                runner_up = best;
                best = index;
            } else if runner_up == usize::MAX || logit > logits[runner_up] {
                runner_up = index;
            }
        }
        Self {
            best,
            runner_up,
            best_logit: logits[best],
            runner_up_logit: logits.get(runner_up).copied().unwrap_or(f32::NEG_INFINITY),
            logits: keep.then(|| logits.to_vec()),
        }
    }

    /// How far ahead of the runner-up the best token was.
    fn margin(&self) -> f32 {
        self.best_logit - self.runner_up_logit
    }

    /// The largest difference between logits both steps hold: all of
    /// them when both kept them, else those of the two best tokens that
    /// the steps have in common.
    fn difference(&self, other: &Step) -> Option<f32> {
        if let (Some(a), Some(b)) = (&self.logits, &other.logits) {
            return a.iter().zip(b).map(|(x, y)| (x - y).abs()).reduce(f32::max);
        }
        let mine = [
            (self.best, self.best_logit),
            (self.runner_up, self.runner_up_logit),
        ];
        let theirs = [
            (other.best, other.best_logit),
            (other.runner_up, other.runner_up_logit),
        ];
        mine.iter()
            .flat_map(|(token, logit)| {
                theirs
                    .iter()
                    .filter(move |(other_token, _)| other_token == token)
                    .map(move |(_, other_logit)| (logit - other_logit).abs())
            })
            .reduce(f32::max)
    }
}

/// One generation, step by step.
#[derive(Default)]
struct Trace {
    steps: Vec<Step>,
    prompt_tokens: usize,
    /// Time to the first token: the pass that reads the prompt.
    prefill: Duration,
    /// Time from the first token to the last.
    decode: Duration,
}

/// How a run of a prompt compares with a reference run of it.
struct Comparison {
    /// First step at which the tokens differ.
    parted: Option<usize>,
    /// Margin of the reference at that step.
    margin: f32,
    /// Largest logit difference over the steps up to and including it
    /// (over every step when the tokens are equal).
    max_difference: f32,
    tokens: usize,
}

impl Comparison {
    fn of(reference: &Trace, run: &Trace) -> Self {
        let parted = reference
            .steps
            .iter()
            .zip(&run.steps)
            .position(|(a, b)| a.best != b.best)
            .or_else(|| {
                let shorter = reference.steps.len().min(run.steps.len());
                (reference.steps.len() != run.steps.len()).then_some(shorter)
            });
        let compared = parted.map_or(reference.steps.len(), |step| step + 1);
        let max_difference = reference
            .steps
            .iter()
            .zip(&run.steps)
            .take(compared)
            .filter_map(|(a, b)| a.difference(b))
            .fold(0.0, f32::max);
        Self {
            parted,
            margin: parted
                .and_then(|step| reference.steps.get(step))
                .map_or(f32::NAN, Step::margin),
            max_difference,
            tokens: reference.steps.len(),
        }
    }

    fn same(&self) -> bool {
        self.parted.is_none()
    }

    /// Whether the tokens part at a step that rounding can decide either
    /// way: the reference's two best logits are closer together than the
    /// two runs' logits are to each other, times [`NEAR_TIE_FACTOR`], and
    /// the runs' logits are no further apart than [`ROUNDING_LIMIT`].
    fn near_tie(&self) -> bool {
        self.parted.is_some()
            && self.max_difference <= ROUNDING_LIMIT
            && self.margin <= NEAR_TIE_FACTOR * self.max_difference
    }
}

/// How the runs of a set of prompts compare with their reference runs.
#[derive(Debug, Default, PartialEq)]
struct Verdict {
    same: usize,
    near_ties: usize,
    /// Prompts whose tokens part at a clear margin.
    wrong: usize,
    max_difference: f32,
}

/// Compare `runs` with `reference`, prompt by prompt, printing one line
/// each.
fn judge(name: &str, reference: &[Trace], runs: &[Trace]) -> Verdict {
    assert_eq!(reference.len(), runs.len(), "{name}: one run per prompt");
    let mut verdict = Verdict::default();
    for (index, (reference, run)) in reference.iter().zip(runs).enumerate() {
        let comparison = Comparison::of(reference, run);
        verdict.max_difference = verdict.max_difference.max(comparison.max_difference);
        match comparison.parted {
            None => {
                verdict.same += 1;
                println!(
                    "{name}: prompt {index} equal over {} tokens, max |logit difference| {:.2e}",
                    comparison.tokens, comparison.max_difference,
                );
            }
            Some(step) => {
                let kind = if comparison.near_tie() {
                    verdict.near_ties += 1;
                    "near tie"
                } else {
                    verdict.wrong += 1;
                    "DIFFERS"
                };
                println!(
                    "{name}: prompt {index} {kind} at step {step} of {}: reference margin there \
                     {:.2e}, max |logit difference| up to there {:.2e}",
                    comparison.tokens, comparison.margin, comparison.max_difference,
                );
            }
        }
    }
    verdict
}

/// One request.
struct Prompt {
    text: String,
    system: Option<String>,
}

impl Prompt {
    fn prepare(&self, pipeline: &TextGenerationPipeline) -> PreparedInput {
        pipeline
            .prepare(TextGenerationInput {
                text: self.text.clone(),
                system: self.system.clone(),
            })
            .expect("prepare")
    }
}

/// Generate every prompt alone, keeping every logit of the first
/// `logit_steps` steps.
fn solo(
    pipeline: &mut TextGenerationPipeline,
    prompts: &[Prompt],
    max_new_tokens: usize,
    logit_steps: usize,
) -> Vec<Trace> {
    prompts
        .iter()
        .map(|prompt| {
            let prepared = prompt.prepare(pipeline);
            let mut trace = Trace {
                prompt_tokens: prepared.sequence_length,
                ..Trace::default()
            };
            let started = Instant::now();
            let mut first = None;
            pipeline
                .generate(
                    &prepared,
                    &GenerationControl::default(),
                    max_new_tokens,
                    &mut |logits| {
                        first.get_or_insert_with(|| started.elapsed());
                        let keep = trace.steps.len() < logit_steps;
                        trace.steps.push(Step::of(logits, keep));
                    },
                )
                .expect("solo generation");
            trace.prefill = first.unwrap_or_default();
            trace.decode = started.elapsed().saturating_sub(trace.prefill);
            trace
        })
        .collect()
}

/// Generate the prompts through a batch, joining in `order`, each
/// `stagger` steps after the one before (or as soon as it fits; 0 joins
/// as many at once as fit).
///
/// Returns the traces by prompt, the most rows the batch held, and the
/// time it all took.
fn batched(
    pipeline: &mut TextGenerationPipeline,
    prompts: &[Prompt],
    order: &[usize],
    stagger: usize,
    logit_steps: usize,
) -> (Vec<Trace>, usize, Duration) {
    let seen: Rc<RefCell<HashMap<u64, Vec<Step>>>> = Rc::default();
    let mut batch = GenerationBatch::new();
    let observed = Rc::clone(&seen);
    batch.observe_logits(move |id, logits| {
        let mut seen = observed.borrow_mut();
        let steps = seen.entry(id).or_default();
        let keep = steps.len() < logit_steps;
        steps.push(Step::of(logits, keep));
    });
    let mut prepared: Vec<Option<PreparedInput>> = prompts
        .iter()
        .map(|prompt| Some(prompt.prepare(pipeline)))
        .collect();
    let prompt_tokens: Vec<usize> = prepared
        .iter()
        .map(|prompt| prompt.as_ref().map_or(0, |prompt| prompt.sequence_length))
        .collect();
    let mut waiting: VecDeque<usize> = order.iter().copied().collect();
    let mut since_join = stagger;
    let mut widest = 0;
    let mut results: HashMap<u64, Result<Execution, CoreError>> = HashMap::new();
    let mut record = |progress: BatchProgress| {
        results.extend(
            progress
                .finished
                .into_iter()
                .map(|done| (done.id, done.result)),
        );
    };
    let started = Instant::now();
    while !(waiting.is_empty() && batch.is_empty()) {
        let mut joining = Vec::new();
        let mut joining_tokens = Vec::new();
        while let Some(&next) = waiting.front() {
            let due = batch.is_empty() || since_join >= stagger;
            if !due || !pipeline.admits(&batch, &joining_tokens, prompt_tokens[next]) {
                break;
            }
            waiting.pop_front();
            joining_tokens.push(prompt_tokens[next]);
            joining.push(GenerationRequest {
                id: next as u64,
                prepared: PreparedData::Text(prepared[next].take().expect("joined once")),
                control: GenerationControl::default(),
            });
            if stagger > 0 {
                since_join = 0;
                break;
            }
        }
        if !joining.is_empty() {
            record(pipeline.join(&mut batch, joining));
        }
        widest = widest.max(batch.len());
        if !batch.is_empty() {
            record(pipeline.step(&mut batch));
            since_join += 1;
        }
    }
    let elapsed = started.elapsed();

    let mut seen = seen.borrow_mut();
    let traces = (0..prompts.len())
        .map(|index| {
            results
                .remove(&(index as u64))
                .expect("request finished")
                .expect("batched generation");
            Trace {
                steps: seen.remove(&(index as u64)).expect("request ran"),
                prompt_tokens: prompt_tokens[index],
                ..Trace::default()
            }
        })
        .collect();
    (traces, widest, elapsed)
}

/// Tokens per second, for a report line.
fn rate(tokens: usize, elapsed: Duration) -> f64 {
    if elapsed.is_zero() {
        return 0.0;
    }
    tokens as f64 / elapsed.as_secs_f64()
}

fn fixture(path: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(path)
}

/// A model directory around one graph of the tiny Qwen3 fixture.
fn tiny_qwen3(graph: &str) -> TempDir {
    let dir = tempfile::tempdir().expect("temp dir");
    for (from, to) in [
        (format!("text_generation/tiny_qwen3/{graph}"), "model.onnx"),
        (
            "text_generation/tiny_qwen3/config.json".to_string(),
            "config.json",
        ),
        (
            "text_generation/tokenizer_config.json".to_string(),
            "tokenizer_config.json",
        ),
        ("seq2seq/tokenizer.json".to_string(), "tokenizer.json"),
    ] {
        std::fs::copy(fixture(&from), dir.path().join(to)).expect("copy fixture");
    }
    dir
}

const FLOAT32: &str = "model.onnx";
const HALF_PRECISION: [&str; 3] = [
    "model_fp16.onnx",
    "model_fp16_logits32.onnx",
    "model_logits16.onnx",
];
const MAX_NEW_TOKENS: usize = 48;

fn tiny_prompts() -> Vec<Prompt> {
    [
        "b b",
        "f g a",
        "l k j i h g",
        "a b c d e f g h i j k l a b c",
        "h i",
        "f g h",
        "k k k k k k k k k k k k k k k k k k k k k k k k k k k k",
        "d",
    ]
    .into_iter()
    .map(|text| Prompt {
        text: text.to_string(),
        system: None,
    })
    .collect()
}

fn tiny_pipeline(dir: &TempDir, width: usize, device_cache: bool) -> TextGenerationPipeline {
    let options = TextGenerationOptions::new()
        .with_max_new_tokens(MAX_NEW_TOKENS)
        .with_max_batch(width)
        .with_device_cache(device_cache);
    TextGenerationPipeline::with_options(dir.path(), &ExecutionProvider::Cpu, &options)
        .expect("load")
}

#[test]
fn steps_are_compared_by_their_logits_and_their_two_best_tokens() {
    // Arrange
    let reference = Step::of(&[0.0, 3.0, 2.5, -1.0], true);
    let rounded = Step::of(&[0.0, 2.9, 2.75, -1.0], true);
    let flipped = Step::of(&[0.0, 2.9, 3.0, -1.0], false);

    // Assert -- best and runner-up, and the margin between them.
    assert_eq!((reference.best, reference.runner_up), (1, 2));
    assert_eq!(reference.margin(), 0.5);
    assert_eq!((flipped.best, flipped.runner_up), (2, 1));
    // All logits when both kept them, else the two best in common.
    assert_eq!(reference.difference(&rounded), Some(0.25));
    assert_eq!(reference.difference(&flipped), Some(0.5));

    // A trace that parts where the margin is small against the logit
    // differences is a near tie; one that parts at a clear margin is not.
    let trace = |steps: Vec<Step>| Trace {
        steps,
        ..Trace::default()
    };
    let tie = Comparison::of(
        &trace(vec![
            Step::of(&[1.0, 0.0], true),
            Step::of(&[0.0, 3.0, 2.5], true),
        ]),
        &trace(vec![
            Step::of(&[1.0, 0.0], true),
            Step::of(&[0.0, 2.9, 3.0], true),
        ]),
    );
    assert_eq!(tie.parted, Some(1));
    assert_eq!((tie.margin, tie.max_difference), (0.5, 0.5));
    assert!(tie.near_tie() && !tie.same());
    let clear = Comparison::of(
        &trace(vec![Step::of(&[0.0, 9.0, 2.5], true)]),
        &trace(vec![Step::of(&[0.0, 2.0, 2.5], false)]),
    );
    assert_eq!(clear.parted, Some(0));
    assert_eq!((clear.margin, clear.max_difference), (6.5, 7.0));
    assert!(!clear.near_tie(), "logits 7 apart are not rounding");
    let longer = Comparison::of(
        &trace(vec![Step::of(&[1.0, 0.0], true)]),
        &trace(vec![
            Step::of(&[1.0, 0.0], true),
            Step::of(&[1.0, 0.0], true),
        ]),
    );
    assert_eq!(longer.parted, Some(1));
    assert!(!longer.near_tie());
}

#[test]
fn device_cache_logits_equal_host_cache_logits_bit_for_bit() {
    let prompts = tiny_prompts();
    let all: Vec<usize> = (0..prompts.len()).collect();
    let reversed: Vec<usize> = all.iter().rev().copied().collect();

    for graph in [FLOAT32].into_iter().chain(HALF_PRECISION) {
        // Arrange
        let dir = tiny_qwen3(graph);
        let mut host = tiny_pipeline(&dir, 4, false);
        let mut device = tiny_pipeline(&dir, 4, true);
        assert!(device.device.is_some() && host.device.is_none());
        let exact = |name: &str, reference: &[Trace], runs: &[Trace]| {
            let verdict = judge(&format!("{graph} {name}"), reference, runs);
            assert_eq!(
                verdict,
                Verdict {
                    same: prompts.len(),
                    ..Verdict::default()
                },
                "{graph} {name}: the same graph on the same tensors"
            );
        };

        // Act / Assert -- alone.
        let from_host = solo(&mut host, &prompts, MAX_NEW_TOKENS, usize::MAX);
        let from_device = solo(&mut device, &prompts, MAX_NEW_TOKENS, usize::MAX);
        exact("solo", &from_host, &from_device);

        // Together, and joining one by one while others run and leave.
        for (name, order, stagger) in [
            ("together", &all, 0),
            ("reversed", &reversed, 0),
            ("staggered", &all, 3),
        ] {
            let (from_host, widest_host, _) =
                batched(&mut host, &prompts, order, stagger, usize::MAX);
            let (from_device, widest_device, _) =
                batched(&mut device, &prompts, order, stagger, usize::MAX);
            assert_eq!(widest_device, widest_host, "{graph} {name}");
            assert!(widest_device >= 2, "{graph} {name}: a batch formed");
            exact(name, &from_host, &from_device);
        }
    }
}

#[test]
fn half_precision_batches_part_from_solo_only_at_near_ties() {
    let prompts = tiny_prompts();
    let all: Vec<usize> = (0..prompts.len()).collect();
    let reversed: Vec<usize> = all.iter().rev().copied().collect();

    for graph in HALF_PRECISION {
        for width in [4, 8] {
            // Arrange
            let dir = tiny_qwen3(graph);
            let mut pipeline = tiny_pipeline(&dir, width, false);
            let alone = solo(&mut pipeline, &prompts, MAX_NEW_TOKENS, usize::MAX);

            for (name, order, stagger) in [
                ("together", &all, 0),
                ("reversed", &reversed, 0),
                ("staggered", &all, 3),
            ] {
                // Act
                let (together, _, _) = batched(&mut pipeline, &prompts, order, stagger, usize::MAX);

                // Assert -- a padded row is computed at other positions,
                // so its float16 logits differ a little; tokens may only
                // part where that decides between two candidates.
                let verdict = judge(&format!("{graph} width {width} {name}"), &alone, &together);
                assert_eq!(
                    verdict.wrong, 0,
                    "{graph} width {width} {name}: {verdict:?}"
                );
                assert!(
                    verdict.same >= 6,
                    "{graph} width {width} {name}: {verdict:?}"
                );
                assert!(
                    verdict.max_difference < 0.25,
                    "{graph} width {width} {name}: {verdict:?}"
                );
            }
        }
    }
}

/// A GPU against the CPU on a real model.
///
/// Needs a GPU, a build with `--features cuda` and an exported model, so
/// it never runs by default. `docs/gpu-verification.md` has the command,
/// the environment variables and what a pass means; in short, for every
/// graph named in `HEPHAESTUS_TEXTGEN_GPU_GRAPHS` it
///
/// 1. generates every prompt alone on the CPU provider (the reference;
///    skipped with `HEPHAESTUS_TEXTGEN_GPU_SKIP_CPU=1`);
/// 2. loads the graph on the CUDA provider with the cache on the device,
///    which fails unless the session computes on the GPU and returns its
///    cache there;
/// 3. generates every prompt alone on the GPU and compares it with the
///    CPU run, step by step;
/// 4. generates them through a batch three times (together, reversed,
///    joining three steps apart) and compares each with step 3;
/// 5. generates them alone on the GPU with the cache in host memory and
///    compares that with step 3 (the same kernels: equal tokens);
/// 6. prints tokens per second for the prompt pass and for decoding,
///    for steps 3, 4 and 5.
///
/// The test fails if a load fails, the cache is not on the device, or
/// tokens part anywhere but at a near tie (see [`Comparison::near_tie`]).
/// Near ties are counted and printed; they are for a person to judge.
#[test]
#[ignore = "requires a GPU, the cuda feature, an exported model and a prompts file"]
fn gpu_generation_matches_cpu_on_a_real_model() {
    // Arrange
    let variable = |name: &str| std::env::var(name).ok();
    let model_dir = variable("HEPHAESTUS_TEXTGEN_BATCH_MODEL_DIR")
        .expect("HEPHAESTUS_TEXTGEN_BATCH_MODEL_DIR must be set");
    let prompts_path = variable("HEPHAESTUS_TEXTGEN_BATCH_PROMPTS")
        .expect("HEPHAESTUS_TEXTGEN_BATCH_PROMPTS must be set");
    let number = |name: &str, default: usize| {
        variable(name).map_or(default, |value| value.parse().expect("a number"))
    };
    let width = number("HEPHAESTUS_TEXTGEN_BATCH_WIDTH", 4);
    let logit_steps = number("HEPHAESTUS_TEXTGEN_BATCH_LOGIT_STEPS", 8);
    let graphs = variable("HEPHAESTUS_TEXTGEN_GPU_GRAPHS")
        .unwrap_or_else(|| "model_fp16.onnx,model.onnx".to_string());
    let skip_cpu = variable("HEPHAESTUS_TEXTGEN_GPU_SKIP_CPU").is_some_and(|value| value == "1");
    let provider: ExecutionProvider = variable("HEPHAESTUS_TEXTGEN_GPU_PROVIDER")
        .unwrap_or_else(|| "cuda".to_string())
        .parse()
        .expect("an execution provider");
    let file: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(prompts_path).expect("read prompts"))
            .expect("parse prompts");
    let max_new_tokens = usize::try_from(file["max_new_tokens"].as_u64().expect("max_new_tokens"))
        .expect("limit fits usize");
    let prompts: Vec<Prompt> = file["prompts"]
        .as_array()
        .expect("prompts array")
        .iter()
        .map(|prompt| Prompt {
            text: prompt["text"].as_str().expect("prompt text").to_string(),
            system: prompt["system"].as_str().map(str::to_string),
        })
        .collect();
    let load = |graph: &str, provider: &ExecutionProvider, width: usize, device: Option<bool>| {
        let mut options = TextGenerationOptions::new()
            .with_max_new_tokens(max_new_tokens)
            .with_max_batch(width)
            .with_graph_file(graph);
        if let Some(device) = device {
            options = options.with_device_cache(device);
        }
        TextGenerationPipeline::with_options(Path::new(&model_dir), provider, &options)
            .unwrap_or_else(|error| panic!("{graph} on {provider}: {error}"))
    };
    let speed = |name: &str, traces: &[Trace]| {
        let prompt_tokens: usize = traces.iter().map(|trace| trace.prompt_tokens).sum();
        let prefill: Duration = traces.iter().map(|trace| trace.prefill).sum();
        // The first token comes out of the prompt pass.
        let decoded: usize = traces
            .iter()
            .map(|trace| trace.steps.len().saturating_sub(1))
            .sum();
        let decode: Duration = traces.iter().map(|trace| trace.decode).sum();
        println!(
            "{name}: prefill {prompt_tokens} tokens in {:.3} s = {:.0} tokens/s; decode \
             {decoded} tokens in {:.3} s = {:.1} tokens/s",
            prefill.as_secs_f64(),
            rate(prompt_tokens, prefill),
            decode.as_secs_f64(),
            rate(decoded, decode),
        );
    };
    let all: Vec<usize> = (0..prompts.len()).collect();
    let reversed: Vec<usize> = all.iter().rev().copied().collect();
    let mut wrong = 0;
    let mut near_ties = 0;
    let mut tally = |verdict: Verdict| {
        wrong += verdict.wrong;
        near_ties += verdict.near_ties;
    };

    for graph in graphs
        .split(',')
        .map(str::trim)
        .filter(|graph| !graph.is_empty())
    {
        // Act -- 1. the CPU reference.
        let reference = (!skip_cpu).then(|| {
            let mut cpu = load(graph, &ExecutionProvider::Cpu, 1, None);
            let reference = solo(&mut cpu, &prompts, max_new_tokens, logit_steps);
            for (index, trace) in reference.iter().enumerate() {
                println!(
                    "{graph}: prompt {index}: {} prompt tokens, {} generated on the CPU",
                    trace.prompt_tokens,
                    trace.steps.len()
                );
            }
            speed(&format!("{graph} cpu solo"), &reference);
            reference
        });

        // 2. The provider under test, cache on the device.
        //    (With the CPU as the provider under test, to try this test
        //    out without a GPU, the device cache has to be asked for.)
        let asked = (provider == ExecutionProvider::Cpu).then_some(true);
        let mut gpu = load(graph, &provider, width, asked);
        assert!(
            gpu.keeps_cache_on_device(),
            "{graph}: {provider} should keep the cache on the device by default"
        );
        assert_eq!(
            gpu.max_batch(),
            width,
            "{graph}: the graph must be able to batch: {:?}",
            gpu.batch_check()
        );
        let device = gpu.device.as_ref().map(|device| device.device().name());
        println!("{graph}: session on {provider}, cache on {device:?}");
        println!(
            "{graph}: startup check of batching: {:?}",
            gpu.batch_check()
        );

        // 3. Alone, against the CPU.
        let on_device = solo(&mut gpu, &prompts, max_new_tokens, logit_steps);
        speed(
            &format!("{graph} {provider} solo, device cache"),
            &on_device,
        );
        match &reference {
            Some(reference) => {
                tally(judge(
                    &format!("{graph} {provider} vs cpu"),
                    reference,
                    &on_device,
                ));
            }
            None => println!("{graph}: CPU reference skipped; {provider} is not compared with it"),
        }

        // 4. Through a batch, against alone on the same provider.
        for (name, order, stagger) in [
            ("together", &all, 0),
            ("reversed", &reversed, 0),
            ("staggered", &all, 3),
        ] {
            let (together, widest, elapsed) =
                batched(&mut gpu, &prompts, order, stagger, logit_steps);
            let generated: usize = together.iter().map(|trace| trace.steps.len()).sum();
            println!(
                "{graph} {provider} batch {name} (widest {widest}): {generated} tokens in {:.3} s \
                 = {:.1} tokens/s, prompt passes included",
                elapsed.as_secs_f64(),
                rate(generated, elapsed),
            );
            tally(judge(
                &format!("{graph} {provider} batch {name} vs solo"),
                &on_device,
                &together,
            ));
        }
        drop(gpu);

        // 5. Alone with the cache in host memory, against the device cache.
        let mut host_cache = load(graph, &provider, 1, Some(false));
        assert!(!host_cache.keeps_cache_on_device());
        let on_host = solo(&mut host_cache, &prompts, max_new_tokens, logit_steps);
        speed(&format!("{graph} {provider} solo, host cache"), &on_host);
        tally(judge(
            &format!("{graph} {provider} host cache vs device cache"),
            &on_device,
            &on_host,
        ));
    }

    // Assert
    println!("near ties: {near_ties}; clear differences: {wrong}");
    assert_eq!(
        wrong, 0,
        "tokens differ at a clear margin; see the lines above"
    );
}
