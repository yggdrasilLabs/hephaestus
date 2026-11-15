//! Text generation pipeline: decoder-only instruction models.
//!
//! Serves the Optimum export of a causal language model: one graph
//! (`model.onnx`) that takes the tokens so far and returns next-token
//! logits, plus, when exported with a key/value cache, `past_key_values.*`
//! inputs and matching `present.*` outputs.
//!
//! A request is one user message and an optional system message. The
//! model's own chat template turns them into the prompt; the prompt is
//! tokenized and the model continues it greedily -- the most likely token
//! at every step -- until it emits an end token or reaches the token
//! limit. With a cache the prompt is read once and every later step feeds
//! only the newest token; without one every step re-reads everything.
//! Both loops pick the same tokens.
//!
//! As in encoder-decoder mode, everything a specific model could get wrong
//! is checked when the pipeline is constructed, so a misfit fails at
//! startup with a message naming the problem.

use std::borrow::Cow;
use std::path::Path;
use std::time::{Duration, Instant};

use ort::session::{Session, SessionInputValue};
use ort::value::{DynValue, Tensor, TensorElementType};
use tokenizers::Tokenizer;

use crate::ep::ExecutionProvider;
use crate::error::CoreError;

use super::chat_template::ChatTemplate;
use super::encoder_decoder::{
    TensorSpec, argmax_excluding, extend_with_repeating_tokens, int64_tensor, parse_token_id,
    read_json, spec_names,
};
use super::{
    Pipeline, PipelineOutput, PreparedData, PreparedInput, build_onnx_session, resolve_onnx_path,
};

/// Smallest accepted input token cap: below this no prompt fits.
pub const TEXT_GENERATION_MIN_INPUT_TOKENS: usize = 16;

/// Absolute ceiling on generated tokens per request, whatever the model or
/// the operator asks for.
pub const TEXT_GENERATION_MAX_NEW_TOKENS_LIMIT: usize = 4096;

/// Input token cap when [`TextGenerationOptions`] does not set one. Counts
/// the whole prompt: template, system message and user message.
const DEFAULT_MAX_INPUT_TOKENS: usize = 4096;

/// Generated-token limit when neither the options nor the model's
/// `generation_config.json` set `max_new_tokens`.
const DEFAULT_MAX_NEW_TOKENS: usize = 256;

/// Graph file loaded when [`TextGenerationOptions`] does not name one.
const DEFAULT_GRAPH_FILE: &str = "model.onnx";

/// Prefix of the key/value cache inputs.
const PAST_INPUT_PREFIX: &str = "past_key_values.";
/// Prefix of the key/value cache outputs; the rest of the name matches the
/// input it feeds on the next step.
const PRESENT_OUTPUT_PREFIX: &str = "present.";

/// How often a prompt above the input cap is shortened and measured again.
/// One pass is the rule; a second covers tokens that merge differently at
/// the new end of the text.
const MAX_TRUNCATION_PASSES: usize = 4;

/// Generation settings that change which token greedy decoding picks and
/// that this pipeline does not implement. Their presence is logged.
const UNSUPPORTED_GENERATION_SETTINGS: [&str; 9] = [
    "bad_words_ids",
    "begin_suppress_tokens",
    "encoder_repetition_penalty",
    "forced_bos_token_id",
    "forced_eos_token_id",
    "min_length",
    "min_new_tokens",
    "sequence_bias",
    "suppress_tokens",
];

/// Optional settings for [`TextGenerationPipeline`].
///
/// ```
/// use std::time::Duration;
///
/// use hephaestus_core::TextGenerationOptions;
///
/// let options = TextGenerationOptions::new()
///     .with_max_input_tokens(2048)
///     .with_max_new_tokens(200)
///     .with_system_prompt("Answer in one sentence.")
///     .with_timeout(Duration::from_secs(60));
/// assert_ne!(options, TextGenerationOptions::default());
/// ```
#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[must_use = "options do nothing until passed to TextGenerationPipeline::with_options"]
pub struct TextGenerationOptions {
    max_input_tokens: Option<usize>,
    max_new_tokens: Option<usize>,
    timeout: Option<Duration>,
    system_prompt: Option<String>,
    kv_cache: Option<bool>,
    graph_file: Option<String>,
}

impl TextGenerationOptions {
    /// Options that leave every setting at its default.
    pub fn new() -> Self {
        Self::default()
    }

    /// Accept prompts of at most `tokens` tokens, counting the template,
    /// the system message and the user message.
    ///
    /// A longer prompt has the end of its user message cut off. Unset, the
    /// cap is 4096 tokens, or less when the model's context is smaller.
    pub fn with_max_input_tokens(mut self, tokens: usize) -> Self {
        self.max_input_tokens = Some(tokens);
        self
    }

    /// Generate at most `tokens` tokens per request.
    ///
    /// Unset, the limit is the `max_new_tokens` of the model's
    /// `generation_config.json`, else 256.
    pub fn with_max_new_tokens(mut self, tokens: usize) -> Self {
        self.max_new_tokens = Some(tokens);
        self
    }

    /// Abort a generation that runs longer than `timeout`.
    ///
    /// The budget covers one request, or one batch when requests arrive
    /// batched. Unset, generation is bounded by the token limit alone.
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    /// Use `prompt` as the system message of requests that bring none.
    ///
    /// An empty string is the same as not setting one.
    pub fn with_system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(prompt.into()).filter(|prompt| !prompt.is_empty());
        self
    }

    /// Load the graph named `file` instead of `model.onnx`.
    ///
    /// Repositories with several precision variants keep them side by side
    /// (`model.onnx`, `model_fp16.onnx`, `model_q4.onnx`, ...). The file
    /// is looked up in `onnx/` and then in the model directory, like
    /// `model.onnx`.
    pub fn with_graph_file(mut self, file: impl Into<String>) -> Self {
        self.graph_file = Some(file.into());
        self
    }

    /// Switch the key/value cache on or off.
    ///
    /// Unset, the cache is used whenever the graph has one. `false` forces
    /// the loop that re-reads the whole sequence at every step. Both loops
    /// generate the same tokens; the cached one does far less work.
    pub fn with_kv_cache(mut self, enabled: bool) -> Self {
        self.kv_cache = Some(enabled);
        self
    }
}

/// One request to a [`TextGenerationPipeline`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TextGenerationInput {
    /// The user message.
    pub text: String,
    /// The system message for this request. `None` uses the pipeline's
    /// default system prompt; an empty string sends no system message.
    pub system: Option<String>,
}

impl From<String> for TextGenerationInput {
    fn from(text: String) -> Self {
        Self { text, system: None }
    }
}

/// One key/value cache input and the output that feeds it next step.
#[derive(Debug, Clone, PartialEq, Eq)]
struct PastSlot {
    /// Input name (`past_key_values.<layer>.<key|value>`).
    input: String,
    /// Output name (`present.<layer>.<key|value>`).
    output: String,
    /// Shape of the tensor fed before anything is cached:
    /// `(1, heads, 0, head_dim)`.
    empty_dims: [usize; 4],
}

/// How to feed the graph and where to read its results.
#[derive(Debug, Clone, PartialEq, Eq)]
struct GraphIo {
    /// Whether the graph declares an `attention_mask` input.
    feeds_attention_mask: bool,
    /// Whether the graph declares a `position_ids` input.
    feeds_position_ids: bool,
    /// Key/value cache slots; empty for a graph exported without a cache.
    past: Vec<PastSlot>,
}

impl GraphIo {
    /// Derive the wiring from the graph's declared inputs and outputs.
    ///
    /// `config` (the model's `config.json`) supplies the cache dimensions
    /// the graph leaves dynamic.
    ///
    /// # Errors
    ///
    /// Returns [`CoreError::ModelValidation`] if the graph has no
    /// `input_ids` input or no `logits` output, declares an input this
    /// pipeline cannot feed, has a cache input without a matching output,
    /// or uses element types other than int64 tokens and float32 tensors.
    fn from_specs(
        inputs: &[TensorSpec],
        outputs: &[TensorSpec],
        config: &serde_json::Value,
    ) -> Result<Self, CoreError> {
        let require = |spec: &TensorSpec, expected: TensorElementType| {
            if spec.element_type == Some(expected) {
                return Ok(());
            }
            Err(CoreError::ModelValidation(format!(
                "tensor '{}' has element type {:?}, but text generation needs {expected:?}",
                spec.name, spec.element_type,
            )))
        };

        let mut has_input_ids = false;
        let mut feeds_attention_mask = false;
        let mut feeds_position_ids = false;
        let mut past = Vec::new();
        for spec in inputs {
            match spec.name.as_str() {
                "input_ids" => {
                    require(spec, TensorElementType::Int64)?;
                    has_input_ids = true;
                }
                "attention_mask" => {
                    require(spec, TensorElementType::Int64)?;
                    feeds_attention_mask = true;
                }
                "position_ids" => {
                    require(spec, TensorElementType::Int64)?;
                    feeds_position_ids = true;
                }
                name => {
                    let Some(slot) = name.strip_prefix(PAST_INPUT_PREFIX) else {
                        return Err(CoreError::ModelValidation(format!(
                            "graph input '{name}' is not supported by text generation; graph \
                             inputs are: {:?}",
                            spec_names(inputs),
                        )));
                    };
                    require(spec, TensorElementType::Float32)?;
                    let output = format!("{PRESENT_OUTPUT_PREFIX}{slot}");
                    let present = outputs
                        .iter()
                        .find(|candidate| candidate.name == output)
                        .ok_or_else(|| {
                            CoreError::ModelValidation(format!(
                                "graph has cache input '{name}' but no output '{output}' to feed it",
                            ))
                        })?;
                    require(present, TensorElementType::Float32)?;
                    past.push(PastSlot {
                        input: spec.name.clone(),
                        output,
                        empty_dims: empty_cache_dims(spec, config)?,
                    });
                }
            }
        }
        if !has_input_ids {
            return Err(CoreError::ModelValidation(format!(
                "graph does not accept input 'input_ids'; graph inputs are: {:?}",
                spec_names(inputs),
            )));
        }

        let logits = outputs
            .iter()
            .find(|spec| spec.name == "logits")
            .ok_or_else(|| {
                CoreError::ModelValidation(format!(
                    "graph does not produce output 'logits'; graph outputs are: {:?}",
                    spec_names(outputs),
                ))
            })?;
        require(logits, TensorElementType::Float32)?;

        Ok(Self {
            feeds_attention_mask,
            feeds_position_ids,
            past,
        })
    }
}

/// Shape of an empty cache tensor for a `(batch, heads, sequence,
/// head_dim)` cache input: batch 1, sequence 0.
///
/// Heads and head size are taken from the graph when it fixes them, and
/// from `config.json` otherwise.
fn empty_cache_dims(spec: &TensorSpec, config: &serde_json::Value) -> Result<[usize; 4], CoreError> {
    let [_, heads, _, head_dim] = spec.dims.as_slice() else {
        return Err(CoreError::ModelValidation(format!(
            "cache input '{}' has {} dimensions, expected (batch, heads, sequence, head_dim)",
            spec.name,
            spec.dims.len(),
        )));
    };
    let config_value = |keys: &[&str]| {
        keys.iter().find_map(|key| {
            config
                .get(*key)
                .and_then(serde_json::Value::as_u64)
                .and_then(|value| usize::try_from(value).ok())
                .filter(|value| *value > 0)
        })
    };
    let fixed = |dim: i64| usize::try_from(dim).ok().filter(|dim| *dim > 0);

    let attention_heads = config_value(&["num_attention_heads", "n_head"]);
    let heads = fixed(*heads)
        .or_else(|| config_value(&["num_key_value_heads"]))
        .or(attention_heads);
    let head_dim = fixed(*head_dim)
        .or_else(|| config_value(&["head_dim"]))
        .or_else(|| {
            let hidden = config_value(&["hidden_size", "n_embd"])?;
            attention_heads.map(|heads| hidden / heads).filter(|dim| *dim > 0)
        });
    match (heads, head_dim) {
        (Some(heads), Some(head_dim)) => Ok([1, heads, 0, head_dim]),
        _ => Err(CoreError::ModelValidation(format!(
            "cannot determine the shape of cache input '{}': the graph declares {:?} and \
             config.json does not give the number of heads and their size",
            spec.name, spec.dims,
        ))),
    }
}

/// Generation settings resolved from the model's own configuration files.
///
/// Each value is taken from `generation_config.json` when that file
/// declares it and from `config.json` otherwise.
#[derive(Debug, Clone, PartialEq)]
struct GenerationSettings {
    /// Tokens that end generation. Empty when the model declares none.
    eos_token_ids: Vec<i64>,
    /// Hard upper bound on generated tokens.
    max_new_tokens: usize,
    /// Penalty on tokens already in the sequence; 1.0 disables it.
    repetition_penalty: f32,
    /// No sequence of this many tokens may occur twice; 0 disables the rule.
    no_repeat_ngram_size: usize,
}

impl GenerationSettings {
    /// Resolve generation settings for a model.
    ///
    /// # Errors
    ///
    /// Returns [`CoreError::ModelValidation`] if a declared value has the
    /// wrong type or range, and [`CoreError::Config`] if the explicit
    /// token limit is zero or above
    /// [`TEXT_GENERATION_MAX_NEW_TOKENS_LIMIT`].
    fn resolve(
        config: &serde_json::Value,
        generation_config: Option<&serde_json::Value>,
        max_new_tokens: Option<usize>,
    ) -> Result<Self, CoreError> {
        let lookup = |key: &str| {
            generation_config
                .and_then(|generation| generation.get(key))
                .filter(|value| !value.is_null())
                .or_else(|| config.get(key).filter(|value| !value.is_null()))
        };
        let vocab_size = config.get("vocab_size").and_then(serde_json::Value::as_i64);

        let eos_token_ids = match lookup("eos_token_id") {
            Some(serde_json::Value::Array(ids)) => ids
                .iter()
                .map(|value| parse_token_id("eos_token_id", value, vocab_size))
                .collect::<Result<Vec<i64>, CoreError>>()?,
            Some(value) => vec![parse_token_id("eos_token_id", value, vocab_size)?],
            None => Vec::new(),
        };

        let repetition_penalty = match lookup("repetition_penalty") {
            Some(value) => value
                .as_f64()
                .filter(|penalty| penalty.is_finite() && *penalty > 0.0)
                .map(|penalty| penalty as f32)
                .ok_or_else(|| {
                    CoreError::ModelValidation(format!(
                        "model configuration value 'repetition_penalty' is not a positive \
                         number: {value}",
                    ))
                })?,
            None => 1.0,
        };

        let no_repeat_ngram_size = match lookup("no_repeat_ngram_size") {
            Some(value) => value
                .as_u64()
                .and_then(|size| usize::try_from(size).ok())
                .ok_or_else(|| {
                    CoreError::ModelValidation(format!(
                        "model configuration value 'no_repeat_ngram_size' is not a non-negative \
                         integer: {value}",
                    ))
                })?,
            None => 0,
        };

        let limit = TEXT_GENERATION_MAX_NEW_TOKENS_LIMIT;
        let max_new_tokens = match max_new_tokens {
            Some(explicit) if explicit == 0 || explicit > limit => {
                return Err(CoreError::Config(format!(
                    "text generation max new tokens must be between 1 and {limit} (got {explicit})",
                )));
            }
            Some(explicit) => explicit,
            // Only `generation_config.json` counts here: `max_length` and
            // the `config.json` defaults describe total sequence length
            // and default to 20 tokens, which no instruction model means.
            None => generation_config
                .and_then(|generation| generation.get("max_new_tokens"))
                .and_then(serde_json::Value::as_u64)
                .and_then(|value| usize::try_from(value).ok())
                .filter(|value| *value > 0)
                .unwrap_or(DEFAULT_MAX_NEW_TOKENS)
                .min(limit),
        };

        if let Some(generation) = generation_config {
            let ignored: Vec<&str> = UNSUPPORTED_GENERATION_SETTINGS
                .into_iter()
                .filter(|key| {
                    generation.get(*key).is_some_and(|value| {
                        !value.is_null() && value.as_u64() != Some(0) && value.as_f64() != Some(1.0)
                    })
                })
                .collect();
            if !ignored.is_empty() {
                tracing::warn!(
                    settings = ?ignored,
                    "generation_config.json sets options that greedy text generation ignores"
                );
            }
        }

        Ok(Self {
            eos_token_ids,
            max_new_tokens,
            repetition_penalty,
            no_repeat_ngram_size,
        })
    }
}

/// Resolve the prompt token cap for a model.
///
/// The prompt and the generated tokens share the model's context
/// (`max_position_embeddings` / `n_positions`), so the cap leaves room for
/// `max_new_tokens`. An unset cap is [`DEFAULT_MAX_INPUT_TOKENS`], lowered
/// to what the context allows.
///
/// # Errors
///
/// Returns [`CoreError::Config`] if an explicit cap is below
/// [`TEXT_GENERATION_MIN_INPUT_TOKENS`] or does not fit the context
/// together with `max_new_tokens`, or if the context is too small for any
/// prompt.
fn resolve_max_input_tokens(
    config: &serde_json::Value,
    explicit: Option<usize>,
    max_new_tokens: usize,
) -> Result<usize, CoreError> {
    let min = TEXT_GENERATION_MIN_INPUT_TOKENS;
    let context = ["max_position_embeddings", "n_positions"]
        .iter()
        .find_map(|key| config.get(*key).and_then(serde_json::Value::as_u64))
        .and_then(|limit| usize::try_from(limit).ok());
    let room = context.map(|context| context.saturating_sub(max_new_tokens));

    match (explicit, room) {
        (Some(cap), _) if cap < min => Err(CoreError::Config(format!(
            "text generation max input tokens {cap} is below the minimum of {min}",
        ))),
        (Some(cap), Some(room)) if cap > room => Err(CoreError::Config(format!(
            "text generation max input tokens {cap} plus max new tokens {max_new_tokens} exceed \
             the model's context of {} tokens",
            room + max_new_tokens,
        ))),
        (Some(cap), _) => Ok(cap),
        (None, Some(room)) if room < min => Err(CoreError::Config(format!(
            "max new tokens {max_new_tokens} leave no room for a prompt in the model's context \
             of {} tokens",
            room + max_new_tokens,
        ))),
        (None, Some(room)) => Ok(room.min(DEFAULT_MAX_INPUT_TOKENS)),
        (None, None) => Ok(DEFAULT_MAX_INPUT_TOKENS),
    }
}

/// Apply the Hugging Face repetition penalty to next-token logits.
///
/// Every token already in the sequence -- prompt included -- has its logit
/// divided by `penalty` when positive and multiplied by it otherwise.
fn penalize_repeats(logits: &mut [f32], tokens: &[i64], penalty: f32) {
    // A token that occurs twice must be penalized once.
    let mut seen = vec![false; logits.len()];
    for &token in tokens {
        let Some(index) = usize::try_from(token).ok().filter(|index| *index < logits.len()) else {
            continue;
        };
        if std::mem::replace(&mut seen[index], true) {
            continue;
        }
        let logit = logits[index];
        logits[index] = if logit > 0.0 { logit / penalty } else { logit * penalty };
    }
}

/// Fail with [`CoreError::Timeout`] if the deadline has passed.
fn ensure_before(deadline: Option<Instant>, generated: usize) -> Result<(), CoreError> {
    match deadline {
        Some(deadline) if Instant::now() >= deadline => Err(CoreError::Timeout(format!(
            "text generation exceeded its time budget after {generated} tokens",
        ))),
        _ => Ok(()),
    }
}

/// Text generation pipeline for decoder-only instruction models.
///
/// The model directory must contain:
/// - `onnx/model.onnx` or `model.onnx` (with its external data, if any),
///   or the graph file named in the options
/// - `tokenizer.json`
/// - `config.json`; `generation_config.json` is read when present
/// - a chat template: `chat_template.jinja`, or `chat_template` in
///   `tokenizer_config.json`
pub struct TextGenerationPipeline {
    session: Session,
    tokenizer: Tokenizer,
    template: ChatTemplate,
    io: GraphIo,
    settings: GenerationSettings,
    /// Cap on prompt tokens.
    max_input_tokens: usize,
    /// System message of requests that bring none.
    system_prompt: Option<String>,
    /// Time budget for one request.
    timeout: Option<Duration>,
    /// Whether generation keeps a key/value cache between steps.
    kv_cache: bool,
    /// Empty cache tensors fed before anything is cached, one per slot.
    empty_past: Vec<Tensor<f32>>,
}

impl TextGenerationPipeline {
    /// Construct a text generation pipeline with default options.
    ///
    /// # Errors
    ///
    /// See [`TextGenerationPipeline::with_options`].
    pub fn new(model_dir: &Path, ep: &ExecutionProvider) -> Result<Self, CoreError> {
        Self::with_options(model_dir, ep, &TextGenerationOptions::default())
    }

    /// Construct a text generation pipeline with explicit options.
    ///
    /// # Errors
    ///
    /// Returns [`CoreError`] if a required file is missing or invalid, the
    /// model has no usable chat template, the graph does not have the
    /// inputs and outputs this pipeline can drive, or an option is out of
    /// range for the model.
    pub fn with_options(
        model_dir: &Path,
        ep: &ExecutionProvider,
        options: &TextGenerationOptions,
    ) -> Result<Self, CoreError> {
        // 1. Configuration and template: validated before the (slow) graph
        //    load.
        let config = read_json(&model_dir.join("config.json"))?;
        let generation_config_path = model_dir.join("generation_config.json");
        let generation_config = if generation_config_path.exists() {
            Some(read_json(&generation_config_path)?)
        } else {
            None
        };
        let settings =
            GenerationSettings::resolve(&config, generation_config.as_ref(), options.max_new_tokens)?;
        let max_input_tokens =
            resolve_max_input_tokens(&config, options.max_input_tokens, settings.max_new_tokens)?;
        let template = ChatTemplate::load(model_dir)?;

        // 2. Tokenizer. The prompt is cut to size before tokenization and
        //    one request is one sequence, so neither truncate nor pad.
        let mut tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json"))
            .map_err(|e| CoreError::Tokenization(e.to_string()))?;
        tokenizer
            .with_truncation(None)
            .map_err(|e| CoreError::Tokenization(e.to_string()))?;
        tokenizer.with_padding(None);

        // 3. Graph, wired by the names it declares.
        let graph_file = options.graph_file.as_deref().unwrap_or(DEFAULT_GRAPH_FILE);
        let model_path = resolve_onnx_path(model_dir, graph_file)?;
        let session = build_onnx_session(&model_path, ep)?;
        let io = GraphIo::from_specs(
            &TensorSpec::from_outlets(session.inputs()),
            &TensorSpec::from_outlets(session.outputs()),
            &config,
        )?;
        let kv_cache = options.kv_cache.unwrap_or(true) && !io.past.is_empty();
        let empty_past = io
            .past
            .iter()
            .map(|slot| {
                Tensor::<f32>::from_array((slot.empty_dims, Vec::<f32>::new())).map_err(|e| {
                    CoreError::ModelLoad(format!(
                        "cannot create an empty tensor for cache input '{}': {e}",
                        slot.input,
                    ))
                })
            })
            .collect::<Result<Vec<_>, CoreError>>()?;

        let pipeline = Self {
            session,
            tokenizer,
            template,
            io,
            settings,
            max_input_tokens,
            system_prompt: options.system_prompt.clone(),
            timeout: options.timeout,
            kv_cache,
            empty_past,
        };

        // 4. The template must render what requests will send; a model
        //    that refuses a system message fails here, not on a request.
        pipeline.prepare_prompt(TextGenerationInput::from("Hello".to_string()))?;

        tracing::info!(
            model = %model_path.display(),
            kv_cache = pipeline.kv_cache,
            cache_tensors = pipeline.io.past.len(),
            position_ids = pipeline.io.feeds_position_ids,
            max_input_tokens,
            max_new_tokens = pipeline.settings.max_new_tokens,
            eos_token_ids = ?pipeline.settings.eos_token_ids,
            repetition_penalty = pipeline.settings.repetition_penalty,
            no_repeat_ngram_size = pipeline.settings.no_repeat_ngram_size,
            default_system_prompt = pipeline.system_prompt.is_some(),
            "text generation pipeline loaded (greedy decoding)"
        );
        if pipeline.settings.eos_token_ids.is_empty() {
            tracing::warn!(
                max_new_tokens = pipeline.settings.max_new_tokens,
                "model declares no eos_token_id; every generation runs to the token limit"
            );
        }
        if !pipeline.kv_cache {
            tracing::warn!(
                "text generation runs without a key/value cache; every token re-reads the whole prompt"
            );
        }

        Ok(pipeline)
    }

    /// Whether generation keeps a key/value cache between steps.
    pub fn uses_kv_cache(&self) -> bool {
        self.kv_cache
    }

    /// Run one short generation so the first request does not pay for
    /// lazy initialisation. Reads a small prompt and generates one token.
    ///
    /// # Errors
    ///
    /// Returns [`CoreError`] if the prompt cannot be built or the graph
    /// fails to run.
    pub fn warm_up(&mut self) -> Result<(), CoreError> {
        let prepared = self.prepare_prompt(TextGenerationInput::from("Hello".to_string()))?;
        self.generate(&prepared, None, 1, &mut |_| {}).map(|_| ())
    }

    /// Run generation for a batch, one request after another.
    ///
    /// Generation is token by token, so requests cannot share one padded
    /// forward pass the way the single-pass profiles do. Returns one
    /// result per request, in order. The time budget covers the batch.
    pub(super) fn execute_batch_sequential(
        &mut self,
        batch: Vec<PreparedData>,
    ) -> Vec<Result<PipelineOutput, CoreError>> {
        let deadline = self.deadline_from_now();
        batch
            .into_iter()
            .map(|item| match item {
                PreparedData::Text(prepared) => self
                    .generate_text(&prepared, deadline)
                    .map(PipelineOutput::TextGeneration),
                PreparedData::Audio(_) => Err(CoreError::InvalidInput(
                    "text pipeline received audio prepared data in batch".to_string(),
                )),
            })
            .collect()
    }

    fn deadline_from_now(&self) -> Option<Instant> {
        self.timeout.map(|timeout| Instant::now() + timeout)
    }

    /// Render and tokenize the prompt, shortening the user message when
    /// the prompt is above the input cap.
    fn prepare_prompt(&self, input: TextGenerationInput) -> Result<PreparedInput, CoreError> {
        let system = match input.system {
            Some(system) => Some(system).filter(|system| !system.is_empty()),
            None => self.system_prompt.clone(),
        };
        let cap = self.max_input_tokens;
        let mut text = input.text;

        for _ in 0..MAX_TRUNCATION_PASSES {
            let prompt = self.template.render(system.as_deref(), &text)?;
            // The template writes every special token itself.
            let encoding = self
                .tokenizer
                .encode(prompt.as_str(), false)
                .map_err(|e| CoreError::Tokenization(e.to_string()))?;
            let length = encoding.len();
            if length == 0 {
                return Err(CoreError::InvalidInput(
                    "chat template rendered an empty prompt".to_string(),
                ));
            }
            if length <= cap {
                return Ok(PreparedInput {
                    input_ids: encoding.get_ids().iter().map(|&id| i64::from(id)).collect(),
                    attention_mask: vec![1; length],
                    sequence_length: length,
                    encoding: None,
                });
            }
            text = shorten_user_text(&prompt, encoding.get_offsets(), &text, length - cap)
                .ok_or_else(|| {
                    CoreError::InvalidInput(format!(
                        "prompt has {length} tokens, more than the limit of {cap}, and cannot \
                         be shortened by cutting the user text",
                    ))
                })?;
            tracing::debug!(
                prompt_tokens = length,
                max_input_tokens = cap,
                "prompt above the input cap; user text truncated"
            );
        }
        Err(CoreError::InvalidInput(format!(
            "prompt does not fit the limit of {cap} tokens",
        )))
    }

    /// Generate and decode the reply for one prepared prompt.
    fn generate_text(
        &mut self,
        prepared: &PreparedInput,
        deadline: Option<Instant>,
    ) -> Result<String, CoreError> {
        let max_new_tokens = self.settings.max_new_tokens;
        let generated = self.generate(prepared, deadline, max_new_tokens, &mut |_| {})?;
        self.tokenizer
            .decode(&generated, true)
            .map_err(|e| CoreError::Inference(e.to_string()))
    }

    /// Greedily continue a prompt.
    ///
    /// Returns the generated token IDs without the prompt. The sequence
    /// ends with an end token when the model produced one and holds at
    /// most `max_new_tokens` tokens in every case. `observe` sees the
    /// next-token logits of every step, before any penalty.
    ///
    /// # Errors
    ///
    /// Returns [`CoreError::InvalidInput`] for an empty prompt,
    /// [`CoreError::Timeout`] if `deadline` passes, and
    /// [`CoreError::Inference`] if the graph fails or returns a tensor of
    /// an unexpected shape.
    fn generate(
        &mut self,
        prepared: &PreparedInput,
        deadline: Option<Instant>,
        max_new_tokens: usize,
        observe: &mut dyn FnMut(&[f32]),
    ) -> Result<Vec<u32>, CoreError> {
        let prompt_len = prepared.sequence_length;
        if prompt_len == 0 {
            return Err(CoreError::InvalidInput(
                "text generation prompt produced no tokens".to_string(),
            ));
        }
        let started = Instant::now();
        let Self {
            session,
            io,
            settings,
            kv_cache,
            empty_past,
            ..
        } = self;
        let kv_cache = *kv_cache;

        let capacity = prompt_len + max_new_tokens;
        let mut tokens: Vec<i64> = Vec::with_capacity(capacity);
        tokens.extend_from_slice(&prepared.input_ids);
        // Every position is attended to; positions count from zero.
        let ones = vec![1_i64; capacity];
        let positions: Vec<i64> = (0..).take(capacity).collect();
        // Cache tensors in slot order, and the tokens they cover.
        let mut cache: Vec<DynValue> = Vec::new();
        let mut cached_len = 0_usize;
        let mut banned = Vec::new();
        let mut penalized = Vec::new();
        let mut first_step = None;

        for step in 0..max_new_tokens {
            ensure_before(deadline, step)?;
            // With a cache, feed only what it does not cover yet (one
            // token after the first step). Without, feed everything.
            let fed_from = if kv_cache { cached_len } else { 0 };
            let fed = &tokens[fed_from..];

            let mut inputs: Vec<(Cow<'_, str>, SessionInputValue<'_>)> =
                Vec::with_capacity(3 + io.past.len());
            inputs.push((
                Cow::Borrowed("input_ids"),
                int64_tensor(fed.len(), fed)?.into(),
            ));
            if io.feeds_attention_mask {
                inputs.push((
                    Cow::Borrowed("attention_mask"),
                    int64_tensor(tokens.len(), &ones[..tokens.len()])?.into(),
                ));
            }
            if io.feeds_position_ids {
                inputs.push((
                    Cow::Borrowed("position_ids"),
                    int64_tensor(fed.len(), &positions[fed_from..tokens.len()])?.into(),
                ));
            }
            for (index, slot) in io.past.iter().enumerate() {
                let value: SessionInputValue<'_> = if fed_from > 0 {
                    (&cache[index]).into()
                } else {
                    (&empty_past[index]).into()
                };
                inputs.push((Cow::Borrowed(slot.input.as_str()), value));
            }

            let mut outputs = session
                .run(inputs)
                .map_err(|e| CoreError::Inference(format!("text generation run failed: {e}")))?;
            let logits = outputs.get("logits").ok_or_else(|| {
                CoreError::Inference("graph did not return output 'logits'".to_string())
            })?;
            let (shape, data) = logits
                .try_extract_tensor::<f32>()
                .map_err(|e| CoreError::Inference(e.to_string()))?;
            let vocab_size = match shape.iter().copied().collect::<Vec<i64>>().as_slice() {
                [1, _, vocab] => usize::try_from(*vocab).ok().filter(|&size| size > 0),
                _ => None,
            }
            .ok_or_else(|| {
                CoreError::Inference(format!(
                    "expected logits of shape (1, seq, vocab), got {shape:?}",
                ))
            })?;
            // Logits of the last position predict the next token.
            let last_position = data.len().checked_sub(vocab_size).ok_or_else(|| {
                CoreError::Inference(format!(
                    "logits hold {} values, fewer than the vocabulary size {vocab_size}",
                    data.len(),
                ))
            })?;
            let next_logits = &data[last_position..];
            observe(next_logits);

            banned.clear();
            extend_with_repeating_tokens(&mut banned, &tokens, settings.no_repeat_ngram_size);
            let token = if settings.repetition_penalty == 1.0 {
                argmax_excluding(next_logits, &banned)?
            } else {
                penalized.clear();
                penalized.extend_from_slice(next_logits);
                penalize_repeats(&mut penalized, &tokens, settings.repetition_penalty);
                argmax_excluding(&penalized, &banned)?
            };
            let token = i64::try_from(token).map_err(|_| {
                CoreError::Inference(format!("token index {token} does not fit a token ID"))
            })?;

            if kv_cache {
                cache.clear();
                for slot in &io.past {
                    let value = outputs.remove(slot.output.as_str()).ok_or_else(|| {
                        CoreError::Inference(format!(
                            "graph did not return output '{}'",
                            slot.output
                        ))
                    })?;
                    cache.push(value);
                }
                cached_len = tokens.len();
            }
            drop(outputs);
            first_step.get_or_insert_with(|| started.elapsed());

            tokens.push(token);
            if settings.eos_token_ids.contains(&token) {
                break;
            }
        }

        let generated = &tokens[prompt_len..];
        let prompt_elapsed = first_step.unwrap_or_default();
        tracing::debug!(
            prompt_tokens = prompt_len,
            generated_tokens = generated.len(),
            reached_token_limit = generated.len() == max_new_tokens,
            prompt_ms = prompt_elapsed.as_secs_f64() * 1000.0,
            decode_ms = (started.elapsed() - prompt_elapsed).as_secs_f64() * 1000.0,
            "text generation finished"
        );

        generated
            .iter()
            .map(|&token| {
                u32::try_from(token).map_err(|_| {
                    CoreError::Inference(format!("invalid token ID {token} in generated text"))
                })
            })
            .collect()
    }
}

/// Cut the end off the user text so the prompt loses `excess` tokens.
///
/// `prompt` is the rendered prompt and `offsets` the byte span of each of
/// its tokens. The user text is located in the prompt (its last
/// occurrence: the user message comes after the system message), and the
/// cut is made where the first surplus token of that text begins. Returns
/// `None` when the text cannot be found in the prompt -- the template
/// rewrote it -- or holds too few tokens to make up the excess.
fn shorten_user_text(
    prompt: &str,
    offsets: &[(usize, usize)],
    text: &str,
    excess: usize,
) -> Option<String> {
    // A template may trim the message; look for the trimmed text then.
    let (needle, start) = match prompt.rfind(text) {
        Some(start) if !text.is_empty() => (text, start),
        _ => {
            let trimmed = text.trim();
            (trimmed, prompt.rfind(trimmed).filter(|_| !trimmed.is_empty())?)
        }
    };
    let end = start + needle.len();
    let inside: Vec<usize> = offsets
        .iter()
        .filter(|(token_start, token_end)| *token_start >= start && *token_end <= end)
        .map(|(token_start, _)| *token_start)
        .collect();
    // At least one token of the user text must remain.
    let keep = inside.len().checked_sub(excess).filter(|keep| *keep > 0)?;
    let mut cut = inside[keep] - start;
    while !needle.is_char_boundary(cut) {
        cut -= 1;
    }
    Some(needle[..cut].to_string()).filter(|shortened| !shortened.is_empty())
}

impl Pipeline for TextGenerationPipeline {
    type Input = TextGenerationInput;
    type Prepared = PreparedInput;
    type Output = String;

    fn prepare(&self, input: TextGenerationInput) -> Result<PreparedInput, CoreError> {
        self.prepare_prompt(input)
    }

    fn execute(&mut self, prepared: PreparedInput) -> Result<String, CoreError> {
        let deadline = self.deadline_from_now();
        self.generate_text(&prepared, deadline)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use serde_json::json;

    fn spec(name: &str, element_type: TensorElementType, dims: &[i64]) -> TensorSpec {
        TensorSpec {
            name: name.to_string(),
            element_type: Some(element_type),
            dims: dims.to_vec(),
        }
    }

    fn int64(name: &str) -> TensorSpec {
        spec(name, TensorElementType::Int64, &[-1, -1])
    }

    fn cache(name: &str) -> TensorSpec {
        spec(name, TensorElementType::Float32, &[-1, 3, -1, 64])
    }

    fn logits() -> TensorSpec {
        spec("logits", TensorElementType::Float32, &[-1, -1, 49152])
    }

    /// Inputs and outputs of an Optimum `text-generation-with-past` export
    /// with one layer (names as on a real SmolLM2 export).
    fn optimum_specs() -> (Vec<TensorSpec>, Vec<TensorSpec>) {
        (
            vec![
                int64("input_ids"),
                int64("attention_mask"),
                int64("position_ids"),
                cache("past_key_values.0.key"),
                cache("past_key_values.0.value"),
            ],
            vec![logits(), cache("present.0.key"), cache("present.0.value")],
        )
    }

    #[test]
    fn optimum_export_is_wired_by_name() {
        // Arrange
        let (inputs, outputs) = optimum_specs();

        // Act
        let io = GraphIo::from_specs(&inputs, &outputs, &json!({})).expect("wiring");

        // Assert
        assert!(io.feeds_attention_mask);
        assert!(io.feeds_position_ids);
        assert_eq!(
            io.past,
            vec![
                PastSlot {
                    input: "past_key_values.0.key".to_string(),
                    output: "present.0.key".to_string(),
                    empty_dims: [1, 3, 0, 64],
                },
                PastSlot {
                    input: "past_key_values.0.value".to_string(),
                    output: "present.0.value".to_string(),
                    empty_dims: [1, 3, 0, 64],
                },
            ]
        );
    }

    #[test]
    fn graph_without_cache_or_position_ids_is_accepted() {
        // Arrange -- a plain `text-generation` export of a GPT-2 style model.
        let inputs = vec![int64("input_ids"), int64("attention_mask")];

        // Act
        let io = GraphIo::from_specs(&inputs, &[logits()], &json!({})).expect("wiring");

        // Assert
        assert!(io.past.is_empty());
        assert!(!io.feeds_position_ids);
    }

    #[test]
    fn unsupported_graphs_are_rejected_with_the_reason() {
        let (inputs, outputs) = optimum_specs();
        let with_input = |extra: TensorSpec| {
            let mut inputs = inputs.clone();
            inputs.push(extra);
            inputs
        };
        let cases: Vec<(&str, Vec<TensorSpec>, Vec<TensorSpec>, &str)> = vec![
            (
                "legacy merged decoder",
                with_input(spec("use_cache_branch", TensorElementType::Bool, &[1])),
                outputs.clone(),
                "use_cache_branch",
            ),
            (
                "embeddings input",
                with_input(spec("inputs_embeds", TensorElementType::Float32, &[-1, -1, 8])),
                outputs.clone(),
                "inputs_embeds",
            ),
            (
                "half-precision cache",
                with_input(spec("past_key_values.1.key", TensorElementType::Float16, &[-1, 3, -1, 64])),
                outputs.clone(),
                "Float16",
            ),
            (
                "cache input without output",
                with_input(cache("past_key_values.1.key")),
                outputs.clone(),
                "present.1.key",
            ),
            ("no token input", inputs[1..].to_vec(), outputs.clone(), "input_ids"),
            ("no logits", inputs.clone(), outputs[1..].to_vec(), "logits"),
        ];

        for (name, inputs, outputs, needle) in cases {
            // Act
            let result = GraphIo::from_specs(&inputs, &outputs, &json!({}));

            // Assert
            let err = result.err().unwrap_or_else(|| panic!("{name} should be rejected"));
            assert!(matches!(err, CoreError::ModelValidation(_)), "{name}: {err:?}");
            assert!(err.to_string().contains(needle), "{name}: {err}");
        }
    }

    #[test]
    fn dynamic_cache_dimensions_come_from_the_config() {
        // Arrange -- a graph that leaves heads and head size dynamic.
        let dynamic = spec("past_key_values.0.key", TensorElementType::Float32, &[-1, -1, -1, -1]);
        let llama = json!({ "hidden_size": 576, "num_attention_heads": 9, "num_key_value_heads": 3 });
        let gpt2 = json!({ "n_embd": 768, "n_head": 12 });

        // Act / Assert -- grouped-query attention caches the KV heads only.
        assert_eq!(empty_cache_dims(&dynamic, &llama).expect("llama"), [1, 3, 0, 64]);
        assert_eq!(empty_cache_dims(&dynamic, &gpt2).expect("gpt2"), [1, 12, 0, 64]);
        assert!(matches!(
            empty_cache_dims(&dynamic, &json!({})),
            Err(CoreError::ModelValidation(_))
        ));
        let three_dims = spec("past_key_values.0.key", TensorElementType::Float32, &[-1, -1, 64]);
        assert!(empty_cache_dims(&three_dims, &llama).is_err());
    }

    #[test]
    fn generation_settings_prefer_the_generation_config() {
        // Arrange -- Qwen-style: several end tokens and sampling settings.
        let config = json!({ "eos_token_id": 7, "vocab_size": 100 });
        let generation = json!({
            "do_sample": true,
            "eos_token_id": [5, 6],
            "repetition_penalty": 1.05,
            "temperature": 0.7,
            "top_p": 0.8,
            "max_new_tokens": 64,
        });

        // Act
        let settings = GenerationSettings::resolve(&config, Some(&generation), None).expect("settings");

        // Assert
        assert_eq!(settings.eos_token_ids, vec![5, 6]);
        assert_eq!(settings.repetition_penalty, 1.05);
        assert_eq!(settings.no_repeat_ngram_size, 0);
        assert_eq!(settings.max_new_tokens, 64);
    }

    #[test]
    fn generation_settings_fall_back_to_the_model_config_and_defaults() {
        // Arrange -- `max_length` and `max_new_tokens` in config.json do
        // not count: they default to 20 total tokens.
        let config = json!({ "eos_token_id": 2, "max_length": 20, "max_new_tokens": 5 });

        // Act
        let from_config = GenerationSettings::resolve(&config, None, None).expect("settings");
        let explicit = GenerationSettings::resolve(&config, None, Some(1000)).expect("settings");
        let no_eos = GenerationSettings::resolve(&json!({}), None, None).expect("settings");

        // Assert
        assert_eq!(from_config.eos_token_ids, vec![2]);
        assert_eq!(from_config.max_new_tokens, DEFAULT_MAX_NEW_TOKENS);
        assert_eq!(from_config.repetition_penalty, 1.0);
        assert_eq!(explicit.max_new_tokens, 1000);
        assert!(no_eos.eos_token_ids.is_empty());
    }

    #[test]
    fn generation_settings_reject_unusable_values() {
        let limit = TEXT_GENERATION_MAX_NEW_TOKENS_LIMIT;
        // Explicit limits out of range are the operator's error.
        for tokens in [0, limit + 1] {
            let result = GenerationSettings::resolve(&json!({}), None, Some(tokens));
            assert!(matches!(result, Err(CoreError::Config(_))), "{tokens}");
        }
        // A model's own limit above the ceiling is clamped, not refused.
        let clamped =
            GenerationSettings::resolve(&json!({}), Some(&json!({ "max_new_tokens": 100_000 })), None)
                .expect("settings");
        assert_eq!(clamped.max_new_tokens, limit);
        // Malformed model values are the model's error.
        for generation in [
            json!({ "repetition_penalty": 0 }),
            json!({ "repetition_penalty": "high" }),
            json!({ "no_repeat_ngram_size": -1 }),
            json!({ "eos_token_id": -5 }),
            json!({ "eos_token_id": [2, "x"] }),
        ] {
            let result = GenerationSettings::resolve(&json!({}), Some(&generation), None);
            assert!(matches!(result, Err(CoreError::ModelValidation(_))), "{generation}");
        }
        let outside_vocab =
            GenerationSettings::resolve(&json!({ "vocab_size": 10, "eos_token_id": 10 }), None, None);
        assert!(matches!(outside_vocab, Err(CoreError::ModelValidation(_))));
    }

    #[test]
    fn input_cap_leaves_room_for_generation_in_the_context() {
        let context = json!({ "max_position_embeddings": 8192 });
        let small = json!({ "n_positions": 1024 });
        let none = json!({});

        // Default: 4096, lowered to what the context leaves.
        assert_eq!(resolve_max_input_tokens(&context, None, 256).expect("cap"), 4096);
        assert_eq!(resolve_max_input_tokens(&small, None, 256).expect("cap"), 768);
        assert_eq!(resolve_max_input_tokens(&none, None, 256).expect("cap"), 4096);
        // Explicit: taken as given when it fits.
        assert_eq!(resolve_max_input_tokens(&context, Some(7936), 256).expect("cap"), 7936);
        assert_eq!(resolve_max_input_tokens(&none, Some(20_000), 256).expect("cap"), 20_000);
        // Explicit: refused when it does not.
        for (config, cap, new) in [(&context, 7937, 256), (&small, 1024, 1), (&none, 15, 1)] {
            let result = resolve_max_input_tokens(config, Some(cap), new);
            assert!(matches!(result, Err(CoreError::Config(_))), "{cap}/{new}");
        }
        // No room left for any prompt.
        assert!(matches!(
            resolve_max_input_tokens(&small, None, 1020),
            Err(CoreError::Config(_))
        ));
    }

    #[test]
    fn repetition_penalty_follows_the_hugging_face_rule() {
        // Arrange -- token 1 occurs twice, token 9 is outside the logits.
        let mut logits = vec![2.0, 4.0, -3.0, 0.0, 1.0];
        let tokens = [1, 2, 1, 3, 9, -1];

        // Act
        penalize_repeats(&mut logits, &tokens, 2.0);

        // Assert -- positive halved, negative doubled, each token once.
        assert_eq!(logits, vec![2.0, 2.0, -6.0, 0.0, 1.0]);
    }

    #[test]
    fn user_text_is_cut_at_a_token_start() {
        // Arrange -- prompt "<sys> hello brave new world <end>", one token
        // per word; the user text spans tokens 1..=4.
        let prompt = "<sys> hello brave new world <end>";
        let offsets = [(0, 5), (6, 11), (12, 17), (18, 21), (22, 27), (28, 33)];
        let text = "hello brave new world";

        // Act / Assert
        assert_eq!(
            shorten_user_text(prompt, &offsets, text, 1).as_deref(),
            Some("hello brave new ")
        );
        assert_eq!(shorten_user_text(prompt, &offsets, text, 3).as_deref(), Some("hello "));
        // Nothing of the text would remain: not something to cut.
        assert_eq!(shorten_user_text(prompt, &offsets, text, 4), None);
        assert_eq!(shorten_user_text(prompt, &offsets, text, 9), None);
    }

    #[test]
    fn user_text_is_found_after_an_equal_system_message_and_when_trimmed() {
        // Arrange -- the same words as system and user message: the cut
        // belongs to the later one. Offsets: one token per word.
        let prompt = "a b | a b";
        let offsets = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)];

        // Act / Assert
        assert_eq!(shorten_user_text(prompt, &offsets, "a b", 1).as_deref(), Some("a "));
        // A template that trims the message still allows the cut.
        assert_eq!(shorten_user_text(prompt, &offsets, "  a b \n", 1).as_deref(), Some("a "));
        // A template that rewrites the message does not.
        assert_eq!(shorten_user_text(prompt, &offsets, "x y", 1), None);
        assert_eq!(shorten_user_text(prompt, &offsets, "", 1), None);
    }

    #[test]
    fn cut_never_splits_a_character() {
        // Arrange -- byte-level tokens can start inside a multi-byte
        // character: "é" is two bytes, split across tokens 1 and 2.
        let prompt = "aé b";
        let offsets = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)];

        // Act
        let shortened = shorten_user_text(prompt, &offsets, prompt, 3);

        // Assert -- the cut moves back to the start of "é".
        assert_eq!(shortened.as_deref(), Some("a"));
    }

    #[test]
    fn options_default_changes_nothing() {
        let options = TextGenerationOptions::new();
        assert_eq!(options, TextGenerationOptions::default());
        assert_eq!(options.max_input_tokens, None);
        assert_eq!(options.max_new_tokens, None);
        assert_eq!(options.system_prompt, None);
        // An empty system prompt is no system prompt.
        assert_eq!(TextGenerationOptions::new().with_system_prompt(""), options);
    }

    /// Token-for-token parity with a Hugging Face reference on a real model.
    ///
    /// Needs an exported model and a cases file, so it never runs by
    /// default. Point `HEPHAESTUS_TEXTGEN_PARITY_MODEL_DIR` at an Optimum
    /// `text-generation-with-past` export and
    /// `HEPHAESTUS_TEXTGEN_PARITY_CASES` at a JSON file of the form
    /// `{"max_new_tokens": N, "cases": [{"name", "system", "text",
    /// "prompt", "input_ids", "token_ids"}]}`, where `prompt` and
    /// `input_ids` come from `tokenizer.apply_chat_template(...,
    /// add_generation_prompt=True)` and `token_ids` are the tokens
    /// `generate(do_sample=False)` appended. Then run:
    ///
    /// `cargo test -p hephaestus-core --release --lib text_generation_matches_reference -- --ignored --nocapture`
    ///
    /// For every case the rendered prompt, the prompt token IDs and the
    /// generated token IDs must equal the reference. The first
    /// `HEPHAESTUS_TEXTGEN_PARITY_UNCACHED_STEPS` steps (default 8) are
    /// also run without the key/value cache: tokens must be equal and the
    /// largest logit difference between the two loops is printed.
    #[test]
    #[ignore = "requires an exported model and reference cases (see doc comment)"]
    fn text_generation_matches_reference_cases() {
        // Arrange
        let model_dir = std::env::var("HEPHAESTUS_TEXTGEN_PARITY_MODEL_DIR")
            .expect("HEPHAESTUS_TEXTGEN_PARITY_MODEL_DIR must be set");
        let cases_path = std::env::var("HEPHAESTUS_TEXTGEN_PARITY_CASES")
            .expect("HEPHAESTUS_TEXTGEN_PARITY_CASES must be set");
        let uncached_steps: usize = std::env::var("HEPHAESTUS_TEXTGEN_PARITY_UNCACHED_STEPS")
            .map_or(8, |steps| steps.parse().expect("step count"));
        let cases: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(cases_path).expect("read cases"))
                .expect("parse cases");
        let max_new_tokens =
            usize::try_from(cases["max_new_tokens"].as_u64().expect("max_new_tokens"))
                .expect("limit fits usize");
        let ids = |value: &serde_json::Value| -> Vec<i64> {
            value
                .as_array()
                .expect("id list")
                .iter()
                .map(|id| id.as_i64().expect("integer id"))
                .collect()
        };
        let load = |kv_cache: bool| {
            let options = TextGenerationOptions::new()
                .with_max_new_tokens(max_new_tokens)
                .with_kv_cache(kv_cache);
            TextGenerationPipeline::with_options(Path::new(&model_dir), &ExecutionProvider::Cpu, &options)
                .expect("load")
        };
        let mut cached = load(true);
        let mut uncached = load(false);
        assert!(cached.uses_kv_cache(), "export must have a key/value cache");

        for case in cases["cases"].as_array().expect("cases array") {
            let name = case["name"].as_str().expect("case name");
            let input = TextGenerationInput {
                text: case["text"].as_str().expect("case text").to_string(),
                system: case["system"].as_str().map(str::to_string),
            };

            // Act
            let system = input.system.clone();
            let prompt = cached
                .template
                .render(system.as_deref(), &input.text)
                .expect("render");
            let prepared = cached.prepare(input).expect("prepare");
            let mut cached_logits: Vec<Vec<f32>> = Vec::new();
            let started = Instant::now();
            let generated = cached
                .generate(&prepared, None, max_new_tokens, &mut |logits| {
                    if cached_logits.len() < uncached_steps {
                        cached_logits.push(logits.to_vec());
                    }
                })
                .expect("generate");
            let cached_ms = started.elapsed().as_secs_f64() * 1000.0;
            let steps = uncached_steps.min(generated.len());
            let mut max_logit_diff = 0.0_f32;
            let mut step = 0;
            let started = Instant::now();
            let without_cache = uncached
                .generate(&prepared, None, steps, &mut |logits| {
                    for (a, b) in logits.iter().zip(&cached_logits[step]) {
                        max_logit_diff = max_logit_diff.max((a - b).abs());
                    }
                    step += 1;
                })
                .expect("generate without cache");
            let uncached_ms = started.elapsed().as_secs_f64() * 1000.0;
            let text = cached.tokenizer.decode(&generated, true).expect("decode");
            println!(
                "{name}: {} prompt tokens, {} generated in {cached_ms:.0} ms; first {steps} steps \
                 without cache {uncached_ms:.0} ms, max |logit difference| {max_logit_diff:.2e}",
                prepared.sequence_length,
                generated.len(),
            );
            println!("{name}: {text}");

            // Assert
            assert_eq!(prompt, case["prompt"].as_str().expect("prompt"), "{name}: rendered prompt differs");
            assert_eq!(prepared.input_ids, ids(&case["input_ids"]), "{name}: prompt token IDs differ");
            let generated: Vec<i64> = generated.into_iter().map(i64::from).collect();
            assert_eq!(generated, ids(&case["token_ids"]), "{name}: generated token IDs differ");
            let without_cache: Vec<i64> = without_cache.into_iter().map(i64::from).collect();
            assert_eq!(without_cache, generated[..steps], "{name}: uncached token IDs differ");
            if let Some(reference_text) = case["reference_text"].as_str() {
                assert_eq!(text, reference_text, "{name}: decoded text differs");
            }
        }
    }
}
