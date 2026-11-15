//! Encoder-decoder seq2seq mode: greedy autoregressive decoding.
//!
//! Serves the standard Optimum export of a text seq2seq model:
//! `encoder_model.onnx` plus `decoder_model_merged.onnx` and/or
//! `decoder_model.onnx`. The encoder runs once per input; the decoder then
//! runs once per generated token.
//!
//! With a merged decoder the loop keeps a key/value cache: the first step
//! runs the graph's uncached branch over the start tokens and keeps its
//! `present.*` outputs, and every later step feeds only the newest token
//! plus those tensors through the cached branch. Without a merged decoder
//! (or with the cache switched off) every step re-reads all tokens
//! generated so far. Both loops pick the same tokens.
//!
//! Everything a specific model could get wrong is checked when the
//! pipeline is constructed -- file layout, graph input/output names and
//! element types, generation settings -- so a misfit fails at startup with
//! a message naming the problem, not at the first request.

use std::borrow::Cow;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use ort::session::{Session, SessionInputValue};
use ort::value::{DynValue, Outlet, TensorElementType, TensorRef};
use tokenizers::Tokenizer;

use crate::ep::ExecutionProvider;
use crate::error::CoreError;

use super::seq2seq::Seq2SeqOptions;
use super::{PreparedInput, build_onnx_session};

/// Encoder graph file name in an Optimum seq2seq export.
const ENCODER_FILE: &str = "encoder_model.onnx";
/// Decoder graph without key/value cache inputs (the uncached loop).
const DECODER_FILE: &str = "decoder_model.onnx";
/// Decoder graph holding both the cached and uncached branches.
const MERGED_DECODER_FILE: &str = "decoder_model_merged.onnx";

/// Input token cap used when [`Seq2SeqOptions`] does not set one.
///
/// Equal to the cap every other text pipeline applies, so an unset option
/// behaves like the rest of the runtime.
pub(super) const DEFAULT_MAX_INPUT_TOKENS: usize = 512;

/// Smallest accepted input token cap.
///
/// The tokenizer subtracts the special tokens it adds from the cap, so a
/// cap below that count cannot hold any text.
pub const SEQ2SEQ_MIN_INPUT_TOKENS: usize = 8;

/// Absolute ceiling on generated tokens per input, whatever the model or
/// the operator asks for.
pub const SEQ2SEQ_MAX_NEW_TOKENS_LIMIT: usize = 4096;

/// `max_length` assumed when neither `generation_config.json` nor
/// `config.json` declares one. Matches the Hugging Face default.
const DEFAULT_MAX_LENGTH: usize = 20;

/// Decoder inputs that carry the encoder output, in lookup order.
const ENCODER_HIDDEN_INPUT_NAMES: [&str; 3] = [
    "encoder_hidden_states",
    "encoder_output",
    "encoder_last_hidden_state",
];

/// Decoder inputs that carry the tokens generated so far, in lookup order.
const DECODER_TOKEN_INPUT_NAMES: [&str; 2] = ["input_ids", "decoder_input_ids"];

/// Prefix of the key/value cache inputs of a merged decoder.
const PAST_INPUT_PREFIX: &str = "past_key_values.";
/// Prefix of the key/value cache outputs; the rest of the name matches
/// the input it feeds on the next step.
const PRESENT_OUTPUT_PREFIX: &str = "present.";
/// Marks a cache tensor that holds encoder cross-attention state.
const ENCODER_CACHE_MARKER: &str = ".encoder.";

/// Paths of the graphs that make up an encoder-decoder export.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct EncoderDecoderFiles {
    /// Path to `encoder_model.onnx`.
    encoder: PathBuf,
    /// Path to the decoder graph for the uncached loop, next to the
    /// encoder: `decoder_model.onnx`, else `decoder_model_merged.onnx`.
    decoder: PathBuf,
    /// Path to `decoder_model_merged.onnx` when it exists next to the
    /// encoder. The cached loop runs this graph.
    merged_decoder: Option<PathBuf>,
}

impl EncoderDecoderFiles {
    /// Locate an encoder-decoder export inside a model directory.
    ///
    /// Looks in `{model_dir}/onnx/` first and then in `{model_dir}/`, the
    /// same order the single-file lookup uses. Both graphs must sit in the
    /// same directory. For the uncached loop `decoder_model.onnx` is
    /// preferred over `decoder_model_merged.onnx`: without a key/value
    /// cache the plain decoder is exactly the graph that is needed, while
    /// the merged one only adds a second, unused branch and dummy cache
    /// inputs. A merged decoder in that directory is reported separately,
    /// for the cached loop.
    ///
    /// Returns `Ok(None)` when no encoder file exists, so the caller can
    /// report the missing model the way it always has.
    ///
    /// # Errors
    ///
    /// Returns [`CoreError::ModelLoad`] if an encoder exists but no decoder
    /// sits next to it.
    pub(super) fn locate(model_dir: &Path) -> Result<Option<Self>, CoreError> {
        let candidates = [model_dir.join("onnx"), model_dir.to_path_buf()];
        let mut encoder_without_decoder = None;

        for dir in candidates {
            let encoder = dir.join(ENCODER_FILE);
            if !encoder.exists() {
                continue;
            }
            let decoder = [DECODER_FILE, MERGED_DECODER_FILE]
                .into_iter()
                .map(|name| dir.join(name))
                .find(|path| path.exists());
            match decoder {
                Some(decoder) => {
                    let merged_decoder = Some(dir.join(MERGED_DECODER_FILE)).filter(|p| p.exists());
                    return Ok(Some(Self {
                        encoder,
                        decoder,
                        merged_decoder,
                    }));
                }
                None => encoder_without_decoder = Some(encoder),
            }
        }

        match encoder_without_decoder {
            Some(encoder) => Err(CoreError::ModelLoad(format!(
                "found encoder '{}' but neither '{DECODER_FILE}' nor '{MERGED_DECODER_FILE}' next to it",
                encoder.display(),
            ))),
            None => Ok(None),
        }
    }
}

/// Name, element type and declared shape of one graph input or output.
///
/// A plain copy of what the ONNX session reports, so the name rules below
/// can be tested without loading a model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct TensorSpec {
    pub(super) name: String,
    /// `None` when the value is not a tensor (a sequence or a map).
    pub(super) element_type: Option<TensorElementType>,
    /// Declared dimensions; `-1` marks a dynamic dimension.
    pub(super) dims: Vec<i64>,
}

impl TensorSpec {
    fn from_outlet(outlet: &Outlet) -> Self {
        let dtype = outlet.dtype();
        Self {
            name: outlet.name().to_string(),
            element_type: dtype.tensor_type(),
            dims: dtype
                .tensor_shape()
                .map(|shape| shape.iter().copied().collect())
                .unwrap_or_default(),
        }
    }

    pub(super) fn from_outlets(outlets: &[Outlet]) -> Vec<Self> {
        outlets.iter().map(Self::from_outlet).collect()
    }
}

/// List the names of a set of graph inputs or outputs for an error message.
pub(super) fn spec_names(specs: &[TensorSpec]) -> Vec<&str> {
    specs.iter().map(|spec| spec.name.as_str()).collect()
}

/// Require that a graph input or output has the element type this mode feeds.
fn require_element_type(
    graph: &str,
    spec: &TensorSpec,
    expected: TensorElementType,
) -> Result<(), CoreError> {
    if spec.element_type == Some(expected) {
        return Ok(());
    }
    Err(CoreError::ModelValidation(format!(
        "{graph} tensor '{}' has element type {:?}, but encoder-decoder mode needs {expected:?}",
        spec.name, spec.element_type,
    )))
}

/// How to feed the encoder graph and where to read its result.
#[derive(Debug, Clone, PartialEq, Eq)]
struct EncoderIo {
    /// Whether the encoder declares an `attention_mask` input.
    feeds_attention_mask: bool,
    /// Name of the hidden-state output handed to the decoder.
    hidden_output: String,
}

impl EncoderIo {
    /// Derive the encoder wiring from the graph's declared inputs and outputs.
    ///
    /// # Errors
    ///
    /// Returns [`CoreError::ModelValidation`] if the encoder has no
    /// `input_ids` input, declares an input this mode cannot feed (for
    /// example the `input_features` of a speech model), has no outputs, or
    /// uses element types other than int64 inputs and a float32 output.
    fn from_specs(inputs: &[TensorSpec], outputs: &[TensorSpec]) -> Result<Self, CoreError> {
        let mut has_input_ids = false;
        let mut feeds_attention_mask = false;
        for spec in inputs {
            match spec.name.as_str() {
                "input_ids" => has_input_ids = true,
                "attention_mask" => feeds_attention_mask = true,
                other => {
                    return Err(CoreError::ModelValidation(format!(
                        "encoder input '{other}' is not supported by encoder-decoder mode; \
                         encoder inputs are: {:?}",
                        spec_names(inputs),
                    )));
                }
            }
            require_element_type("encoder", spec, TensorElementType::Int64)?;
        }
        if !has_input_ids {
            return Err(CoreError::ModelValidation(format!(
                "encoder does not accept input 'input_ids'; encoder inputs are: {:?}",
                spec_names(inputs),
            )));
        }

        let hidden = outputs
            .iter()
            .find(|spec| spec.name == "last_hidden_state")
            .or_else(|| outputs.first())
            .ok_or_else(|| {
                CoreError::ModelValidation("encoder declares no output tensors".to_string())
            })?;
        require_element_type("encoder", hidden, TensorElementType::Float32)?;

        Ok(Self {
            feeds_attention_mask,
            hidden_output: hidden.name.clone(),
        })
    }
}

/// A key/value cache input of a merged decoder.
///
/// The uncached branch ignores the values, but ONNX Runtime still needs a
/// tensor of a valid shape for every declared input, so it is fed zeros.
/// The cached branch is fed the matching `present.*` output of an earlier
/// step.
#[derive(Debug, Clone, PartialEq, Eq)]
struct PastInput {
    name: String,
    /// Declared shape with every dynamic dimension replaced by 1.
    dims: Vec<usize>,
    /// Name of the float32 output that feeds this input on the next step
    /// (`present.<rest of the input name>`); `None` when the graph has none.
    present_output: Option<String>,
    /// Whether this is encoder cross-attention state. It does not change
    /// between steps, and the cached branch of an Optimum merged decoder
    /// returns an empty tensor for it, so the first step's tensor is kept.
    from_encoder: bool,
}

impl PastInput {
    fn element_count(&self) -> usize {
        self.dims.iter().product()
    }
}

/// Extra inputs of a merged decoder.
#[derive(Debug, Clone, PartialEq, Eq)]
struct MergedInputs {
    /// Name of the boolean branch selector.
    use_cache_branch: String,
    /// Key/value cache inputs.
    past: Vec<PastInput>,
}

impl MergedInputs {
    /// Whether the graph can run the cached loop: it has cache inputs and
    /// returns a tensor to feed back into every one of them.
    fn supports_cache(&self) -> bool {
        !self.past.is_empty() && self.past.iter().all(|past| past.present_output.is_some())
    }
}

/// How to feed the decoder graph and where to read its result.
#[derive(Debug, Clone, PartialEq, Eq)]
struct DecoderIo {
    /// Input receiving the tokens generated so far.
    tokens_input: String,
    /// Input receiving the encoder hidden states.
    encoder_hidden_input: String,
    /// Input receiving the encoder attention mask, when declared.
    encoder_mask_input: Option<String>,
    /// Present when the graph is a merged decoder.
    merged: Option<MergedInputs>,
    /// Output holding next-token logits.
    logits_output: String,
}

impl DecoderIo {
    /// Derive the decoder wiring from the graph's declared inputs and outputs.
    ///
    /// # Errors
    ///
    /// Returns [`CoreError::ModelValidation`] if a required input or the
    /// `logits` output is missing, if the graph declares an input this mode
    /// cannot feed, if it has cache inputs without a `use_cache_branch`
    /// selector (a cache-only decoder), or if an element type does not
    /// match what this mode feeds.
    fn from_specs(inputs: &[TensorSpec], outputs: &[TensorSpec]) -> Result<Self, CoreError> {
        let find = |names: &[&str]| {
            names
                .iter()
                .find_map(|name| inputs.iter().find(|spec| spec.name == *name))
        };

        let tokens = find(&DECODER_TOKEN_INPUT_NAMES).ok_or_else(|| {
            CoreError::ModelValidation(format!(
                "decoder must accept one of {DECODER_TOKEN_INPUT_NAMES:?}; decoder inputs are: {:?}",
                spec_names(inputs),
            ))
        })?;
        require_element_type("decoder", tokens, TensorElementType::Int64)?;

        let hidden = find(&ENCODER_HIDDEN_INPUT_NAMES).ok_or_else(|| {
            CoreError::ModelValidation(format!(
                "decoder must accept one of {ENCODER_HIDDEN_INPUT_NAMES:?}; decoder inputs are: {:?}",
                spec_names(inputs),
            ))
        })?;
        require_element_type("decoder", hidden, TensorElementType::Float32)?;

        let mut encoder_mask_input = None;
        let mut use_cache_branch = None;
        let mut past = Vec::new();
        for spec in inputs {
            let name = spec.name.as_str();
            if name == tokens.name || name == hidden.name {
                continue;
            }
            if name == "encoder_attention_mask" {
                require_element_type("decoder", spec, TensorElementType::Int64)?;
                encoder_mask_input = Some(spec.name.clone());
            } else if name == "use_cache_branch" {
                require_element_type("decoder", spec, TensorElementType::Bool)?;
                use_cache_branch = Some(spec.name.clone());
            } else if let Some(slot) = name.strip_prefix(PAST_INPUT_PREFIX) {
                require_element_type("decoder", spec, TensorElementType::Float32)?;
                let present = format!("{PRESENT_OUTPUT_PREFIX}{slot}");
                past.push(PastInput {
                    name: spec.name.clone(),
                    present_output: outputs
                        .iter()
                        .any(|output| {
                            output.name == present
                                && output.element_type == Some(TensorElementType::Float32)
                        })
                        .then_some(present),
                    from_encoder: name.contains(ENCODER_CACHE_MARKER),
                    dims: spec
                        .dims
                        .iter()
                        .map(|&dim| usize::try_from(dim).ok().filter(|&d| d > 0).unwrap_or(1))
                        .collect(),
                });
            } else {
                return Err(CoreError::ModelValidation(format!(
                    "decoder input '{name}' is not supported by encoder-decoder mode; \
                     decoder inputs are: {:?}",
                    spec_names(inputs),
                )));
            }
        }

        let merged = match (use_cache_branch, past.is_empty()) {
            (Some(use_cache_branch), _) => Some(MergedInputs {
                use_cache_branch,
                past,
            }),
            (None, true) => None,
            (None, false) => {
                return Err(CoreError::ModelValidation(format!(
                    "decoder needs key/value cache inputs but has no 'use_cache_branch' selector, \
                     so it cannot run without a cache; decoder inputs are: {:?}",
                    spec_names(inputs),
                )));
            }
        };

        let logits = outputs
            .iter()
            .find(|spec| spec.name == "logits")
            .ok_or_else(|| {
                CoreError::ModelValidation(format!(
                    "decoder does not produce output 'logits'; decoder outputs are: {:?}",
                    spec_names(outputs),
                ))
            })?;
        require_element_type("decoder", logits, TensorElementType::Float32)?;

        Ok(Self {
            tokens_input: tokens.name.clone(),
            encoder_hidden_input: hidden.name.clone(),
            encoder_mask_input,
            merged,
            logits_output: logits.name.clone(),
        })
    }
}

/// Generation settings resolved from the model's own configuration files.
///
/// Each value is taken from `generation_config.json` when that file
/// declares it and from `config.json` otherwise.
#[derive(Debug, Clone, PartialEq, Eq)]
struct GenerationSettings {
    /// First decoder token; never part of the output.
    decoder_start_token_id: i64,
    /// Tokens that end generation. Empty when the model declares none.
    eos_token_ids: Vec<i64>,
    /// Padding token. Read for completeness; an unbatched greedy loop
    /// never pads.
    pad_token_id: Option<i64>,
    /// Token forced as the first generated token.
    forced_bos_token_id: Option<i64>,
    /// Token forced as the last token when the length limit is reached.
    forced_eos_token_id: Option<i64>,
    /// No sequence of this many tokens may occur twice; 0 disables the rule.
    no_repeat_ngram_size: usize,
    /// Hard upper bound on generated tokens (start token excluded).
    max_new_tokens: usize,
}

impl GenerationSettings {
    /// Resolve generation settings for a model.
    ///
    /// `max_new_tokens` is the operator's explicit limit, if any. Without
    /// one the limit is the model's `max_new_tokens`, else its
    /// `max_length - 1` (`max_length` counts the start token), else the
    /// Hugging Face default. A derived limit is clamped to the decoder's
    /// position limit and to [`SEQ2SEQ_MAX_NEW_TOKENS_LIMIT`].
    ///
    /// # Errors
    ///
    /// Returns [`CoreError::ModelValidation`] if no decoder start token can
    /// be determined, a token ID is not a valid vocabulary index,
    /// `no_repeat_ngram_size` is not a non-negative integer, or the length
    /// limit is zero. Returns [`CoreError::Config`] if the explicit limit
    /// is zero or exceeds what the model or the runtime allows.
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
        let token_id = |key: &str| -> Result<Option<i64>, CoreError> {
            lookup(key)
                .map(|value| parse_token_id(key, value, vocab_size))
                .transpose()
        };

        let decoder_start_token_id = match token_id("decoder_start_token_id")? {
            Some(id) => id,
            None => token_id("bos_token_id")?.ok_or_else(|| {
                CoreError::ModelValidation(
                    "model configuration declares neither 'decoder_start_token_id' nor \
                     'bos_token_id'"
                        .to_string(),
                )
            })?,
        };

        let eos_token_ids = match lookup("eos_token_id") {
            Some(serde_json::Value::Array(ids)) => ids
                .iter()
                .map(|value| parse_token_id("eos_token_id", value, vocab_size))
                .collect::<Result<Vec<i64>, CoreError>>()?,
            Some(value) => vec![parse_token_id("eos_token_id", value, vocab_size)?],
            None => Vec::new(),
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

        let position_limit = config
            .get("max_position_embeddings")
            .and_then(serde_json::Value::as_u64)
            .and_then(|limit| usize::try_from(limit).ok());
        // The start token occupies one decoder position.
        let ceiling = position_limit.map_or(SEQ2SEQ_MAX_NEW_TOKENS_LIMIT, |limit| {
            limit.saturating_sub(1).min(SEQ2SEQ_MAX_NEW_TOKENS_LIMIT)
        });

        let max_new_tokens = match max_new_tokens {
            Some(0) => {
                return Err(CoreError::Config(
                    "seq2seq max new tokens must be at least 1".to_string(),
                ));
            }
            Some(explicit) if explicit > ceiling => {
                return Err(CoreError::Config(format!(
                    "seq2seq max new tokens {explicit} exceeds the limit of {ceiling} for this model",
                )));
            }
            Some(explicit) => explicit,
            None => {
                let length = |key: &str| {
                    lookup(key)
                        .and_then(serde_json::Value::as_u64)
                        .and_then(|value| usize::try_from(value).ok())
                };
                let declared = length("max_new_tokens").unwrap_or_else(|| {
                    length("max_length")
                        .unwrap_or(DEFAULT_MAX_LENGTH)
                        .saturating_sub(1)
                });
                if declared == 0 {
                    return Err(CoreError::ModelValidation(
                        "model configuration leaves no room for generated tokens \
                         ('max_length' must be at least 2)"
                            .to_string(),
                    ));
                }
                declared.min(ceiling)
            }
        };

        Ok(Self {
            decoder_start_token_id,
            eos_token_ids,
            pad_token_id: token_id("pad_token_id")?,
            forced_bos_token_id: token_id("forced_bos_token_id")?,
            forced_eos_token_id: token_id("forced_eos_token_id")?,
            no_repeat_ngram_size,
            max_new_tokens,
        })
    }
}

/// Parse a token ID from a configuration value and bounds-check it.
pub(super) fn parse_token_id(
    key: &str,
    value: &serde_json::Value,
    vocab_size: Option<i64>,
) -> Result<i64, CoreError> {
    let id = value.as_i64().filter(|id| *id >= 0).ok_or_else(|| {
        CoreError::ModelValidation(format!(
            "model configuration value '{key}' is not a non-negative integer: {value}",
        ))
    })?;
    if let Some(vocab_size) = vocab_size
        && id >= vocab_size
    {
        return Err(CoreError::ModelValidation(format!(
            "model configuration value '{key}' = {id} is outside the vocabulary of {vocab_size} tokens",
        )));
    }
    Ok(id)
}

/// Read and parse a JSON file from the model directory.
pub(super) fn read_json(path: &Path) -> Result<serde_json::Value, CoreError> {
    let text = std::fs::read_to_string(path)?;
    Ok(serde_json::from_str(&text)?)
}

/// Resolve the input token cap for a model.
///
/// An unset cap defaults to [`DEFAULT_MAX_INPUT_TOKENS`], lowered to the
/// model's position limit when that is smaller.
///
/// # Errors
///
/// Returns [`CoreError::Config`] if an explicit cap is below
/// [`SEQ2SEQ_MIN_INPUT_TOKENS`] or above the model's position limit.
fn resolve_max_input_tokens(
    config: &serde_json::Value,
    explicit: Option<usize>,
) -> Result<usize, CoreError> {
    let position_limit = config
        .get("max_position_embeddings")
        .and_then(serde_json::Value::as_u64)
        .and_then(|limit| usize::try_from(limit).ok());

    match explicit {
        Some(cap) if cap < SEQ2SEQ_MIN_INPUT_TOKENS => Err(CoreError::Config(format!(
            "seq2seq max input tokens {cap} is below the minimum of {SEQ2SEQ_MIN_INPUT_TOKENS}",
        ))),
        Some(cap) => match position_limit {
            Some(limit) if cap > limit => Err(CoreError::Config(format!(
                "seq2seq max input tokens {cap} exceeds the model's position limit of {limit}",
            ))),
            _ => Ok(cap),
        },
        None => Ok(position_limit.map_or(DEFAULT_MAX_INPUT_TOKENS, |limit| {
            limit.clamp(SEQ2SEQ_MIN_INPUT_TOKENS, DEFAULT_MAX_INPUT_TOKENS)
        })),
    }
}

/// Run greedy decoding until an end token, the length limit or the deadline.
///
/// `next_token` receives the decoder sequence so far (start token first)
/// and the tokens that may not come next, and returns the most likely
/// permitted token. It is not called for forced tokens: their value does
/// not depend on the model.
///
/// Returns the generated tokens without the start token. The sequence ends
/// with an end token when the model produced one; it holds at most
/// `settings.max_new_tokens` tokens in every case, so a model that never
/// emits an end token cannot run away.
///
/// # Errors
///
/// Returns [`CoreError::Timeout`] once `deadline` has passed, and any
/// error `next_token` returns.
fn greedy_decode<F>(
    settings: &GenerationSettings,
    deadline: Option<Instant>,
    mut next_token: F,
) -> Result<Vec<i64>, CoreError>
where
    F: FnMut(&[i64], &[i64]) -> Result<i64, CoreError>,
{
    let max_new_tokens = settings.max_new_tokens;
    let mut tokens = Vec::with_capacity(max_new_tokens + 1);
    tokens.push(settings.decoder_start_token_id);
    let mut banned = Vec::new();

    for step in 0..max_new_tokens {
        // The forced end token wins when both rules apply to one step,
        // matching the order in which Hugging Face applies them.
        let forced = match settings.forced_eos_token_id {
            Some(eos) if step + 1 == max_new_tokens => Some(eos),
            _ if step == 0 => settings.forced_bos_token_id,
            _ => None,
        };
        let token = match forced {
            Some(token) => token,
            None => {
                ensure_before(deadline, tokens.len() - 1)?;
                banned.clear();
                extend_with_repeating_tokens(&mut banned, &tokens, settings.no_repeat_ngram_size);
                next_token(&tokens, &banned)?
            }
        };
        tokens.push(token);
        if settings.eos_token_ids.contains(&token) {
            break;
        }
    }

    tokens.remove(0);
    Ok(tokens)
}

/// Collect the tokens that would repeat an n-gram already in `tokens`.
///
/// A token is collected when appending it would make the last
/// `ngram_size` tokens equal to a run that already occurs in `tokens`
/// (the Hugging Face `no_repeat_ngram_size` rule; the start token and
/// forced tokens count as part of the sequence). Nothing is collected
/// when `ngram_size` is 0 or the sequence is still too short.
pub(super) fn extend_with_repeating_tokens(banned: &mut Vec<i64>, tokens: &[i64], ngram_size: usize) {
    if ngram_size == 0 || tokens.len() + 1 < ngram_size {
        return;
    }
    let prefix_len = ngram_size - 1;
    let prefix = &tokens[tokens.len() - prefix_len..];
    banned.extend(
        tokens
            .windows(ngram_size)
            .filter(|ngram| &ngram[..prefix_len] == prefix)
            .map(|ngram| ngram[prefix_len]),
    );
}

/// Fail with [`CoreError::Timeout`] if the deadline has passed.
fn ensure_before(deadline: Option<Instant>, generated: usize) -> Result<(), CoreError> {
    match deadline {
        Some(deadline) if Instant::now() >= deadline => Err(CoreError::Timeout(format!(
            "seq2seq generation exceeded its time budget after {generated} tokens",
        ))),
        _ => Ok(()),
    }
}

/// Index of the largest value among permitted tokens; the first wins a tie.
///
/// A banned token scores negative infinity, as it does in Hugging Face
/// generation, so it is only chosen when nothing scores higher.
///
/// # Errors
///
/// Returns [`CoreError::Inference`] if the slice is empty or holds only
/// NaN values.
pub(super) fn argmax_excluding(logits: &[f32], banned: &[i64]) -> Result<usize, CoreError> {
    let mut best: Option<(usize, f32)> = None;
    for (index, &value) in logits.iter().enumerate() {
        if value.is_nan() {
            continue;
        }
        let is_banned =
            !banned.is_empty() && i64::try_from(index).is_ok_and(|token| banned.contains(&token));
        let value = if is_banned { f32::NEG_INFINITY } else { value };
        if best.is_none_or(|(_, best_value)| value > best_value) {
            best = Some((index, value));
        }
    }
    best.map(|(index, _)| index)
        .ok_or_else(|| CoreError::Inference("decoder produced no comparable logits".to_string()))
}

/// The decoder half of an encoder-decoder seq2seq model.
///
/// Owns the decoder session and everything needed to drive generation;
/// the encoder session and the tokenizer stay with the pipeline.
pub(super) struct EncoderDecoder {
    decoder: Session,
    encoder_io: EncoderIo,
    decoder_io: DecoderIo,
    settings: GenerationSettings,
    /// Time budget for one input, measured from the start of its encoder pass.
    timeout: Option<Duration>,
    /// Zeros backing the dummy cache inputs of a merged decoder.
    past_zeros: Vec<f32>,
    /// Whether generation keeps a key/value cache between decoder steps.
    kv_cache: bool,
}

impl EncoderDecoder {
    /// Load an encoder-decoder export.
    ///
    /// Returns the encoder session, the tokenizer (truncating at the
    /// resolved input cap, with padding disabled) and the decoder state.
    ///
    /// # Errors
    ///
    /// Returns [`CoreError`] if a file is missing or invalid, a graph does
    /// not have the inputs and outputs this mode can drive, or the
    /// generation settings or options are unusable for this model.
    pub(super) fn load(
        files: &EncoderDecoderFiles,
        model_dir: &Path,
        ep: &ExecutionProvider,
        options: &Seq2SeqOptions,
    ) -> Result<(Session, Tokenizer, Self), CoreError> {
        // 1. Model configuration: validated before the (slow) graph loads.
        let config = read_json(&model_dir.join("config.json"))?;
        let generation_config_path = model_dir.join("generation_config.json");
        let generation_config = if generation_config_path.exists() {
            Some(read_json(&generation_config_path)?)
        } else {
            None
        };
        let settings = GenerationSettings::resolve(
            &config,
            generation_config.as_ref(),
            options.max_new_tokens(),
        )?;
        let max_input_tokens = resolve_max_input_tokens(&config, options.max_input_tokens())?;

        // 2. Tokenizer: one input at a time, so never pad.
        let mut tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json"))
            .map_err(|e| CoreError::Tokenization(e.to_string()))?;
        tokenizer
            .with_truncation(Some(tokenizers::TruncationParams {
                max_length: max_input_tokens,
                ..Default::default()
            }))
            .map_err(|e| CoreError::Tokenization(e.to_string()))?;
        tokenizer.with_padding(None);

        // 3. Graphs, wired by the names each one declares.
        let encoder = build_onnx_session(&files.encoder, ep)?;
        let encoder_io = EncoderIo::from_specs(
            &TensorSpec::from_outlets(encoder.inputs()),
            &TensorSpec::from_outlets(encoder.outputs()),
        )?;
        let (decoder_path, decoder, decoder_io) = load_decoder(files, ep, options.kv_cache())?;
        let kv_cache = options.kv_cache()
            && decoder_io
                .merged
                .as_ref()
                .is_some_and(MergedInputs::supports_cache);

        let past_zeros = decoder_io.merged.as_ref().map_or_else(Vec::new, |merged| {
            let longest = merged
                .past
                .iter()
                .map(PastInput::element_count)
                .max()
                .unwrap_or(0);
            vec![0.0; longest]
        });

        tracing::info!(
            encoder = %files.encoder.display(),
            decoder = %decoder_path.display(),
            merged_decoder = decoder_io.merged.is_some(),
            kv_cache,
            max_input_tokens,
            max_new_tokens = settings.max_new_tokens,
            decoder_start_token_id = settings.decoder_start_token_id,
            eos_token_ids = ?settings.eos_token_ids,
            pad_token_id = ?settings.pad_token_id,
            forced_bos_token_id = ?settings.forced_bos_token_id,
            forced_eos_token_id = ?settings.forced_eos_token_id,
            no_repeat_ngram_size = settings.no_repeat_ngram_size,
            "seq2seq encoder-decoder mode loaded (greedy decoding)"
        );
        if settings.eos_token_ids.is_empty() {
            tracing::warn!(
                max_new_tokens = settings.max_new_tokens,
                "model declares no eos_token_id; every generation runs to the token limit"
            );
        }

        Ok((
            encoder,
            tokenizer,
            Self {
                decoder,
                encoder_io,
                decoder_io,
                settings,
                timeout: options.timeout(),
                past_zeros,
                kv_cache,
            },
        ))
    }

    /// Whether generation keeps a key/value cache between decoder steps.
    pub(super) fn uses_kv_cache(&self) -> bool {
        self.kv_cache
    }

    /// The deadline for an input whose processing starts now.
    pub(super) fn deadline_from_now(&self) -> Option<Instant> {
        self.timeout.map(|timeout| Instant::now() + timeout)
    }

    /// Generate output tokens for one tokenized input.
    ///
    /// Runs the encoder once, then decodes greedily, with a key/value
    /// cache when the decoder supports one. Returns the generated token IDs
    /// without the decoder start token.
    ///
    /// # Errors
    ///
    /// Returns [`CoreError::InvalidInput`] for an empty token sequence,
    /// [`CoreError::Timeout`] if `deadline` passes, and
    /// [`CoreError::Inference`] if a graph fails or returns a tensor of an
    /// unexpected shape.
    pub(super) fn generate(
        &mut self,
        encoder: &mut Session,
        prepared: &PreparedInput,
        deadline: Option<Instant>,
    ) -> Result<Vec<u32>, CoreError> {
        let input_len = prepared.sequence_length;
        if input_len == 0 {
            return Err(CoreError::InvalidInput(
                "seq2seq input produced no tokens".to_string(),
            ));
        }
        ensure_before(deadline, 0)?;
        let started = Instant::now();

        // Encoder pass.
        let mut encoder_inputs: Vec<(Cow<'_, str>, SessionInputValue<'_>)> = Vec::with_capacity(2);
        encoder_inputs.push((
            Cow::Borrowed("input_ids"),
            int64_tensor(input_len, &prepared.input_ids)?.into(),
        ));
        if self.encoder_io.feeds_attention_mask {
            encoder_inputs.push((
                Cow::Borrowed("attention_mask"),
                int64_tensor(input_len, &prepared.attention_mask)?.into(),
            ));
        }
        let encoder_outputs = encoder
            .run(encoder_inputs)
            .map_err(|e| CoreError::Inference(format!("encoder run failed: {e}")))?;
        let hidden = encoder_outputs
            .get(&self.encoder_io.hidden_output)
            .ok_or_else(|| {
                CoreError::Inference(format!(
                    "encoder did not return output '{}'",
                    self.encoder_io.hidden_output,
                ))
            })?;
        let (hidden_shape, hidden_data) = hidden
            .try_extract_tensor::<f32>()
            .map_err(|e| CoreError::Inference(e.to_string()))?;
        let hidden_dims: Vec<i64> = hidden_shape.iter().copied().collect();
        if hidden_dims.len() != 3 || hidden_dims[0] != 1 {
            return Err(CoreError::Inference(format!(
                "expected encoder output of shape (1, seq, hidden), got {hidden_dims:?}",
            )));
        }
        let encoder_elapsed = started.elapsed();

        // Decode loop: one decoder pass per generated token.
        let Self {
            decoder,
            decoder_io,
            settings,
            past_zeros,
            kv_cache,
            ..
        } = self;
        let kv_cache = *kv_cache;
        // Cache tensors in the order of the decoder's cache inputs, and the
        // number of decoder tokens they cover. Local to this input.
        let mut cache: Vec<DynValue> = Vec::new();
        let mut cached_len = 0_usize;
        let generated = greedy_decode(settings, deadline, |tokens, banned| {
            // The cached branch takes exactly the one token the cache does
            // not cover yet. Anything else (the first step, which may hold
            // a forced token after the start token) runs the uncached
            // branch over the whole sequence and starts the cache afresh.
            let reuse = kv_cache && cached_len > 0 && tokens.len() == cached_len + 1;
            let fed_tokens = if reuse { &tokens[cached_len..] } else { tokens };
            let branch = [reuse];

            let mut inputs: Vec<(Cow<'_, str>, SessionInputValue<'_>)> =
                Vec::with_capacity(3 + decoder_io.merged.as_ref().map_or(0, |m| 1 + m.past.len()));
            inputs.push((
                Cow::Borrowed(decoder_io.tokens_input.as_str()),
                int64_tensor(fed_tokens.len(), fed_tokens)?.into(),
            ));
            inputs.push((
                Cow::Borrowed(decoder_io.encoder_hidden_input.as_str()),
                TensorRef::from_array_view((hidden_dims.as_slice(), hidden_data))
                    .map_err(|e| CoreError::Inference(e.to_string()))?
                    .into(),
            ));
            if let Some(name) = &decoder_io.encoder_mask_input {
                inputs.push((
                    Cow::Borrowed(name.as_str()),
                    int64_tensor(input_len, &prepared.attention_mask)?.into(),
                ));
            }
            if let Some(merged) = &decoder_io.merged {
                inputs.push((
                    Cow::Borrowed(merged.use_cache_branch.as_str()),
                    TensorRef::from_array_view(([1_usize], branch.as_slice()))
                        .map_err(|e| CoreError::Inference(e.to_string()))?
                        .into(),
                ));
                for (index, past) in merged.past.iter().enumerate() {
                    let value: SessionInputValue<'_> = if reuse {
                        (&cache[index]).into()
                    } else {
                        TensorRef::from_array_view((
                            past.dims.as_slice(),
                            &past_zeros[..past.element_count()],
                        ))
                        .map_err(|e| CoreError::Inference(e.to_string()))?
                        .into()
                    };
                    inputs.push((Cow::Borrowed(past.name.as_str()), value));
                }
            }

            let mut outputs = decoder
                .run(inputs)
                .map_err(|e| CoreError::Inference(format!("decoder run failed: {e}")))?;
            let logits = outputs.get(&decoder_io.logits_output).ok_or_else(|| {
                CoreError::Inference(format!(
                    "decoder did not return output '{}'",
                    decoder_io.logits_output,
                ))
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
                    "expected decoder logits of shape (1, seq, vocab), got {shape:?}",
                ))
            })?;
            // Logits of the last position predict the next token.
            let last_position = data.len().checked_sub(vocab_size).ok_or_else(|| {
                CoreError::Inference(format!(
                    "decoder logits hold {} values, fewer than the vocabulary size {vocab_size}",
                    data.len(),
                ))
            })?;
            let token = argmax_excluding(&data[last_position..], banned)?;

            if kv_cache && let Some(merged) = &decoder_io.merged {
                if !reuse {
                    cache.clear();
                }
                for (index, past) in merged.past.iter().enumerate() {
                    // The cached branch returns no encoder cross-attention
                    // state; the tensors of the first step stay in place.
                    if reuse && past.from_encoder {
                        continue;
                    }
                    let value = take_cache_tensor(&mut outputs, past)?;
                    if reuse {
                        cache[index] = value;
                    } else {
                        cache.push(value);
                    }
                }
                cached_len = tokens.len();
            }

            i64::try_from(token).map_err(|_| {
                CoreError::Inference(format!("token index {token} does not fit a token ID"))
            })
        })?;

        tracing::debug!(
            input_tokens = input_len,
            generated_tokens = generated.len(),
            reached_token_limit = generated.len() == settings.max_new_tokens,
            encoder_ms = encoder_elapsed.as_secs_f64() * 1000.0,
            decode_ms = (started.elapsed() - encoder_elapsed).as_secs_f64() * 1000.0,
            "seq2seq generation finished"
        );

        generated
            .into_iter()
            .map(|token| {
                u32::try_from(token).map_err(|_| {
                    CoreError::Inference(format!("invalid token ID {token} in seq2seq output"))
                })
            })
            .collect()
    }
}

/// Load the decoder graph and derive its wiring.
///
/// With `kv_cache` set, a merged decoder that can feed its cache back is
/// preferred. When a plain decoder sits next to a merged one that cannot
/// be used that way (or cannot be loaded at all), the plain decoder runs
/// instead, exactly as it does for an export without a merged decoder.
///
/// Returns the path of the graph that was loaded, its session and wiring.
///
/// # Errors
///
/// Returns [`CoreError`] if the decoder that ends up chosen cannot be
/// loaded or does not have the inputs and outputs this mode can drive.
fn load_decoder(
    files: &EncoderDecoderFiles,
    ep: &ExecutionProvider,
    kv_cache: bool,
) -> Result<(PathBuf, Session, DecoderIo), CoreError> {
    let open = |path: &Path| -> Result<(Session, DecoderIo), CoreError> {
        let session = build_onnx_session(path, ep)?;
        let io = DecoderIo::from_specs(
            &TensorSpec::from_outlets(session.inputs()),
            &TensorSpec::from_outlets(session.outputs()),
        )?;
        Ok((session, io))
    };

    let separate_merged = files
        .merged_decoder
        .as_ref()
        .filter(|merged| kv_cache && **merged != files.decoder);
    if let Some(merged) = separate_merged {
        match open(merged) {
            Ok((session, io)) if io.merged.as_ref().is_some_and(MergedInputs::supports_cache) => {
                return Ok((merged.clone(), session, io));
            }
            Ok(_) => tracing::warn!(
                merged_decoder = %merged.display(),
                decoder = %files.decoder.display(),
                "merged decoder cannot feed a key/value cache back; decoding without a cache"
            ),
            Err(error) => tracing::warn!(
                merged_decoder = %merged.display(),
                decoder = %files.decoder.display(),
                %error,
                "merged decoder could not be used; decoding without a cache"
            ),
        }
    }

    let (session, io) = open(&files.decoder)?;
    Ok((files.decoder.clone(), session, io))
}

/// Take the tensor that feeds `past` on the next step out of the outputs.
///
/// # Errors
///
/// Returns [`CoreError::Inference`] if the decoder did not return the
/// tensor or returned it without elements: an empty tensor fed back as
/// cache would silently change what the model attends to.
fn take_cache_tensor(
    outputs: &mut ort::session::SessionOutputs<'_>,
    past: &PastInput,
) -> Result<DynValue, CoreError> {
    let name = past.present_output.as_deref().ok_or_else(|| {
        CoreError::Inference(format!(
            "decoder has no output to feed cache input '{}'",
            past.name,
        ))
    })?;
    let value = outputs
        .remove(name)
        .ok_or_else(|| CoreError::Inference(format!("decoder did not return output '{name}'")))?;
    let is_empty = value
        .dtype()
        .tensor_shape()
        .is_none_or(|shape| shape.contains(&0));
    if is_empty {
        return Err(CoreError::Inference(format!(
            "decoder returned an empty cache tensor '{name}'",
        )));
    }
    Ok(value)
}

/// Borrow a token slice as a `(1, len)` int64 tensor.
pub(super) fn int64_tensor(len: usize, data: &[i64]) -> Result<TensorRef<'_, i64>, CoreError> {
    TensorRef::from_array_view(([1, len], data)).map_err(|e| CoreError::Inference(e.to_string()))
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

    fn float32(name: &str) -> TensorSpec {
        spec(name, TensorElementType::Float32, &[-1, -1, -1])
    }

    fn settings(max_new_tokens: usize) -> GenerationSettings {
        GenerationSettings {
            decoder_start_token_id: 2,
            eos_token_ids: vec![2],
            pad_token_id: Some(1),
            forced_bos_token_id: None,
            forced_eos_token_id: None,
            no_repeat_ngram_size: 0,
            max_new_tokens,
        }
    }

    /// The published BART configuration this mode was first built for.
    fn bart_config() -> serde_json::Value {
        json!({
            "bos_token_id": 0,
            "decoder_start_token_id": 2,
            "eos_token_id": 2,
            "forced_bos_token_id": 0,
            "forced_eos_token_id": 2,
            "max_length": 142,
            "max_position_embeddings": 1024,
            "min_length": 56,
            "no_repeat_ngram_size": 3,
            "num_beams": 4,
            "pad_token_id": 1,
            "vocab_size": 50264
        })
    }

    // --- File layout ---

    #[test]
    fn locate_returns_none_without_encoder_file() {
        // Arrange
        let dir = tempfile::tempdir().expect("temp dir");

        // Act
        let files = EncoderDecoderFiles::locate(dir.path()).expect("lookup should not fail");

        // Assert
        assert_eq!(files, None);
    }

    #[test]
    fn locate_prefers_plain_decoder_over_merged() {
        // Arrange
        let dir = tempfile::tempdir().expect("temp dir");
        for name in [ENCODER_FILE, DECODER_FILE, MERGED_DECODER_FILE] {
            std::fs::write(dir.path().join(name), b"").expect("write");
        }

        // Act
        let files = EncoderDecoderFiles::locate(dir.path())
            .expect("lookup should not fail")
            .expect("export should be found");

        // Assert
        assert_eq!(files.encoder, dir.path().join(ENCODER_FILE));
        assert_eq!(files.decoder, dir.path().join(DECODER_FILE));
    }

    #[test]
    fn locate_falls_back_to_merged_decoder() {
        // Arrange
        let dir = tempfile::tempdir().expect("temp dir");
        for name in [ENCODER_FILE, MERGED_DECODER_FILE] {
            std::fs::write(dir.path().join(name), b"").expect("write");
        }

        // Act
        let files = EncoderDecoderFiles::locate(dir.path())
            .expect("lookup should not fail")
            .expect("export should be found");

        // Assert
        assert_eq!(files.decoder, dir.path().join(MERGED_DECODER_FILE));
    }

    #[test]
    fn locate_reports_merged_decoder_next_to_plain_decoder() {
        // Arrange
        let dir = tempfile::tempdir().expect("temp dir");
        for name in [ENCODER_FILE, DECODER_FILE, MERGED_DECODER_FILE] {
            std::fs::write(dir.path().join(name), b"").expect("write");
        }

        // Act
        let files = EncoderDecoderFiles::locate(dir.path())
            .expect("lookup should not fail")
            .expect("export should be found");

        // Assert -- the cached loop gets the merged graph, the uncached
        // loop keeps the plain one.
        assert_eq!(
            files.merged_decoder,
            Some(dir.path().join(MERGED_DECODER_FILE))
        );
        assert_eq!(files.decoder, dir.path().join(DECODER_FILE));
    }

    #[test]
    fn locate_reports_no_merged_decoder_when_only_plain_exists() {
        // Arrange
        let dir = tempfile::tempdir().expect("temp dir");
        for name in [ENCODER_FILE, DECODER_FILE] {
            std::fs::write(dir.path().join(name), b"").expect("write");
        }

        // Act
        let files = EncoderDecoderFiles::locate(dir.path())
            .expect("lookup should not fail")
            .expect("export should be found");

        // Assert
        assert_eq!(files.merged_decoder, None);
    }

    #[test]
    fn locate_ignores_merged_decoder_outside_the_encoder_directory() {
        // Arrange -- the export is in onnx/, a stray merged file is not.
        let dir = tempfile::tempdir().expect("temp dir");
        let onnx = dir.path().join("onnx");
        std::fs::create_dir(&onnx).expect("mkdir");
        for name in [ENCODER_FILE, DECODER_FILE] {
            std::fs::write(onnx.join(name), b"").expect("write");
        }
        std::fs::write(dir.path().join(MERGED_DECODER_FILE), b"").expect("write");

        // Act
        let files = EncoderDecoderFiles::locate(dir.path())
            .expect("lookup should not fail")
            .expect("export should be found");

        // Assert
        assert_eq!(files.merged_decoder, None);
    }

    #[test]
    fn locate_prefers_onnx_subdirectory() {
        // Arrange -- a complete export in both locations.
        let dir = tempfile::tempdir().expect("temp dir");
        let onnx = dir.path().join("onnx");
        std::fs::create_dir(&onnx).expect("mkdir");
        for parent in [dir.path(), onnx.as_path()] {
            for name in [ENCODER_FILE, DECODER_FILE] {
                std::fs::write(parent.join(name), b"").expect("write");
            }
        }

        // Act
        let files = EncoderDecoderFiles::locate(dir.path())
            .expect("lookup should not fail")
            .expect("export should be found");

        // Assert
        assert_eq!(files.encoder, onnx.join(ENCODER_FILE));
        assert_eq!(files.decoder, onnx.join(DECODER_FILE));
    }

    #[test]
    fn locate_rejects_encoder_without_decoder() {
        // Arrange
        let dir = tempfile::tempdir().expect("temp dir");
        std::fs::write(dir.path().join(ENCODER_FILE), b"").expect("write");

        // Act
        let result = EncoderDecoderFiles::locate(dir.path());

        // Assert
        let message = result.expect_err("missing decoder should fail").to_string();
        assert!(
            message.contains(DECODER_FILE),
            "unexpected error: {message}"
        );
    }

    // --- Encoder wiring ---

    #[test]
    fn encoder_io_accepts_ids_and_mask() {
        // Arrange
        let inputs = [int64("input_ids"), int64("attention_mask")];
        let outputs = [float32("last_hidden_state")];

        // Act
        let io = EncoderIo::from_specs(&inputs, &outputs).expect("valid encoder");

        // Assert
        assert!(io.feeds_attention_mask);
        assert_eq!(io.hidden_output, "last_hidden_state");
    }

    #[test]
    fn encoder_io_rejects_speech_encoder() {
        // Arrange -- a Whisper encoder takes audio features, not tokens.
        let inputs = [float32("input_features")];
        let outputs = [float32("last_hidden_state")];

        // Act
        let result = EncoderIo::from_specs(&inputs, &outputs);

        // Assert
        let message = result.expect_err("speech encoder should fail").to_string();
        assert!(
            message.contains("input_features"),
            "unexpected error: {message}"
        );
    }

    #[test]
    fn encoder_io_rejects_int32_inputs() {
        // Arrange
        let inputs = [spec("input_ids", TensorElementType::Int32, &[-1, -1])];
        let outputs = [float32("last_hidden_state")];

        // Act
        let result = EncoderIo::from_specs(&inputs, &outputs);

        // Assert
        assert!(matches!(result, Err(CoreError::ModelValidation(_))));
    }

    // --- Decoder wiring ---

    #[test]
    fn decoder_io_reads_bart_names() {
        // Arrange -- Optimum's BART decoder_model.onnx.
        let inputs = [
            int64("encoder_attention_mask"),
            int64("input_ids"),
            float32("encoder_hidden_states"),
        ];
        let outputs = [float32("logits"), float32("present.0.decoder.key")];

        // Act
        let io = DecoderIo::from_specs(&inputs, &outputs).expect("valid decoder");

        // Assert
        assert_eq!(io.tokens_input, "input_ids");
        assert_eq!(io.encoder_hidden_input, "encoder_hidden_states");
        assert_eq!(
            io.encoder_mask_input.as_deref(),
            Some("encoder_attention_mask")
        );
        assert_eq!(io.merged, None);
        assert_eq!(io.logits_output, "logits");
    }

    #[test]
    fn decoder_io_accepts_decoder_without_encoder_mask() {
        // Arrange
        let inputs = [int64("decoder_input_ids"), float32("encoder_output")];
        let outputs = [float32("logits")];

        // Act
        let io = DecoderIo::from_specs(&inputs, &outputs).expect("valid decoder");

        // Assert
        assert_eq!(io.tokens_input, "decoder_input_ids");
        assert_eq!(io.encoder_hidden_input, "encoder_output");
        assert_eq!(io.encoder_mask_input, None);
    }

    #[test]
    fn decoder_io_sizes_dummy_cache_inputs_for_merged_decoder() {
        // Arrange -- dynamic batch and sequence dimensions around fixed
        // head count and head size.
        let inputs = [
            int64("input_ids"),
            float32("encoder_hidden_states"),
            spec("use_cache_branch", TensorElementType::Bool, &[1]),
            spec(
                "past_key_values.0.decoder.key",
                TensorElementType::Float32,
                &[-1, 16, -1, 64],
            ),
        ];
        let outputs = [float32("logits")];

        // Act
        let io = DecoderIo::from_specs(&inputs, &outputs).expect("valid merged decoder");

        // Assert
        let merged = io.merged.expect("merged inputs should be detected");
        assert_eq!(merged.use_cache_branch, "use_cache_branch");
        assert_eq!(merged.past.len(), 1);
        assert_eq!(merged.past[0].dims, vec![1, 16, 1, 64]);
        assert_eq!(merged.past[0].element_count(), 1024);
    }

    /// The inputs and outputs of Optimum's merged BART decoder, one layer.
    fn merged_bart_specs() -> (Vec<TensorSpec>, Vec<TensorSpec>) {
        let cache = |name: &str| spec(name, TensorElementType::Float32, &[-1, 16, -1, 64]);
        let slots = [
            "decoder.key",
            "decoder.value",
            "encoder.key",
            "encoder.value",
        ];
        let mut inputs = vec![
            int64("encoder_attention_mask"),
            int64("input_ids"),
            float32("encoder_hidden_states"),
        ];
        inputs.extend(
            slots
                .iter()
                .map(|slot| cache(&format!("past_key_values.0.{slot}"))),
        );
        inputs.push(spec("use_cache_branch", TensorElementType::Bool, &[1]));
        let mut outputs = vec![float32("logits")];
        outputs.extend(slots.iter().map(|slot| cache(&format!("present.0.{slot}"))));
        (inputs, outputs)
    }

    #[test]
    fn decoder_io_pairs_cache_inputs_with_present_outputs() {
        // Arrange
        let (inputs, outputs) = merged_bart_specs();

        // Act
        let io = DecoderIo::from_specs(&inputs, &outputs).expect("valid merged decoder");

        // Assert -- every input has its output; encoder slots are marked.
        let merged = io.merged.expect("merged inputs should be detected");
        assert!(merged.supports_cache());
        let pairs: Vec<(&str, Option<&str>, bool)> = merged
            .past
            .iter()
            .map(|past| {
                (
                    past.name.as_str(),
                    past.present_output.as_deref(),
                    past.from_encoder,
                )
            })
            .collect();
        assert_eq!(
            pairs,
            vec![
                (
                    "past_key_values.0.decoder.key",
                    Some("present.0.decoder.key"),
                    false
                ),
                (
                    "past_key_values.0.decoder.value",
                    Some("present.0.decoder.value"),
                    false
                ),
                (
                    "past_key_values.0.encoder.key",
                    Some("present.0.encoder.key"),
                    true
                ),
                (
                    "past_key_values.0.encoder.value",
                    Some("present.0.encoder.value"),
                    true
                ),
            ]
        );
    }

    #[test]
    fn decoder_io_does_not_support_cache_when_a_present_output_is_missing() {
        // Arrange
        let (inputs, mut outputs) = merged_bart_specs();
        outputs.retain(|output| output.name != "present.0.decoder.value");

        // Act
        let io = DecoderIo::from_specs(&inputs, &outputs).expect("valid merged decoder");

        // Assert -- still a usable merged decoder, for the uncached loop.
        let merged = io.merged.expect("merged inputs should be detected");
        assert!(!merged.supports_cache());
    }

    #[test]
    fn decoder_io_does_not_support_cache_when_a_present_output_is_not_float32() {
        // Arrange
        let (inputs, mut outputs) = merged_bart_specs();
        for output in &mut outputs {
            if output.name == "present.0.encoder.key" {
                output.element_type = Some(TensorElementType::Float16);
            }
        }

        // Act
        let io = DecoderIo::from_specs(&inputs, &outputs).expect("valid merged decoder");

        // Assert
        assert!(!io.merged.expect("merged inputs").supports_cache());
    }

    #[test]
    fn decoder_io_does_not_support_cache_without_cache_inputs() {
        // Arrange -- a branch selector but nothing to cache.
        let inputs = [
            int64("input_ids"),
            float32("encoder_hidden_states"),
            spec("use_cache_branch", TensorElementType::Bool, &[1]),
        ];
        let outputs = [float32("logits")];

        // Act
        let io = DecoderIo::from_specs(&inputs, &outputs).expect("valid merged decoder");

        // Assert
        assert!(!io.merged.expect("merged inputs").supports_cache());
    }

    #[test]
    fn decoder_io_rejects_cache_only_decoder() {
        // Arrange -- decoder_with_past_model.onnx has no branch selector.
        let inputs = [
            int64("input_ids"),
            float32("encoder_hidden_states"),
            float32("past_key_values.0.decoder.key"),
        ];
        let outputs = [float32("logits")];

        // Act
        let result = DecoderIo::from_specs(&inputs, &outputs);

        // Assert
        let message = result
            .expect_err("cache-only decoder should fail")
            .to_string();
        assert!(
            message.contains("use_cache_branch"),
            "unexpected error: {message}"
        );
    }

    #[test]
    fn decoder_io_rejects_unknown_input() {
        // Arrange
        let inputs = [
            int64("input_ids"),
            float32("encoder_hidden_states"),
            int64("decoder_attention_mask"),
        ];
        let outputs = [float32("logits")];

        // Act
        let result = DecoderIo::from_specs(&inputs, &outputs);

        // Assert
        let message = result.expect_err("unknown input should fail").to_string();
        assert!(
            message.contains("decoder_attention_mask"),
            "unexpected error: {message}"
        );
    }

    #[test]
    fn decoder_io_rejects_missing_logits() {
        // Arrange
        let inputs = [int64("input_ids"), float32("encoder_hidden_states")];
        let outputs = [float32("sequences")];

        // Act
        let result = DecoderIo::from_specs(&inputs, &outputs);

        // Assert
        let message = result.expect_err("missing logits should fail").to_string();
        assert!(message.contains("logits"), "unexpected error: {message}");
    }

    // --- Generation settings ---

    #[test]
    fn settings_read_bart_config() {
        // Arrange
        let config = bart_config();

        // Act
        let resolved = GenerationSettings::resolve(&config, None, None).expect("valid config");

        // Assert -- max_length 142 counts the start token.
        assert_eq!(
            resolved,
            GenerationSettings {
                decoder_start_token_id: 2,
                eos_token_ids: vec![2],
                pad_token_id: Some(1),
                forced_bos_token_id: Some(0),
                forced_eos_token_id: Some(2),
                no_repeat_ngram_size: 3,
                max_new_tokens: 141,
            }
        );
    }

    #[test]
    fn settings_default_to_no_ngram_rule() {
        // Arrange
        let config = json!({ "decoder_start_token_id": 0 });

        // Act
        let resolved = GenerationSettings::resolve(&config, None, None).expect("valid config");

        // Assert
        assert_eq!(resolved.no_repeat_ngram_size, 0);
    }

    #[test]
    fn settings_reject_negative_ngram_size() {
        // Arrange
        let config = json!({ "decoder_start_token_id": 0, "no_repeat_ngram_size": -1 });

        // Act
        let result = GenerationSettings::resolve(&config, None, None);

        // Assert
        assert!(matches!(result, Err(CoreError::ModelValidation(_))));
    }

    #[test]
    fn settings_prefer_generation_config_over_config() {
        // Arrange
        let config = bart_config();
        let generation = json!({
            "decoder_start_token_id": 0,
            "eos_token_id": [2, 5],
            "forced_bos_token_id": null,
            "max_length": 64
        });

        // Act
        let resolved =
            GenerationSettings::resolve(&config, Some(&generation), None).expect("valid config");

        // Assert -- a null entry falls back to config.json.
        assert_eq!(resolved.decoder_start_token_id, 0);
        assert_eq!(resolved.eos_token_ids, vec![2, 5]);
        assert_eq!(resolved.forced_bos_token_id, Some(0));
        assert_eq!(resolved.max_new_tokens, 63);
    }

    #[test]
    fn settings_fall_back_to_bos_for_decoder_start() {
        // Arrange
        let config = json!({ "bos_token_id": 7, "eos_token_id": 3 });

        // Act
        let resolved = GenerationSettings::resolve(&config, None, None).expect("valid config");

        // Assert
        assert_eq!(resolved.decoder_start_token_id, 7);
    }

    #[test]
    fn settings_reject_config_without_start_token() {
        // Arrange
        let config = json!({ "eos_token_id": 3 });

        // Act
        let result = GenerationSettings::resolve(&config, None, None);

        // Assert
        let message = result
            .expect_err("missing start token should fail")
            .to_string();
        assert!(
            message.contains("decoder_start_token_id"),
            "unexpected error: {message}"
        );
    }

    #[test]
    fn settings_reject_token_outside_vocabulary() {
        // Arrange
        let config = json!({ "decoder_start_token_id": 9, "vocab_size": 9 });

        // Act
        let result = GenerationSettings::resolve(&config, None, None);

        // Assert
        assert!(matches!(result, Err(CoreError::ModelValidation(_))));
    }

    #[test]
    fn settings_default_to_hugging_face_max_length() {
        // Arrange -- no max_length anywhere.
        let config = json!({ "decoder_start_token_id": 0 });

        // Act
        let resolved = GenerationSettings::resolve(&config, None, None).expect("valid config");

        // Assert
        assert_eq!(resolved.max_new_tokens, DEFAULT_MAX_LENGTH - 1);
        assert!(resolved.eos_token_ids.is_empty());
    }

    #[test]
    fn settings_use_explicit_max_new_tokens() {
        // Arrange
        let config = bart_config();

        // Act
        let resolved =
            GenerationSettings::resolve(&config, None, Some(300)).expect("valid override");

        // Assert
        assert_eq!(resolved.max_new_tokens, 300);
    }

    #[test]
    fn settings_reject_explicit_limit_beyond_decoder_positions() {
        // Arrange -- 1024 positions leave room for 1023 generated tokens.
        let config = bart_config();

        // Act
        let result = GenerationSettings::resolve(&config, None, Some(1024));

        // Assert
        assert!(matches!(result, Err(CoreError::Config(_))));
    }

    #[test]
    fn settings_reject_explicit_limit_of_zero() {
        // Arrange
        let config = bart_config();

        // Act
        let result = GenerationSettings::resolve(&config, None, Some(0));

        // Assert
        assert!(matches!(result, Err(CoreError::Config(_))));
    }

    #[test]
    fn settings_clamp_declared_limit_to_absolute_ceiling() {
        // Arrange -- a model that asks for far more than the runtime allows.
        let config = json!({ "decoder_start_token_id": 0, "max_length": 1_000_000 });

        // Act
        let resolved = GenerationSettings::resolve(&config, None, None).expect("valid config");

        // Assert
        assert_eq!(resolved.max_new_tokens, SEQ2SEQ_MAX_NEW_TOKENS_LIMIT);
    }

    // --- Input token cap ---

    #[test]
    fn input_cap_defaults_to_512() {
        // Arrange
        let config = bart_config();

        // Act
        let cap = resolve_max_input_tokens(&config, None).expect("valid cap");

        // Assert -- the same cap every other text pipeline uses.
        assert_eq!(cap, 512);
        assert_eq!(cap, DEFAULT_MAX_INPUT_TOKENS);
    }

    #[test]
    fn input_cap_default_shrinks_to_small_position_limit() {
        // Arrange
        let config = json!({ "max_position_embeddings": 128 });

        // Act
        let cap = resolve_max_input_tokens(&config, None).expect("valid cap");

        // Assert
        assert_eq!(cap, 128);
    }

    #[test]
    fn input_cap_accepts_explicit_value_within_position_limit() {
        // Arrange
        let config = bart_config();

        // Act
        let cap = resolve_max_input_tokens(&config, Some(1024)).expect("valid cap");

        // Assert
        assert_eq!(cap, 1024);
    }

    #[test]
    fn input_cap_rejects_explicit_value_beyond_position_limit() {
        // Arrange
        let config = bart_config();

        // Act
        let result = resolve_max_input_tokens(&config, Some(1025));

        // Assert
        assert!(matches!(result, Err(CoreError::Config(_))));
    }

    #[test]
    fn input_cap_rejects_explicit_value_below_minimum() {
        // Arrange
        let config = bart_config();

        // Act
        let result = resolve_max_input_tokens(&config, Some(SEQ2SEQ_MIN_INPUT_TOKENS - 1));

        // Assert
        assert!(matches!(result, Err(CoreError::Config(_))));
    }

    // --- Greedy loop ---

    #[test]
    fn greedy_decode_stops_at_end_token() {
        // Arrange -- the model emits 10, 11, then the end token.
        let settings = settings(50);
        let mut script = [10, 11, 2, 99].into_iter();

        // Act
        let generated = greedy_decode(&settings, None, |_, _| Ok(script.next().unwrap_or(99)))
            .expect("decode should succeed");

        // Assert -- the end token is kept, nothing after it is requested.
        assert_eq!(generated, vec![10, 11, 2]);
        assert_eq!(script.next(), Some(99));
    }

    #[test]
    fn greedy_decode_is_bounded_when_model_never_stops() {
        // Arrange -- a model that never emits the end token.
        let settings = settings(25);
        let mut calls = 0;

        // Act
        let generated = greedy_decode(&settings, None, |_, _| {
            calls += 1;
            Ok(7)
        })
        .expect("decode should succeed");

        // Assert -- exactly the token limit, and no extra decoder passes.
        assert_eq!(generated, vec![7; 25]);
        assert_eq!(calls, 25);
    }

    #[test]
    fn greedy_decode_feeds_growing_sequence_starting_with_start_token() {
        // Arrange
        let settings = settings(3);
        let mut seen: Vec<Vec<i64>> = Vec::new();

        // Act
        greedy_decode(&settings, None, |tokens, _| {
            seen.push(tokens.to_vec());
            Ok(40 + i64::try_from(tokens.len()).expect("small length"))
        })
        .expect("decode should succeed");

        // Assert
        assert_eq!(seen, vec![vec![2], vec![2, 41], vec![2, 41, 42]]);
    }

    #[test]
    fn greedy_decode_forces_first_and_last_tokens_without_running_model() {
        // Arrange -- BART-style forced tokens around a model that never stops.
        let mut settings = settings(4);
        settings.forced_bos_token_id = Some(0);
        settings.forced_eos_token_id = Some(2);
        let mut calls = 0;

        // Act
        let generated = greedy_decode(&settings, None, |_, _| {
            calls += 1;
            Ok(7)
        })
        .expect("decode should succeed");

        // Assert -- forced tokens cost no decoder pass.
        assert_eq!(generated, vec![0, 7, 7, 2]);
        assert_eq!(calls, 2);
    }

    #[test]
    fn greedy_decode_prefers_forced_end_token_when_limit_is_one() {
        // Arrange
        let mut settings = settings(1);
        settings.forced_bos_token_id = Some(0);
        settings.forced_eos_token_id = Some(2);

        // Act
        let generated =
            greedy_decode(&settings, None, |_, _| Ok(7)).expect("decode should succeed");

        // Assert
        assert_eq!(generated, vec![2]);
    }

    #[test]
    fn greedy_decode_times_out_at_deadline() {
        // Arrange -- a deadline that has already passed.
        let settings = settings(1000);
        let deadline = Some(Instant::now());

        // Act
        let result = greedy_decode(&settings, deadline, |_, _| Ok(7));

        // Assert
        assert!(matches!(result, Err(CoreError::Timeout(_))));
    }

    #[test]
    fn greedy_decode_propagates_model_errors() {
        // Arrange
        let settings = settings(10);

        // Act
        let result = greedy_decode(&settings, None, |_, _| {
            Err(CoreError::Inference("decoder run failed".to_string()))
        });

        // Assert
        assert!(matches!(result, Err(CoreError::Inference(_))));
    }

    // --- No-repeat n-gram rule ---

    fn repeating_tokens(tokens: &[i64], ngram_size: usize) -> Vec<i64> {
        let mut banned = Vec::new();
        extend_with_repeating_tokens(&mut banned, tokens, ngram_size);
        banned
    }

    #[test]
    fn ngram_rule_is_off_at_size_zero() {
        assert!(repeating_tokens(&[5, 5, 5, 5], 0).is_empty());
    }

    #[test]
    fn ngram_rule_waits_for_enough_tokens() {
        // Arrange -- one token cannot be the two-token prefix of a trigram.
        let tokens = [2];

        // Act / Assert
        assert!(repeating_tokens(&tokens, 3).is_empty());
    }

    #[test]
    fn ngram_rule_bans_token_completing_a_seen_trigram() {
        // Arrange -- "7 8 9" has occurred and the sequence ends in "7 8".
        let tokens = [2, 7, 8, 9, 4, 7, 8];

        // Act
        let banned = repeating_tokens(&tokens, 3);

        // Assert
        assert_eq!(banned, vec![9]);
    }

    #[test]
    fn ngram_rule_allows_a_run_once_and_bans_its_repeat() {
        // Arrange -- the start token counts as part of the sequence.
        let before_run = [2, 0, 0];
        let after_run = [2, 0, 0, 0];

        // Act / Assert -- "0 0 0" may form once, but not twice.
        assert!(repeating_tokens(&before_run, 3).is_empty());
        assert_eq!(repeating_tokens(&after_run, 3), vec![0]);
    }

    #[test]
    fn ngram_rule_of_size_one_bans_every_seen_token() {
        // Arrange
        let tokens = [2, 7, 8];

        // Act
        let banned = repeating_tokens(&tokens, 1);

        // Assert
        assert_eq!(banned, vec![2, 7, 8]);
    }

    #[test]
    fn greedy_decode_passes_banned_tokens_to_the_model_step() {
        // Arrange -- a model that would repeat token 0 forever; the rule
        // lets "0 0 0" form once and then forces a different token.
        let mut settings = settings(6);
        settings.forced_bos_token_id = Some(0);
        settings.no_repeat_ngram_size = 3;

        // Act
        let generated = greedy_decode(&settings, None, |_, banned| {
            Ok(if banned.contains(&0) { 9 } else { 0 })
        })
        .expect("decode should succeed");

        // Assert
        assert_eq!(generated, vec![0, 0, 0, 9, 0, 0]);
    }

    // --- Argmax ---

    #[test]
    fn argmax_returns_first_of_equal_maxima() {
        // Arrange
        let logits = [0.5, 2.0, -1.0, 2.0];

        // Act
        let index = argmax_excluding(&logits, &[]).expect("non-empty logits");

        // Assert
        assert_eq!(index, 1);
    }

    #[test]
    fn argmax_skips_banned_tokens() {
        // Arrange
        let logits = [0.5, 2.0, -1.0, 1.5];

        // Act
        let index = argmax_excluding(&logits, &[1]).expect("non-empty logits");

        // Assert
        assert_eq!(index, 3);
    }

    #[test]
    fn argmax_skips_nan_values() {
        // Arrange
        let logits = [f32::NAN, 1.0, f32::NAN, 3.0];

        // Act
        let index = argmax_excluding(&logits, &[]).expect("comparable logits");

        // Assert
        assert_eq!(index, 3);
    }

    #[test]
    fn argmax_rejects_empty_logits() {
        // Arrange
        let logits: [f32; 0] = [];

        // Act
        let result = argmax_excluding(&logits, &[]);

        // Assert
        assert!(matches!(result, Err(CoreError::Inference(_))));
    }
}
