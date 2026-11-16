//! Batched text generation: several requests share each forward pass.
//!
//! A [`GenerationBatch`] holds the generations that are running together.
//! Its owner drives it with two calls on the pipeline:
//! [`TextGenerationPipeline::join`] reads the prompts of new requests and
//! adds them to the batch, and [`TextGenerationPipeline::step`] generates
//! one token for every running request. Both return the requests that
//! ended in that call, so each is answered when it ends, not when the
//! batch does. Requests join between any two steps.
//!
//! Every request gets the tokens the single-request loop picks for it.
//! That rests on how rows are laid out, and graphs differ in the layout
//! they can read. There are two ([`RowLayout`]); which one a graph gets is
//! not guessed from its inputs but tried when the pipeline is built
//! ([`TextGenerationPipeline::check_batching`], below).
//!
//! # Rows that end at the same column ([`RowLayout::EndAligned`])
//!
//! For graphs that apply `attention_mask` as a mask and count positions by
//! column: Optimum exports.
//!
//! - **Rows end at the same column.** The batch is a rectangle of
//!   `columns` cached positions. A row's tokens fill its last columns and
//!   the columns before them are padding (left padding), so the newest
//!   token of every row is in the newest column and the next-token logits
//!   of every row are at the same place.
//! - **Padding is masked and its cache is zero.** `attention_mask` is 0
//!   over a row's padding and 1 over its tokens, cached ones included.
//!   The cache entries the graph computed for padding tokens are replaced
//!   by zeros when the row joins.
//! - **A row never changes columns.** A graph without a `position_ids`
//!   input numbers positions by column, and the cache holds keys that
//!   were computed with those numbers. A row padded by `p` columns is the
//!   same sequence at positions `p, p + 1, ...`: a model with relative
//!   (rotary) positions gives it the same attention, but only as long as
//!   its cached entries and its new tokens keep counting from the same
//!   column. So the batch never drops or inserts a column while it has
//!   rows: it grows by one column per step, and starts again from nothing
//!   when the last row has left. For the same reason the prompt of a
//!   joining request is read in a forward pass of its own that is padded
//!   to the batch's width -- not read alone and shifted afterwards.
//! - **Positions that are fed are the row's own.** A graph with a
//!   `position_ids` input is fed `0, 1, 2, ...` over each row's tokens,
//!   exactly as when it runs alone.
//!
//! A request can therefore join a running batch only if its prompt is no
//! longer than the batch is wide; a longer one waits until the batch has
//! grown or emptied. [`TextGenerationPipeline::admits`] decides, and also
//! refuses a prompt so short that most of its row would be padding.
//!
//! # Rows that start at column 0 ([`RowLayout::StartAligned`])
//!
//! For graphs that read `attention_mask` only as a length: exports of the
//! ONNX Runtime GenAI model builder (most onnx-community language models),
//! whose attention is the fused `GroupQueryAttention` operator. Such a
//! graph takes the number of ones in a mask row as the row's length `n`
//! and nothing else from it. It reads the row's cache from columns
//! `0..n - fed`, numbers the fed tokens `n - fed..n` (rotary positions),
//! writes their keys to those columns and attends to no other column.
//! Where the zeros of a mask row are makes no difference, so a row padded
//! on the left is read as if its padding were tokens -- the layout above
//! gives such a graph wrong tokens from the first one.
//!
//! - **Rows start at column 0** and have their own lengths; the columns
//!   after a row's last token are unused. The mask row of a row with `n`
//!   tokens is `n` ones, then zeros.
//! - **A prompt is read alone**, in the very forward pass the
//!   single-request loop runs, one pass per joining request. Its cache is
//!   then copied next to the others, with zeros on the right up to the
//!   batch's width.
//! - **The batch is as wide as its longest row.** It grows by a column
//!   per step and is cut back when the longest row leaves. Any prompt may
//!   join at any time: a longer one widens the batch.
//!
//! A row is here computed at its own positions, exactly as alone.
//!
//! # Common to both
//!
//! When a row leaves, the cache of the others is copied to a tensor with
//! fewer rows; when rows join, to one with more. Between such changes the
//! cache a step returns is fed to the next step untouched. The copy is
//! made where the cache lives: by the host for a cache in host memory, on
//! the device for one that stays there (`kv_cache.rs`, `device_run.rs`).
//! Both forward passes ([`TextGenerationPipeline::read_prompts`] and
//! [`TextGenerationPipeline::decode_step`]) run either through
//! `Session::run` or, with the cache on the device, through the
//! pipeline's `DeviceRun`; rows, padding, masks and positions are the
//! same in both.
//!
//! What cannot be promised is bit-equal logits on a real model: a row in
//! a batch is computed in other tensor shapes (and, end-aligned, at other
//! positions) than the same request alone, so its logits differ in the
//! last bits and a near tie between two tokens can fall the other way.
//! The fixture graphs compute in whole numbers, where the outputs are
//! exactly equal.
//!
//! # The check at startup
//!
//! No layout is right for every graph, and a wrong one does not fail: it
//! returns fluent wrong text. So a pipeline with a batch width above one
//! proves its layout before it serves
//! ([`TextGenerationPipeline::check_batching`]): three short prompts of
//! different lengths are generated alone and then together, one joining
//! after another, in each layout until one reproduces the logits of the
//! solo runs. If none does, an error is logged and the width is set to
//! one. [`TextGenerationPipeline::batch_check`] tells what happened.

use std::borrow::Cow;
use std::collections::VecDeque;
use std::time::{Duration, Instant};

use ort::session::{Session, SessionInputValue};
use ort::value::{DynValue, TensorRef};
use tokenizers::Tokenizer;

use crate::error::CoreError;

use super::control::{Execution, GenerationControl, TokenUsage, tokens_per_second};
use super::device_run::DeviceRun;
use super::float::float_tensor;
use super::kv_cache::{CacheHome, KvCache, RowPick};
use super::text_generation::{
    GenerationSettings, GraphIo, TextGenerationInput, TextGenerationPipeline, TokenScratch,
};
use super::{PipelineOutput, PreparedData, PreparedInput};

/// User message of the prompt the startup check generates from.
const CHECK_TEXT: &str = "Name three colours, then count from one to five.";
/// Tokens the startup check generates per prompt.
const CHECK_TOKENS: usize = 3;
/// Shortest prompt the startup check cuts its three prompts from.
const CHECK_MIN_PROMPT: usize = 8;
/// Longest one: the check has to stay cheap whatever the template adds.
const CHECK_MAX_PROMPT: usize = 48;
/// How far the logits of a row in a batch may be from those of the same
/// request alone, as a share of the spread of the solo logits (largest
/// minus smallest). Rounding moves them by far less than a hundredth of
/// that; a layout the graph cannot read moves them by most of it.
const CHECK_TOLERANCE: f32 = 0.05;

/// One request for [`TextGenerationPipeline::join`].
pub struct GenerationRequest {
    /// The caller's name for the request; returned in
    /// [`FinishedGeneration::id`].
    pub id: u64,
    /// The prepared prompt.
    pub prepared: PreparedData,
    /// The request's own limits, checked before every one of its steps.
    pub control: GenerationControl,
}

/// A request that left the batch.
pub struct FinishedGeneration {
    /// [`GenerationRequest::id`] of the request.
    pub id: u64,
    /// The generated text with its token counts, or why the request
    /// stopped: [`CoreError::Cancelled`] or [`CoreError::Timeout`] from
    /// its control, or a failure of the model.
    pub result: Result<Execution, CoreError>,
    /// Time from joining to leaving.
    pub elapsed: Duration,
}

/// What one [`join`](TextGenerationPipeline::join) or
/// [`step`](TextGenerationPipeline::step) did.
pub struct BatchProgress {
    /// Requests that ended in this call, in batch order.
    pub finished: Vec<FinishedGeneration>,
    /// Sequences in the call's forward pass; 0 if it ran none.
    pub rows: usize,
}

/// How the rows of a batch lie in its columns; see the module
/// documentation of `generation_batch.rs`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RowLayout {
    /// Rows end at the same column and are padded on the left. For graphs
    /// that apply `attention_mask` as a mask (Optimum exports).
    EndAligned,
    /// Rows start at column 0 and have their own lengths. For graphs that
    /// read `attention_mask` as a length (ONNX Runtime GenAI exports).
    StartAligned,
}

/// What the startup check of batched generation found; see
/// [`TextGenerationPipeline::batch_check`].
#[derive(Debug, Clone, PartialEq)]
pub enum BatchCheck {
    /// The batch width is one: there was nothing to check.
    NotNeeded,
    /// Generating together reproduced generating alone, with this layout.
    Passed(RowLayout),
    /// No layout reproduced generating alone, and the batch width was set
    /// to one. Holds what differed, per layout.
    Failed(String),
}

/// One running generation.
struct Row {
    id: u64,
    control: GenerationControl,
    /// Prompt and generated tokens. The last one is not in the cache yet.
    tokens: Vec<i64>,
    prompt_len: usize,
    /// Columns of padding before the row's first token; 0 in a layout
    /// whose rows start at column 0.
    padding: usize,
    started: Instant,
    /// Time the prompt took, for the log line.
    prompt_elapsed: Duration,
}

impl Row {
    fn generated(&self) -> usize {
        self.tokens.len() - self.prompt_len
    }

    /// Tokens in the cache: all but the newest.
    fn cached(&self) -> usize {
        self.tokens.len() - 1
    }

    /// Whether the row has just ended: an end token, or the token limit.
    fn is_complete(&self, settings: &GenerationSettings) -> bool {
        self.generated() >= settings.max_new_tokens
            || self
                .tokens
                .last()
                .is_some_and(|token| settings.eos_token_ids.contains(token))
    }

    fn fail(self, error: CoreError) -> FinishedGeneration {
        FinishedGeneration {
            id: self.id,
            result: Err(error),
            elapsed: self.started.elapsed(),
        }
    }

    /// Decode the generated tokens and report them.
    fn complete(self, tokenizer: &Tokenizer, max_new_tokens: usize) -> FinishedGeneration {
        let generated = &self.tokens[self.prompt_len..];
        let elapsed = self.started.elapsed();
        tracing::debug!(
            prompt_tokens = self.prompt_len,
            generated_tokens = generated.len(),
            reached_token_limit = generated.len() == max_new_tokens,
            padding = self.padding,
            prompt_ms = self.prompt_elapsed.as_secs_f64() * 1000.0,
            decode_ms = elapsed.saturating_sub(self.prompt_elapsed).as_secs_f64() * 1000.0,
            tokens_per_second = tokens_per_second(generated.len(), elapsed),
            "text generation finished (batched)"
        );
        let result = generated
            .iter()
            .map(|&token| {
                u32::try_from(token).map_err(|_| {
                    CoreError::Inference(format!("invalid token ID {token} in generated text"))
                })
            })
            .collect::<Result<Vec<u32>, CoreError>>()
            .and_then(|ids| {
                tokenizer
                    .decode(&ids, true)
                    .map_err(|e| CoreError::Inference(e.to_string()))
            })
            .map(|text| Execution {
                output: PipelineOutput::TextGeneration(text),
                usage: Some(TokenUsage {
                    prompt_tokens: self.prompt_len,
                    completion_tokens: generated.len(),
                }),
            });
        FinishedGeneration {
            id: self.id,
            result,
            elapsed,
        }
    }
}

/// Sees the next-token logits of every row at every step, before any
/// penalty: `(request id, logits)`.
type LogitsObserver = Box<dyn FnMut(u64, &[f32])>;

/// The generations that are running together; see the module
/// documentation of `generation_batch.rs` for the layout.
///
/// Belongs to the pipeline it was filled by: do not move a batch that has
/// rows to another pipeline.
#[derive(Default)]
pub struct GenerationBatch {
    rows: Vec<Row>,
    /// The cache of `rows`, over `columns` positions; `None` without rows.
    cache: Option<KvCache>,
    /// Positions in the cache. Every row has one more token to feed.
    columns: usize,
    scratch: TokenScratch,
    /// float16 logits of one row, widened.
    widened: Vec<f32>,
    observer: Option<LogitsObserver>,
}

impl GenerationBatch {
    /// A batch with no generation in it.
    pub fn new() -> Self {
        Self::default()
    }

    /// Generations running.
    pub fn len(&self) -> usize {
        self.rows.len()
    }

    /// Whether nothing is running.
    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    /// Cached positions per row, padding included; 0 when empty.
    pub fn columns(&self) -> usize {
        self.columns
    }

    /// Call `observer` with the next-token logits of every row.
    pub(super) fn observe_logits(&mut self, observer: impl FnMut(u64, &[f32]) + 'static) {
        self.observer = Some(Box::new(observer));
    }

    /// Forget every row.
    fn clear(&mut self) {
        self.rows.clear();
        self.cache = None;
        self.columns = 0;
    }

    /// Keep the cache rows `keep`; `self.rows` already holds exactly
    /// those rows, in that order.
    fn keep_cache_rows(
        &mut self,
        keep: &[usize],
        layout: RowLayout,
        home: CacheHome<'_>,
    ) -> Result<(), CoreError> {
        match &self.cache {
            _ if keep.is_empty() => self.clear(),
            Some(cache) if keep.len() < cache.rows() => {
                // Rows that start at column 0 need no column beyond the
                // longest of them; rows that end together keep theirs.
                let columns = match layout {
                    RowLayout::EndAligned => self.columns,
                    RowLayout::StartAligned => longest_cached(&self.rows),
                };
                let picks: Vec<RowPick> = keep
                    .iter()
                    .zip(&self.rows)
                    .map(|(&index, row)| match layout {
                        RowLayout::EndAligned => RowPick::copy(index),
                        RowLayout::StartAligned => RowPick::first(index, row.cached()),
                    })
                    .collect();
                self.cache = Some(KvCache::gather(&[(cache, &picks)], columns, home)?);
                self.columns = columns;
            }
            _ => {}
        }
        Ok(())
    }

    /// End every row with `error` and empty the batch.
    fn fail_all(&mut self, error: &CoreError, finished: &mut Vec<FinishedGeneration>) {
        let message = match error {
            CoreError::Inference(message) => message.clone(),
            other => other.to_string(),
        };
        finished.extend(
            self.rows
                .drain(..)
                .map(|row| row.fail(CoreError::Inference(message.clone()))),
        );
        self.clear();
    }
}

/// Cached tokens of the longest of `rows`.
fn longest_cached(rows: &[Row]) -> usize {
    rows.iter().map(Row::cached).max().unwrap_or(0)
}

/// Whether a prompt of `prompt` tokens may be a row of a batch `columns`
/// wide whose rows end at the same column.
///
/// It must fit, the batch must not be wider than the longest prompt the
/// pipeline accepts (so no row ever reaches further than a request alone
/// can), and its padding must not exceed what the row itself may come to
/// hold (`prompt + max_new_tokens`): padding costs memory and attention
/// like tokens do, and this keeps a row within twice its own worst case.
fn fits(prompt: usize, columns: usize, max_new_tokens: usize, max_input_tokens: usize) -> bool {
    prompt > 0
        && prompt <= columns
        && columns <= max_input_tokens
        && columns - prompt <= prompt + max_new_tokens
}

/// A `(rows, columns)` int64 tensor over `data`.
fn int64_rows(rows: usize, columns: usize, data: &[i64]) -> Result<TensorRef<'_, i64>, CoreError> {
    TensorRef::from_array_view(([rows, columns], data))
        .map_err(|e| CoreError::Inference(e.to_string()))
}

/// Pick the next token of every row from the logits of a forward pass.
///
/// `logits` is a host tensor of shape `(rows, fed, vocab)`, `float32` or
/// `float16`: all `fed` positions the pass was given, or (`fed` = 1) only
/// the last one, as a bound run returns it. Tokens are picked from
/// `float32` values either way.
///
/// The outer error is a pass whose logits cannot be read at all; an inner
/// one is a row for which no token can be chosen.
fn next_tokens(
    logits: &DynValue,
    rows: &[Row],
    fed: usize,
    settings: &GenerationSettings,
    batch: &mut BatchScratch<'_>,
) -> Result<Vec<Result<i64, CoreError>>, CoreError> {
    let (shape, data) = float_tensor(logits)?;
    let dims: Vec<usize> = shape
        .iter()
        .map(|&dim| usize::try_from(dim).unwrap_or(0))
        .collect();
    let vocab_size = match dims.as_slice() {
        [batch, positions, vocab]
            if *batch == rows.len()
                && *positions == fed
                && *vocab > 0
                && data.len() == batch * positions * vocab =>
        {
            *vocab
        }
        _ => {
            return Err(CoreError::Inference(format!(
                "expected logits of shape ({}, {fed}, vocab), got {shape:?}",
                rows.len(),
            )));
        }
    };
    Ok(rows
        .iter()
        .enumerate()
        .map(|(index, row)| {
            // Logits of the last position predict the next token.
            let from = (index * fed + fed - 1) * vocab_size;
            let next_logits = data.widen(from..from + vocab_size, batch.widened);
            if let Some(observer) = batch.observer {
                observer(row.id, next_logits);
            }
            settings.next_token(next_logits, &row.tokens, batch.scratch)
        })
        .collect())
}

/// The buffers of a [`GenerationBatch`] that picking tokens needs.
struct BatchScratch<'a> {
    scratch: &'a mut TokenScratch,
    widened: &'a mut Vec<f32>,
    observer: &'a mut Option<LogitsObserver>,
}

/// What one forward pass feeds.
struct Feeds<'a> {
    /// The cache of the positions already read.
    cache: &'a KvCache,
    /// Positions the cache will cover after the pass.
    columns: usize,
    rows: usize,
    /// Positions fed per row.
    fed: usize,
    input_ids: &'a [i64],
    /// `rows` x (cached + fed) values.
    attention_mask: &'a [i64],
    /// `rows` x `fed` values; fed only if the graph takes them.
    position_ids: &'a [i64],
}

/// One forward pass over `feeds`, through `Session::run` or, with the
/// cache on the device, through `device`.
///
/// Returns each row's next token and the cache the pass produced.
fn forward(
    session: &mut Session,
    device: Option<&mut DeviceRun>,
    io: &GraphIo,
    feeds: &Feeds<'_>,
    rows: &[Row],
    settings: &GenerationSettings,
    scratch: &mut BatchScratch<'_>,
) -> Result<(Vec<Result<i64, CoreError>>, KvCache), CoreError> {
    let Feeds {
        cache,
        columns,
        rows: count,
        fed,
        input_ids,
        attention_mask,
        position_ids,
    } = *feeds;
    let input_ids = int64_rows(count, fed, input_ids)?;
    let attention_mask = int64_rows(count, attention_mask.len() / count.max(1), attention_mask)?;
    let position_ids = int64_rows(count, fed, position_ids)?;

    if let Some(device) = device {
        device.begin();
        device.feed("input_ids", &*input_ids)?;
        device.feed("attention_mask", &*attention_mask)?;
        if io.feeds_position_ids {
            device.feed("position_ids", &*position_ids)?;
        }
        for (slot, value) in io.past.iter().zip(cache.tensors()) {
            device.feed(slot.input.as_str(), value)?;
        }
        let (last, present) = device.run(session, &io.past)?;
        let tokens = next_tokens(&last, rows, 1, settings, scratch)?;
        return Ok((tokens, KvCache::from_tensors(present, count, columns)));
    }

    let mut inputs: Vec<(Cow<'_, str>, SessionInputValue<'_>)> =
        Vec::with_capacity(3 + io.past.len());
    inputs.push((Cow::Borrowed("input_ids"), input_ids.into()));
    inputs.push((Cow::Borrowed("attention_mask"), attention_mask.into()));
    if io.feeds_position_ids {
        inputs.push((Cow::Borrowed("position_ids"), position_ids.into()));
    }
    for (slot, value) in io.past.iter().zip(cache.tensors()) {
        inputs.push((Cow::Borrowed(slot.input.as_str()), value.into()));
    }

    let mut outputs = session
        .run(inputs)
        .map_err(|e| CoreError::Inference(format!("text generation run failed: {e}")))?;
    let logits = outputs
        .get("logits")
        .ok_or_else(|| CoreError::Inference("graph did not return output 'logits'".to_string()))?;
    let tokens = next_tokens(logits, rows, fed, settings, scratch)?;
    let cache = KvCache::from_outputs(&mut outputs, &io.past, count, columns)?;
    Ok((tokens, cache))
}

impl TextGenerationPipeline {
    /// How many generations may share a forward pass: the width asked for
    /// with [`TextGenerationOptions::with_max_batch`](super::TextGenerationOptions::with_max_batch),
    /// or 1 if the graph cannot batch.
    pub fn max_batch(&self) -> usize {
        self.max_batch
    }

    /// Whether a prompt of `prompt_tokens` tokens may join `batch` now,
    /// together with the prompts of `joining` tokens that will join in the
    /// same [`join`](Self::join) call.
    ///
    /// `false` when the batch is full, or when its rows end at the same
    /// column and the prompt does not fit the batch's width (see the
    /// module documentation); the request then waits. A request is always
    /// admitted to an empty batch on its own.
    pub fn admits(&self, batch: &GenerationBatch, joining: &[usize], prompt_tokens: usize) -> bool {
        if batch.len() + joining.len() >= self.max_batch {
            return false;
        }
        if batch.is_empty() && joining.is_empty() {
            return true;
        }
        if self.layout == RowLayout::StartAligned {
            // A row has its own length: any prompt fits.
            return true;
        }
        let prompts = || joining.iter().copied().chain([prompt_tokens]);
        let columns = if batch.is_empty() {
            prompts().max().unwrap_or(0)
        } else {
            batch.columns
        };
        prompts().all(|prompt| {
            fits(
                prompt,
                columns,
                self.settings.max_new_tokens,
                self.prompt.max_input_tokens,
            )
        })
    }

    /// Read the prompts of `requests` and add them to `batch`.
    ///
    /// One forward pass over the new prompts only, padded to the batch's
    /// width -- or, where rows start at column 0, one pass per prompt,
    /// unpadded. The running generations are not part of it and do not
    /// advance. It yields each new request's first token. A request that
    /// ends there (end token, a limit of one token, a failure, a control
    /// that already says stop) is returned and never enters the batch.
    ///
    /// The caller asks [`admits`](Self::admits) first. A request that does
    /// not fit is failed here, not queued.
    pub fn join(
        &mut self,
        batch: &mut GenerationBatch,
        requests: Vec<GenerationRequest>,
    ) -> BatchProgress {
        let deadline = self.deadline_from_now();
        let layout = self.layout;
        let mut finished = Vec::new();
        let mut entrants: Vec<Row> = Vec::with_capacity(requests.len());
        // The width end-aligned rows are padded to.
        let columns = if layout == RowLayout::StartAligned {
            usize::MAX
        } else if batch.is_empty() {
            requests
                .iter()
                .filter_map(|request| request.prepared.token_count())
                .max()
                .unwrap_or(0)
        } else {
            batch.columns
        };
        for request in requests {
            let started = Instant::now();
            let GenerationRequest {
                id,
                prepared,
                control,
            } = request;
            let control = control.limited_to(deadline);
            let refused = match &prepared {
                PreparedData::Audio(_) => Some(CoreError::InvalidInput(
                    "text pipeline received audio prepared data in batch".to_string(),
                )),
                PreparedData::Text(prompt) if prompt.sequence_length == 0 => {
                    Some(CoreError::InvalidInput(
                        "text generation prompt produced no tokens".to_string(),
                    ))
                }
                PreparedData::Text(prompt) if prompt.sequence_length > columns => {
                    Some(CoreError::Inference(format!(
                        "a prompt of {} tokens cannot join a running batch {columns} columns wide",
                        prompt.sequence_length,
                    )))
                }
                PreparedData::Text(_) => control.check("text generation", 0).err(),
            };
            match (refused, prepared) {
                (None, PreparedData::Text(prompt)) => entrants.push(Row {
                    id,
                    control,
                    padding: match layout {
                        RowLayout::EndAligned => columns - prompt.sequence_length,
                        RowLayout::StartAligned => 0,
                    },
                    prompt_len: prompt.sequence_length,
                    tokens: prompt.input_ids,
                    started,
                    prompt_elapsed: Duration::ZERO,
                }),
                (refused, _) => finished.push(FinishedGeneration {
                    id,
                    result: Err(refused.unwrap_or_else(|| {
                        CoreError::InvalidInput("request is not a text prompt".to_string())
                    })),
                    elapsed: started.elapsed(),
                }),
            }
        }
        if entrants.is_empty() {
            return BatchProgress { finished, rows: 0 };
        }
        let rows = entrants.len();

        // The caches the prompts were read into, and for every entrant
        // its first token and its place: (cache, row of that cache).
        let mut caches: Vec<KvCache> = Vec::new();
        let mut read: Vec<(Result<i64, CoreError>, usize, usize)> = Vec::with_capacity(rows);
        match layout {
            RowLayout::EndAligned => match self.read_prompts(batch, &entrants, columns) {
                Ok((cache, tokens)) => {
                    caches.push(cache);
                    read.extend(
                        tokens
                            .into_iter()
                            .enumerate()
                            .map(|(row, token)| (token, 0, row)),
                    );
                }
                Err(error) => {
                    // The running generations were not in this pass: only
                    // the new requests fail.
                    let message = error.to_string();
                    finished.extend(
                        entrants
                            .into_iter()
                            .map(|row| row.fail(CoreError::Inference(message.clone()))),
                    );
                    return BatchProgress { finished, rows };
                }
            },
            RowLayout::StartAligned => {
                for row in &entrants {
                    let alone = std::slice::from_ref(row);
                    let token = match self.read_prompts(batch, alone, row.prompt_len) {
                        Ok((cache, mut tokens)) => {
                            caches.push(cache);
                            tokens.pop().unwrap_or_else(|| {
                                Err(CoreError::Inference(
                                    "the prompt pass returned no token".to_string(),
                                ))
                            })
                        }
                        Err(error) => Err(CoreError::Inference(error.to_string())),
                    };
                    read.push((token, caches.len().saturating_sub(1), 0));
                }
            }
        }

        let settings = &self.settings;
        let mut picks: Vec<Vec<RowPick>> = caches.iter().map(|_| Vec::new()).collect();
        let mut joined = Vec::with_capacity(rows);
        for (mut row, (token, cache, index)) in entrants.into_iter().zip(read) {
            match token {
                Err(error) => finished.push(row.fail(error)),
                Ok(token) => {
                    row.prompt_elapsed = row.started.elapsed();
                    row.tokens.push(token);
                    if row.is_complete(settings) {
                        finished
                            .push(row.complete(&self.prompt.tokenizer, settings.max_new_tokens));
                    } else {
                        picks[cache].push(match layout {
                            RowLayout::EndAligned => RowPick::after_padding(index, row.padding),
                            RowLayout::StartAligned => RowPick::first(index, row.prompt_len),
                        });
                        joined.push(row);
                    }
                }
            }
        }
        if joined.is_empty() {
            return BatchProgress { finished, rows };
        }
        let columns = match layout {
            RowLayout::EndAligned => columns,
            RowLayout::StartAligned => longest_cached(&batch.rows).max(longest_cached(&joined)),
        };

        // The cache of one pass whose rows all go on, whole, is used as
        // it is; everything else is copied into place.
        let as_returned = batch.is_empty()
            && caches.len() == 1
            && picks[0].len() == caches[0].rows()
            && caches[0].columns() == columns
            && picks[0].iter().all(|pick| pick.is_whole(columns));
        let cache = match caches.pop() {
            Some(cache) if as_returned => Ok(cache),
            last => {
                caches.extend(last);
                let running: Vec<RowPick> = (0..batch.len()).map(RowPick::copy).collect();
                let mut parts: Vec<(&KvCache, &[RowPick])> = Vec::with_capacity(1 + caches.len());
                if let Some(cache) = &batch.cache {
                    parts.push((cache, &running[..]));
                }
                parts.extend(
                    caches
                        .iter()
                        .zip(&picks)
                        .map(|(cache, picks)| (cache, &picks[..])),
                );
                KvCache::gather(&parts, columns, self.cache_home())
            }
        };
        match cache {
            Ok(cache) => {
                batch.cache = Some(cache);
                batch.columns = columns;
                batch.rows.extend(joined);
            }
            Err(error) => {
                let message = error.to_string();
                finished.extend(
                    joined
                        .into_iter()
                        .map(|row| row.fail(CoreError::Inference(message.clone()))),
                );
            }
        }
        BatchProgress { finished, rows }
    }

    /// Generate one token for every generation in `batch`.
    ///
    /// First asks each row's control whether to go on; a row that must
    /// stop leaves with [`CoreError::Cancelled`] or [`CoreError::Timeout`]
    /// and the others run without it. Rows that produce an end token or
    /// reach the token limit leave with their text. A failure of the
    /// forward pass itself ends every row in it.
    pub fn step(&mut self, batch: &mut GenerationBatch) -> BatchProgress {
        let mut finished = Vec::new();

        // 1. Limits, as the single-request loop checks them before a step.
        let mut keep = Vec::with_capacity(batch.len());
        for (index, row) in std::mem::take(&mut batch.rows).into_iter().enumerate() {
            match row.control.check("text generation", row.generated()) {
                Ok(()) => {
                    keep.push(index);
                    batch.rows.push(row);
                }
                Err(error) => finished.push(row.fail(error)),
            }
        }
        let layout = self.layout;
        if let Err(error) = batch.keep_cache_rows(&keep, layout, self.cache_home()) {
            batch.fail_all(&error, &mut finished);
        }
        if batch.is_empty() {
            return BatchProgress { finished, rows: 0 };
        }
        let rows = batch.len();

        // 2. One forward pass over the newest token of every row.
        let tokens = match self.decode_step(batch) {
            Ok(tokens) => tokens,
            Err(error) => {
                batch.fail_all(&error, &mut finished);
                return BatchProgress { finished, rows };
            }
        };

        // 3. Rows that ended leave; the cache keeps the others.
        let settings = &self.settings;
        let mut keep = Vec::with_capacity(rows);
        let advanced = std::mem::take(&mut batch.rows).into_iter().zip(tokens);
        for (index, (mut row, token)) in advanced.enumerate() {
            match token {
                Err(error) => finished.push(row.fail(error)),
                Ok(token) => {
                    row.tokens.push(token);
                    if row.is_complete(settings) {
                        finished
                            .push(row.complete(&self.prompt.tokenizer, settings.max_new_tokens));
                    } else {
                        keep.push(index);
                        batch.rows.push(row);
                    }
                }
            }
        }
        if let Err(error) = batch.keep_cache_rows(&keep, layout, self.cache_home()) {
            batch.fail_all(&error, &mut finished);
        }
        BatchProgress { finished, rows }
    }

    /// Where this pipeline's cache tensors live.
    fn cache_home(&mut self) -> CacheHome<'_> {
        match self.device.as_mut() {
            Some(device) => CacheHome::Device(device),
            None => CacheHome::Host(self.io.cache_type),
        }
    }

    /// Run a batch of requests through a [`GenerationBatch`], up to the
    /// batch width at a time, each joining as soon as there is room and
    /// its prompt fits. Returns one result per request, in order.
    ///
    /// The time budget covers the batch; `controls[i]`, when present,
    /// limits request `i` further.
    pub(super) fn execute_batch_together(
        &mut self,
        requests: Vec<PreparedData>,
        controls: &[GenerationControl],
    ) -> Vec<Result<Execution, CoreError>> {
        let deadline = self.deadline_from_now();
        let unlimited = GenerationControl::default();
        let mut results: Vec<Option<Result<Execution, CoreError>>> =
            requests.iter().map(|_| None).collect();
        let mut waiting: VecDeque<GenerationRequest> = requests
            .into_iter()
            .enumerate()
            .map(|(index, prepared)| GenerationRequest {
                id: index as u64,
                prepared,
                control: controls
                    .get(index)
                    .unwrap_or(&unlimited)
                    .limited_to(deadline),
            })
            .collect();
        let mut batch = GenerationBatch::new();
        let mut record = |progress: BatchProgress| {
            for done in progress.finished {
                if let Some(slot) = usize::try_from(done.id)
                    .ok()
                    .and_then(|i| results.get_mut(i))
                {
                    *slot = Some(done.result);
                }
            }
        };

        loop {
            let mut joining = Vec::new();
            let mut prompts = Vec::new();
            while let Some(next) = waiting.front() {
                let prompt = next.prepared.token_count().unwrap_or(0);
                if !self.admits(&batch, &prompts, prompt) {
                    break;
                }
                prompts.push(prompt);
                joining.extend(waiting.pop_front());
            }
            if !joining.is_empty() {
                record(self.join(&mut batch, joining));
            }
            if batch.is_empty() {
                if waiting.is_empty() {
                    break;
                }
                // An empty batch admits whoever is first.
                continue;
            }
            record(self.step(&mut batch));
        }

        results
            .into_iter()
            .map(|result| {
                result.unwrap_or_else(|| {
                    Err(CoreError::Inference(
                        "batched generation lost a request".to_string(),
                    ))
                })
            })
            .collect()
    }

    /// The forward pass that reads the prompts of `entrants`, each padded
    /// on the left to `columns` (a prompt read alone has no padding).
    ///
    /// Returns the cache of that pass (one row per entrant, padding not
    /// yet zeroed) and each entrant's first token.
    fn read_prompts(
        &mut self,
        batch: &mut GenerationBatch,
        entrants: &[Row],
        columns: usize,
    ) -> Result<(KvCache, Vec<Result<i64, CoreError>>), CoreError> {
        let Self {
            session,
            io,
            settings,
            device,
            ..
        } = self;
        let rows = entrants.len();
        // Any token will do under the mask.
        let padding_token = settings.eos_token_ids.first().copied().unwrap_or(0);
        let mut input_ids = Vec::with_capacity(rows * columns);
        let mut attention_mask = Vec::with_capacity(rows * columns);
        let mut position_ids = Vec::with_capacity(rows * columns);
        for row in entrants {
            input_ids.extend(std::iter::repeat_n(padding_token, row.padding));
            input_ids.extend_from_slice(&row.tokens);
            attention_mask.extend(std::iter::repeat_n(0_i64, row.padding));
            attention_mask.extend(std::iter::repeat_n(1_i64, row.prompt_len));
            // The row's own positions; padding gets a valid one.
            position_ids.extend(std::iter::repeat_n(0_i64, row.padding));
            position_ids.extend((0_i64..).take(row.prompt_len));
        }
        let empty = KvCache::empty(&io.past, rows, io.cache_type)?;
        let feeds = Feeds {
            cache: &empty,
            columns,
            rows,
            fed: columns,
            input_ids: &input_ids,
            attention_mask: &attention_mask,
            position_ids: &position_ids,
        };
        let (tokens, cache) = forward(
            session,
            device.as_mut(),
            io,
            &feeds,
            entrants,
            settings,
            &mut BatchScratch {
                scratch: &mut batch.scratch,
                widened: &mut batch.widened,
                observer: &mut batch.observer,
            },
        )?;
        Ok((cache, tokens))
    }

    /// The forward pass of one step: feeds the newest token of every row
    /// with the batch's cache, stores the cache it returns and returns
    /// each row's next token.
    fn decode_step(
        &mut self,
        batch: &mut GenerationBatch,
    ) -> Result<Vec<Result<i64, CoreError>>, CoreError> {
        let Self {
            session,
            io,
            settings,
            device,
            layout,
            ..
        } = self;
        let rows = batch.rows.len();
        let columns = batch.columns + 1;
        let mut input_ids = Vec::with_capacity(rows);
        let mut attention_mask = Vec::with_capacity(rows * columns);
        let mut position_ids = Vec::with_capacity(rows);
        for row in &batch.rows {
            let newest = row.tokens.len() - 1;
            input_ids.push(row.tokens[newest]);
            match layout {
                RowLayout::EndAligned => {
                    attention_mask.extend(std::iter::repeat_n(0_i64, row.padding));
                    attention_mask.extend(std::iter::repeat_n(1_i64, columns - row.padding));
                }
                // As many ones as the row has tokens, the newest included.
                RowLayout::StartAligned => {
                    attention_mask.extend(std::iter::repeat_n(1_i64, row.tokens.len()));
                    attention_mask.extend(std::iter::repeat_n(0_i64, columns - row.tokens.len()));
                }
            }
            position_ids.push(i64::try_from(newest).map_err(|_| {
                CoreError::Inference(format!("position {newest} does not fit a position ID"))
            })?);
        }
        let cache = batch.cache.take().ok_or_else(|| {
            CoreError::Inference("generation batch has rows but no cache".to_string())
        })?;

        let feeds = Feeds {
            cache: &cache,
            columns,
            rows,
            fed: 1,
            input_ids: &input_ids,
            attention_mask: &attention_mask,
            position_ids: &position_ids,
        };
        let (tokens, present) = forward(
            session,
            device.as_mut(),
            io,
            &feeds,
            &batch.rows,
            settings,
            &mut BatchScratch {
                scratch: &mut batch.scratch,
                widened: &mut batch.widened,
                observer: &mut batch.observer,
            },
        )?;
        batch.cache = Some(present);
        batch.columns = columns;
        Ok(tokens)
    }
}

/// The next-token logits of every step of one generation.
type Trace = Vec<Vec<f32>>;

/// Compare the logits of requests generated together (`batched`) with
/// those of the same requests alone (`solo`), request by request and step
/// by step.
///
/// Returns the largest difference found, as a share of the spread of the
/// solo logits of its step.
///
/// # Errors
///
/// Returns what differs: logits further apart than [`CHECK_TOLERANCE`],
/// logits that are not numbers, or a different number of steps. Two runs
/// that pick different tokens from logits within the tolerance (a near
/// tie) are compared up to that step only.
fn compare_traces(solo: &[Trace], batched: &[Trace]) -> Result<f32, String> {
    let mut largest = 0.0_f32;
    for (request, (alone, together)) in solo.iter().zip(batched).enumerate() {
        let mut parted = false;
        for (step, (a, b)) in alone.iter().zip(together).enumerate() {
            let (mut low, mut high) = (f32::INFINITY, f32::NEG_INFINITY);
            let mut difference = 0.0_f32;
            let mut numbers = a.len() == b.len() && !a.is_empty();
            for (x, y) in a.iter().zip(b) {
                numbers &= x.is_finite() && y.is_finite();
                low = low.min(*x);
                high = high.max(*x);
                difference = difference.max((x - y).abs());
            }
            if !numbers {
                return Err(format!(
                    "request {request}, token {step}: the logits are not {} numbers in both runs",
                    a.len(),
                ));
            }
            let share = difference / (high - low).max(f32::MIN_POSITIVE);
            if share > CHECK_TOLERANCE {
                return Err(format!(
                    "request {request}, token {step}: logits differ by {difference:.3e}, {:.0}% of \
                     their spread; alone the model picks token {}, in the batch {}",
                    share * 100.0,
                    best(a),
                    best(b),
                ));
            }
            largest = largest.max(share);
            if best(a) != best(b) {
                parted = true;
                break;
            }
        }
        if !parted && alone.len() != together.len() {
            return Err(format!(
                "request {request}: {} tokens alone, {} in the batch",
                alone.len(),
                together.len(),
            ));
        }
    }
    if solo.len() != batched.len() {
        return Err(format!(
            "{} requests alone, {} in the batch",
            solo.len(),
            batched.len()
        ));
    }
    Ok(largest)
}

/// Index of the largest of `logits`.
fn best(logits: &[f32]) -> usize {
    let mut best = 0;
    for (index, logit) in logits.iter().enumerate() {
        if *logit > logits[best] {
            best = index;
        }
    }
    best
}

impl TextGenerationPipeline {
    /// What the check of batched generation found when the pipeline was
    /// built: whether generating together reproduces generating alone on
    /// this graph and provider, and with which row layout. After
    /// [`BatchCheck::Failed`] the batch width is one.
    pub fn batch_check(&self) -> &BatchCheck {
        &self.batch_check
    }

    /// Prove that this pipeline may batch, or stop it from batching.
    ///
    /// Generates [`CHECK_TOKENS`] tokens for three prompts of different
    /// lengths alone, then together -- one joining after another -- in
    /// each [`RowLayout`] until one reproduces the logits of the solo
    /// runs. That layout is kept. If none does, the batch width becomes
    /// one and an error is logged: the pipeline then serves every request
    /// alone, which is always right.
    ///
    /// Costs three prompt passes and six single-token passes alone, and
    /// about as many per layout tried, all over prompts of at most
    /// [`CHECK_MAX_PROMPT`] tokens.
    pub(super) fn check_batching(&mut self) -> &BatchCheck {
        if self.max_batch <= 1 {
            self.batch_check = BatchCheck::NotNeeded;
            return &self.batch_check;
        }
        let started = Instant::now();
        // Few tokens per prompt, for the solo and the batched runs alike.
        let max_new_tokens = std::mem::replace(&mut self.settings.max_new_tokens, CHECK_TOKENS);
        let outcome = self.find_layout();
        self.settings.max_new_tokens = max_new_tokens;
        let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
        self.batch_check = match outcome {
            Ok((layout, difference)) => {
                self.layout = layout;
                tracing::info!(
                    ?layout,
                    max_batch = self.max_batch,
                    largest_logit_difference = difference,
                    elapsed_ms,
                    "batched text generation reproduces single-request generation"
                );
                BatchCheck::Passed(layout)
            }
            Err(detail) => {
                tracing::error!(
                    requested_max_batch = self.max_batch,
                    detail = %detail,
                    elapsed_ms,
                    "batched text generation does not reproduce single-request generation on \
                     this graph; generating one request at a time (TEXT_GENERATION_MAX_BATCH is \
                     ignored)"
                );
                self.layout = RowLayout::EndAligned;
                self.max_batch = 1;
                BatchCheck::Failed(detail)
            }
        };
        &self.batch_check
    }

    /// The first layout in which the check prompts get, together, the
    /// logits they get alone, with the largest difference seen.
    fn find_layout(&mut self) -> Result<(RowLayout, f32), String> {
        let prompts = self.check_prompts()?;
        let mut solo: Vec<Trace> = Vec::with_capacity(prompts.len());
        for prompt in &prompts {
            let mut trace = Trace::new();
            self.generate(
                prompt,
                &GenerationControl::default(),
                CHECK_TOKENS,
                &mut |logits| trace.push(logits.to_vec()),
            )
            .map_err(|e| format!("cannot generate alone: {e}"))?;
            solo.push(trace);
        }
        let mut failures = Vec::new();
        for layout in [RowLayout::EndAligned, RowLayout::StartAligned] {
            self.layout = layout;
            match self
                .check_together(&prompts)
                .and_then(|batched| compare_traces(&solo, &batched))
            {
                Ok(difference) => return Ok((layout, difference)),
                Err(detail) => failures.push(format!("{layout:?}: {detail}")),
            }
        }
        let lengths: Vec<usize> = prompts.iter().map(|p| p.sequence_length).collect();
        Err(format!(
            "requests 0, 1, 2 are prompts of {lengths:?} tokens -- {}",
            failures.join("; ")
        ))
    }

    /// Three prompts of different lengths, longest first: the tokens of a
    /// rendered chat prompt, and its last three quarters and last half.
    /// Any token sequence does for the check; these are at least tokens
    /// the model knows.
    fn check_prompts(&self) -> Result<Vec<PreparedInput>, String> {
        let render = |text: &str| {
            self.prompt.prepare(TextGenerationInput {
                text: text.to_string(),
                // No system message: the default one may be long.
                system: Some(String::new()),
            })
        };
        let mut tokens = render(CHECK_TEXT)
            .or_else(|_| render("Hello"))
            .map_err(|e| format!("cannot build a prompt: {e}"))?
            .input_ids;
        while !tokens.is_empty() && tokens.len() < CHECK_MIN_PROMPT {
            tokens.extend_from_within(..);
        }
        let longest = CHECK_MAX_PROMPT.min(self.prompt.max_input_tokens);
        let cut = tokens.len().saturating_sub(longest);
        let tokens = &tokens[cut..];
        if tokens.len() < 4 {
            return Err(format!("a prompt of {} tokens is too short", tokens.len()));
        }
        Ok([0, tokens.len() / 4, tokens.len() / 2]
            .into_iter()
            .map(|from| {
                let input_ids = tokens[from..].to_vec();
                PreparedInput {
                    attention_mask: vec![1; input_ids.len()],
                    sequence_length: input_ids.len(),
                    input_ids,
                    encoding: None,
                }
            })
            .collect())
    }

    /// Generate for `prompts` (longest first) through a batch in the
    /// pipeline's current layout and return each one's logits.
    ///
    /// The shortest starts; the longest and then the middle one join, each
    /// a step after the one before or as soon as the batch admits it. So
    /// in either layout a request joins a batch that is running, a row
    /// shares a pass with a longer one, and rows leave before others.
    fn check_together(&mut self, prompts: &[PreparedInput]) -> Result<Vec<Trace>, String> {
        use std::cell::RefCell;
        use std::rc::Rc;

        let traces: Rc<RefCell<Vec<Trace>>> =
            Rc::new(RefCell::new(vec![Trace::new(); prompts.len()]));
        let mut batch = GenerationBatch::new();
        let seen = Rc::clone(&traces);
        batch.observe_logits(move |id, logits| {
            let index = usize::try_from(id).unwrap_or(usize::MAX);
            if let Some(trace) = seen.borrow_mut().get_mut(index) {
                trace.push(logits.to_vec());
            }
        });
        let mut waiting: VecDeque<usize> = (0..prompts.len()).rev().collect();
        // The shortest first, then the longest, then the rest.
        if waiting.len() > 2 {
            waiting.swap(1, 2);
        }
        let mut failed = None;
        let mut record = |progress: BatchProgress| {
            for done in progress.finished {
                if let Err(error) = done.result {
                    failed.get_or_insert(format!("request {} failed: {error}", done.id));
                }
            }
        };
        let mut joined_running = false;
        // Every pass either starts a request or generates a token.
        let mut passes = 2 * prompts.len() * (CHECK_TOKENS + 1);
        while !(waiting.is_empty() && batch.is_empty()) {
            passes = passes
                .checked_sub(1)
                .ok_or("the batch did not come to an end")?;
            let admitted = waiting
                .front()
                .copied()
                .filter(|&next| self.admits(&batch, &[], prompts[next].sequence_length));
            if let Some(next) = admitted {
                waiting.pop_front();
                joined_running |= !batch.is_empty();
                let request = GenerationRequest {
                    id: next as u64,
                    prepared: PreparedData::Text(PreparedInput {
                        input_ids: prompts[next].input_ids.clone(),
                        attention_mask: prompts[next].attention_mask.clone(),
                        sequence_length: prompts[next].sequence_length,
                        encoding: None,
                    }),
                    control: GenerationControl::default(),
                };
                record(self.join(&mut batch, vec![request]));
            }
            if !batch.is_empty() {
                record(self.step(&mut batch));
            }
        }
        drop(batch);
        if let Some(failed) = failed {
            return Err(failed);
        }
        if !joined_running {
            return Err("no request could join a running batch".to_string());
        }
        Ok(traces.take())
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::collections::HashMap;
    use std::path::Path;
    use std::rc::Rc;

    use super::*;
    use crate::ep::ExecutionProvider;
    use crate::pipeline::{Pipeline, PreparedInput, TextGenerationInput, TextGenerationOptions};

    #[test]
    fn prompt_fits_a_batch_that_is_wide_enough_and_not_mostly_padding() {
        // 10 new tokens at most, prompts of at most 100 tokens.
        let fits = |prompt, columns| fits(prompt, columns, 10, 100);

        // Exactly as wide as the prompt, or wider.
        assert!(fits(20, 20));
        assert!(fits(20, 21));
        // Not narrower: the row would have to start before column 0.
        assert!(!fits(20, 19));
        // Padding up to prompt + new tokens, and no more.
        assert!(fits(20, 50));
        assert!(!fits(20, 51));
        // Never wider than the longest prompt the pipeline accepts.
        assert!(fits(60, 100));
        assert!(!fits(60, 101));
        // An empty prompt fits nowhere.
        assert!(!fits(0, 0));
        assert!(!fits(0, 5));
    }

    #[test]
    fn batched_logits_are_compared_with_solo_logits_step_by_step() {
        // Arrange -- two requests of two and one steps, over 4 tokens.
        let solo: Vec<Trace> = vec![
            vec![vec![0.0, 4.0, 1.0, -6.0], vec![2.0, 0.5, 9.0, -1.0]],
            vec![vec![1.0, 1.5, -8.0, 0.0]],
        ];
        let moved = |by: f32| {
            let mut batched = solo.clone();
            batched[0][1][3] += by;
            batched
        };

        // Act / Assert -- equal, and apart by rounding: the share of the
        // spread (10 at that step) is reported.
        assert_eq!(compare_traces(&solo, &solo), Ok(0.0));
        let share = compare_traces(&solo, &moved(0.01)).expect("rounding");
        assert!((share - 0.001).abs() < 1e-6, "{share}");
        // Apart by more than the tolerance, even where the best token is
        // the same: the request and the step are named.
        let error = compare_traces(&solo, &moved(0.6)).expect_err("too far");
        assert!(error.contains("request 0, token 1"), "{error}");
        assert!(error.contains("6% of their spread"), "{error}");
        assert!(error.contains("picks token 2, in the batch 2"), "{error}");
        // Logits that are not numbers never pass.
        for bad in [f32::NAN, f32::INFINITY] {
            let mut batched = solo.clone();
            batched[1][0][0] = bad;
            let error = compare_traces(&solo, &batched).expect_err("not a number");
            assert!(error.contains("request 1, token 0"), "{error}");
        }
        // A step or a request too few or too many.
        let mut short = solo.clone();
        short[0].pop();
        let error = compare_traces(&solo, &short).expect_err("a step is missing");
        assert!(
            error.contains("request 0: 2 tokens alone, 1 in the batch"),
            "{error}"
        );
        assert!(compare_traces(&short, &solo).is_err());
        assert!(compare_traces(&solo, &solo[..1]).is_err());
        let different_vocabulary = vec![vec![vec![0.0, 4.0, 1.0]], solo[1].clone()];
        assert!(compare_traces(&solo, &different_vocabulary).is_err());
    }

    #[test]
    fn near_tie_that_falls_the_other_way_ends_the_comparison_of_that_request() {
        // Arrange -- two candidates 0.001 apart alone; in the batch the
        // other one wins, and the two runs go different ways from there.
        let solo: Vec<Trace> = vec![vec![vec![5.0, 5.001, -5.0], vec![9.0, 0.0, -1.0]]];
        let batched: Vec<Trace> = vec![vec![vec![5.001, 5.0, -5.0], vec![-3.0, 7.0, 0.0]]];
        let ended_early: Vec<Trace> = vec![vec![vec![5.001, 5.0, -5.0]]];

        // Act / Assert -- rounding, not a wrong layout.
        assert!(compare_traces(&solo, &batched).is_ok());
        assert!(compare_traces(&solo, &ended_early).is_ok());
    }

    /// What one step of a generation looked like: the token picked, how
    /// far ahead of the runner-up its logit was, and the logits themselves
    /// for the first steps.
    struct Step {
        best: usize,
        margin: f32,
        logits: Option<Vec<f32>>,
    }

    impl Step {
        fn of(logits: &[f32], keep: bool) -> Self {
            let mut best = 0;
            let mut runner_up = f32::NEG_INFINITY;
            for (index, &logit) in logits.iter().enumerate().skip(1) {
                if logit > logits[best] {
                    runner_up = logits[best];
                    best = index;
                } else if logit > runner_up {
                    runner_up = logit;
                }
            }
            Self {
                best,
                margin: logits[best] - runner_up,
                logits: keep.then(|| logits.to_vec()),
            }
        }
    }

    /// Batched against solo generation on a real model.
    ///
    /// Needs an exported model, so it never runs by default. Set
    ///
    /// - `HEPHAESTUS_TEXTGEN_BATCH_MODEL_DIR`: a model directory as the
    ///   server loads it (`config.json`, `tokenizer.json`, the chat
    ///   template, `onnx/<graph>` with its external data);
    /// - `HEPHAESTUS_TEXTGEN_BATCH_PROMPTS`: a JSON file
    ///   `{"max_new_tokens": N, "prompts": [{"text": "...", "system": "..."}]}`
    ///   (`system` optional) with prompts of clearly different lengths;
    /// - optionally `HEPHAESTUS_TEXTGEN_BATCH_GRAPH` (for example
    ///   `model_q4.onnx`; default `model.onnx`),
    ///   `HEPHAESTUS_TEXTGEN_BATCH_WIDTH` (default 4) and
    ///   `HEPHAESTUS_TEXTGEN_BATCH_LOGIT_STEPS` (default 8: steps whose
    ///   full logits are compared);
    ///
    /// and run
    ///
    /// `cargo test -p hephaestus-core --release --lib batched_generation_matches_solo_on_a_real_model -- --ignored --nocapture`
    ///
    /// Every prompt is generated alone, then all are generated through a
    /// batch three times: in the given order, in reverse, and with each
    /// request joining three steps after the one before (or as soon as it
    /// fits). For every prompt and run it prints whether the tokens are
    /// those of the solo run, the largest difference between solo and
    /// batched logits over the first steps, and, if the tokens differ, the
    /// step where they part and how far apart the two best logits of the
    /// solo run were there. The test fails if any token differs.
    ///
    /// A difference at a step where the solo margin is of the size of the
    /// logit differences is a near tie decided by rounding. A difference
    /// at a clear margin, or logit differences far above 1e-2, mean the
    /// row layout the startup check chose (printed first) does not hold
    /// on these prompts: do not run that model with a batch width above
    /// one, and keep the output of this test.
    ///
    /// Loading the pipeline runs the startup check itself; if it finds no
    /// layout the test stops there and prints what the check found.
    #[test]
    #[ignore = "requires an exported model and a prompts file (see doc comment)"]
    fn batched_generation_matches_solo_on_a_real_model() {
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
        let file: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(prompts_path).expect("read prompts"))
                .expect("parse prompts");
        let max_new_tokens =
            usize::try_from(file["max_new_tokens"].as_u64().expect("max_new_tokens"))
                .expect("limit fits usize");
        let mut options = TextGenerationOptions::new()
            .with_max_new_tokens(max_new_tokens)
            .with_max_batch(width);
        if let Some(graph) = variable("HEPHAESTUS_TEXTGEN_BATCH_GRAPH") {
            options = options.with_graph_file(graph);
        }
        let mut pipeline = TextGenerationPipeline::with_options(
            Path::new(&model_dir),
            &ExecutionProvider::Cpu,
            &options,
        )
        .expect("load");
        println!("startup check: {:?}", pipeline.batch_check());
        assert_eq!(
            pipeline.max_batch(),
            width,
            "the graph must be able to batch: {:?}",
            pipeline.batch_check()
        );
        let prepare = |pipeline: &TextGenerationPipeline| -> Vec<PreparedInput> {
            file["prompts"]
                .as_array()
                .expect("prompts array")
                .iter()
                .map(|prompt| {
                    pipeline
                        .prepare(TextGenerationInput {
                            text: prompt["text"].as_str().expect("prompt text").to_string(),
                            system: prompt["system"].as_str().map(str::to_string),
                        })
                        .expect("prepare")
                })
                .collect()
        };

        // Act -- alone.
        let mut solo: Vec<Vec<Step>> = Vec::new();
        for prepared in prepare(&pipeline) {
            let mut steps = Vec::new();
            let tokens = pipeline
                .generate(
                    &prepared,
                    &GenerationControl::default(),
                    max_new_tokens,
                    &mut |logits| {
                        let keep = steps.len() < logit_steps;
                        steps.push(Step::of(logits, keep));
                    },
                )
                .expect("solo generation");
            println!(
                "prompt {}: {} prompt tokens, {} generated alone",
                solo.len(),
                prepared.sequence_length,
                tokens.len()
            );
            solo.push(steps);
        }

        // Act -- together: (name, order, steps between joins).
        let count = solo.len();
        let runs: [(&str, Vec<usize>, usize); 3] = [
            ("together", (0..count).collect(), 0),
            ("reversed", (0..count).rev().collect(), 0),
            ("staggered", (0..count).collect(), 3),
        ];
        let mut all_equal = true;
        for (name, order, stagger) in runs {
            let seen: Rc<RefCell<HashMap<u64, Vec<Step>>>> = Rc::default();
            let mut batch = GenerationBatch::new();
            let observed = Rc::clone(&seen);
            batch.observe_logits(move |id, logits| {
                let mut seen = observed.borrow_mut();
                let steps = seen.entry(id).or_default();
                let keep = steps.len() < logit_steps;
                steps.push(Step::of(logits, keep));
            });
            let mut prepared: Vec<Option<PreparedInput>> =
                prepare(&pipeline).into_iter().map(Some).collect();
            let mut waiting: VecDeque<usize> = order.into();
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
            while !(waiting.is_empty() && batch.is_empty()) {
                let mut joining = Vec::new();
                let mut prompts = Vec::new();
                while let Some(&next) = waiting.front() {
                    let tokens = prepared[next]
                        .as_ref()
                        .map_or(0, |prompt| prompt.sequence_length);
                    let due = batch.is_empty() || since_join >= stagger;
                    if !due || !pipeline.admits(&batch, &prompts, tokens) {
                        break;
                    }
                    waiting.pop_front();
                    prompts.push(tokens);
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

            // Report.
            let seen = seen.borrow();
            for (index, alone) in solo.iter().enumerate() {
                let together = &seen[&(index as u64)];
                results
                    .remove(&(index as u64))
                    .expect("request finished")
                    .expect("batched generation");
                let mut max_logit_diff = 0.0_f32;
                for (a, b) in alone.iter().zip(together) {
                    if let (Some(a), Some(b)) = (&a.logits, &b.logits) {
                        for (x, y) in a.iter().zip(b) {
                            max_logit_diff = max_logit_diff.max((x - y).abs());
                        }
                    }
                }
                let parted = alone
                    .iter()
                    .zip(together)
                    .position(|(a, b)| a.best != b.best)
                    .or_else(|| {
                        (alone.len() != together.len()).then(|| alone.len().min(together.len()))
                    });
                match parted {
                    None => println!(
                        "{name} (widest {widest}): prompt {index} equal over {} tokens, max |logit \
                         difference| over the first {logit_steps} steps {max_logit_diff:.2e}",
                        alone.len(),
                    ),
                    Some(step) => {
                        all_equal = false;
                        println!(
                            "{name} (widest {widest}): prompt {index} DIFFERS at step {step} of {}: \
                             solo margin there {:.2e}, max |logit difference| over the first \
                             {logit_steps} steps {max_logit_diff:.2e}",
                            alone.len(),
                            alone.get(step).map_or(f32::NAN, |step| step.margin),
                        );
                    }
                }
            }
        }

        // Assert
        assert!(
            all_equal,
            "batched tokens differ from solo tokens; see the lines above"
        );
    }
}
