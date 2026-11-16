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
//! That rests on how rows are laid out:
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
//! What cannot be promised is bit-equal logits on a real model: a padded
//! row is computed at other positions and in other tensor shapes than the
//! same request alone, so its logits differ in the last bits and a near
//! tie between two tokens can fall the other way. The fixture graphs
//! compute in whole numbers, where the outputs are exactly equal.

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
use super::text_generation::{GenerationSettings, GraphIo, TextGenerationPipeline, TokenScratch};
use super::{PipelineOutput, PreparedData};

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

/// One running generation.
struct Row {
    id: u64,
    control: GenerationControl,
    /// Prompt and generated tokens. The last one is not in the cache yet.
    tokens: Vec<i64>,
    prompt_len: usize,
    /// Columns of padding before the row's first token.
    padding: usize,
    started: Instant,
    /// Time the prompt took, for the log line.
    prompt_elapsed: Duration,
}

impl Row {
    fn generated(&self) -> usize {
        self.tokens.len() - self.prompt_len
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
    #[cfg(test)]
    pub(super) fn observe_logits(&mut self, observer: impl FnMut(u64, &[f32]) + 'static) {
        self.observer = Some(Box::new(observer));
    }

    /// Forget every row.
    fn clear(&mut self) {
        self.rows.clear();
        self.cache = None;
        self.columns = 0;
    }

    /// Keep the cache rows in `keep`; `self.rows` already holds exactly
    /// those rows, in that order.
    fn keep_cache_rows(&mut self, keep: &[RowPick], home: CacheHome<'_>) -> Result<(), CoreError> {
        match &self.cache {
            _ if keep.is_empty() => self.clear(),
            Some(cache) if keep.len() < cache.rows() => {
                self.cache = Some(KvCache::gather(&[(cache, keep)], home)?);
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

/// Whether a prompt of `prompt` tokens may be a row of a batch `columns`
/// wide.
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
    /// `false` when the batch is full, or when the prompt does not fit the
    /// batch's width (see the module documentation); the request then
    /// waits. A request is always admitted to an empty batch on its own.
    pub fn admits(&self, batch: &GenerationBatch, joining: &[usize], prompt_tokens: usize) -> bool {
        if batch.len() + joining.len() >= self.max_batch {
            return false;
        }
        if batch.is_empty() && joining.is_empty() {
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
    /// width; the running generations are not part of it and do not
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
        let mut finished = Vec::new();
        let mut entrants: Vec<Row> = Vec::with_capacity(requests.len());
        let columns = if batch.is_empty() {
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
                    padding: columns - prompt.sequence_length,
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

        let (prefilled, tokens) = match self.read_prompts(batch, &entrants, columns) {
            Ok(read) => read,
            Err(error) => {
                // The running generations were not in this pass: only the
                // new requests fail.
                let message = error.to_string();
                finished.extend(
                    entrants
                        .into_iter()
                        .map(|row| row.fail(CoreError::Inference(message.clone()))),
                );
                return BatchProgress { finished, rows };
            }
        };

        let settings = &self.settings;
        let mut picks = Vec::with_capacity(rows);
        let mut joined = Vec::with_capacity(rows);
        for (index, (mut row, token)) in entrants.into_iter().zip(tokens).enumerate() {
            match token {
                Err(error) => finished.push(row.fail(error)),
                Ok(token) => {
                    row.prompt_elapsed = row.started.elapsed();
                    row.tokens.push(token);
                    if row.is_complete(settings) {
                        finished
                            .push(row.complete(&self.prompt.tokenizer, settings.max_new_tokens));
                    } else {
                        picks.push(RowPick {
                            row: index,
                            zero_columns: row.padding,
                        });
                        joined.push(row);
                    }
                }
            }
        }
        if joined.is_empty() {
            return BatchProgress { finished, rows };
        }

        // The cache of one pass over unpadded prompts that all go on is
        // used as it is; everything else is copied into place.
        let as_returned = batch.is_empty()
            && picks.len() == prefilled.rows()
            && picks.iter().all(|pick| pick.zero_columns == 0);
        let cache = if as_returned {
            Ok(prefilled)
        } else {
            let running: Vec<RowPick> = (0..batch.len()).map(RowPick::copy).collect();
            let home = self.cache_home();
            match &batch.cache {
                Some(cache) => KvCache::gather(&[(cache, &running), (&prefilled, &picks)], home),
                None => KvCache::gather(&[(&prefilled, &picks)], home),
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
                    keep.push(RowPick::copy(index));
                    batch.rows.push(row);
                }
                Err(error) => finished.push(row.fail(error)),
            }
        }
        if let Err(error) = batch.keep_cache_rows(&keep, self.cache_home()) {
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
                        keep.push(RowPick::copy(index));
                        batch.rows.push(row);
                    }
                }
            }
        }
        if let Err(error) = batch.keep_cache_rows(&keep, self.cache_home()) {
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
    /// on the left to `columns`.
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
            attention_mask.extend(std::iter::repeat_n(0_i64, row.padding));
            attention_mask.extend(std::iter::repeat_n(1_i64, columns - row.padding));
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
    /// graph does not treat a padded row as the same sequence (for
    /// example absolute positions without a `position_ids` input): do not
    /// run that model with a batch width above one.
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
        assert_eq!(
            pipeline.max_batch(),
            width,
            "the graph must be able to batch"
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
