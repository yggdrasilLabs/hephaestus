//! Channel-based dynamic request batcher (D-06).
//!
//! When dynamic batching is enabled, HTTP requests are submitted to a
//! bounded [`mpsc`] channel. A background [`batcher_loop`] task collects
//! requests over a configurable time window and runs batched ONNX
//! inference via [`PipelineKind::execute_batch`]. Results are fanned
//! back to individual requests through [`oneshot`] channels.
//!
//! When batching is disabled (default), none of this machinery is
//! instantiated -- zero overhead per D-07.

use std::sync::Arc;
use std::time::Duration;

use hephaestus_core::{CoreError, PipelineOutput, PreparedData};
use tokio::sync::{mpsc, oneshot};

use crate::state::AppState;

/// A single inference request queued for batching.
pub struct BatchRequest {
    /// The tokenized and prepared input ready for inference.
    pub prepared: PreparedData,
    /// One-shot channel to send the result back to the HTTP handler.
    pub reply: oneshot::Sender<Result<PipelineOutput, CoreError>>,
}

/// Handle for submitting requests to the batcher.
///
/// Cloneable -- the HTTP handler holds a reference through `AppState`.
/// Internally wraps a bounded `mpsc::Sender` per
/// [`rules/async-bounded-channel.md`].
#[derive(Clone)]
pub struct Batcher {
    tx: mpsc::Sender<BatchRequest>,
}

impl Batcher {
    /// Create a new batcher with a bounded channel.
    ///
    /// Channel capacity is `4 * max_batch_size` per
    /// [`rules/async-bounded-channel.md`] -- enough to absorb burst
    /// traffic while still applying backpressure under sustained load.
    ///
    /// Returns the `Batcher` handle and the `mpsc::Receiver` to be
    /// passed to [`batcher_loop`].
    pub fn new(max_batch_size: usize) -> (Self, mpsc::Receiver<BatchRequest>) {
        let capacity = 4 * max_batch_size;
        let (tx, rx) = mpsc::channel(capacity);
        (Self { tx }, rx)
    }

    /// Submit a prepared input for batched inference.
    ///
    /// Creates a oneshot channel, sends the request into the batch
    /// queue, and awaits the result. Backpressure is applied when the
    /// channel is full -- the `.send().await` suspends until space is
    /// available.
    ///
    /// # Errors
    ///
    /// Returns `CoreError::Inference` if the batch channel is closed
    /// (batcher task panicked or shut down) or if the oneshot reply
    /// channel is dropped without a response.
    pub async fn submit(
        &self,
        prepared: PreparedData,
    ) -> Result<PipelineOutput, CoreError> {
        let (reply_tx, reply_rx) = oneshot::channel();

        self.tx
            .send(BatchRequest {
                prepared,
                reply: reply_tx,
            })
            .await
            .map_err(|_| {
                CoreError::Inference("batcher channel closed".to_string())
            })?;

        reply_rx
            .await
            .map_err(|_| {
                CoreError::Inference(
                    "batcher reply channel dropped without response".to_string(),
                )
            })?
    }
}

/// Background batch collection and execution loop.
///
/// Waits for the first request, then collects additional requests
/// until either `max_batch_size` is reached or `max_wait` elapses.
/// Write-locks the pipeline RwLock only during `execute_batch`, which runs
/// on a blocking thread ([`AppState::run_inference`]) -- never
/// during the collection phase per [`rules/anti-lock-across-await.md`].
///
/// On any `execute_batch` failure, the error is sent to ALL pending
/// oneshot channels so no request hangs. Uses `let _ = reply.send()`
/// to handle dropped receivers (HTTP request may have timed out).
pub async fn batcher_loop(
    mut rx: mpsc::Receiver<BatchRequest>,
    state: Arc<AppState>,
    max_batch_size: usize,
    max_wait: Duration,
) {
    loop {
        // Wait for the first request (blocks until one arrives or channel closes).
        let first = match rx.recv().await {
            Some(req) => req,
            None => break, // Channel closed, shut down.
        };

        let mut batch = Vec::with_capacity(max_batch_size);
        batch.push(first);

        // Collect more requests until batch is full or deadline fires.
        let deadline = tokio::time::sleep(max_wait);
        tokio::pin!(deadline);

        loop {
            if batch.len() >= max_batch_size {
                break;
            }

            tokio::select! {
                biased;
                maybe_req = rx.recv() => {
                    match maybe_req {
                        Some(req) => batch.push(req),
                        None => break, // Channel closed mid-collection.
                    }
                }
                () = &mut deadline => break,
            }
        }

        // Extract prepared inputs and reply channels.
        let mut replies: Vec<oneshot::Sender<Result<PipelineOutput, CoreError>>> =
            Vec::with_capacity(batch.len());
        let mut inputs: Vec<PreparedData> = Vec::with_capacity(batch.len());
        for req in batch {
            inputs.push(req.prepared);
            replies.push(req.reply);
        }

        // Write-lock pipeline only during execute_batch (not during
        // collection). The batch runs on a blocking thread, which holds the
        // lock until it finishes.
        let results = state
            .run_inference(move |pipeline| pipeline.execute_batch(inputs))
            .await;

        // Fan results back to individual requests.
        match results {
            Ok(results) => {
                for (reply, result) in replies.into_iter().zip(results) {
                    let _ = reply.send(result);
                }
            }
            Err(e) => {
                for reply in replies {
                    let _ = reply.send(Err(CoreError::Inference(e.to_string())));
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_batcher_submit_sends_to_channel() {
        // Arrange -- verify the round-trip through the oneshot channel.
        let (batcher, mut rx) = Batcher::new(8);

        // Act -- spawn submit in background, receive from channel.
        let handle = tokio::spawn(async move {
            let _ = batcher.submit(hephaestus_core::PreparedData::Text(
                hephaestus_core::PreparedInput::new_for_test(
                    vec![101, 2023, 102],
                    vec![1, 1, 1],
                    3,
                ),
            )).await;
        });

        // Assert -- the request should appear on the receiver.
        let req = tokio::time::timeout(Duration::from_secs(1), rx.recv())
            .await
            .expect("should receive within timeout")
            .expect("channel should have a request");

        // Verify request arrived. Send a reply to complete the round trip.
        let _ = req.reply.send(Ok(hephaestus_core::PipelineOutput::Classifier {
            label: "TEST".to_string(),
            score: 0.99,
        }));

        handle.await.expect("submit task should complete");
    }

    #[tokio::test]
    async fn test_batcher_channel_is_bounded() {
        // Arrange -- max_batch_size = 4, so capacity should be 16.
        let (batcher, _rx) = Batcher::new(4);

        // Act -- fill the channel to capacity with try_send.
        let mut sent = 0;
        for _ in 0..20 {
            let (reply_tx, _reply_rx) = oneshot::channel();
            let prepared = hephaestus_core::PreparedData::Text(
                hephaestus_core::PreparedInput::new_for_test(
                    vec![101],
                    vec![1],
                    1,
                ),
            );
            match batcher.tx.try_send(BatchRequest {
                prepared,
                reply: reply_tx,
            }) {
                Ok(()) => sent += 1,
                Err(mpsc::error::TrySendError::Full(_)) => break,
                Err(mpsc::error::TrySendError::Closed(_)) => break,
            }
        }

        // Assert -- should have sent exactly 16 (4 * max_batch_size).
        assert_eq!(sent, 16, "bounded channel capacity should be 4 * max_batch_size = 16");
    }
}
