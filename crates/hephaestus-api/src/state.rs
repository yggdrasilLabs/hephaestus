//! Shared application state for the HTTP serving layer.
//!
//! [`AppState`] holds the inference pipeline, readiness flag, and
//! runtime metadata. It is wrapped in [`Arc`] and passed to every
//! axum handler via the [`State`](axum::extract::State) extractor.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use hephaestus_core::{CoreError, InferenceInput, PipelineKind, PreparedData, Preparer};
use metrics_exporter_prometheus::PrometheusHandle;
use tokio::sync::RwLock;

use crate::batcher::{BatchRequest, Batcher};
use crate::error::ApiError;
use crate::worker::{InferenceTicket, InferenceWorker};

/// Shared application state for all HTTP handlers.
///
/// Constructed once at startup in the binary crate and shared via
/// `Arc<AppState>` through axum's state extractor. Fields are private
/// to enforce the Ousterhout deep-module principle -- callers interact
/// through controlled accessors rather than reaching into internals.
pub struct AppState {
    /// The inference pipeline, guarded by a tokio RwLock.
    ///
    /// Behind an [`Arc`] shared with the inference worker thread, which
    /// takes the write lock for every job ([`PipelineKind::execute`]
    /// requires `&mut self`). Nothing else locks it once the server is up:
    /// requests are prepared with `preparer`, so tokenization never waits
    /// for a running model (SC-02).
    pipeline: Arc<RwLock<PipelineKind>>,

    /// Prepares request inputs (tokenization) without the pipeline lock.
    preparer: Preparer,

    /// The thread that runs the model, and the bounded queue in front of it.
    worker: InferenceWorker,

    /// Readiness flag. Starts `false`; flipped to `true` after the
    /// warmup inference pass succeeds (D-05). On SIGTERM, flipped
    /// back to `false` so the readiness probe returns 503 (D-07).
    ready: AtomicBool,

    /// Model identifier from configuration (e.g., `distilbert-base-uncased-finetuned-sst-2-english`).
    model_id: String,

    /// Process start time, used to compute `uptime_s` in health responses (D-06).
    start_time: Instant,

    /// Per-request inference timeout duration (D-12, CORE-04).
    request_timeout: Duration,

    /// Prometheus metrics handle for rendering `/metrics` endpoint (OBSV-01).
    metrics_handle: PrometheusHandle,

    /// Optional dynamic batcher handle. `Some` when `BATCH_ENABLED=true`;
    /// `None` when batching is disabled (default, zero overhead per D-07),
    /// and when the pipeline batches its generations itself
    /// (`TEXT_GENERATION_MAX_BATCH` above one).
    batcher: Option<Batcher>,

    /// Window duration in seconds for streaming audio chunking (D-10).
    window_size_secs: f32,

    /// Overlap duration in seconds between consecutive audio windows (D-10).
    overlap_secs: f32,
}

impl AppState {
    /// Construct new application state.
    ///
    /// Pass `Some(batcher)` when dynamic batching is enabled, or `None`
    /// for the default direct-execution path (zero overhead per D-07).
    pub fn new(
        pipeline: PipelineKind,
        model_id: String,
        request_timeout: Duration,
        metrics_handle: PrometheusHandle,
        batcher: Option<Batcher>,
        window_size_secs: f32,
        overlap_secs: f32,
    ) -> Self {
        let preparer = pipeline.preparer();
        // A pipeline that batches its generations itself takes requests
        // one by one, between token steps. The dynamic batcher would hand
        // it closed batches and make every request wait for the batch
        // before it, so it is not used.
        let generation_batch = pipeline.generation_batch_width();
        let batcher = batcher.filter(|_| {
            if generation_batch > 1 {
                tracing::warn!(
                    generation_batch,
                    "dynamic batching is not used: text generation batches its requests itself"
                );
            }
            generation_batch <= 1
        });
        let pipeline = Arc::new(RwLock::new(pipeline));
        let worker = InferenceWorker::spawn(Arc::clone(&pipeline), &model_id, generation_batch);
        Self {
            pipeline,
            preparer,
            worker,
            ready: AtomicBool::new(false),
            model_id,
            start_time: Instant::now(),
            request_timeout,
            metrics_handle,
            batcher,
            window_size_secs,
            overlap_secs,
        }
    }

    /// Let at most `capacity` requests wait for the model at one time
    /// (`INFERENCE_QUEUE_CAPACITY`). Unset, the capacity is
    /// [`DEFAULT_QUEUE_CAPACITY`](crate::worker::DEFAULT_QUEUE_CAPACITY).
    #[must_use]
    pub fn with_inference_queue_capacity(self, capacity: usize) -> Self {
        self.worker.set_capacity(capacity);
        self
    }

    /// Install the test seam that runs on the worker thread, under the
    /// write lock, just before every model run.
    #[cfg(test)]
    pub(crate) fn set_before_inference(&self, hook: impl Fn() + Send + Sync + 'static) {
        self.worker.set_before_inference(hook);
    }

    /// Install the test seam that runs on the worker thread before every
    /// token step of a generation.
    #[cfg(test)]
    pub(crate) fn set_on_generation_step(&self, hook: impl Fn() + Send + Sync + 'static) {
        self.worker.set_on_generation_step(hook);
    }

    /// Requests admitted to the inference queue and not yet started.
    #[cfg(test)]
    pub(crate) fn waiting_requests(&self) -> usize {
        self.worker.waiting()
    }

    /// Generations running together right now.
    #[cfg(test)]
    pub(crate) fn active_generations(&self) -> usize {
        self.worker.active_generations()
    }

    /// Check whether the service is ready to accept inference requests.
    pub fn is_ready(&self) -> bool {
        self.ready.load(Ordering::SeqCst)
    }

    /// Set the readiness flag (e.g., after warmup or on shutdown).
    pub fn set_ready(&self, val: bool) {
        self.ready.store(val, Ordering::SeqCst);
    }

    /// The model identifier from configuration.
    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    /// Seconds since process start (for health probe responses).
    pub fn uptime_secs(&self) -> u64 {
        self.start_time.elapsed().as_secs()
    }

    /// Per-request inference timeout duration.
    pub fn request_timeout(&self) -> Duration {
        self.request_timeout
    }

    /// Render Prometheus exposition text for the `/metrics` endpoint.
    pub fn render_metrics(&self) -> String {
        self.metrics_handle.render()
    }

    /// Prepare a request's input for the model (tokenization or feature
    /// extraction).
    ///
    /// Takes no lock: requests are prepared concurrently, and while the
    /// model is running. Returns what [`PipelineKind::prepare`] returns.
    ///
    /// # Errors
    ///
    /// Returns [`CoreError`] if the input cannot be prepared.
    pub fn prepare(&self, input: impl Into<InferenceInput>) -> Result<PreparedData, CoreError> {
        self.preparer.prepare(input)
    }

    /// Whether requests to the loaded model may carry a system message.
    pub fn accepts_system_prompt(&self) -> bool {
        self.preparer.accepts_system_prompt()
    }

    /// Acquire a shared read lock on the inference pipeline.
    ///
    /// Only for work done before the server takes requests (warmup): it
    /// waits for as long as the model is running. Request paths use
    /// [`AppState::prepare`] instead.
    pub async fn read_pipeline(&self) -> tokio::sync::RwLockReadGuard<'_, PipelineKind> {
        self.pipeline.read().await
    }

    /// Acquire an exclusive write lock on the inference pipeline.
    ///
    /// Only for short work done before the server takes requests (warmup).
    /// The guard is held on the calling task, so running a model under it
    /// blocks a runtime worker thread; request paths use
    /// [`AppState::admit`] instead.
    pub async fn write_pipeline(&self) -> tokio::sync::RwLockWriteGuard<'_, PipelineKind> {
        self.pipeline.write().await
    }

    /// Take a place in the inference queue for a request that must be
    /// answered by `deadline` (`None`: no time limit).
    ///
    /// Call this before preparing the input, so a request that cannot be
    /// queued is refused at once. The model runs on the inference worker
    /// thread ([`InferenceTicket::run`]), never on a runtime thread, so
    /// probes, metrics and other requests keep being served, and there is
    /// never more than one job on the pipeline at a time.
    ///
    /// # Errors
    ///
    /// Returns [`ApiError::QueueFull`] when the queue is at capacity.
    pub fn admit(&self, deadline: Option<Instant>) -> Result<InferenceTicket, ApiError> {
        self.worker.admit(deadline)
    }

    /// Run a batch collected by the dynamic batcher as one job on the
    /// inference worker. Each request is answered on its own channel.
    pub(crate) async fn run_batch(&self, batch: Vec<BatchRequest>) {
        self.worker.run_batch(batch).await;
    }

    /// Begin shutdown of inference: refuse new jobs and fail the queued
    /// ones with [`ApiError::NotReady`]. The job that is running finishes,
    /// unless its deadline passes or its caller goes away first. Does not
    /// touch the readiness flag.
    pub fn shutdown_inference(&self) {
        self.worker.close();
    }

    /// Check whether dynamic batching is enabled.
    ///
    /// Returns `true` when a [`Batcher`] handle was provided at construction
    /// (i.e., `BATCH_ENABLED=true`). When `false`, the handler uses the
    /// direct execution path with zero batching overhead (D-07).
    pub fn is_batching_enabled(&self) -> bool {
        self.batcher.is_some()
    }

    /// Reference to the batcher handle, if batching is enabled.
    pub fn batcher(&self) -> Option<&Batcher> {
        self.batcher.as_ref()
    }

    /// Window duration in seconds for streaming audio chunking.
    pub fn window_size_secs(&self) -> f32 {
        self.window_size_secs
    }

    /// Overlap duration in seconds between consecutive audio windows.
    pub fn overlap_secs(&self) -> f32 {
        self.overlap_secs
    }
}
