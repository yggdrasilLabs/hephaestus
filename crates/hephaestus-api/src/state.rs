//! Shared application state for the HTTP serving layer.
//!
//! [`AppState`] holds the inference pipeline, readiness flag, and
//! runtime metadata. It is wrapped in [`Arc`] and passed to every
//! axum handler via the [`State`](axum::extract::State) extractor.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use hephaestus_core::PipelineKind;
use metrics_exporter_prometheus::PrometheusHandle;
use tokio::sync::RwLock;

use crate::batcher::Batcher;
use crate::error::ApiError;

/// Shared application state for all HTTP handlers.
///
/// Constructed once at startup in the binary crate and shared via
/// `Arc<AppState>` through axum's state extractor. Fields are private
/// to enforce the Ousterhout deep-module principle -- callers interact
/// through controlled accessors rather than reaching into internals.
pub struct AppState {
    /// The inference pipeline, guarded by a tokio RwLock.
    ///
    /// Read lock: [`PipelineKind::prepare`] takes `&self` (tokenization only).
    /// Write lock: [`PipelineKind::execute`] requires `&mut self` (ONNX session).
    /// This allows concurrent tokenization across requests while serializing
    /// inference (SC-02).
    ///
    /// Behind an [`Arc`] so [`AppState::run_inference`] can move an owned
    /// write guard onto a blocking thread.
    pipeline: Arc<RwLock<PipelineKind>>,

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
    /// `None` when batching is disabled (default, zero overhead per D-07).
    batcher: Option<Batcher>,

    /// Window duration in seconds for streaming audio chunking (D-10).
    window_size_secs: f32,

    /// Overlap duration in seconds between consecutive audio windows (D-10).
    overlap_secs: f32,

    /// Test seam: runs on the blocking thread, under the write lock, just
    /// before the inference closure. Lets a test stand in for a slow model.
    #[cfg(test)]
    before_inference: Option<Arc<dyn Fn() + Send + Sync>>,
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
        Self {
            pipeline: Arc::new(RwLock::new(pipeline)),
            ready: AtomicBool::new(false),
            model_id,
            start_time: Instant::now(),
            request_timeout,
            metrics_handle,
            batcher,
            window_size_secs,
            overlap_secs,
            #[cfg(test)]
            before_inference: None,
        }
    }

    /// Install the test seam that runs before every inference closure.
    #[cfg(test)]
    pub(crate) fn set_before_inference(&mut self, hook: impl Fn() + Send + Sync + 'static) {
        self.before_inference = Some(Arc::new(hook));
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

    /// Acquire a shared read lock on the inference pipeline.
    ///
    /// Used for [`PipelineKind::prepare`] which takes `&self` (tokenization).
    /// Multiple concurrent readers are allowed, enabling parallel tokenization
    /// across requests (SC-02).
    pub async fn read_pipeline(&self) -> tokio::sync::RwLockReadGuard<'_, PipelineKind> {
        self.pipeline.read().await
    }

    /// Acquire an exclusive write lock on the inference pipeline.
    ///
    /// Only for short work done before the server takes requests (warmup).
    /// The guard is held on the calling task, so running a model under it
    /// blocks a runtime worker thread; request paths use
    /// [`AppState::run_inference`] instead.
    pub async fn write_pipeline(&self) -> tokio::sync::RwLockWriteGuard<'_, PipelineKind> {
        self.pipeline.write().await
    }

    /// Run model inference under the pipeline write lock, off the runtime.
    ///
    /// Waits for the write lock without blocking, then runs `f` on tokio's
    /// blocking thread pool, so a long generation never occupies a runtime
    /// worker thread and probes, metrics and other requests keep being
    /// served. Used for [`PipelineKind::execute`] and
    /// [`PipelineKind::execute_batch`], which take `&mut self`.
    ///
    /// The blocking thread owns the write guard. If the caller stops
    /// waiting (request timeout, client gone), `f` still runs to its end
    /// and only then releases the lock, so there is never more than one
    /// inference on the pipeline at a time.
    ///
    /// `f` runs inside the caller's tracing span.
    ///
    /// # Errors
    ///
    /// Returns [`ApiError::Internal`] if the blocking task panicked or the
    /// runtime shut down before it ran.
    pub async fn run_inference<R, F>(&self, f: F) -> Result<R, ApiError>
    where
        F: FnOnce(&mut PipelineKind) -> R + Send + 'static,
        R: Send + 'static,
    {
        let mut pipeline = Arc::clone(&self.pipeline).write_owned().await;
        let span = tracing::Span::current();
        #[cfg(test)]
        let before_inference = self.before_inference.clone();
        tokio::task::spawn_blocking(move || {
            let _entered = span.enter();
            #[cfg(test)]
            if let Some(hook) = before_inference {
                hook();
            }
            f(&mut pipeline)
        })
        .await
        .map_err(|e| ApiError::Internal(format!("inference task failed: {e}")))
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
