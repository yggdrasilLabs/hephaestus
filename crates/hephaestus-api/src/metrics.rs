//! Prometheus metrics with a deep-module timer abstraction (D-09).
//!
//! [`StageTimer`] hides all `metrics` crate interaction behind a minimal
//! interface. Pipeline stages call [`StageTimer::time`] to record
//! per-stage latency histograms; the handler calls [`StageTimer::finish_request`]
//! for overall request metrics. Callers never touch `metrics` macros directly.
//!
//! [`install_recorder`] installs the global Prometheus metrics recorder
//! and returns a [`PrometheusHandle`] used by [`metrics_handler`] to render
//! the `/metrics` scrape endpoint.

use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::extract::State;
use metrics_exporter_prometheus::PrometheusHandle;

use crate::state::AppState;

/// Install the global Prometheus metrics recorder.
///
/// Must be called exactly once at startup, before any metrics are recorded.
/// Returns the [`PrometheusHandle`] that renders Prometheus exposition text.
///
/// # Errors
///
/// Returns an error if the recorder has already been installed (double init).
pub fn install_recorder() -> Result<PrometheusHandle, anyhow::Error> {
    let handle = metrics_exporter_prometheus::PrometheusBuilder::new()
        .install_recorder()
        .map_err(|e| anyhow::anyhow!("failed to install Prometheus recorder: {e}"))?;
    Ok(handle)
}

/// Deep-module timer abstraction for per-stage inference metrics (D-09).
///
/// Wraps the `metrics` crate so that pipeline stages and handlers never
/// interact with metrics macros directly. All histograms and counters
/// carry a `model_id` label per D-10.
///
/// Cloning is cheap (a shared `model_id`), so a stage that runs on another
/// thread can take its own copy.
#[derive(Clone)]
pub struct StageTimer {
    model_id: Arc<str>,
}

impl StageTimer {
    /// Create a new timer for the given model.
    pub fn new(model_id: impl Into<Arc<str>>) -> Self {
        Self {
            model_id: model_id.into(),
        }
    }

    /// Time a pipeline stage and record its duration.
    ///
    /// Records `hephaestus_stage_duration_seconds` with labels
    /// `stage` and `model_id`. Stage values are `"tokenization"`,
    /// `"inference"`, or `"postprocessing"` per D-08.
    pub fn time<T>(&self, stage: &'static str, f: impl FnOnce() -> T) -> T {
        let start = Instant::now();
        let result = f();
        self.record(stage, start.elapsed());
        result
    }

    /// Record the duration of a stage that was timed elsewhere: a
    /// generation that shared its forward passes with others.
    pub fn record(&self, stage: &'static str, elapsed: Duration) {
        metrics::histogram!(
            "hephaestus_stage_duration_seconds",
            "stage" => stage,
            "model_id" => self.model_id.clone(),
        )
        .record(elapsed.as_secs_f64());
    }

    /// Record overall request completion metrics.
    ///
    /// Records `hephaestus_request_duration_seconds` histogram and
    /// `hephaestus_requests_total` counter with `model_id` and `status`
    /// labels per D-10.
    pub fn finish_request(&self, start: Instant, success: bool) {
        let elapsed = start.elapsed().as_secs_f64();
        metrics::histogram!(
            "hephaestus_request_duration_seconds",
            "model_id" => self.model_id.clone(),
        )
        .record(elapsed);
        metrics::counter!(
            "hephaestus_requests_total",
            "model_id" => self.model_id.clone(),
            "status" => if success { "ok" } else { "error" },
        )
        .increment(1);
    }
}

/// GET /metrics -- render Prometheus exposition text.
///
/// Reads the [`PrometheusHandle`] from [`AppState`] and returns the
/// pre-computed metrics text for Prometheus scraping (OBSV-01).
pub async fn metrics_handler(State(state): State<Arc<AppState>>) -> String {
    state.render_metrics()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stage_timer_new_accepts_model_id() {
        let timer = StageTimer::new("test-model");
        assert_eq!(&*timer.model_id, "test-model");
    }

    #[test]
    fn stage_timer_time_returns_closure_result() {
        // Arrange
        let timer = StageTimer::new("test-model");

        // Act -- the closure returns a value; time() must pass it through.
        let result = timer.time("tokenization", || 42);

        // Assert
        assert_eq!(result, 42);
    }

    #[test]
    fn stage_timer_time_returns_result_type() {
        // Arrange
        let timer = StageTimer::new("test-model");

        // Act -- verify it works with Result types (common in pipeline calls).
        let result: Result<&str, &str> = timer.time("inference", || Ok("output"));

        // Assert
        assert_eq!(result.unwrap(), "output");
    }
}
