//! HTTP request handlers for inference and health probes.
//!
//! All handlers receive shared [`AppState`] via axum's
//! [`State`](axum::extract::State) extractor.

use std::sync::Arc;
use std::time::Instant;

use axum::Json;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::IntoResponse;
use serde::Deserialize;

use hephaestus_core::{InferenceInput, PipelineKind};

use crate::error::ApiError;
use crate::metrics::StageTimer;
use crate::state::AppState;

/// JSON request body for the inference endpoint (D-01).
#[derive(Debug, Deserialize)]
pub struct InferRequest {
    /// The text to run inference on.
    pub text: String,
    /// System message for this request. Read by text generation models
    /// only; every other profile ignores it, whatever its JSON type, as it
    /// ignores any field it does not know.
    #[serde(default)]
    pub system: Option<serde_json::Value>,
}

/// Turn a request into the input its pipeline takes.
///
/// For a pipeline without a system message this is the text alone, so the
/// `system` field changes nothing there. For text generation the field
/// must be a string (or absent / `null`, which selects the server's
/// default system prompt).
///
/// # Errors
///
/// Returns [`ApiError::BadRequest`] if a text generation request carries
/// a `system` value that is not a string.
fn request_input(
    pipeline: &PipelineKind,
    text: String,
    system: Option<serde_json::Value>,
) -> Result<InferenceInput, ApiError> {
    if !pipeline.accepts_system_prompt() {
        return Ok(InferenceInput::Text(text));
    }
    match system {
        None | Some(serde_json::Value::Null) => Ok(InferenceInput::Text(text)),
        Some(serde_json::Value::String(system)) => {
            Ok(InferenceInput::TextWithSystem { text, system })
        }
        Some(_) => Err(ApiError::BadRequest(
            "system field must be a string".to_string(),
        )),
    }
}

/// POST /infer -- run inference on the loaded model.
///
/// Validates readiness, acquires the pipeline lock, runs tokenization
/// and inference, and returns a model-determined JSON result (D-04, D-05).
///
/// The response shape depends on the loaded model profile:
/// - Classifier: `{"label": "...", "score": ..., "model_id": "...", "latency_ms": ...}`
/// - Embeddings: `{"embedding": [...], "model_id": "...", "latency_ms": ...}`
/// - Seq2seq and text generation: `{"generated_text": "...", "model_id": "...", "latency_ms": ...}`
///
/// # Errors
///
/// Returns [`ApiError`] variants mapped to HTTP status codes per D-03:
/// - 503 if the service is not ready
/// - 400 if the request text is empty
/// - 422 if tokenization fails
/// - 504 if inference times out (D-14)
/// - 500 if inference fails
#[tracing::instrument(skip(state, req), fields(text_len = req.text.len()))]
pub async fn infer(
    State(state): State<Arc<AppState>>,
    Json(req): Json<InferRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    // Gate on readiness (D-05).
    if !state.is_ready() {
        return Err(ApiError::NotReady);
    }

    // Validate input (T-02-01).
    if req.text.is_empty() {
        return Err(ApiError::BadRequest("text field must not be empty".to_string()));
    }

    let request_start = Instant::now();
    let timer = StageTimer::new(state.model_id());

    // Wrap inference in a request-level timeout (D-12, D-14, CORE-04).
    // Uses tokio::time::timeout (not tower-http TimeoutLayer) for full
    // control over the 504 response body per Pitfall 4.
    let result = tokio::time::timeout(state.request_timeout(), async {
        if state.is_batching_enabled() {
            // Batching path (D-06): prepare under read lock, drop lock, submit to batcher.
            // Pipeline RwLock is NOT held across the batcher submit await
            // per rules/anti-lock-across-await.md.
            let prepared = {
                let pipeline = state.read_pipeline().await;
                let input = request_input(&pipeline, req.text, req.system)?;
                timer.time("tokenization", || pipeline.prepare(input))?
            }; // Read lock dropped here before submit.

            let pipeline_output = state
                .batcher()
                .ok_or(ApiError::Internal("batcher not available".to_string()))?
                .submit(prepared)
                .await
                .map_err(ApiError::from)?;
            Ok::<_, ApiError>(pipeline_output.to_json())
        } else {
            // Direct path (D-07): read lock for prepare, write lock for execute.
            // Splitting the lock allows other requests to tokenize concurrently
            // while only inference holds exclusive access (SC-02).
            let prepared = {
                let pipeline = state.read_pipeline().await;
                let input = request_input(&pipeline, req.text, req.system)?;
                timer.time("tokenization", || pipeline.prepare(input))?
            }; // Read lock dropped here.
            // The model runs on a blocking thread, which holds the write
            // lock until it finishes, so this task only waits.
            let stage_timer = timer.clone();
            let pipeline_output = state
                .run_inference(move |pipeline| {
                    stage_timer.time("inference", || pipeline.execute(prepared))
                })
                .await??;
            Ok::<_, ApiError>(pipeline_output.to_json())
        }
    })
    .await;

    let mut output = match result {
        Ok(inner) => match inner {
            Ok(output) => {
                timer.finish_request(request_start, true);
                output
            }
            Err(e) => {
                let latency_ms = request_start.elapsed().as_millis() as u64;
                timer.finish_request(request_start, false);
                tracing::warn!(
                    model_id = %state.model_id(),
                    latency_ms,
                    status = "error",
                    "inference request failed"
                );
                return Err(e);
            }
        },
        Err(_elapsed) => {
            let latency_ms = request_start.elapsed().as_millis() as u64;
            timer.finish_request(request_start, false);
            tracing::warn!(
                model_id = %state.model_id(),
                latency_ms,
                status = "timeout",
                "inference request timed out"
            );
            return Err(ApiError::Timeout);
        }
    };

    let latency_ms = request_start.elapsed().as_millis() as u64;

    // Insert model_id and latency_ms into the model-determined output (D-05).
    if let Some(obj) = output.as_object_mut() {
        obj.insert(
            "model_id".to_string(),
            serde_json::Value::String(state.model_id().to_string()),
        );
        obj.insert(
            "latency_ms".to_string(),
            serde_json::Value::Number(serde_json::Number::from(latency_ms)),
        );
    }

    tracing::info!(
        model_id = %state.model_id(),
        latency_ms,
        status = "success",
        "inference request completed"
    );

    Ok(Json(output))
}

/// GET /healthz/live -- liveness probe (D-05, D-06).
///
/// Always returns 200 OK with service metadata. Used as the
/// Kubernetes liveness probe -- the pod is alive if the process
/// is running and can respond to HTTP.
pub async fn liveness(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "status": "ok",
        "model_id": state.model_id(),
        "uptime_s": state.uptime_secs(),
    }))
}

/// GET /healthz/ready -- readiness probe (D-05, D-06, D-07).
///
/// Returns 200 after warmup completes; 503 before warmup or after
/// SIGTERM flips readiness to false.
pub async fn readiness(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    if state.is_ready() {
        (
            StatusCode::OK,
            Json(serde_json::json!({
                "status": "ok",
                "model_id": state.model_id(),
                "uptime_s": state.uptime_secs(),
            })),
        )
    } else {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({
                "status": "not_ready",
                "model_id": state.model_id(),
            })),
        )
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::time::Duration;

    use axum::body::Body;
    use axum::http::Request;
    use hephaestus_core::{ExecutionProvider, Seq2SeqPipeline};
    use tower::ServiceExt;

    use super::*;

    /// State around a real (tiny) encoder-decoder pipeline, built from the
    /// fixture graphs of `hephaestus-core`. `before_inference` runs on the
    /// inference thread before each run, to stand in for a slow model.
    fn fixture_state(
        request_timeout: Duration,
        before_inference: impl Fn() + Send + Sync + 'static,
    ) -> (Arc<AppState>, tempfile::TempDir) {
        let fixtures =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../hephaestus-core/tests/fixtures/seq2seq");
        let dir = tempfile::tempdir().expect("temp dir");
        for file in ["tokenizer.json", "encoder_model.onnx", "decoder_model.onnx"] {
            std::fs::copy(fixtures.join(file), dir.path().join(file)).expect("copy fixture");
        }
        let config = serde_json::json!({
            "architectures": ["BartForConditionalGeneration"],
            "bos_token_id": 0,
            "decoder_start_token_id": 2,
            "eos_token_id": 2,
            "max_length": 32,
            "max_position_embeddings": 64,
            "pad_token_id": 1,
            "vocab_size": 16
        });
        std::fs::write(dir.path().join("config.json"), config.to_string()).expect("write config");
        let pipeline = Seq2SeqPipeline::new(dir.path(), &ExecutionProvider::Cpu)
            .expect("pipeline should load");

        // A recorder that is not installed globally: only its handle is needed.
        let metrics_handle = metrics_exporter_prometheus::PrometheusBuilder::new()
            .build_recorder()
            .handle();
        let mut state = AppState::new(
            PipelineKind::Seq2Seq(pipeline),
            "fixture/seq2seq".to_string(),
            request_timeout,
            metrics_handle,
            None,
            30.0,
            5.0,
        );
        state.set_before_inference(before_inference);
        state.set_ready(true);
        (Arc::new(state), dir)
    }

    fn infer_request(text: &str) -> Request<Body> {
        Request::post("/infer")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::json!({ "text": text }).to_string()))
            .expect("request should build")
    }

    fn get(path: &str) -> Request<Body> {
        Request::get(path)
            .body(Body::empty())
            .expect("request should build")
    }

    // A single-threaded runtime: if inference ran on the runtime thread,
    // nothing else could be polled until it finished.
    #[tokio::test(flavor = "current_thread")]
    async fn probes_and_metrics_answer_while_inference_is_running() {
        // Arrange -- every inference takes at least a second.
        const INFERENCE_TIME: Duration = Duration::from_secs(1);
        let started = Arc::new(AtomicBool::new(false));
        let hook_started = Arc::clone(&started);
        let (state, _dir) = fixture_state(Duration::from_secs(30), move || {
            hook_started.store(true, Ordering::SeqCst);
            std::thread::sleep(INFERENCE_TIME);
        });
        let app = crate::build_router(state);

        // Act -- start a generation, wait until the model is running, then
        // call the probes and metrics on the same runtime.
        let infer = tokio::spawn(app.clone().oneshot(infer_request("a b c")));
        while !started.load(Ordering::SeqCst) {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        let asked = Instant::now();
        let mut statuses = Vec::new();
        for path in ["/healthz/live", "/healthz/ready", "/metrics"] {
            let response = app.clone().oneshot(get(path)).await.expect("infallible");
            statuses.push(response.status());
        }
        let answered_in = asked.elapsed();
        let still_running = !infer.is_finished();

        // Assert -- all three answered at once, during the generation.
        assert_eq!(statuses, [StatusCode::OK; 3]);
        assert!(
            still_running,
            "probes were only served after inference finished"
        );
        assert!(
            answered_in < INFERENCE_TIME / 4,
            "probes took {answered_in:?} during inference"
        );

        // The generation itself is unaffected.
        let response = infer.await.expect("task should join").expect("infallible");
        assert_eq!(response.status(), StatusCode::OK);
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("body should read");
        let body: serde_json::Value = serde_json::from_slice(&body).expect("JSON body");
        assert_eq!(body["generated_text"], "a b c");
        assert_eq!(body["model_id"], "fixture/seq2seq");
    }

    #[tokio::test(flavor = "current_thread")]
    async fn timed_out_request_keeps_the_pipeline_locked_until_its_inference_ends() {
        // Arrange -- inference outlasts the request timeout; the hook
        // counts how many inferences run at the same moment.
        let running = Arc::new(AtomicUsize::new(0));
        let most_at_once = Arc::new(AtomicUsize::new(0));
        let runs = Arc::new(AtomicUsize::new(0));
        let (hook_running, hook_most, hook_runs) = (
            Arc::clone(&running),
            Arc::clone(&most_at_once),
            Arc::clone(&runs),
        );
        let (state, _dir) = fixture_state(Duration::from_millis(100), move || {
            let now = hook_running.fetch_add(1, Ordering::SeqCst) + 1;
            hook_most.fetch_max(now, Ordering::SeqCst);
            hook_runs.fetch_add(1, Ordering::SeqCst);
            std::thread::sleep(Duration::from_millis(400));
            hook_running.fetch_sub(1, Ordering::SeqCst);
        });
        let app = crate::build_router(Arc::clone(&state));

        // Act -- the request gives up after 100 ms; its inference goes on.
        let asked = Instant::now();
        let response = app
            .oneshot(infer_request("a b c"))
            .await
            .expect("infallible");
        let timed_out_after = asked.elapsed();
        let abandoned_still_running = running.load(Ordering::SeqCst) == 1;
        // A second inference, with no timeout of its own, must wait for it.
        let prepared = state
            .read_pipeline()
            .await
            .prepare("d e f".to_string())
            .expect("prepare should succeed");
        let output = state
            .run_inference(move |pipeline| pipeline.execute(prepared))
            .await
            .expect("inference task should finish")
            .expect("execute should succeed");

        // Assert
        assert_eq!(response.status(), StatusCode::GATEWAY_TIMEOUT);
        assert!(
            timed_out_after < Duration::from_millis(350),
            "504 after {timed_out_after:?}"
        );
        assert!(
            abandoned_still_running,
            "abandoned inference should still be running"
        );
        assert_eq!(runs.load(Ordering::SeqCst), 2, "both inferences ran");
        assert_eq!(most_at_once.load(Ordering::SeqCst), 1, "never two at once");
        assert_eq!(
            output.to_json(),
            serde_json::json!({ "generated_text": "d e f" })
        );
    }

    #[test]
    fn infer_request_deserializes_from_json() {
        // Arrange
        let json = r#"{"text": "hello world"}"#;

        // Act
        let req: InferRequest = serde_json::from_str(json).expect("should deserialize");

        // Assert
        assert_eq!(req.text, "hello world");
    }

    #[test]
    fn model_determined_output_accepts_model_id_and_latency() {
        // Arrange -- simulate what PipelineKind::execute returns
        let mut output = serde_json::json!({
            "label": "POSITIVE",
            "score": 0.95,
        });

        // Act -- insert model_id and latency_ms as the handler does
        if let Some(obj) = output.as_object_mut() {
            obj.insert(
                "model_id".to_string(),
                serde_json::Value::String("test-model".to_string()),
            );
            obj.insert(
                "latency_ms".to_string(),
                serde_json::Value::Number(serde_json::Number::from(12_u64)),
            );
        }

        // Assert
        assert_eq!(output["label"], "POSITIVE");
        let score = output["score"].as_f64().expect("score should be a number");
        assert!((score - 0.95).abs() < 1e-6, "score should be ~0.95, got {score}");
        assert_eq!(output["model_id"], "test-model");
        assert_eq!(output["latency_ms"], 12);
    }

    #[test]
    fn embeddings_output_accepts_model_id_and_latency() {
        // Arrange -- simulate embeddings PipelineKind::execute output
        let mut output = serde_json::json!({
            "embedding": [0.1, 0.2, 0.3],
        });

        // Act
        if let Some(obj) = output.as_object_mut() {
            obj.insert(
                "model_id".to_string(),
                serde_json::Value::String("all-MiniLM-L6-v2".to_string()),
            );
            obj.insert(
                "latency_ms".to_string(),
                serde_json::Value::Number(serde_json::Number::from(5_u64)),
            );
        }

        // Assert
        assert!(output["embedding"].is_array());
        assert_eq!(output["model_id"], "all-MiniLM-L6-v2");
        assert_eq!(output["latency_ms"], 5);
    }

    #[test]
    fn infer_request_accepts_any_system_value() {
        // A request that carried a `system` field of any JSON type was
        // accepted before the field had a meaning; it still is.
        for (json, expected) in [
            (r#"{"text": "hi"}"#, None),
            (r#"{"text": "hi", "system": null}"#, None),
            (r#"{"text": "hi", "system": "Be brief."}"#, Some(serde_json::json!("Be brief."))),
            (r#"{"text": "hi", "system": 5}"#, Some(serde_json::json!(5))),
            (r#"{"text": "hi", "system": {"a": [1]}}"#, Some(serde_json::json!({"a": [1]}))),
            (r#"{"text": "hi", "other": true, "system": ""}"#, Some(serde_json::json!(""))),
        ] {
            // Act
            let req: InferRequest = serde_json::from_str(json).expect("should deserialize");

            // Assert
            assert_eq!(req.text, "hi", "{json}");
            assert_eq!(req.system, expected, "{json}");
        }
    }
}
