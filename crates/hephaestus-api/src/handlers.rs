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

use hephaestus_core::InferenceInput;

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
    state: &AppState,
    text: String,
    system: Option<serde_json::Value>,
) -> Result<InferenceInput, ApiError> {
    if !state.accepts_system_prompt() {
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
/// Validates readiness, tokenizes the input, queues it for the inference
/// worker, and returns a model-determined JSON result (D-04, D-05).
///
/// The response shape depends on the loaded model profile:
/// - Classifier: `{"label": "...", "score": ..., "model_id": "...", "latency_ms": ...}`
/// - Embeddings: `{"embedding": [...], "model_id": "...", "latency_ms": ...}`
/// - Seq2seq and text generation: `{"generated_text": "...", "model_id": "...", "latency_ms": ...}`,
///   plus `prompt_tokens` and `completion_tokens` when the model generates
///   token by token (text generation, encoder-decoder seq2seq)
///
/// # Errors
///
/// Returns [`ApiError`] variants mapped to HTTP status codes per D-03:
/// - 503 if the service is not ready, or (with `Retry-After`) if the
///   inference queue is full
/// - 400 if the request text is empty
/// - 413 if the prompt of a text generation request exceeds the input
///   token cap
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
    // A job still queued at this moment is dropped instead of run.
    let deadline = request_start + state.request_timeout();

    // Wrap inference in a request-level timeout (D-12, D-14, CORE-04).
    // Uses tokio::time::timeout (not tower-http TimeoutLayer) for full
    // control over the 504 response body per Pitfall 4.
    let result = tokio::time::timeout(state.request_timeout(), async {
        if state.is_batching_enabled() {
            // Batching path (D-06): prepare, then submit to the batcher.
            let input = request_input(&state, req.text, req.system)?;
            let prepared = timer.time("tokenization", || state.prepare(input))?;

            let pipeline_output = state
                .batcher()
                .ok_or(ApiError::Internal("batcher not available".to_string()))?
                .submit(prepared, Some(deadline))
                .await?;
            Ok::<_, ApiError>((pipeline_output.to_json(), pipeline_output.usage))
        } else {
            // Direct path (D-07): take a place in the inference queue first,
            // so a request that cannot be queued is refused at once, then
            // prepare. Neither waits for a running model (SC-02).
            let ticket = state.admit(Some(deadline))?;
            let input = request_input(&state, req.text, req.system)?;
            let prepared = timer.time("tokenization", || state.prepare(input))?;
            // The model runs on the inference worker thread, so this task
            // only waits. Dropping this future (timeout, client gone)
            // takes a job that is still queued out of the queue and stops
            // a generation that is running.
            let pipeline_output = ticket.run(prepared, Some(timer.clone())).await?;
            Ok::<_, ApiError>((pipeline_output.to_json(), pipeline_output.usage))
        }
    })
    .await;

    let (mut output, usage) = match result {
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
        prompt_tokens = usage.map(|usage| usage.prompt_tokens),
        completion_tokens = usage.map(|usage| usage.completion_tokens),
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
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::mpsc;
    use std::time::Duration;

    use axum::body::Body;
    use axum::http::Request;
    use axum::response::Response;
    use hephaestus_core::{
        ExecutionProvider, PipelineKind, Seq2SeqPipeline, TextGenerationOptions,
        TextGenerationPipeline,
    };
    use tower::ServiceExt;

    use super::*;
    use crate::batcher::{Batcher, batcher_loop};

    /// A real (tiny) encoder-decoder pipeline, built from the fixture
    /// graphs of `hephaestus-core`. It generates its input back.
    fn seq2seq_pipeline() -> (PipelineKind, tempfile::TempDir) {
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
        (PipelineKind::Seq2Seq(pipeline), dir)
    }

    /// Ready state around `pipeline`.
    fn state_for(
        pipeline: PipelineKind,
        request_timeout: Duration,
        batcher: Option<Batcher>,
    ) -> AppState {
        // A recorder that is not installed globally: only its handle is needed.
        let metrics_handle = metrics_exporter_prometheus::PrometheusBuilder::new()
            .build_recorder()
            .handle();
        let state = AppState::new(
            pipeline,
            "fixture/model".to_string(),
            request_timeout,
            metrics_handle,
            batcher,
            30.0,
            5.0,
        );
        state.set_ready(true);
        state
    }

    /// State around the seq2seq fixture. `before_inference` runs on the
    /// inference thread before each run, to stand in for a slow model.
    fn fixture_state(
        request_timeout: Duration,
        before_inference: impl Fn() + Send + Sync + 'static,
    ) -> (Arc<AppState>, tempfile::TempDir) {
        let (pipeline, dir) = seq2seq_pipeline();
        let state = state_for(pipeline, request_timeout, None);
        state.set_before_inference(before_inference);
        (Arc::new(state), dir)
    }

    /// A hook that holds every model run until the test lets it go.
    ///
    /// Returns the hook, the number of runs that have started, and the
    /// sender that releases one run per message (dropping it releases all).
    fn gated_hook() -> (
        impl Fn() + Send + Sync + 'static,
        Arc<AtomicUsize>,
        mpsc::Sender<()>,
    ) {
        let (release, gate) = mpsc::channel::<()>();
        let gate = Mutex::new(gate);
        let started = Arc::new(AtomicUsize::new(0));
        let hook_started = Arc::clone(&started);
        let hook = move || {
            hook_started.fetch_add(1, Ordering::SeqCst);
            let _ = gate.lock().expect("gate lock").recv();
        };
        (hook, started, release)
    }

    /// Wait until `condition` holds, failing the test if it never does.
    async fn until(what: &str, condition: impl Fn() -> bool) {
        let asked = Instant::now();
        while !condition() {
            assert!(
                asked.elapsed() < Duration::from_secs(10),
                "timed out waiting for {what}"
            );
            tokio::time::sleep(Duration::from_millis(2)).await;
        }
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

    async fn json_body(response: Response) -> serde_json::Value {
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("body should read");
        serde_json::from_slice(&body).expect("JSON body")
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
        let body = json_body(response).await;
        assert_eq!(body["generated_text"], "a b c");
        assert_eq!(body["model_id"], "fixture/model");
    }

    /// A real (tiny) decoder-only pipeline from the fixture graphs of
    /// `hephaestus-core`, with a prompt cap of 16 tokens. Its model never
    /// emits an end token, so every generation runs for `max_new_tokens`
    /// steps unless something stops it.
    fn text_generation_pipeline(max_new_tokens: usize) -> (PipelineKind, tempfile::TempDir) {
        let fixtures =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../hephaestus-core/tests/fixtures");
        let dir = tempfile::tempdir().expect("temp dir");
        for (from, file) in [
            ("text_generation", "model.onnx"),
            ("text_generation", "tokenizer_config.json"),
            ("seq2seq", "tokenizer.json"),
        ] {
            std::fs::copy(fixtures.join(from).join(file), dir.path().join(file))
                .expect("copy fixture");
        }
        let config = serde_json::json!({
            "architectures": ["LlamaForCausalLM"],
            "bos_token_id": 0,
            "max_position_embeddings": 512,
            "vocab_size": 16
        });
        std::fs::write(dir.path().join("config.json"), config.to_string()).expect("write config");
        let options = TextGenerationOptions::new()
            .with_max_input_tokens(16)
            .with_max_new_tokens(max_new_tokens);
        let pipeline =
            TextGenerationPipeline::with_options(dir.path(), &ExecutionProvider::Cpu, &options)
                .expect("pipeline should load");
        (PipelineKind::TextGeneration(pipeline), dir)
    }

    /// Token steps a generation of [`slow_generation_state`] runs when
    /// nothing stops it.
    const SLOW_GENERATION_STEPS: usize = 400;
    /// Time [`slow_generation_state`] spends on each token step.
    const STEP_TIME: Duration = Duration::from_millis(5);

    /// State around a text generation model whose every token step takes
    /// [`STEP_TIME`] while `slow` is set: two seconds for a whole
    /// generation. Returns the state, the number of token steps asked for
    /// so far, and the `slow` switch.
    fn slow_generation_state(
        request_timeout: Duration,
        batcher: Option<Batcher>,
    ) -> (
        Arc<AppState>,
        Arc<AtomicUsize>,
        Arc<AtomicBool>,
        tempfile::TempDir,
    ) {
        let (pipeline, dir) = text_generation_pipeline(SLOW_GENERATION_STEPS);
        let state = state_for(pipeline, request_timeout, batcher);
        let steps = Arc::new(AtomicUsize::new(0));
        let slow = Arc::new(AtomicBool::new(true));
        let (hook_steps, hook_slow) = (Arc::clone(&steps), Arc::clone(&slow));
        state.set_on_generation_step(move || {
            hook_steps.fetch_add(1, Ordering::SeqCst);
            if hook_slow.load(Ordering::SeqCst) {
                std::thread::sleep(STEP_TIME);
            }
        });
        (Arc::new(state), steps, slow, dir)
    }

    /// Wait until the worker has finished everything queued before this
    /// call: a job that is already past its deadline is answered, without
    /// running, as soon as the worker reaches it.
    async fn worker_reaches_the_end_of_its_queue(state: &AppState) {
        let ticket = state.admit(Some(Instant::now())).expect("queue has room");
        let prepared = state
            .prepare("a".to_string())
            .expect("prepare should succeed");
        let result = ticket.run(prepared, None).await;
        assert!(matches!(result, Err(ApiError::Timeout)));
    }

    /// Assert that a generation was stopped soon after `steps_when_left`
    /// steps, far from its natural end.
    fn assert_stopped_within_a_few_steps(steps: &AtomicUsize, steps_when_left: usize) {
        let steps = steps.load(Ordering::SeqCst);
        assert!(
            steps <= steps_when_left + 4,
            "generation went on for {} steps after its caller left",
            steps - steps_when_left
        );
        assert!(steps < SLOW_GENERATION_STEPS / 4, "{steps} steps ran");
    }

    #[tokio::test(flavor = "current_thread")]
    async fn timed_out_request_stops_its_generation() {
        // Arrange -- a generation of two seconds against a timeout of 100 ms.
        let (state, steps, slow, _dir) = slow_generation_state(Duration::from_millis(100), None);
        let app = crate::build_router(Arc::clone(&state));

        // Act -- the request gives up after 100 ms.
        let asked = Instant::now();
        let response = app.oneshot(infer_request("a b")).await.expect("infallible");
        let timed_out_after = asked.elapsed();
        worker_reaches_the_end_of_its_queue(&state).await;
        let stopped_after = asked.elapsed();
        let steps_run = steps.load(Ordering::SeqCst);
        // The worker is free again: a second generation, at full speed and
        // with no timeout of its own, runs to its end.
        slow.store(false, Ordering::SeqCst);
        let ticket = state.admit(None).expect("queue has room");
        let prepared = state
            .prepare("a b".to_string())
            .expect("prepare should succeed");
        let second = ticket
            .run(prepared, None)
            .await
            .expect("inference should succeed");

        // Assert -- the timed-out generation was cut short, not run out.
        assert_eq!(response.status(), StatusCode::GATEWAY_TIMEOUT);
        assert!(
            timed_out_after < Duration::from_millis(350),
            "504 after {timed_out_after:?}"
        );
        assert!(
            stopped_after < Duration::from_secs(1),
            "generation still running {stopped_after:?} after the request"
        );
        assert!(
            (1..SLOW_GENERATION_STEPS / 4).contains(&steps_run),
            "{steps_run} steps ran"
        );
        let usage = second.usage.expect("text generation counts tokens");
        assert_eq!(usage.completion_tokens, SLOW_GENERATION_STEPS);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn dropped_request_stops_its_running_generation() {
        // Arrange -- a generation of two seconds, well inside the timeout.
        let (state, steps, _slow, _dir) = slow_generation_state(Duration::from_secs(30), None);
        let app = crate::build_router(Arc::clone(&state));
        let request = tokio::spawn(app.oneshot(infer_request("a b")));
        until("the generation to start", || {
            steps.load(Ordering::SeqCst) >= 3
        })
        .await;

        // Act -- the handler future is dropped, as axum does when the
        // client disconnects.
        let steps_when_left = steps.load(Ordering::SeqCst);
        request.abort();
        worker_reaches_the_end_of_its_queue(&state).await;

        // Assert
        assert_stopped_within_a_few_steps(&steps, steps_when_left);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn http_client_disconnect_stops_its_running_generation() {
        use tokio::io::AsyncWriteExt;

        // Arrange -- the real server on a local socket.
        let (state, steps, _slow, _dir) = slow_generation_state(Duration::from_secs(30), None);
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind");
        let address = listener.local_addr().expect("local address");
        let server = tokio::spawn(
            axum::serve(listener, crate::build_router(Arc::clone(&state))).into_future(),
        );
        let body = serde_json::json!({ "text": "a b" }).to_string();
        let request = format!(
            "POST /infer HTTP/1.1\r\nhost: {address}\r\ncontent-type: application/json\r\n\
             content-length: {}\r\n\r\n{body}",
            body.len()
        );
        let mut client = tokio::net::TcpStream::connect(address)
            .await
            .expect("connect");
        client
            .write_all(request.as_bytes())
            .await
            .expect("send request");
        until("the generation to start", || {
            steps.load(Ordering::SeqCst) >= 3
        })
        .await;

        // Act -- the client closes its connection without waiting.
        let steps_when_left = steps.load(Ordering::SeqCst);
        drop(client);
        worker_reaches_the_end_of_its_queue(&state).await;

        // Assert
        assert_stopped_within_a_few_steps(&steps, steps_when_left);
        server.abort();
    }

    #[tokio::test(flavor = "current_thread")]
    async fn dropped_grpc_call_stops_its_running_generation() {
        use hephaestus_proto::v1::inference_service_server::InferenceService;

        // Arrange
        let (state, steps, _slow, _dir) = slow_generation_state(Duration::from_secs(30), None);
        let service = crate::grpc::GrpcInferenceService::new(Arc::clone(&state));
        let call = tokio::spawn(async move {
            service
                .infer(tonic::Request::new(hephaestus_proto::v1::InferRequest {
                    text: "a b".to_string(),
                }))
                .await
        });
        until("the generation to start", || {
            steps.load(Ordering::SeqCst) >= 3
        })
        .await;

        // Act -- the call's future is dropped, as tonic does when the
        // client cancels.
        let steps_when_left = steps.load(Ordering::SeqCst);
        call.abort();
        worker_reaches_the_end_of_its_queue(&state).await;

        // Assert
        assert_stopped_within_a_few_steps(&steps, steps_when_left);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn dropped_batched_request_stops_its_running_generation() {
        // Arrange -- batching on: the request reaches the worker in a batch.
        let (batcher, receiver) = Batcher::new(2);
        let (state, steps, _slow, _dir) =
            slow_generation_state(Duration::from_secs(30), Some(batcher));
        let batcher_task = tokio::spawn(batcher_loop(
            receiver,
            Arc::clone(&state),
            2,
            Duration::from_millis(10),
        ));
        let app = crate::build_router(Arc::clone(&state));
        let request = tokio::spawn(app.oneshot(infer_request("a b")));
        until("the generation to start", || {
            steps.load(Ordering::SeqCst) >= 3
        })
        .await;

        // Act
        let steps_when_left = steps.load(Ordering::SeqCst);
        request.abort();
        worker_reaches_the_end_of_its_queue(&state).await;

        // Assert
        assert_stopped_within_a_few_steps(&steps, steps_when_left);
        batcher_task.abort();
    }

    #[tokio::test(flavor = "current_thread")]
    async fn over_long_prompt_is_refused_with_413_even_while_the_model_is_busy() {
        use hephaestus_proto::v1::inference_service_server::InferenceService;

        // Arrange -- a prompt cap of 16 tokens; the model is held by a
        // first request.
        let (hook, started, release) = gated_hook();
        let (pipeline, _dir) = text_generation_pipeline(4);
        let state = state_for(pipeline, Duration::from_secs(30), None);
        state.set_before_inference(hook);
        let state = Arc::new(state);
        let app = crate::build_router(Arc::clone(&state));
        let first = tokio::spawn(app.clone().oneshot(infer_request("a b")));
        until("the first run", || started.load(Ordering::SeqCst) == 1).await;
        // 40 words; the chat template adds three tokens.
        let long_text = ["a b c d e f g h i j"; 4].join(" ");

        // Act
        let asked = Instant::now();
        let response = app
            .clone()
            .oneshot(infer_request(&long_text))
            .await
            .expect("infallible");
        let refused_in = asked.elapsed();
        let grpc = crate::grpc::GrpcInferenceService::new(Arc::clone(&state))
            .infer(tonic::Request::new(hephaestus_proto::v1::InferRequest {
                text: long_text.clone(),
            }))
            .await;
        let queued_after = state.waiting_requests();

        // Assert -- a client error that states both numbers, at once.
        assert_eq!(response.status(), StatusCode::PAYLOAD_TOO_LARGE);
        assert!(
            refused_in < Duration::from_secs(1),
            "413 after {refused_in:?}"
        );
        let body = json_body(response).await;
        assert_eq!(body["error"]["code"], "INPUT_TOO_LONG");
        assert_eq!(
            body["error"]["message"],
            "input too long: the prompt has 43 tokens, the limit is 16"
        );
        let status = grpc.expect_err("gRPC call should fail");
        assert_eq!(status.code(), tonic::Code::InvalidArgument);
        assert_eq!(
            status.message(),
            "input too long: the prompt has 43 tokens, the limit is 16"
        );
        assert_eq!(queued_after, 0, "the refused requests left the queue");
        assert_eq!(
            started.load(Ordering::SeqCst),
            1,
            "nothing was truncated and run"
        );

        // The request that fits is unaffected.
        drop(release);
        let first = first.await.expect("task should join").expect("infallible");
        assert_eq!(first.status(), StatusCode::OK);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn generation_responses_carry_token_counts() {
        // Arrange -- text generation that runs for exactly 6 tokens, and
        // the seq2seq fixture, which echoes its input.
        let (pipeline, _text_generation_dir) = text_generation_pipeline(6);
        let text_generation =
            crate::build_router(Arc::new(state_for(pipeline, Duration::from_secs(30), None)));
        let (state, _seq2seq_dir) = fixture_state(Duration::from_secs(30), || {});
        let seq2seq = crate::build_router(state);

        // Act
        let text_generation = text_generation
            .oneshot(infer_request("a b"))
            .await
            .expect("infallible");
        let seq2seq = seq2seq
            .oneshot(infer_request("a b c"))
            .await
            .expect("infallible");

        // Assert -- the prompt is "c a b d e" (template around the text);
        // the seq2seq input is <s> a b c </s>, its output "a b c" </s>.
        assert_eq!(text_generation.status(), StatusCode::OK);
        let body = json_body(text_generation).await;
        assert_eq!(body["prompt_tokens"], 5);
        assert_eq!(body["completion_tokens"], 6);
        assert!(body["generated_text"].is_string());
        assert_eq!(seq2seq.status(), StatusCode::OK);
        let mut body = json_body(seq2seq).await;
        assert!(body["latency_ms"].is_u64());
        body.as_object_mut().expect("object").remove("latency_ms");
        assert_eq!(
            body,
            serde_json::json!({
                "generated_text": "a b c",
                "prompt_tokens": 5,
                "completion_tokens": 4,
                "model_id": "fixture/model",
            })
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn full_queue_is_refused_at_once_with_retry_after() {
        // Arrange -- room for one waiting request; the model is held.
        let (hook, started, release) = gated_hook();
        let (pipeline, _dir) = seq2seq_pipeline();
        let state =
            state_for(pipeline, Duration::from_secs(30), None).with_inference_queue_capacity(1);
        state.set_before_inference(hook);
        let state = Arc::new(state);
        let app = crate::build_router(Arc::clone(&state));

        // Act -- one request running, one waiting, then a third.
        let first = tokio::spawn(app.clone().oneshot(infer_request("a b c")));
        until("the first run", || started.load(Ordering::SeqCst) == 1).await;
        let second = tokio::spawn(app.clone().oneshot(infer_request("d e f")));
        until("the second request to queue", || {
            state.waiting_requests() == 1
        })
        .await;
        let asked = Instant::now();
        let third = app
            .clone()
            .oneshot(infer_request("g h i"))
            .await
            .expect("infallible");
        let refused_in = asked.elapsed();

        // Assert -- refused without waiting for the model.
        assert_eq!(third.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(third.headers()["retry-after"], "1");
        assert!(
            refused_in < Duration::from_secs(1),
            "503 after {refused_in:?}"
        );
        let body = json_body(third).await;
        assert_eq!(body["error"]["code"], "QUEUE_FULL");
        assert_eq!(started.load(Ordering::SeqCst), 1, "only one run started");

        // The two accepted requests are served once the model is free.
        drop(release);
        for (request, expected) in [(first, "a b c"), (second, "d e f")] {
            let response = request
                .await
                .expect("task should join")
                .expect("infallible");
            assert_eq!(response.status(), StatusCode::OK);
            assert_eq!(json_body(response).await["generated_text"], expected);
        }
        assert_eq!(state.waiting_requests(), 0);
        // With the queue empty again, a new request is accepted.
        let response = app
            .oneshot(infer_request("j k l"))
            .await
            .expect("infallible");
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn request_that_times_out_in_the_queue_never_runs() {
        // Arrange -- the first run outlasts the request timeout.
        let runs = Arc::new(AtomicUsize::new(0));
        let hook_runs = Arc::clone(&runs);
        let (state, _dir) = fixture_state(Duration::from_millis(150), move || {
            if hook_runs.fetch_add(1, Ordering::SeqCst) == 0 {
                std::thread::sleep(Duration::from_millis(600));
            }
        });
        let app = crate::build_router(Arc::clone(&state));

        // Act -- the second request waits behind the first and times out
        // there.
        let first = tokio::spawn(app.clone().oneshot(infer_request("a b c")));
        until("the first run", || runs.load(Ordering::SeqCst) == 1).await;
        let second = app
            .clone()
            .oneshot(infer_request("d e f"))
            .await
            .expect("infallible");
        let queue_after_timeout = state.waiting_requests();
        let first = first.await.expect("task should join").expect("infallible");
        // A request sent once the model is free again runs normally.
        tokio::time::sleep(Duration::from_millis(600)).await;
        let third = app
            .oneshot(infer_request("g h i"))
            .await
            .expect("infallible");

        // Assert
        assert_eq!(first.status(), StatusCode::GATEWAY_TIMEOUT);
        assert_eq!(second.status(), StatusCode::GATEWAY_TIMEOUT);
        assert_eq!(queue_after_timeout, 0, "the dead job left the queue");
        assert_eq!(third.status(), StatusCode::OK);
        assert_eq!(json_body(third).await["generated_text"], "g h i");
        assert_eq!(
            runs.load(Ordering::SeqCst),
            2,
            "the queued request never ran"
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn job_past_its_deadline_is_answered_with_a_timeout_and_not_run() {
        // Arrange -- a caller that keeps waiting past its deadline.
        let runs = Arc::new(AtomicUsize::new(0));
        let hook_runs = Arc::clone(&runs);
        let (state, _dir) = fixture_state(Duration::from_secs(30), move || {
            hook_runs.fetch_add(1, Ordering::SeqCst);
        });
        let ticket = state.admit(Some(Instant::now())).expect("queue has room");
        let prepared = state
            .prepare("a b c".to_string())
            .expect("prepare should succeed");

        // Act
        let result = ticket.run(prepared, None).await;

        // Assert
        assert!(matches!(result, Err(ApiError::Timeout)));
        assert_eq!(runs.load(Ordering::SeqCst), 0);
        assert_eq!(state.waiting_requests(), 0);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn queued_request_whose_caller_goes_away_never_runs() {
        // Arrange -- the model is held by a first request.
        let (hook, started, release) = gated_hook();
        let (state, _dir) = fixture_state(Duration::from_secs(30), hook);
        let app = crate::build_router(Arc::clone(&state));
        let first = tokio::spawn(app.clone().oneshot(infer_request("a b c")));
        until("the first run", || started.load(Ordering::SeqCst) == 1).await;

        // Act -- a second request queues, then its client disconnects
        // (axum drops the handler future).
        let second = tokio::spawn(app.clone().oneshot(infer_request("d e f")));
        until("the second request to queue", || {
            state.waiting_requests() == 1
        })
        .await;
        second.abort();
        until("the dead job to leave the queue", || {
            state.waiting_requests() == 0
        })
        .await;
        drop(release);
        let first = first.await.expect("task should join").expect("infallible");
        let third = app
            .oneshot(infer_request("g h i"))
            .await
            .expect("infallible");

        // Assert -- only the first and third requests ran.
        assert_eq!(first.status(), StatusCode::OK);
        assert_eq!(third.status(), StatusCode::OK);
        assert_eq!(started.load(Ordering::SeqCst), 2);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn never_more_than_one_job_runs_at_once() {
        // Arrange -- the hook counts how many runs overlap.
        let running = Arc::new(AtomicUsize::new(0));
        let most_at_once = Arc::new(AtomicUsize::new(0));
        let runs = Arc::new(AtomicUsize::new(0));
        let (hook_running, hook_most, hook_runs) = (
            Arc::clone(&running),
            Arc::clone(&most_at_once),
            Arc::clone(&runs),
        );
        let (state, _dir) = fixture_state(Duration::from_secs(30), move || {
            let now = hook_running.fetch_add(1, Ordering::SeqCst) + 1;
            hook_most.fetch_max(now, Ordering::SeqCst);
            hook_runs.fetch_add(1, Ordering::SeqCst);
            std::thread::sleep(Duration::from_millis(20));
            hook_running.fetch_sub(1, Ordering::SeqCst);
        });
        let app = crate::build_router(state);
        let texts = ["a", "b c", "d e f", "g", "h i", "j k l", "a c e", "b d"];

        // Act -- eight requests at the same time.
        let requests: Vec<_> = texts
            .iter()
            .map(|text| tokio::spawn(app.clone().oneshot(infer_request(text))))
            .collect();

        // Assert -- every request got its own answer, one run at a time.
        for (request, text) in requests.into_iter().zip(texts) {
            let response = request
                .await
                .expect("task should join")
                .expect("infallible");
            assert_eq!(response.status(), StatusCode::OK);
            assert_eq!(json_body(response).await["generated_text"], text);
        }
        assert_eq!(runs.load(Ordering::SeqCst), texts.len());
        assert_eq!(most_at_once.load(Ordering::SeqCst), 1);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn shutdown_fails_queued_jobs_and_lets_the_running_one_finish() {
        // Arrange -- one request running (held), two waiting.
        let (hook, started, release) = gated_hook();
        let (state, _dir) = fixture_state(Duration::from_secs(30), hook);
        let app = crate::build_router(Arc::clone(&state));
        let running = tokio::spawn(app.clone().oneshot(infer_request("a b c")));
        until("the first run", || started.load(Ordering::SeqCst) == 1).await;
        let queued: Vec<_> = ["d e f", "g h i"]
            .iter()
            .map(|text| tokio::spawn(app.clone().oneshot(infer_request(text))))
            .collect();
        until("two requests to queue", || state.waiting_requests() == 2).await;

        // Act -- shutdown begins while the model is still held.
        state.shutdown_inference();
        let mut queued_statuses = Vec::new();
        for request in queued {
            let response = request
                .await
                .expect("task should join")
                .expect("infallible");
            queued_statuses.push((response.status(), json_body(response).await));
        }
        let running_was_unfinished = !running.is_finished();
        // Readiness is the caller's to flip; a request that still gets in
        // is refused by the closed queue.
        let late = app
            .clone()
            .oneshot(infer_request("j k l"))
            .await
            .expect("infallible");
        drop(release);
        let running = running
            .await
            .expect("task should join")
            .expect("infallible");

        // Assert
        for (status, body) in queued_statuses {
            assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
            assert_eq!(body["error"]["code"], "NOT_READY");
        }
        assert!(
            running_was_unfinished,
            "queued jobs failed before the running one ended"
        );
        assert_eq!(late.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert!(
            state.is_ready(),
            "shutdown_inference leaves readiness alone"
        );
        assert_eq!(running.status(), StatusCode::OK);
        assert_eq!(json_body(running).await["generated_text"], "a b c");
        assert_eq!(started.load(Ordering::SeqCst), 1, "queued jobs never ran");
        assert_eq!(state.waiting_requests(), 0);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn batched_requests_run_through_the_worker() {
        // Arrange -- batching on: requests go batcher -> worker.
        let runs = Arc::new(AtomicUsize::new(0));
        let hook_runs = Arc::clone(&runs);
        let (pipeline, _dir) = seq2seq_pipeline();
        let (batcher, receiver) = Batcher::new(4);
        let state = state_for(pipeline, Duration::from_secs(30), Some(batcher));
        state.set_before_inference(move || {
            hook_runs.fetch_add(1, Ordering::SeqCst);
        });
        let state = Arc::new(state);
        let batcher_task = tokio::spawn(batcher_loop(
            receiver,
            Arc::clone(&state),
            4,
            Duration::from_millis(200),
        ));
        let app = crate::build_router(Arc::clone(&state));
        let texts = ["a b c", "d e", "f", "g h i j"];

        // Act -- four requests at once fill one batch.
        let requests: Vec<_> = texts
            .iter()
            .map(|text| tokio::spawn(app.clone().oneshot(infer_request(text))))
            .collect();

        // Assert -- each got its own answer, from a single job.
        for (request, text) in requests.into_iter().zip(texts) {
            let response = request
                .await
                .expect("task should join")
                .expect("infallible");
            assert_eq!(response.status(), StatusCode::OK);
            assert_eq!(json_body(response).await["generated_text"], text);
        }
        assert_eq!(runs.load(Ordering::SeqCst), 1, "one batch, one job");
        batcher_task.abort();
    }

    /// A text generation pipeline over the batch fixture graph of
    /// `hephaestus-core` (`model_batch.onnx`), in which a wrong mask,
    /// padding or cache changes the reply. `width` generations may run
    /// together. With `end_token` replies end by themselves after 1 to 27
    /// tokens; without, every reply runs to `max_new_tokens`.
    fn batch_generation_pipeline(
        width: usize,
        max_new_tokens: usize,
        end_token: bool,
    ) -> (PipelineKind, tempfile::TempDir) {
        let fixtures =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../hephaestus-core/tests/fixtures");
        let dir = tempfile::tempdir().expect("temp dir");
        for (from, file, to) in [
            ("text_generation", "model_batch.onnx", "model.onnx"),
            ("text_generation", "tokenizer_config.json", "tokenizer_config.json"),
            ("seq2seq", "tokenizer.json", "tokenizer.json"),
        ] {
            std::fs::copy(fixtures.join(from).join(file), dir.path().join(to))
                .expect("copy fixture");
        }
        let mut config = serde_json::json!({
            "architectures": ["LlamaForCausalLM"],
            "bos_token_id": 0,
            "max_position_embeddings": 512,
            "vocab_size": 16
        });
        if end_token {
            config["eos_token_id"] = 2.into();
        }
        std::fs::write(dir.path().join("config.json"), config.to_string()).expect("write config");
        let options = TextGenerationOptions::new()
            .with_max_new_tokens(max_new_tokens)
            .with_max_batch(width);
        let pipeline =
            TextGenerationPipeline::with_options(dir.path(), &ExecutionProvider::Cpu, &options)
                .expect("pipeline should load");
        (PipelineKind::TextGeneration(pipeline), dir)
    }

    /// Requests whose prompts and replies differ in length.
    const BATCH_TEXTS: [&str; 8] = [
        "h i",
        "a",
        "l k j i h g",
        "a b c d e f g h i j k l a b c",
        "k",
        "b b",
        "e f g h i j k l",
        "j k l a",
    ];

    /// State around [`batch_generation_pipeline`] whose every token step
    /// takes `step_time` per running generation. Returns the state and the
    /// largest number of generations seen running together.
    fn batch_generation_state(
        width: usize,
        max_new_tokens: usize,
        end_token: bool,
        step_time: Duration,
        batcher: Option<Batcher>,
    ) -> (Arc<AppState>, Arc<AtomicUsize>, tempfile::TempDir) {
        let (pipeline, dir) = batch_generation_pipeline(width, max_new_tokens, end_token);
        let state = Arc::new(state_for(pipeline, Duration::from_secs(30), batcher));
        let most = Arc::new(AtomicUsize::new(0));
        let (hook_state, hook_most) = (Arc::downgrade(&state), Arc::clone(&most));
        state.set_on_generation_step(move || {
            if let Some(state) = hook_state.upgrade() {
                hook_most.fetch_max(state.active_generations(), Ordering::SeqCst);
            }
            std::thread::sleep(step_time);
        });
        (state, most, dir)
    }

    /// The `/infer` response body for `text`, without its latency.
    async fn infer_body(app: &axum::Router, text: &str) -> serde_json::Value {
        let response = app
            .clone()
            .oneshot(infer_request(text))
            .await
            .expect("infallible");
        assert_eq!(response.status(), StatusCode::OK, "'{text}'");
        let mut body = json_body(response).await;
        body.as_object_mut().expect("object").remove("latency_ms");
        body
    }

    /// What every text of `texts` is answered with when it is generated
    /// alone, by a server without batching.
    async fn solo_bodies(
        max_new_tokens: usize,
        end_token: bool,
        texts: &[&str],
    ) -> Vec<serde_json::Value> {
        let (pipeline, _dir) = batch_generation_pipeline(1, max_new_tokens, end_token);
        let state = Arc::new(state_for(pipeline, Duration::from_secs(30), None));
        let app = crate::build_router(state);
        let mut bodies = Vec::new();
        for text in texts {
            bodies.push(infer_body(&app, text).await);
        }
        bodies
    }

    #[tokio::test(flavor = "current_thread")]
    async fn concurrent_generations_share_a_batch_and_get_their_solo_responses() {
        // Arrange -- three places, sixteen requests at once.
        let width = 3;
        let texts: Vec<&str> = BATCH_TEXTS.iter().chain(&BATCH_TEXTS).copied().collect();
        let expected = solo_bodies(40, true, &texts).await;
        let (state, most, _dir) =
            batch_generation_state(width, 40, true, Duration::from_millis(1), None);
        let app = crate::build_router(Arc::clone(&state));

        // Act
        let requests: Vec<_> = texts
            .iter()
            .map(|text| {
                let app = app.clone();
                let text = (*text).to_string();
                tokio::spawn(async move { infer_body(&app, &text).await })
            })
            .collect();

        // Assert -- text and token counts are each request's own.
        for ((request, text), expected) in requests.into_iter().zip(&texts).zip(&expected) {
            let body = request.await.expect("task should join");
            assert_eq!(&body, expected, "'{text}'");
            assert!(body["prompt_tokens"].is_u64() && body["completion_tokens"].is_u64());
        }
        assert_eq!(
            most.load(Ordering::SeqCst),
            width,
            "the batch should fill up and never hold more than its width"
        );
        until("the batch to empty", || state.active_generations() == 0).await;
        assert_eq!(state.waiting_requests(), 0);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn request_joins_a_generation_that_is_already_running() {
        // Arrange -- "b b" runs for 27 tokens; "a" arrives a few in.
        let expected = solo_bodies(40, true, &["b b", "a"]).await;
        let (state, most, _dir) =
            batch_generation_state(2, 40, true, Duration::from_millis(5), None);
        let app = crate::build_router(Arc::clone(&state));

        // Act
        let first = {
            let app = app.clone();
            tokio::spawn(async move { infer_body(&app, "b b").await })
        };
        until("the first generation to run", || {
            state.active_generations() == 1
        })
        .await;
        let second = infer_body(&app, "a").await;
        let first_was_running = !first.is_finished();
        let first = first.await.expect("task should join");

        // Assert -- the second did not wait for the first to end, and
        // neither changed the other.
        assert!(first_was_running, "the second request was answered first");
        assert_eq!(most.load(Ordering::SeqCst), 2, "both ran together");
        assert_eq!(first, expected[0]);
        assert_eq!(second, expected[1]);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn abandoned_generation_leaves_the_batch_to_a_waiting_request() {
        // Arrange -- two places, replies of 60 tokens, three requests.
        let expected = solo_bodies(60, false, &["b b", "h i"]).await;
        let (state, most, _dir) =
            batch_generation_state(2, 60, false, Duration::from_millis(3), None);
        let run = |text: &str| {
            let state = Arc::clone(&state);
            let text = text.to_string();
            tokio::spawn(async move {
                let ticket = state.admit(None).expect("queue has room");
                let prepared = state.prepare(text).expect("prepare should succeed");
                ticket.run(prepared, None).await
            })
        };

        // Act -- two run, the third waits for a place.
        let stays = run("b b");
        let leaves = run("a");
        until("two generations to run", || state.active_generations() == 2).await;
        let waits = run("h i");
        until("the third request to wait", || state.waiting_requests() == 1).await;
        tokio::time::sleep(Duration::from_millis(20)).await;
        let waited_for_a_place = state.waiting_requests() == 1 && !waits.is_finished();
        // One caller goes away: its place goes to the waiting request.
        leaves.abort();
        until("the third request to start", || state.waiting_requests() == 0).await;
        let stays_was_running = !stays.is_finished();
        let stays = stays.await.expect("task should join").expect("generation");
        let waits = waits.await.expect("task should join").expect("generation");

        // Assert
        assert!(waited_for_a_place, "a full batch takes no third request");
        assert!(stays_was_running, "the place was freed before the batch ended");
        assert_eq!(most.load(Ordering::SeqCst), 2);
        for (execution, expected) in [(stays, &expected[0]), (waits, &expected[1])] {
            let json = execution.to_json();
            assert_eq!(json["generated_text"], expected["generated_text"]);
            assert_eq!(json["prompt_tokens"], expected["prompt_tokens"]);
            assert_eq!(json["completion_tokens"], 60);
        }
    }

    #[tokio::test(flavor = "current_thread")]
    async fn generation_past_its_deadline_times_out_alone() {
        // Arrange -- two generations of 60 slow tokens; one has 60 ms.
        let expected = solo_bodies(60, false, &["b b"]).await;
        let (state, _most, _dir) =
            batch_generation_state(2, 60, false, Duration::from_millis(3), None);
        let run = |text: &str, deadline: Option<Duration>| {
            let state = Arc::clone(&state);
            let text = text.to_string();
            tokio::spawn(async move {
                let ticket = state
                    .admit(deadline.map(|after| Instant::now() + after))
                    .expect("queue has room");
                let prepared = state.prepare(text).expect("prepare should succeed");
                ticket.run(prepared, None).await
            })
        };

        // Act
        let unlimited = run("b b", None);
        let limited = run("a", Some(Duration::from_millis(60)));
        until("two generations to run", || state.active_generations() == 2).await;
        let limited = limited.await.expect("task should join");
        let unlimited_was_running = !unlimited.is_finished();
        let unlimited = unlimited.await.expect("task should join").expect("generation");

        // Assert
        assert!(matches!(limited, Err(ApiError::Timeout)), "{:?}", limited.err());
        assert!(unlimited_was_running, "the timeout was answered at once");
        let json = unlimited.to_json();
        assert_eq!(json["generated_text"], expected[0]["generated_text"]);
        assert_eq!(json["completion_tokens"], 60);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn full_queue_is_refused_while_a_batch_is_running() {
        // Arrange -- two places in the batch, one in the queue; the model
        // is held at its first token step.
        let (hook, started, release) = gated_hook();
        let (pipeline, _dir) = batch_generation_pipeline(2, 4, false);
        let state =
            state_for(pipeline, Duration::from_secs(30), None).with_inference_queue_capacity(1);
        state.set_on_generation_step(hook);
        let state = Arc::new(state);
        let app = crate::build_router(Arc::clone(&state));

        // Act -- one request is being read, one waits, a third arrives.
        let first = tokio::spawn(app.clone().oneshot(infer_request("a b")));
        until("the first generation", || started.load(Ordering::SeqCst) == 1).await;
        let second = tokio::spawn(app.clone().oneshot(infer_request("b b")));
        until("the second request to queue", || {
            state.waiting_requests() == 1
        })
        .await;
        let third = app
            .clone()
            .oneshot(infer_request("h i"))
            .await
            .expect("infallible");

        // Assert
        assert_eq!(third.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(third.headers()["retry-after"], "1");
        assert_eq!(json_body(third).await["error"]["code"], "QUEUE_FULL");
        drop(release);
        for request in [first, second] {
            let response = request.await.expect("task should join").expect("infallible");
            assert_eq!(response.status(), StatusCode::OK);
            assert_eq!(json_body(response).await["completion_tokens"], 4);
        }
        assert_eq!(state.waiting_requests(), 0);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn shutdown_fails_waiting_requests_and_lets_the_running_batch_finish() {
        // Arrange -- the hook holds the worker every time it asks a
        // generation whether to go on: once when a request joins, then
        // once per running generation at every token step.
        let (hook, asked, release) = gated_hook();
        let (pipeline, _dir) = batch_generation_pipeline(2, 4, false);
        let state = state_for(pipeline, Duration::from_secs(30), None);
        state.set_on_generation_step(hook);
        let state = Arc::new(state);
        let app = crate::build_router(Arc::clone(&state));
        let let_go = |times: usize| {
            for _ in 0..times {
                release.send(()).expect("worker is waiting");
            }
        };
        // A full batch of two: the first joins and takes a step, then the
        // second joins.
        let first = tokio::spawn(app.clone().oneshot(infer_request("a b")));
        until("the first generation", || asked.load(Ordering::SeqCst) == 1).await;
        let second = tokio::spawn(app.clone().oneshot(infer_request("b b")));
        until("the second request to queue", || {
            state.waiting_requests() == 1
        })
        .await;
        let_go(3);
        until("both generations to run", || state.active_generations() == 2).await;
        // Two more wait. After one more step the worker has taken one of
        // them off the queue and holds it until a place is free.
        let waiting: Vec<_> = ["h i", "k"]
            .iter()
            .map(|text| tokio::spawn(app.clone().oneshot(infer_request(text))))
            .collect();
        until("two requests to wait", || state.waiting_requests() == 2).await;
        let_go(2);
        until("the next step", || asked.load(Ordering::SeqCst) == 6).await;

        // Act
        state.shutdown_inference();
        let late = app
            .clone()
            .oneshot(infer_request("j k l"))
            .await
            .expect("infallible");
        until("the queued request to be refused", || {
            waiting.iter().any(|request| request.is_finished())
        })
        .await;
        let running_were_unfinished = !first.is_finished() && !second.is_finished();
        drop(release);

        // Assert -- both waiting requests fail, the one the worker held
        // included; the running ones finish.
        assert_eq!(late.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert!(running_were_unfinished);
        for request in waiting {
            let response = request.await.expect("task should join").expect("infallible");
            assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
            assert_eq!(json_body(response).await["error"]["code"], "NOT_READY");
        }
        for request in [first, second] {
            let response = request.await.expect("task should join").expect("infallible");
            assert_eq!(response.status(), StatusCode::OK);
            assert_eq!(json_body(response).await["completion_tokens"], 4);
        }
        assert_eq!(state.waiting_requests(), 0);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn dynamic_batcher_is_bypassed_when_generation_batches_itself() {
        // Arrange -- BATCH_ENABLED and a generation batch width together.
        let texts = ["b b", "a", "h i"];
        let expected = solo_bodies(40, true, &texts).await;
        let (batcher, receiver) = Batcher::new(4);
        let (state, most, _dir) =
            batch_generation_state(2, 40, true, Duration::from_millis(2), Some(batcher));
        let batcher_task = tokio::spawn(batcher_loop(
            receiver,
            Arc::clone(&state),
            4,
            Duration::from_millis(200),
        ));
        let app = crate::build_router(Arc::clone(&state));

        // Act
        let requests: Vec<_> = texts
            .iter()
            .map(|text| {
                let app = app.clone();
                let text = (*text).to_string();
                tokio::spawn(async move { infer_body(&app, &text).await })
            })
            .collect();

        // Assert -- served by the worker's own batch; the batcher's task
        // has nothing to wait for and ends.
        assert!(!state.is_batching_enabled());
        for (request, expected) in requests.into_iter().zip(&expected) {
            assert_eq!(&request.await.expect("task should join"), expected);
        }
        assert_eq!(most.load(Ordering::SeqCst), 2);
        batcher_task.await.expect("batcher loop should end");
    }

    #[tokio::test(flavor = "current_thread")]
    async fn dynamic_batcher_still_feeds_generation_of_width_one() {
        // Arrange -- BATCH_ENABLED without a generation batch width.
        let expected = solo_bodies(40, true, &["b b", "a"]).await;
        let (batcher, receiver) = Batcher::new(2);
        let (pipeline, _dir) = batch_generation_pipeline(1, 40, true);
        let state = Arc::new(state_for(pipeline, Duration::from_secs(30), Some(batcher)));
        let batcher_task = tokio::spawn(batcher_loop(
            receiver,
            Arc::clone(&state),
            2,
            Duration::from_millis(50),
        ));
        let app = crate::build_router(Arc::clone(&state));

        // Act
        let first = {
            let app = app.clone();
            tokio::spawn(async move { infer_body(&app, "b b").await })
        };
        let second = infer_body(&app, "a").await;

        // Assert
        assert!(state.is_batching_enabled());
        assert_eq!(first.await.expect("task should join"), expected[0]);
        assert_eq!(second, expected[1]);
        batcher_task.abort();
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
