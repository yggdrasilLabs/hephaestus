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
            let pipeline_output = {
                let mut pipeline = state.write_pipeline().await;
                timer.time("inference", || pipeline.execute(prepared))?
            }; // Write lock dropped here.
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
    use super::*;

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
