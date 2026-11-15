//! gRPC InferenceService implementation.
//!
//! [`GrpcInferenceService`] implements the tonic-generated
//! [`InferenceService`](hephaestus_proto::v1::inference_service_server::InferenceService)
//! trait, mirroring the HTTP handler control flow from [`handlers::infer`](crate::handlers::infer).
//! Results are returned as opaque JSON bytes in `result_json`, matching
//! the REST API response payload for all model profiles.

use std::sync::Arc;
use std::time::Instant;

use tonic::{Request, Response, Status};

use hephaestus_proto::v1::inference_service_server::InferenceService;
use hephaestus_proto::v1::{InferRequest, InferResponse};

use crate::error::ApiError;
use crate::metrics::StageTimer;
use crate::state::AppState;

/// gRPC implementation of the Hephaestus InferenceService.
///
/// Holds a reference to shared [`AppState`] and follows the same
/// control flow as the HTTP handler: readiness check, input validation,
/// timeout-wrapped inference (through the inference queue or the batcher),
/// and model-determined JSON output serialized into `result_json` bytes.
pub struct GrpcInferenceService {
    state: Arc<AppState>,
}

impl GrpcInferenceService {
    /// Create a new gRPC inference service backed by the given application state.
    pub fn new(state: Arc<AppState>) -> Self {
        Self { state }
    }
}

#[tonic::async_trait]
impl InferenceService for GrpcInferenceService {
    async fn infer(
        &self,
        request: Request<InferRequest>,
    ) -> Result<Response<InferResponse>, Status> {
        let req = request.into_inner();

        // Gate on readiness (matches HTTP handler).
        if !self.state.is_ready() {
            return Err(ApiError::NotReady.into());
        }

        // Validate input.
        if req.text.is_empty() {
            return Err(ApiError::BadRequest("text field must not be empty".into()).into());
        }

        let request_start = Instant::now();
        let timer = StageTimer::new(self.state.model_id());
        // A job still queued at this moment is dropped instead of run.
        let deadline = request_start + self.state.request_timeout();

        // Wrap inference in a request-level timeout (mirrors HTTP handler).
        let result = tokio::time::timeout(self.state.request_timeout(), async {
            if self.state.is_batching_enabled() {
                // Batching path: prepare, then submit to the batcher.
                let prepared = timer.time("tokenization", || self.state.prepare(req.text))?;

                let pipeline_output = self
                    .state
                    .batcher()
                    .ok_or(ApiError::Internal("batcher not available".into()))?
                    .submit(prepared, Some(deadline))
                    .await?;
                Ok::<_, ApiError>(pipeline_output.to_json())
            } else {
                // Direct path: a place in the inference queue first (refused
                // at once when it is full), then prepare.
                let ticket = self.state.admit(Some(deadline))?;
                let prepared = timer.time("tokenization", || self.state.prepare(req.text))?;
                // The model runs on the inference worker thread, so this
                // task only waits.
                let pipeline_output = ticket.run(prepared, Some(timer.clone())).await?;
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
                        model_id = %self.state.model_id(),
                        latency_ms,
                        status = "error",
                        "gRPC inference request failed"
                    );
                    return Err(e.into());
                }
            },
            Err(_elapsed) => {
                let latency_ms = request_start.elapsed().as_millis() as u64;
                timer.finish_request(request_start, false);
                tracing::warn!(
                    model_id = %self.state.model_id(),
                    latency_ms,
                    status = "timeout",
                    "gRPC inference request timed out"
                );
                return Err(ApiError::Timeout.into());
            }
        };

        let latency_ms = request_start.elapsed().as_millis() as u64;

        // Insert model_id and latency_ms into the model-determined output
        // (same enrichment as the HTTP handler).
        if let Some(obj) = output.as_object_mut() {
            obj.insert(
                "model_id".to_string(),
                serde_json::Value::String(self.state.model_id().to_string()),
            );
            obj.insert(
                "latency_ms".to_string(),
                serde_json::Value::Number(serde_json::Number::from(latency_ms)),
            );
        }

        tracing::info!(
            model_id = %self.state.model_id(),
            latency_ms,
            status = "success",
            "gRPC inference request completed"
        );

        // Serialize the enriched JSON output into opaque bytes.
        let result_json = serde_json::to_vec(&output).map_err(|e| {
            tracing::error!(error = %e, "failed to serialize inference output");
            Status::internal("internal server error")
        })?;

        Ok(Response::new(InferResponse {
            model_id: self.state.model_id().to_string(),
            latency_ms,
            result_json,
        }))
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn classifier_result_json_roundtrips() {
        // Arrange -- simulate classifier output with model_id/latency injected
        let output = serde_json::json!({
            "label": "POSITIVE",
            "score": 0.9998,
            "model_id": "test/classifier",
            "latency_ms": 5,
        });

        // Act -- serialize to result_json bytes as the handler does
        let result_json = serde_json::to_vec(&output).expect("serialize should succeed");

        // Assert -- deserialize back and verify fields
        let parsed: serde_json::Value =
            serde_json::from_slice(&result_json).expect("result_json should be valid JSON");
        assert_eq!(parsed["label"], "POSITIVE");
        let score = parsed["score"].as_f64().expect("score should be a number");
        assert!((score - 0.9998).abs() < 1e-6);
        assert_eq!(parsed["model_id"], "test/classifier");
        assert_eq!(parsed["latency_ms"], 5);
    }

    #[test]
    fn embedding_result_json_roundtrips() {
        // Arrange -- simulate embeddings output
        let output = serde_json::json!({
            "embedding": [0.1, 0.2, 0.3, 0.4],
            "model_id": "test/embeddings",
            "latency_ms": 14,
        });

        // Act
        let result_json = serde_json::to_vec(&output).expect("serialize should succeed");

        // Assert
        let parsed: serde_json::Value =
            serde_json::from_slice(&result_json).expect("result_json should be valid JSON");
        assert!(parsed["embedding"].is_array());
        assert_eq!(parsed["embedding"].as_array().unwrap().len(), 4);
        assert_eq!(parsed["model_id"], "test/embeddings");
        assert_eq!(parsed["latency_ms"], 14);
    }
}
