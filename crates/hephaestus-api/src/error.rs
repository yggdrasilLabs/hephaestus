//! API error types and HTTP response mapping.
//!
//! [`ApiError`] wraps [`CoreError`](hephaestus_core::CoreError) and maps
//! each variant to a structured JSON error response with the appropriate
//! HTTP status code per D-03.

use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use hephaestus_core::CoreError;
use thiserror::Error;

use crate::worker::QUEUE_FULL_RETRY_AFTER_SECS;

/// Errors produced by the HTTP serving layer.
///
/// Each variant maps to a specific HTTP status code and a machine-parseable
/// error code in the JSON response body (D-03).
#[derive(Error, Debug)]
pub enum ApiError {
    /// Tokenization failed (maps to 422 Unprocessable Entity).
    #[error("tokenization failed: {0}")]
    Tokenization(String),

    /// ONNX inference failed (maps to 500 Internal Server Error).
    #[error("inference failed: {0}")]
    Inference(String),

    /// Model load or validation error (maps to 500 Internal Server Error).
    #[error("model error: {0}")]
    Model(String),

    /// Request-level inference timeout exceeded (maps to 504 Gateway Timeout, D-14).
    #[error("inference timeout")]
    Timeout,

    /// Service not ready -- warmup not complete or shutdown in progress (maps to 503).
    #[error("service not ready")]
    NotReady,

    /// The inference queue is at capacity (maps to 503 with `Retry-After`).
    #[error("inference queue full")]
    QueueFull,

    /// Client sent an invalid request (maps to 400 Bad Request).
    #[error("bad request: {0}")]
    BadRequest(String),

    /// Catch-all internal error (maps to 500 Internal Server Error).
    #[error("internal error: {0}")]
    Internal(String),
}

impl From<CoreError> for ApiError {
    fn from(err: CoreError) -> Self {
        match err {
            CoreError::Tokenization(msg) => Self::Tokenization(msg),
            CoreError::Inference(msg) => Self::Inference(msg),
            CoreError::InvalidInput(msg) => Self::BadRequest(msg),
            CoreError::ModelLoad(msg) | CoreError::ModelValidation(msg) => Self::Model(msg),
            CoreError::Config(msg) => Self::Internal(msg),
            CoreError::Timeout(_) => Self::Timeout,
            CoreError::Io(e) => Self::Internal(e.to_string()),
            CoreError::JsonParse(e) => Self::Internal(e.to_string()),
        }
    }
}

impl From<ApiError> for tonic::Status {
    fn from(err: ApiError) -> Self {
        match err {
            ApiError::NotReady => tonic::Status::unavailable("service not ready"),
            // UNAVAILABLE, like the HTTP 503: transient, and the code that
            // clients and meshes retry with backoff. RESOURCE_EXHAUSTED
            // reads as a per-caller quota and is usually not retried.
            ApiError::QueueFull => tonic::Status::unavailable("inference queue full"),
            ApiError::BadRequest(msg) => tonic::Status::invalid_argument(msg),
            ApiError::Tokenization(msg) => {
                tonic::Status::invalid_argument(format!("tokenization failed: {msg}"))
            }
            ApiError::Timeout => tonic::Status::deadline_exceeded("inference timeout"),
            // Internal errors: log server-side, return generic message to clients
            // (same information-hiding as the HTTP handler).
            ApiError::Inference(ref msg) => {
                tracing::error!(error = %msg, "gRPC inference error");
                tonic::Status::internal("internal server error")
            }
            ApiError::Model(ref msg) => {
                tracing::error!(error = %msg, "gRPC model error");
                tonic::Status::internal("internal server error")
            }
            ApiError::Internal(ref msg) => {
                tracing::error!(error = %msg, "gRPC internal error");
                tonic::Status::internal("internal server error")
            }
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let (status, code) = match &self {
            Self::Tokenization(_) => (StatusCode::UNPROCESSABLE_ENTITY, "TOKENIZATION_FAILED"),
            Self::Inference(_) => (StatusCode::INTERNAL_SERVER_ERROR, "INFERENCE_FAILED"),
            Self::Model(_) => (StatusCode::INTERNAL_SERVER_ERROR, "MODEL_ERROR"),
            Self::Timeout => (StatusCode::GATEWAY_TIMEOUT, "INFERENCE_TIMEOUT"),
            Self::NotReady => (StatusCode::SERVICE_UNAVAILABLE, "NOT_READY"),
            Self::QueueFull => (StatusCode::SERVICE_UNAVAILABLE, "QUEUE_FULL"),
            Self::BadRequest(_) => (StatusCode::BAD_REQUEST, "BAD_REQUEST"),
            Self::Internal(_) => (StatusCode::INTERNAL_SERVER_ERROR, "INTERNAL_ERROR"),
        };

        // Log detailed error server-side for debugging; return generic
        // message to clients for server errors to avoid leaking internal
        // paths and system details (information disclosure).
        if status.is_server_error() {
            tracing::error!(error = %self, "request failed");
        }

        let client_message = match &self {
            Self::Internal(_) | Self::Inference(_) | Self::Model(_) => {
                "internal server error".to_string()
            }
            other => other.to_string(),
        };

        let body = serde_json::json!({
            "error": {
                "code": code,
                "message": client_message,
            }
        });

        let mut response = (status, axum::Json(body)).into_response();
        if matches!(self, Self::QueueFull) {
            response.headers_mut().insert(
                axum::http::header::RETRY_AFTER,
                axum::http::HeaderValue::from(QUEUE_FULL_RETRY_AFTER_SECS),
            );
        }
        response
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::to_bytes;

    #[tokio::test]
    async fn api_error_tokenization_maps_to_422() {
        // Arrange
        let err = ApiError::Tokenization("bad token".to_string());

        // Act
        let response = err.into_response();

        // Assert
        assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY);
        let body = to_bytes(response.into_body(), 1024).await.unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["error"]["code"], "TOKENIZATION_FAILED");
    }

    #[tokio::test]
    async fn api_error_timeout_maps_to_504() {
        // Arrange
        let err = ApiError::Timeout;

        // Act
        let response = err.into_response();

        // Assert
        assert_eq!(response.status(), StatusCode::GATEWAY_TIMEOUT);
        let body = to_bytes(response.into_body(), 1024).await.unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["error"]["code"], "INFERENCE_TIMEOUT");
    }

    #[tokio::test]
    async fn api_error_not_ready_maps_to_503() {
        // Arrange
        let err = ApiError::NotReady;

        // Act
        let response = err.into_response();

        // Assert
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        let body = to_bytes(response.into_body(), 1024).await.unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["error"]["code"], "NOT_READY");
    }

    #[tokio::test]
    async fn api_error_queue_full_maps_to_503_with_retry_after() {
        // Arrange
        let err = ApiError::QueueFull;

        // Act
        let response = err.into_response();

        // Assert
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(response.headers()["retry-after"], "1");
        let body = to_bytes(response.into_body(), 1024).await.unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["error"]["code"], "QUEUE_FULL");
        assert_eq!(json["error"]["message"], "inference queue full");
    }

    #[test]
    fn api_error_queue_full_to_grpc_unavailable() {
        let status: tonic::Status = ApiError::QueueFull.into();
        assert_eq!(status.code(), tonic::Code::Unavailable);
        assert_eq!(status.message(), "inference queue full");
    }

    #[tokio::test]
    async fn core_error_converts_to_api_error() {
        // Arrange
        let core_err = CoreError::Tokenization("test error".to_string());

        // Act
        let api_err: ApiError = core_err.into();

        // Assert
        assert!(matches!(api_err, ApiError::Tokenization(_)));
    }

    #[test]
    fn api_error_not_ready_to_grpc_unavailable() {
        let status: tonic::Status = ApiError::NotReady.into();
        assert_eq!(status.code(), tonic::Code::Unavailable);
    }

    #[test]
    fn api_error_timeout_to_grpc_deadline_exceeded() {
        let status: tonic::Status = ApiError::Timeout.into();
        assert_eq!(status.code(), tonic::Code::DeadlineExceeded);
    }

    #[test]
    fn api_error_tokenization_to_grpc_invalid_argument() {
        let status: tonic::Status = ApiError::Tokenization("bad input".into()).into();
        assert_eq!(status.code(), tonic::Code::InvalidArgument);
        assert!(status.message().contains("tokenization failed"));
    }

    #[test]
    fn api_error_inference_to_grpc_internal() {
        let status: tonic::Status = ApiError::Inference("ort crash".into()).into();
        assert_eq!(status.code(), tonic::Code::Internal);
        assert_eq!(status.message(), "internal server error");
    }
}
