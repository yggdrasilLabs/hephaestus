//! Forge conversion service client.
//!
//! Defines the [`ForgeClient`] trait for requesting model conversion
//! to ONNX format, the [`StubForgeClient`] stub implementation, and
//! the [`HttpForgeClient`] real HTTP client using reqwest.

use std::time::Duration;

use reqwest::Client;
use serde::{Deserialize, Serialize};

use crate::error::ResolveError;

/// Response from the Forge conversion service.
///
/// Contains the storage paths of the converted model files and metadata
/// about the conversion process.
#[derive(Deserialize, Debug)]
pub struct ForgeResponse {
    /// Storage paths, relative to the configured backend root, where the
    /// converted ONNX files were uploaded.
    pub storage_paths: Vec<String>,
    /// Metadata about the conversion process.
    pub metadata: ConversionMetadata,
}

/// Metadata about a Forge conversion operation.
#[derive(Deserialize, Debug)]
pub struct ConversionMetadata {
    /// Model architecture (e.g., `"distilbert"`).
    pub architecture: String,
    /// Original model format before conversion (e.g., `"pytorch"`).
    pub original_format: String,
    /// Time taken for conversion in seconds.
    pub conversion_duration_secs: f64,
    /// Version of the `optimum` library used for conversion.
    pub optimum_version: String,
}

/// Forge service client for converting models to ONNX format (D-10).
///
/// Phase 3 shipped [`StubForgeClient`] which always returns
/// [`ResolveError::ForgeUnavailable`]. Phase 5 adds [`HttpForgeClient`]
/// which POSTs to the Forge service via reqwest.
///
/// The trait follows the Ousterhout deep module pattern with a single
/// `convert()` method hiding all conversion complexity.
#[cfg_attr(test, mockall::automock)]
pub trait ForgeClient: Send + Sync {
    /// Request model conversion to ONNX format.
    ///
    /// Returns a [`ForgeResponse`] containing the storage paths and
    /// conversion metadata on success, or a [`ResolveError`] on failure.
    fn convert(
        &self,
        model_id: &str,
    ) -> impl std::future::Future<Output = Result<ForgeResponse, ResolveError>> + Send;
}

/// Stub Forge client that always returns an unavailable error (D-10).
///
/// Used when `FORGE_URL` is not configured. Returns a clear error
/// message indicating the model has no ONNX export and the Forge
/// service is not available (D-04).
pub struct StubForgeClient;

impl ForgeClient for StubForgeClient {
    async fn convert(&self, model_id: &str) -> Result<ForgeResponse, ResolveError> {
        Err(ResolveError::ForgeUnavailable {
            model_id: model_id.to_string(),
        })
    }
}

/// JSON request body for the Forge `/convert` endpoint.
#[derive(Serialize)]
struct ConvertRequest {
    model_id: String,
}

/// HTTP client for the Forge conversion service.
///
/// Sends POST requests to `{base_url}/convert` with a JSON body
/// containing the model ID. The response is deserialized into a
/// [`ForgeResponse`] with storage paths and conversion metadata.
///
/// Configured with a timeout from `FORGE_TIMEOUT_SECS` (default 600s,
/// per D-04) to prevent unbounded blocking on long conversions.
pub struct HttpForgeClient {
    client: Client,
    base_url: String,
}

impl HttpForgeClient {
    /// Create a new HTTP Forge client.
    ///
    /// # Arguments
    ///
    /// * `base_url` - Base URL of the Forge service (e.g., `"http://forge:8080"`).
    ///   Trailing slashes are trimmed.
    /// * `timeout_secs` - Request timeout in seconds (T-05-R02).
    ///
    /// # Errors
    ///
    /// Returns [`ResolveError::ForgeConversion`] if the HTTP client cannot
    /// be built (e.g., TLS backend initialization failure).
    pub fn new(base_url: &str, timeout_secs: u64) -> Result<Self, ResolveError> {
        let client = Client::builder()
            .timeout(Duration::from_secs(timeout_secs))
            .build()
            .map_err(|e| ResolveError::ForgeConversion {
                model_id: String::new(),
                reason: format!("failed to build HTTP client: {e}"),
            })?;

        Ok(Self {
            client,
            base_url: base_url.trim_end_matches('/').to_string(),
        })
    }
}

impl ForgeClient for HttpForgeClient {
    async fn convert(&self, model_id: &str) -> Result<ForgeResponse, ResolveError> {
        let url = format!("{}/convert", self.base_url);
        let body = ConvertRequest {
            model_id: model_id.to_string(),
        };

        let response = self.client.post(&url).json(&body).send().await.map_err(
            |e| ResolveError::ForgeConversion {
                model_id: model_id.to_string(),
                reason: e.to_string(),
            },
        )?;

        let status = response.status();
        if !status.is_success() {
            let body_text = response.text().await.unwrap_or_default();
            return Err(ResolveError::ForgeConversion {
                model_id: model_id.to_string(),
                reason: format!("HTTP {status}: {body_text}"),
            });
        }

        response.json::<ForgeResponse>().await.map_err(|e| {
            ResolveError::ForgeConversion {
                model_id: model_id.to_string(),
                reason: format!("invalid response: {e}"),
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn stub_forge_returns_forge_unavailable() {
        let client = StubForgeClient;
        let result = client.convert("org/model").await;
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            matches!(err, ResolveError::ForgeUnavailable { ref model_id } if model_id == "org/model"),
            "expected ForgeUnavailable with model_id 'org/model', got: {err:?}"
        );
    }

    #[tokio::test]
    async fn stub_forge_error_message_mentions_model() {
        let client = StubForgeClient;
        let err = client.convert("test-org/test-model").await.unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("test-org/test-model"),
            "error message should contain the model ID: {msg}"
        );
        assert!(
            msg.contains("Forge"),
            "error message should mention Forge: {msg}"
        );
    }

    #[tokio::test]
    async fn stub_forge_error_message_mentions_configuration() {
        let client = StubForgeClient;
        let err = client.convert("some/model").await.unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("FORGE_URL") || msg.contains("not configured"),
            "error message should mention configuration: {msg}"
        );
    }

    #[test]
    fn forge_client_trait_has_single_method() {
        // Compile-time verification: ForgeClient trait has exactly
        // one required method (convert). This test documents the
        // Ousterhout deep module constraint (D-10).
        //
        // If additional methods are added to ForgeClient, this test
        // still compiles but serves as documentation that the trait
        // should remain minimal.
        fn _assert_forge_client_impl<T: ForgeClient>() {}
        _assert_forge_client_impl::<StubForgeClient>();
    }

    #[test]
    fn http_forge_client_stores_base_url() {
        let client = HttpForgeClient::new("http://forge:8080", 600).unwrap();
        assert_eq!(client.base_url, "http://forge:8080");
    }

    #[test]
    fn http_forge_client_trims_trailing_slash() {
        let client = HttpForgeClient::new("http://forge:8080/", 600).unwrap();
        assert_eq!(client.base_url, "http://forge:8080");
    }

    #[test]
    fn forge_response_deserializes_from_json() {
        let json = r#"{
            "storage_paths": ["org/model/model.onnx", "org/model/tokenizer.json"],
            "metadata": {
                "architecture": "distilbert",
                "original_format": "pytorch",
                "conversion_duration_secs": 42.5,
                "optimum_version": "1.17.0"
            }
        }"#;

        let resp: ForgeResponse = serde_json::from_str(json).expect("should deserialize");
        assert_eq!(resp.storage_paths.len(), 2);
        assert_eq!(resp.metadata.architecture, "distilbert");
        assert_eq!(resp.metadata.original_format, "pytorch");
        assert!((resp.metadata.conversion_duration_secs - 42.5).abs() < f64::EPSILON);
        assert_eq!(resp.metadata.optimum_version, "1.17.0");
    }
}
