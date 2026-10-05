//! Model resolver: deep module hiding the 3-tier resolution chain.
//!
//! [`ModelResolver`] exposes a single [`resolve()`](ModelResolver::resolve)
//! method that checks storage cache, falls back to HuggingFace, and finally
//! attempts Forge conversion. All tier details, retry logic, and caching
//! are hidden behind this interface (RSLV-05, D-05).

use std::path::{Path, PathBuf};
use std::time::Duration;

use crate::error::ResolveError;
use crate::forge::{ForgeClient, StubForgeClient};
use crate::hf;
use crate::storage;

/// Model resolver implementing the 3-tier resolution chain.
///
/// Exposes a single `resolve()` method per the Ousterhout deep module
/// pattern (RSLV-05). Callers never see storage/HF/Forge internals.
///
/// Generic over [`ForgeClient`] so the binary can inject either
/// [`StubForgeClient`] (when `FORGE_URL` is unset) or
/// [`HttpForgeClient`](crate::forge::HttpForgeClient) (when configured).
/// Defaults to [`StubForgeClient`] for backward compatibility.
pub struct ModelResolver<F: ForgeClient = StubForgeClient> {
    cache_dir: PathBuf,
    operator: Option<opendal::Operator>,
    forge: F,
}

/// Validate that a model ID contains only allowed characters (T-03-01, T-03-04).
///
/// Allowed characters: alphanumeric, hyphen (`-`), underscore (`_`),
/// forward-slash (`/`), and period (`.`). Path segments equal to `..`
/// are rejected to prevent directory traversal.
///
/// Called as the first operation inside [`ModelResolver::resolve()`]
/// before any tier logic.
pub fn validate_model_id(model_id: &str) -> Result<(), ResolveError> {
    // Reject empty strings.
    if model_id.is_empty() {
        return Err(ResolveError::InvalidModelId {
            model_id: model_id.to_string(),
        });
    }

    // Reject characters outside the allowed set.
    let is_allowed = |c: char| c.is_alphanumeric() || c == '-' || c == '_' || c == '/' || c == '.';
    if !model_id.chars().all(is_allowed) {
        return Err(ResolveError::InvalidModelId {
            model_id: model_id.to_string(),
        });
    }

    // Reject ".." path segments (directory traversal).
    for segment in model_id.split('/') {
        if segment == ".." {
            return Err(ResolveError::InvalidModelId {
                model_id: model_id.to_string(),
            });
        }
    }

    Ok(())
}

impl ModelResolver<StubForgeClient> {
    /// Create a new model resolver with the stub Forge client.
    ///
    /// Used when `FORGE_URL` is not configured. The stub always returns
    /// [`ResolveError::ForgeUnavailable`] for the Forge tier.
    ///
    /// # Arguments
    ///
    /// * `operator` -- OpenDAL operator for storage tier (optional, D-05).
    pub async fn new_with_stub(
        operator: Option<opendal::Operator>,
    ) -> Result<Self, ResolveError> {
        Self::new_with_client(operator, StubForgeClient).await
    }
}

impl<F: ForgeClient> ModelResolver<F> {
    /// Create a new model resolver with a custom Forge client.
    ///
    /// # Arguments
    ///
    /// * `operator` -- OpenDAL operator for storage tier (optional, D-05).
    /// * `forge` -- Forge client implementation (e.g., `HttpForgeClient`
    ///   or `StubForgeClient`).
    pub async fn new_with_client(
        operator: Option<opendal::Operator>,
        forge: F,
    ) -> Result<Self, ResolveError> {
        // Determine cache_dir from HF_HOME or default (D-07).
        // hf-hub uses HF_HOME or ~/.cache/huggingface by default.
        let cache_dir = match std::env::var("HF_HOME") {
            Ok(home) => PathBuf::from(home),
            Err(_) => {
                let home = std::env::var("HOME")
                    .map(PathBuf::from)
                    .unwrap_or_else(|_| PathBuf::from("."));
                home.join(".cache").join("huggingface")
            }
        };

        Ok(Self {
            cache_dir,
            operator,
            forge,
        })
    }

    /// Resolve a model ID to a local directory containing ONNX files.
    ///
    /// Resolution order: storage cache -> HuggingFace -> Forge conversion.
    /// Returns the path to a directory containing `model.onnx` (or
    /// `onnx/model.onnx`), `tokenizer.json`, and `config.json`.
    ///
    /// Validates the model ID before any tier logic (T-03-01).
    pub async fn resolve(&self, model_id: &str) -> Result<PathBuf, ResolveError> {
        // T-03-01: validate model ID before any tier logic.
        validate_model_id(model_id)?;

        // Tier 1: Storage cache (D-05). RetryLayer on the Operator handles retries.
        if let Some(op) = &self.operator {
            let storage_result =
                storage::download_model(op, model_id, &self.cache_dir).await;

            match storage_result {
                Ok(Some(path)) => {
                    tracing::info!(model_id, tier = "storage", path = %path.display(), "model resolved from storage cache");
                    return Ok(path);
                }
                Ok(None) => {
                    tracing::info!(model_id, tier = "storage", "storage cache miss, falling through to HuggingFace");
                }
                Err(e) => {
                    tracing::warn!(model_id, tier = "storage", error = %e, "storage tier failed, falling through to HuggingFace");
                }
            }
        }

        // Tier 2: HuggingFace with retry (D-05).
        let model_id_owned = model_id.to_string();
        let hf_result = with_retry(3, Duration::from_millis(500), move || {
            let id = model_id_owned.clone();
            async move { hf::download_from_hf(&id).await }
        })
        .await;

        match hf_result {
            Ok(model_dir) => {
                tracing::info!(
                    model_id,
                    tier = "huggingface",
                    model_dir = %model_dir.display(),
                    "model resolved from HuggingFace"
                );

                // Spawn background storage cache-back after HF success (D-12).
                self.spawn_cache_back(model_id, &model_dir);

                return Ok(model_dir);
            }
            Err(ResolveError::NoOnnxExport { .. }) => {
                tracing::info!(
                    model_id,
                    tier = "huggingface",
                    "no ONNX export found, falling through to Forge"
                );
                // Fall through to Forge tier.
            }
            Err(e) => {
                return Err(e);
            }
        }

        // Tier 3: Forge conversion (D-09, D-10).
        tracing::info!(model_id, tier = "forge", "attempting Forge conversion");
        let forge_result = self.forge.convert(model_id).await;

        match forge_result {
            Ok(forge_resp) => {
                // Forge converted the model and uploaded it to storage.
                // Log conversion metadata and download the model from storage.
                tracing::info!(
                    model_id,
                    tier = "forge",
                    storage_paths = ?forge_resp.storage_paths,
                    architecture = %forge_resp.metadata.architecture,
                    conversion_duration_secs = forge_resp.metadata.conversion_duration_secs,
                    "Forge conversion succeeded, downloading from storage"
                );

                if let Some(op) = &self.operator {
                    let download_result =
                        storage::download_model(op, model_id, &self.cache_dir).await?;

                    if let Some(path) = download_result {
                        tracing::info!(
                            model_id,
                            tier = "forge",
                            path = %path.display(),
                            "model resolved via Forge conversion"
                        );
                        return Ok(path);
                    }
                }

                // If storage not configured, the Forge result can't be downloaded.
                Err(ResolveError::Storage(format!(
                    "Forge converted model '{model_id}' but storage is not configured to download it"
                )))
            }
            Err(e) => {
                // Forge unavailable or conversion failed.
                tracing::warn!(model_id, tier = "forge", error = %e, "Forge tier failed");
                Err(e)
            }
        }
    }

    /// Spawn a background task to upload model files to storage.
    ///
    /// Fire-and-forget: the upload runs in a separate tokio task.
    /// On failure, logs a warning but does not affect the serving pod.
    /// RetryLayer on the Operator handles transient failures.
    fn spawn_cache_back(&self, model_id: &str, local_dir: &Path) {
        let Some(op) = self.operator.clone() else {
            return;
        };
        let model_id = model_id.to_string();
        let local_dir = local_dir.to_path_buf();

        tokio::spawn(async move {
            match storage::upload_model(&op, &model_id, &local_dir).await {
                Ok(()) => {
                    tracing::info!(
                        model_id,
                        "successfully cached model to storage"
                    );
                }
                Err(e) => {
                    // Upload failure is non-fatal -- log warning and continue.
                    tracing::warn!(
                        model_id,
                        error = %e,
                        "failed to cache model to storage"
                    );
                }
            }
        });
    }
}

/// Trait for errors that can distinguish transient from permanent failures.
///
/// Used by [`with_retry`] to break early on non-transient errors
/// (auth failures, 404s) instead of wasting retry attempts.
pub(crate) trait Transient {
    /// Returns `true` if the error is transient and the operation
    /// should be retried, `false` if retrying would produce the same error.
    fn is_transient(&self) -> bool;
}

impl Transient for ResolveError {
    fn is_transient(&self) -> bool {
        ResolveError::is_transient(self)
    }
}

/// Generic async retry with exponential backoff (D-05).
///
/// Retries the operation up to `max_attempts` times with exponential
/// backoff starting from `base_delay`. Each retry is logged at warn
/// level with attempt number, max_attempts, delay, and error message.
///
/// Breaks early on non-transient errors (auth failures, 404s) to
/// avoid wasting retry attempts on errors that will fail identically.
pub(crate) async fn with_retry<F, Fut, T, E>(
    max_attempts: u32,
    base_delay: Duration,
    mut operation: F,
) -> Result<T, E>
where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = Result<T, E>>,
    E: std::fmt::Display + Transient,
{
    let mut attempt = 0;
    loop {
        attempt += 1;
        match operation().await {
            Ok(result) => return Ok(result),
            Err(e) if attempt >= max_attempts || !e.is_transient() => return Err(e),
            Err(e) => {
                let delay = base_delay * 2u32.pow(attempt - 1);
                tracing::warn!(
                    attempt,
                    max_attempts,
                    delay_ms = delay.as_millis() as u64,
                    error = %e,
                    "retrying after transient error"
                );
                tokio::time::sleep(delay).await;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- validate_model_id tests ---

    #[test]
    fn validate_model_id_accepts_org_model() {
        assert!(validate_model_id("Xenova/distilbert-base").is_ok());
    }

    #[test]
    fn validate_model_id_accepts_model_with_dots_and_hyphens() {
        assert!(validate_model_id("sentence-transformers/all-MiniLM-L6-v2").is_ok());
    }

    #[test]
    fn validate_model_id_accepts_simple_model() {
        assert!(validate_model_id("bert-base-uncased").is_ok());
    }

    #[test]
    fn validate_model_id_accepts_underscores() {
        assert!(validate_model_id("org/my_model_v2").is_ok());
    }

    #[test]
    fn validate_model_id_rejects_path_traversal() {
        let result = validate_model_id("../../../etc/passwd");
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            matches!(err, ResolveError::InvalidModelId { .. }),
            "expected InvalidModelId, got: {err:?}"
        );
    }

    #[test]
    fn validate_model_id_rejects_embedded_traversal() {
        let result = validate_model_id("model/../../secret");
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            ResolveError::InvalidModelId { .. }
        ));
    }

    #[test]
    fn validate_model_id_rejects_shell_metacharacters() {
        let result = validate_model_id("model;rm -rf /");
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            ResolveError::InvalidModelId { .. }
        ));
    }

    #[test]
    fn validate_model_id_rejects_empty_string() {
        let result = validate_model_id("");
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            ResolveError::InvalidModelId { .. }
        ));
    }

    #[test]
    fn validate_model_id_error_contains_model_id() {
        let result = validate_model_id("bad;model");
        let err = result.unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("bad;model"),
            "error message should contain the model_id: {msg}"
        );
    }

    // --- is_transient tests ---

    #[test]
    fn is_transient_hf_auth_error() {
        let err = ResolveError::HuggingFace("Authentication failed: 401".into());
        assert!(!err.is_transient(), "auth errors should not be transient");
    }

    #[test]
    fn is_transient_hf_403_error() {
        let err = ResolveError::HuggingFace("Access denied: 403 Forbidden".into());
        assert!(!err.is_transient(), "403 errors should not be transient");
    }

    #[test]
    fn is_transient_hf_404_error() {
        let err = ResolveError::HuggingFace("Repository not found: 404".into());
        assert!(!err.is_transient(), "404 errors should not be transient");
    }

    #[test]
    fn is_transient_hf_network_error() {
        let err = ResolveError::HuggingFace("connection timed out".into());
        assert!(err.is_transient(), "network errors should be transient");
    }

    #[test]
    fn is_transient_no_onnx_export() {
        let err = ResolveError::NoOnnxExport {
            model_id: "test/model".into(),
        };
        assert!(!err.is_transient(), "no ONNX export should not be transient");
    }

    #[test]
    fn is_transient_storage_not_found() {
        let err = ResolveError::Storage("object not found".into());
        assert!(!err.is_transient(), "storage not found should not be transient");
    }

    #[test]
    fn is_transient_storage_timeout() {
        let err = ResolveError::Storage("connection timed out".into());
        assert!(err.is_transient(), "storage timeout should be transient");
    }

    // --- with_retry tests ---

    /// Test error type that implements Transient (always transient).
    #[derive(Debug)]
    struct TransientError(String);
    impl std::fmt::Display for TransientError {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "{}", self.0)
        }
    }
    impl Transient for TransientError {
        fn is_transient(&self) -> bool {
            true
        }
    }

    /// Test error type that is never transient.
    #[derive(Debug)]
    struct PermanentError(String);
    impl std::fmt::Display for PermanentError {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "{}", self.0)
        }
    }
    impl Transient for PermanentError {
        fn is_transient(&self) -> bool {
            false
        }
    }

    #[tokio::test]
    async fn retry_retries_specified_number_of_times() {
        use std::sync::atomic::{AtomicU32, Ordering};
        use std::sync::Arc;

        let counter = Arc::new(AtomicU32::new(0));
        let counter_clone = counter.clone();

        let result: Result<(), TransientError> = with_retry(
            3,
            Duration::from_millis(1),
            move || {
                let c = counter_clone.clone();
                async move {
                    let attempt = c.fetch_add(1, Ordering::SeqCst) + 1;
                    if attempt < 3 {
                        Err(TransientError(format!("transient error on attempt {attempt}")))
                    } else {
                        Ok(())
                    }
                }
            },
        )
        .await;

        assert!(result.is_ok());
        assert_eq!(counter.load(Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn retry_returns_last_error_on_exhaustion() {
        use std::sync::atomic::{AtomicU32, Ordering};
        use std::sync::Arc;

        let counter = Arc::new(AtomicU32::new(0));
        let counter_clone = counter.clone();

        let result: Result<(), TransientError> = with_retry(
            3,
            Duration::from_millis(1),
            move || {
                let c = counter_clone.clone();
                async move {
                    let attempt = c.fetch_add(1, Ordering::SeqCst) + 1;
                    Err(TransientError(format!("persistent error on attempt {attempt}")))
                }
            },
        )
        .await;

        assert!(result.is_err());
        assert_eq!(counter.load(Ordering::SeqCst), 3);
        let err = result.unwrap_err();
        assert!(
            err.0.contains("attempt 3"),
            "should return last error: {}", err.0
        );
    }

    #[tokio::test]
    async fn retry_succeeds_on_first_attempt() {
        use std::sync::atomic::{AtomicU32, Ordering};
        use std::sync::Arc;

        let counter = Arc::new(AtomicU32::new(0));
        let counter_clone = counter.clone();

        let result: Result<&str, TransientError> = with_retry(
            3,
            Duration::from_millis(1),
            move || {
                let c = counter_clone.clone();
                async move {
                    c.fetch_add(1, Ordering::SeqCst);
                    Ok("success")
                }
            },
        )
        .await;

        assert!(result.is_ok());
        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn retry_breaks_early_on_non_transient_error() {
        use std::sync::atomic::{AtomicU32, Ordering};
        use std::sync::Arc;

        let counter = Arc::new(AtomicU32::new(0));
        let counter_clone = counter.clone();

        let result: Result<(), PermanentError> = with_retry(
            5,
            Duration::from_millis(1),
            move || {
                let c = counter_clone.clone();
                async move {
                    c.fetch_add(1, Ordering::SeqCst);
                    Err(PermanentError("auth failure".into()))
                }
            },
        )
        .await;

        assert!(result.is_err());
        // Should have tried only once (non-transient error breaks immediately).
        assert_eq!(
            counter.load(Ordering::SeqCst),
            1,
            "non-transient error should not retry"
        );
    }

    // --- Error message tests ---

    #[test]
    fn no_onnx_export_error_contains_model_id() {
        let err = ResolveError::NoOnnxExport {
            model_id: "test-org/test-model".to_string(),
        };
        let msg = err.to_string();
        assert!(
            msg.contains("test-org/test-model"),
            "NoOnnxExport Display should contain model_id: {msg}"
        );
    }

    #[tokio::test]
    async fn stub_forge_returns_unavailable() {
        let err = ResolveError::ForgeUnavailable {
            model_id: "test/model".to_string(),
        };
        assert!(matches!(err, ResolveError::ForgeUnavailable { .. }));
        let msg = err.to_string();
        assert!(
            msg.contains("test/model"),
            "ForgeUnavailable should contain model_id: {msg}"
        );
    }

    // --- ModelResolver construction tests ---

    #[tokio::test]
    async fn resolver_new_without_operator() {
        let resolver = ModelResolver::new_with_stub(None).await.unwrap();
        assert!(resolver.operator.is_none());
    }

    #[tokio::test]
    async fn resolver_new_with_operator() {
        let op = opendal::Operator::new(opendal::services::Memory::default()).unwrap();
        let resolver = ModelResolver::new_with_stub(Some(op)).await.unwrap();
        assert!(resolver.operator.is_some());
    }

    // --- 3-tier resolve chain tests ---

    #[tokio::test]
    async fn resolve_rejects_invalid_model_id() {
        let resolver = ModelResolver::new_with_stub(None).await.unwrap();
        let result = resolver.resolve("../bad").await;
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), ResolveError::InvalidModelId { .. }));
    }

    #[tokio::test]
    async fn forge_tier_returns_unavailable_when_no_forge_url() {
        // StubForgeClient always returns ForgeUnavailable.
        let forge = StubForgeClient;
        let result = forge.convert("some/model").await;
        assert!(matches!(
            result.unwrap_err(),
            ResolveError::ForgeUnavailable { ref model_id } if model_id == "some/model"
        ));
    }
}
