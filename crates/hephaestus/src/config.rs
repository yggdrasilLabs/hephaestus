//! Typed configuration loaded from environment variables via `envy`.
//!
//! All configuration comes from environment variables (D-11). This is
//! a k8s-only service -- no CLI parser, no config files.

use std::collections::HashMap;
use std::path::{Component, PathBuf};

use anyhow::{Context, bail};
use hephaestus_core::ExecutionProvider;
use serde::Deserialize;

/// Accepted `STORAGE_TYPE` values (T-06-05 allowlist).
///
/// Every entry other than `none` must have a matching opendal `services-*`
/// cargo feature enabled in the workspace manifest. The test
/// `storage_operator_builds_for_every_allowed_storage_type` enforces this,
/// so adding a backend here without its feature fails CI.
const ALLOWED_STORAGE_TYPES: &[&str] = &["s3", "fs", "gcs", "none"];

/// Runtime configuration deserialized from environment variables.
///
/// # Required
///
/// - `MODEL_ID` -- identifier for the model (e.g., `distilbert-base-uncased-finetuned-sst-2-english`).
///   The binary crashes with a clear error if this is missing (D-13).
///
/// # Optional
///
/// - `MODEL_PATH` -- absolute path to the local directory containing model files.
/// - `EXECUTION_PROVIDER` -- ONNX execution provider (default: `"cpu"`).
/// - `LOG_LEVEL` -- log verbosity (default: `"info"`).
/// - `WARMUP_INPUT` -- custom text for the warmup inference pass.
#[derive(Deserialize, Debug)]
pub struct Config {
    /// Model identifier (required).
    pub model_id: String,

    /// Local directory containing model files (optional).
    #[serde(default)]
    pub model_path: Option<String>,

    /// ONNX execution provider (default: `"cpu"`).
    #[serde(default = "default_ep")]
    pub execution_provider: String,

    /// Log level (default: `"info"`).
    #[serde(default = "default_log_level")]
    pub log_level: String,

    /// Custom warmup inference text (optional).
    #[serde(default)]
    pub warmup_input: Option<String>,

    /// HTTP server listen port (default: 8080, env `PORT`).
    #[serde(default = "default_port")]
    pub port: u16,

    /// Per-request inference timeout in seconds (default: 30, env `REQUEST_TIMEOUT_SECS`).
    #[serde(default = "default_request_timeout_secs")]
    pub request_timeout_secs: u64,

    /// Graceful shutdown drain timeout in seconds (default: 30, env `SHUTDOWN_TIMEOUT_SECS`).
    #[serde(default = "default_shutdown_timeout_secs")]
    pub shutdown_timeout_secs: u64,

    /// OpenTelemetry OTLP exporter endpoint (optional, env `OTEL_EXPORTER_OTLP_ENDPOINT`).
    /// When set, OTel tracing is activated. When absent, only structured JSON logs are emitted.
    ///
    /// Used by `hephaestus_api::telemetry::init` to conditionally activate OTel tracing.
    #[serde(default)]
    pub otel_exporter_otlp_endpoint: Option<String>,

    /// Storage backend type (env `STORAGE_TYPE`).
    /// Accepted values: `s3`, `fs`, `gcs`, `none`.
    /// Defaults to `"s3"` when unset (D-02).
    /// `none` disables the storage tier entirely (D-05).
    #[serde(default = "default_storage_type")]
    pub storage_type: String,

    /// Storage bucket name (env `STORAGE_BUCKET`).
    /// Required for S3 and GCS backends.
    #[serde(default)]
    pub storage_bucket: Option<String>,

    /// Universal path prefix across all backends (env `STORAGE_PREFIX`, D-04).
    /// On S3 becomes a key prefix, on filesystem becomes a subdirectory.
    #[serde(default)]
    pub storage_prefix: Option<String>,

    /// Root directory for filesystem backend (env `STORAGE_ROOT`, D-15).
    /// Required when `STORAGE_TYPE=fs` (D-17).
    #[serde(default)]
    pub storage_root: Option<String>,

    /// AWS region for the S3 backend (env `STORAGE_REGION`).
    /// Ignored by other backends.
    #[serde(default)]
    pub storage_region: Option<String>,

    /// Path to a GCS service-account JSON key file (env `STORAGE_CREDENTIAL_PATH`).
    /// Only applied when `STORAGE_TYPE=gcs`. When unset, OpenDAL falls back to
    /// `GOOGLE_APPLICATION_CREDENTIALS` and then the GKE metadata server
    /// (Workload Identity).
    #[serde(default)]
    pub storage_credential_path: Option<String>,

    /// Forge conversion service URL (optional, env `FORGE_URL`, D-09).
    /// When set, enables the Forge conversion tier for models without ONNX exports.
    #[serde(default)]
    pub forge_url: Option<String>,

    /// Forge conversion service timeout in seconds (default: 600, env `FORGE_TIMEOUT_SECS`).
    #[serde(default = "default_forge_timeout_secs")]
    pub forge_timeout_secs: u64,

    /// Optional model profile override (env `MODEL_PROFILE`, D-02).
    /// When set, takes precedence over auto-detection from config.json.
    /// Accepted values: `classifier`, `embeddings`, `seq2seq`, `token_classifier`.
    #[serde(default)]
    pub model_profile: Option<String>,

    /// Feature extractor for ASR preprocessing (env `FEATURE_EXTRACTOR`, D-09).
    /// `"mel"` computes mel spectrograms for Whisper-style models.
    /// `"none"` passes raw waveform for CTC models like wav2vec2.
    #[serde(default = "default_feature_extractor")]
    pub feature_extractor: String,

    /// Chunking strategy for streaming ASR (env `CHUNKING_STRATEGY`, D-11).
    /// `"windowed"` uses fixed-size windows with overlap for encoder-decoder models.
    /// `"streaming"` passes through for native streaming CTC models.
    #[serde(default = "default_chunking_strategy")]
    pub chunking_strategy: String,

    /// Window duration in seconds for windowed chunking (env `WINDOW_SIZE_SECS`, D-10).
    #[serde(default = "default_window_size_secs")]
    pub window_size_secs: f32,

    /// Overlap duration in seconds between adjacent windows (env `OVERLAP_SECS`, D-10).
    #[serde(default = "default_overlap_secs")]
    pub overlap_secs: f32,

    /// Enable dynamic request batching (env `BATCH_ENABLED`, D-07, BTCH-02).
    /// When false (default), requests flow through the direct path with zero overhead.
    #[serde(default)]
    pub batch_enabled: bool,

    /// Maximum number of requests to collect in a single batch (env `BATCH_MAX_SIZE`, D-09, BTCH-03).
    /// Defaults to 8. Values > 64 or < 1 are rejected at startup.
    #[serde(default = "default_batch_max_size")]
    pub batch_max_size: u32,

    /// Maximum time in milliseconds to wait for a full batch before executing (env `BATCH_MAX_WAIT_MS`, D-09, BTCH-03).
    /// Defaults to 50ms.
    #[serde(default = "default_batch_max_wait_ms")]
    pub batch_max_wait_ms: u64,
}

fn default_storage_type() -> String {
    "s3".to_string()
}

fn default_ep() -> String {
    "cpu".to_string()
}

fn default_log_level() -> String {
    "info".to_string()
}

fn default_port() -> u16 {
    8080
}

fn default_request_timeout_secs() -> u64 {
    30
}

fn default_shutdown_timeout_secs() -> u64 {
    30
}

fn default_batch_max_size() -> u32 {
    8
}

fn default_batch_max_wait_ms() -> u64 {
    50
}

fn default_forge_timeout_secs() -> u64 {
    600
}

fn default_feature_extractor() -> String {
    "none".to_string()
}

fn default_chunking_strategy() -> String {
    "windowed".to_string()
}

fn default_window_size_secs() -> f32 {
    30.0
}

fn default_overlap_secs() -> f32 {
    1.0
}

impl Config {
    /// Load configuration from environment variables.
    ///
    /// # Errors
    ///
    /// Returns an error if `MODEL_ID` is not set or if any env var
    /// fails to deserialize into the expected type.
    pub fn from_env() -> Result<Self, anyhow::Error> {
        envy::from_env::<Self>().context("failed to load config from environment (MODEL_ID is required)")
    }

    /// Resolve and validate the model directory path.
    ///
    /// Validates that the path is absolute and contains no parent-directory
    /// traversal components (`..`) to mitigate T-01-01 path tampering.
    ///
    /// # Errors
    ///
    /// Returns an error if `MODEL_PATH` is not set, the path is relative,
    /// the path contains `..` components, or the path does not exist.
    pub fn model_dir(&self) -> Result<PathBuf, anyhow::Error> {
        let raw = self
            .model_path
            .as_deref()
            .context("MODEL_PATH is not set and no model was resolved automatically")?;

        let path = PathBuf::from(raw);

        // T-01-01: reject relative paths.
        if !path.is_absolute() {
            bail!("MODEL_PATH must be an absolute path, got: {raw}");
        }

        // T-01-01: reject parent-directory traversal.
        for component in path.components() {
            if matches!(component, Component::ParentDir) {
                bail!("MODEL_PATH must not contain '..' components, got: {raw}");
            }
        }

        // Validate that the directory exists.
        if !path.is_dir() {
            bail!("MODEL_PATH does not exist or is not a directory: {raw}");
        }

        Ok(path)
    }

    /// Parse the `execution_provider` string into a typed enum.
    ///
    /// # Errors
    ///
    /// Returns an error if the value is not one of the accepted
    /// execution provider names (`cpu`, `cuda`, `tensorrt`, `coreml`).
    pub fn parsed_execution_provider(&self) -> Result<ExecutionProvider, anyhow::Error> {
        self.execution_provider
            .parse::<ExecutionProvider>()
            .context("invalid EXECUTION_PROVIDER value")
    }

    /// Validate configuration values before resource allocation.
    ///
    /// Checks execution provider, storage type, and batch parameters.
    /// When `batch_enabled` is true, validates that `batch_max_size`
    /// is within [1, 64] and that `batch_max_wait_ms` does not exceed
    /// `request_timeout_secs * 1000`. When batching is disabled,
    /// batch-related validation is skipped entirely.
    ///
    /// # Errors
    ///
    /// Returns an error if any configuration value is out of range.
    pub fn validate(&self) -> Result<(), anyhow::Error> {
        // Validate execution provider early (T-EP-01).
        self.parsed_execution_provider()?;
        // T-06-05: validate storage_type against explicit allowlist.
        if !ALLOWED_STORAGE_TYPES.contains(&self.storage_type.as_str()) {
            bail!(
                "invalid STORAGE_TYPE '{}' -- accepted values: {}",
                self.storage_type,
                ALLOWED_STORAGE_TYPES.join(", "),
            );
        }

        // D-17: STORAGE_ROOT is required when STORAGE_TYPE=fs.
        if self.storage_type == "fs" && self.storage_root.is_none() {
            bail!("STORAGE_ROOT is required when STORAGE_TYPE=fs");
        }

        // T-11-10: validate ASR config fields against explicit allowlists.
        const ALLOWED_FEATURE_EXTRACTORS: &[&str] = &["mel", "none"];
        if !ALLOWED_FEATURE_EXTRACTORS.contains(&self.feature_extractor.as_str()) {
            bail!(
                "invalid FEATURE_EXTRACTOR '{}' -- accepted values: mel, none",
                self.feature_extractor,
            );
        }

        const ALLOWED_CHUNKING_STRATEGIES: &[&str] = &["windowed", "streaming"];
        if !ALLOWED_CHUNKING_STRATEGIES.contains(&self.chunking_strategy.as_str()) {
            bail!(
                "invalid CHUNKING_STRATEGY '{}' -- accepted values: windowed, streaming",
                self.chunking_strategy,
            );
        }

        if self.window_size_secs <= 0.0 || self.window_size_secs > 60.0 {
            bail!(
                "WINDOW_SIZE_SECS must be positive and at most 60.0 (got {})",
                self.window_size_secs,
            );
        }

        if self.overlap_secs < 0.0 || self.overlap_secs >= self.window_size_secs {
            bail!(
                "OVERLAP_SECS must be non-negative and less than WINDOW_SIZE_SECS ({}) (got {})",
                self.window_size_secs,
                self.overlap_secs,
            );
        }

        if self.batch_enabled {
            if self.batch_max_size < 1 || self.batch_max_size > 64 {
                bail!(
                    "batch_max_size must be between 1 and 64 (got {})",
                    self.batch_max_size,
                );
            }
            let timeout_ms = self.request_timeout_secs * 1000;
            if self.batch_max_wait_ms >= timeout_ms {
                bail!(
                    "batch_max_wait_ms ({}) must be less than request_timeout_secs * 1000 ({})",
                    self.batch_max_wait_ms,
                    timeout_ms,
                );
            }
        }
        Ok(())
    }

    /// Build the OpenDAL storage operator for the configured backend (D-01, D-02).
    ///
    /// Returns `Ok(None)` when `STORAGE_TYPE=none`, which disables the storage
    /// tier entirely (D-05). Otherwise maps the storage env config onto OpenDAL
    /// config keys and wraps the operator in a retry layer.
    ///
    /// # Errors
    ///
    /// Returns an error if `STORAGE_TYPE=fs` without `STORAGE_ROOT`, or if
    /// OpenDAL rejects the configuration (e.g. a missing bucket, S3 without a
    /// region, or a backend whose cargo feature is not compiled in).
    pub fn storage_operator(&self) -> Result<Option<opendal::Operator>, anyhow::Error> {
        if self.storage_type == "none" {
            return Ok(None);
        }
        let op = opendal::Operator::via_iter(self.storage_type.as_str(), self.storage_options()?)
            .with_context(|| format!("failed to build {} storage operator", self.storage_type))?
            .layer(opendal::layers::RetryLayer::new().with_max_times(3));
        Ok(Some(op))
    }

    /// Map storage env config onto OpenDAL config keys for the configured backend.
    ///
    /// `bucket` is passed whenever set. `region` is passed only to `s3` and
    /// `credential_path` only to `gcs`, so backend-specific settings never
    /// reach an unintended backend. `root` comes from `STORAGE_ROOT` (joined
    /// with `STORAGE_PREFIX`) for `fs`, and from `/{STORAGE_PREFIX}` for cloud
    /// backends (D-04).
    ///
    /// # Errors
    ///
    /// Returns an error if `STORAGE_TYPE=fs` and `STORAGE_ROOT` is not set (D-17).
    fn storage_options(&self) -> Result<HashMap<String, String>, anyhow::Error> {
        let mut options = HashMap::new();
        if let Some(ref bucket) = self.storage_bucket {
            options.insert("bucket".to_string(), bucket.clone());
        }
        if self.storage_type == "s3"
            && let Some(ref region) = self.storage_region
        {
            options.insert("region".to_string(), region.clone());
        }
        if self.storage_type == "gcs"
            && let Some(ref credential_path) = self.storage_credential_path
        {
            options.insert("credential_path".to_string(), credential_path.clone());
        }
        // D-04: STORAGE_PREFIX/STORAGE_ROOT -> OpenDAL "root" config.
        if self.storage_type == "fs" {
            let root = self
                .storage_root
                .as_deref()
                .context("storage_root is required when storage_type is fs")?;
            let fs_root = match self.storage_prefix.as_deref() {
                Some(prefix) => format!("{root}/{prefix}"),
                None => root.to_string(),
            };
            options.insert("root".to_string(), fs_root);
        } else if let Some(ref prefix) = self.storage_prefix {
            options.insert("root".to_string(), format!("/{prefix}"));
        }
        Ok(options)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: construct a `Config` with the given model_path and sensible
    /// defaults for all other fields. Avoids going through envy so tests
    /// are deterministic and don't mutate process-wide env vars.
    fn config_with_model_path(model_path: Option<&str>) -> Config {
        Config {
            model_id: "test-model".to_string(),
            model_path: model_path.map(String::from),
            execution_provider: "cpu".to_string(),
            log_level: "info".to_string(),
            warmup_input: None,
            port: 8080,
            request_timeout_secs: 30,
            shutdown_timeout_secs: 30,
            otel_exporter_otlp_endpoint: None,
            storage_type: "s3".to_string(),
            storage_bucket: None,
            storage_prefix: None,
            storage_root: None,
            storage_region: None,
            storage_credential_path: None,
            forge_url: None,
            forge_timeout_secs: 600,
            model_profile: None,
            feature_extractor: "none".to_string(),
            chunking_strategy: "windowed".to_string(),
            window_size_secs: 30.0,
            overlap_secs: 1.0,
            batch_enabled: false,
            batch_max_size: 8,
            batch_max_wait_ms: 50,
        }
    }

    #[test]
    fn from_env_with_defaults_has_correct_defaults() {
        // Arrange -- set only MODEL_ID; rely on serde defaults for the rest.
        // Safety: env var mutation is process-global but acceptable in unit
        // tests that are not run in parallel with other env-dependent tests.
        unsafe { std::env::set_var("MODEL_ID", "test-model") };

        // Act
        let config = Config::from_env().expect("should load config with MODEL_ID set");

        // Assert
        assert_eq!(config.model_id, "test-model");
        assert_eq!(config.execution_provider, "cpu");
        assert_eq!(config.log_level, "info");
        assert!(config.model_path.is_none());
        assert!(config.warmup_input.is_none());

        // Cleanup
        unsafe { std::env::remove_var("MODEL_ID") };
    }

    #[test]
    fn model_dir_returns_error_when_model_path_is_none() {
        // Arrange
        let config = config_with_model_path(None);

        // Act
        let result = config.model_dir();

        // Assert
        assert!(result.is_err());
        let msg = format!("{}", result.unwrap_err());
        assert!(msg.contains("MODEL_PATH"), "error should mention MODEL_PATH: {msg}");
    }

    #[test]
    fn model_dir_rejects_relative_path() {
        // Arrange
        let config = config_with_model_path(Some("relative/path"));

        // Act
        let result = config.model_dir();

        // Assert
        assert!(result.is_err());
        let msg = format!("{}", result.unwrap_err());
        assert!(msg.contains("absolute"), "error should mention 'absolute': {msg}");
    }

    #[test]
    fn model_dir_rejects_parent_traversal() {
        // Arrange
        let config = config_with_model_path(Some("/tmp/models/../secret"));

        // Act
        let result = config.model_dir();

        // Assert
        assert!(result.is_err());
        let msg = format!("{}", result.unwrap_err());
        assert!(msg.contains(".."), "error should mention '..': {msg}");
    }

    #[test]
    fn model_dir_accepts_valid_absolute_path() {
        // Arrange
        let tmpdir = tempfile::tempdir().expect("should create temp dir");
        let path_str = tmpdir.path().to_str().expect("path should be valid UTF-8");
        let config = config_with_model_path(Some(path_str));

        // Act
        let result = config.model_dir();

        // Assert
        let dir = result.expect("should accept valid absolute path");
        assert_eq!(dir, tmpdir.path());
    }

    #[test]
    fn model_dir_rejects_nonexistent_path() {
        // Arrange
        let config = config_with_model_path(Some("/nonexistent/path/to/model"));

        // Act
        let result = config.model_dir();

        // Assert
        assert!(result.is_err());
        let msg = format!("{}", result.unwrap_err());
        assert!(
            msg.contains("does not exist"),
            "error should mention 'does not exist': {msg}"
        );
    }

    #[test]
    fn test_batch_config_defaults() {
        // Arrange
        let config = config_with_model_path(None);

        // Assert -- batch fields should have their defaults
        assert!(!config.batch_enabled, "batch_enabled should default to false");
        assert_eq!(config.batch_max_size, 8, "batch_max_size should default to 8");
        assert_eq!(config.batch_max_wait_ms, 50, "batch_max_wait_ms should default to 50");
    }

    #[test]
    fn test_batch_config_custom_values() {
        // Arrange
        let mut config = config_with_model_path(None);
        config.batch_enabled = true;
        config.batch_max_size = 16;
        config.batch_max_wait_ms = 100;

        // Assert
        assert!(config.batch_enabled);
        assert_eq!(config.batch_max_size, 16);
        assert_eq!(config.batch_max_wait_ms, 100);
    }

    #[test]
    fn test_validate_rejects_zero_batch_size() {
        // Arrange
        let mut config = config_with_model_path(None);
        config.batch_enabled = true;
        config.batch_max_size = 0;

        // Act
        let result = config.validate();

        // Assert
        assert!(result.is_err(), "batch_max_size=0 should be rejected");
        let msg = format!("{}", result.unwrap_err());
        assert!(
            msg.contains("between 1 and 64"),
            "error should mention valid range: {msg}"
        );
    }

    #[test]
    fn test_validate_rejects_large_batch_size() {
        // Arrange
        let mut config = config_with_model_path(None);
        config.batch_enabled = true;
        config.batch_max_size = 65;

        // Act
        let result = config.validate();

        // Assert
        assert!(result.is_err(), "batch_max_size=65 should be rejected");
        let msg = format!("{}", result.unwrap_err());
        assert!(
            msg.contains("between 1 and 64"),
            "error should mention valid range: {msg}"
        );
    }

    #[test]
    fn test_validate_accepts_valid_batch_size() {
        // Arrange
        let mut config = config_with_model_path(None);
        config.batch_enabled = true;
        config.batch_max_size = 32;

        // Act
        let result = config.validate();

        // Assert
        assert!(result.is_ok(), "batch_max_size=32 should be accepted");
    }

    #[test]
    fn test_validate_skips_when_batching_disabled() {
        // Arrange -- invalid batch_max_size but batching disabled
        let mut config = config_with_model_path(None);
        config.batch_enabled = false;
        config.batch_max_size = 0;

        // Act
        let result = config.validate();

        // Assert -- should pass because batch validation is skipped
        assert!(result.is_ok(), "validation should skip batch checks when batching disabled");
    }

    #[test]
    fn test_forge_timeout_default() {
        let config = config_with_model_path(None);
        assert_eq!(
            config.forge_timeout_secs, 600,
            "forge_timeout_secs should default to 600"
        );
    }

    #[test]
    fn test_validate_rejects_wait_exceeding_timeout() {
        // Arrange -- batch_max_wait_ms > request_timeout_secs * 1000
        let mut config = config_with_model_path(None);
        config.batch_enabled = true;
        config.batch_max_size = 8;
        config.request_timeout_secs = 5;
        config.batch_max_wait_ms = 6000; // 6s > 5s timeout

        // Act
        let result = config.validate();

        // Assert
        assert!(result.is_err(), "batch_max_wait_ms exceeding timeout should be rejected");
    }

    // --- Storage config tests ---

    #[test]
    fn test_storage_type_defaults_to_s3() {
        let config = config_with_model_path(None);
        assert_eq!(config.storage_type, "s3", "storage_type should default to s3");
    }

    #[test]
    fn test_validate_rejects_invalid_storage_type() {
        let mut config = config_with_model_path(None);
        config.storage_type = "invalid".to_string();

        let result = config.validate();

        assert!(result.is_err(), "invalid storage_type should be rejected");
        let msg = format!("{}", result.unwrap_err());
        assert!(
            msg.contains("accepted values"),
            "error should list accepted values: {msg}"
        );
    }

    #[test]
    fn test_validate_accepts_all_storage_types() {
        for st in &["s3", "fs", "gcs", "none"] {
            let mut config = config_with_model_path(None);
            config.storage_type = st.to_string();
            // fs requires storage_root (D-17)
            if *st == "fs" {
                config.storage_root = Some("/data/models".to_string());
            }
            let result = config.validate();
            assert!(result.is_ok(), "storage_type={st} should be accepted, got: {result:?}");
        }
    }

    #[test]
    fn test_validate_rejects_azblob() {
        // Arrange
        let mut config = config_with_model_path(None);
        config.storage_type = "azblob".to_string();

        // Act
        let result = config.validate();

        // Assert
        assert!(result.is_err(), "azblob should no longer be an accepted storage_type");
        let msg = format!("{}", result.unwrap_err());
        assert!(
            msg.contains("accepted values"),
            "error should list accepted values: {msg}"
        );
    }

    #[test]
    fn storage_operator_builds_for_every_allowed_storage_type() {
        // Arrange -- the tempdir guard must outlive the loop so the fs root exists.
        let tmpdir = tempfile::tempdir().expect("should create temp dir");
        let root = tmpdir.path().to_str().expect("path should be valid UTF-8");

        for st in ALLOWED_STORAGE_TYPES {
            let mut config = config_with_model_path(None);
            config.storage_type = (*st).to_string();
            config.storage_bucket = Some("test-bucket".to_string());
            config.storage_region = Some("us-east-1".to_string());
            config.storage_root = Some(root.to_string());

            // Act
            let result = config.storage_operator();

            // Assert -- every non-none entry needs its opendal services-* feature.
            let operator = result.unwrap_or_else(|e| {
                panic!("storage_type={st} should build an operator, got: {e:#}")
            });
            if *st == "none" {
                assert!(operator.is_none(), "storage_type=none should yield no operator");
            } else {
                assert!(operator.is_some(), "storage_type={st} should yield an operator");
            }
        }
    }

    #[test]
    fn storage_options_passes_region_only_to_s3() {
        for st in &["s3", "gcs", "fs"] {
            // Arrange
            let mut config = config_with_model_path(None);
            config.storage_type = (*st).to_string();
            config.storage_bucket = Some("test-bucket".to_string());
            config.storage_region = Some("us-east-1".to_string());
            config.storage_root = Some("/data/models".to_string());

            // Act
            let options = config
                .storage_options()
                .unwrap_or_else(|e| panic!("storage_options for {st} should succeed: {e:#}"));

            // Assert
            if *st == "s3" {
                assert_eq!(
                    options.get("region").map(String::as_str),
                    Some("us-east-1"),
                    "s3 options should carry the region"
                );
            } else {
                assert!(
                    !options.contains_key("region"),
                    "{st} options must not carry a region: {options:?}"
                );
            }
        }
    }

    #[test]
    fn storage_options_passes_credential_path_only_to_gcs() {
        for st in &["s3", "gcs", "fs"] {
            // Arrange
            let mut config = config_with_model_path(None);
            config.storage_type = (*st).to_string();
            config.storage_bucket = Some("test-bucket".to_string());
            config.storage_root = Some("/data/models".to_string());
            config.storage_credential_path = Some("/var/secrets/gcs/key.json".to_string());

            // Act
            let options = config
                .storage_options()
                .unwrap_or_else(|e| panic!("storage_options for {st} should succeed: {e:#}"));

            // Assert
            if *st == "gcs" {
                assert_eq!(
                    options.get("credential_path").map(String::as_str),
                    Some("/var/secrets/gcs/key.json"),
                    "gcs options should carry the credential path"
                );
            } else {
                assert!(
                    !options.contains_key("credential_path"),
                    "{st} options must not carry a credential_path: {options:?}"
                );
            }
        }
    }

    #[test]
    fn test_parsed_execution_provider_accepts_all_valid() {
        for (input, expected) in &[
            ("cpu", hephaestus_core::ExecutionProvider::Cpu),
            ("cuda", hephaestus_core::ExecutionProvider::Cuda),
            ("tensorrt", hephaestus_core::ExecutionProvider::TensorRt),
            ("coreml", hephaestus_core::ExecutionProvider::CoreMl),
        ] {
            let mut config = config_with_model_path(None);
            config.execution_provider = input.to_string();
            let ep = config
                .parsed_execution_provider()
                .unwrap_or_else(|_| panic!("should parse '{input}'"));
            assert_eq!(ep, *expected, "mismatch for input '{input}'");
        }
    }

    #[test]
    fn test_parsed_execution_provider_rejects_invalid() {
        let mut config = config_with_model_path(None);
        config.execution_provider = "vulkan".to_string();
        let result = config.parsed_execution_provider();
        assert!(result.is_err(), "'vulkan' should be rejected");
    }

    #[test]
    fn test_validate_rejects_invalid_execution_provider() {
        let mut config = config_with_model_path(None);
        config.execution_provider = "bogus".to_string();
        let result = config.validate();
        assert!(result.is_err(), "invalid EP should fail validation");
    }

    #[test]
    fn test_validate_rejects_fs_without_root() {
        let mut config = config_with_model_path(None);
        config.storage_type = "fs".to_string();
        config.storage_root = None;

        let result = config.validate();

        assert!(result.is_err(), "fs without storage_root should be rejected");
        let msg = format!("{}", result.unwrap_err());
        assert!(
            msg.contains("STORAGE_ROOT"),
            "error should mention STORAGE_ROOT: {msg}"
        );
    }

    #[test]
    fn test_validate_accepts_fs_with_root() {
        let mut config = config_with_model_path(None);
        config.storage_type = "fs".to_string();
        config.storage_root = Some("/data/models".to_string());

        let result = config.validate();

        assert!(result.is_ok(), "fs with storage_root should be accepted");
    }

    // --- ASR config tests ---

    #[test]
    fn test_validate_rejects_invalid_feature_extractor() {
        let mut config = config_with_model_path(None);
        config.feature_extractor = "fbank".to_string();

        let result = config.validate();

        assert!(result.is_err(), "invalid feature_extractor should be rejected");
        let msg = format!("{}", result.unwrap_err());
        assert!(
            msg.contains("FEATURE_EXTRACTOR"),
            "error should mention FEATURE_EXTRACTOR: {msg}"
        );
    }

    #[test]
    fn test_validate_accepts_mel_and_none() {
        for fe in &["mel", "none"] {
            let mut config = config_with_model_path(None);
            config.feature_extractor = fe.to_string();
            let result = config.validate();
            assert!(result.is_ok(), "feature_extractor={fe} should be accepted");
        }
    }

    #[test]
    fn test_validate_rejects_invalid_chunking_strategy() {
        let mut config = config_with_model_path(None);
        config.chunking_strategy = "buffered".to_string();

        let result = config.validate();

        assert!(result.is_err(), "invalid chunking_strategy should be rejected");
        let msg = format!("{}", result.unwrap_err());
        assert!(
            msg.contains("CHUNKING_STRATEGY"),
            "error should mention CHUNKING_STRATEGY: {msg}"
        );
    }

    #[test]
    fn test_validate_rejects_negative_window_size() {
        let mut config = config_with_model_path(None);
        config.window_size_secs = -1.0;

        let result = config.validate();

        assert!(result.is_err(), "negative window_size_secs should be rejected");
    }

    #[test]
    fn test_validate_rejects_overlap_exceeding_window() {
        let mut config = config_with_model_path(None);
        config.window_size_secs = 10.0;
        config.overlap_secs = 10.0;

        let result = config.validate();

        assert!(result.is_err(), "overlap_secs >= window_size_secs should be rejected");
    }
}
