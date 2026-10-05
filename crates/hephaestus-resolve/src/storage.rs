//! Backend-agnostic model storage operations via Apache OpenDAL.
//!
//! Implements download and upload for the storage cache tier using
//! OpenDAL's [`Operator`] abstraction. Works identically across S3, GCS,
//! local filesystem, and in-memory backends (RSLV-01, RSLV-04, D-11).
//!
//! Downloads use an atomic temp-dir-then-rename pattern to prevent
//! serving partial files (D-12). Uploads are unconditional with
//! non-fatal failure handling.

use std::path::{Path, PathBuf};

use opendal::{ErrorKind, Operator};

use crate::error::ResolveError;

/// Required model files for a complete model directory.
pub(crate) const MODEL_ONNX: &str = "model.onnx";
pub(crate) const TOKENIZER_JSON: &str = "tokenizer.json";
pub(crate) const CONFIG_JSON: &str = "config.json";
/// Alternative ONNX file location used by some HuggingFace models.
pub(crate) const ONNX_SUBDIR_MODEL: &str = "onnx/model.onnx";

/// Storage cache subdirectory under the HuggingFace home directory.
/// Keeps storage-downloaded models separate from hf-hub's content-addressed
/// blob layout.
pub(crate) const STORAGE_CACHE_SUBDIR: &str = "hephaestus/storage-cache";

/// Construct a storage path from model ID and filename.
///
/// Returns `"{model_id}/{filename}"`. No prefix parameter because
/// the OpenDAL `Operator` root already includes the prefix (D-04,
/// Pitfall 1 from RESEARCH.md).
fn format_storage_path(model_id: &str, filename: &str) -> String {
    format!("{model_id}/{filename}")
}

/// Download a single file from storage by path.
///
/// Returns `Ok(Some(bytes))` on hit, `Ok(None)` on cache miss
/// (`ErrorKind::NotFound`), or `Err` on other storage errors.
async fn download_file(op: &Operator, path: &str) -> Result<Option<Vec<u8>>, ResolveError> {
    match op.read(path).await {
        Ok(data) => Ok(Some(data.to_vec())),
        Err(e) if e.kind() == ErrorKind::NotFound => Ok(None),
        Err(e) => Err(ResolveError::Storage(format!("read {path}: {e}"))),
    }
}

/// Download a model directory from storage into the local cache.
///
/// Constructs storage paths as `{model_id}/{filename}`. Downloads
/// all required files into a temporary directory on the same filesystem
/// as `cache_dir`, then atomically renames to the final path (D-12).
///
/// Returns `Ok(Some(path))` on cache hit, `Ok(None)` on cache miss
/// (file not found for the ONNX file), or `Err` on other errors.
pub(crate) async fn download_model(
    op: &Operator,
    model_id: &str,
    cache_dir: &Path,
) -> Result<Option<PathBuf>, ResolveError> {
    let final_dir = cache_dir.join(STORAGE_CACHE_SUBDIR).join(model_id);

    // Already cached locally -- skip download.
    if final_dir.exists() {
        tracing::debug!(model_id, path = %final_dir.display(), "model already cached locally");
        return Ok(Some(final_dir));
    }

    // Try downloading the ONNX model file.
    // Check model.onnx first, then onnx/model.onnx.
    let mut onnx_filename = None;
    let mut onnx_bytes = None;
    for candidate in &[MODEL_ONNX, ONNX_SUBDIR_MODEL] {
        let path = format_storage_path(model_id, candidate);
        match download_file(op, &path).await? {
            Some(bytes) => {
                onnx_filename = Some(*candidate);
                onnx_bytes = Some(bytes);
                break;
            }
            None => continue,
        }
    }

    // If no ONNX file found, this is a cache miss.
    let (onnx_filename, onnx_data) = match (onnx_filename, onnx_bytes) {
        (Some(f), Some(d)) => (f, d),
        _ => return Ok(None),
    };

    // Download supporting files.
    let tokenizer_path = format_storage_path(model_id, TOKENIZER_JSON);
    let tokenizer_data = match download_file(op, &tokenizer_path).await? {
        Some(bytes) => bytes,
        None => return Ok(None),
    };

    let config_path = format_storage_path(model_id, CONFIG_JSON);
    let config_data = match download_file(op, &config_path).await? {
        Some(bytes) => bytes,
        None => return Ok(None),
    };

    // Atomic download: create temp dir on same filesystem as final path (D-12).
    let parent = final_dir.parent().unwrap_or(cache_dir);
    tokio::fs::create_dir_all(parent).await?;
    let temp_dir = tempfile::TempDir::new_in(parent)
        .map_err(|e| ResolveError::Storage(format!("failed to create temp dir: {e}")))?;

    // Write ONNX file (may be in subdirectory).
    let onnx_path = temp_dir.path().join(onnx_filename);
    if let Some(onnx_parent) = onnx_path.parent() {
        tokio::fs::create_dir_all(onnx_parent).await?;
    }
    tokio::fs::write(&onnx_path, &onnx_data).await?;

    // Write supporting files.
    tokio::fs::write(temp_dir.path().join(TOKENIZER_JSON), &tokenizer_data).await?;
    tokio::fs::write(temp_dir.path().join(CONFIG_JSON), &config_data).await?;

    // Atomic rename (same filesystem guarantees atomicity).
    tokio::fs::rename(temp_dir.path(), &final_dir).await?;
    // Prevent TempDir destructor from removing the now-renamed directory (D-12).
    let _ = temp_dir.keep();

    tracing::info!(model_id, path = %final_dir.display(), "downloaded model from storage cache");
    Ok(Some(final_dir))
}

/// Upload a model directory to storage for cache-back.
///
/// Iterates files in `local_dir`, constructs storage paths as
/// `{model_id}/{filename}`, and uploads each via `Operator::write()`.
/// Upload is unconditional -- no existence check.
///
/// On failure, returns `Err` so the caller can handle retry or
/// log a warning.
pub(crate) async fn upload_model(
    op: &Operator,
    model_id: &str,
    local_dir: &Path,
) -> Result<(), ResolveError> {
    upload_files_recursive(op, model_id, local_dir, local_dir).await
}

/// Recursively upload files from a directory to storage.
async fn upload_files_recursive(
    op: &Operator,
    model_id: &str,
    base_dir: &Path,
    current_dir: &Path,
) -> Result<(), ResolveError> {
    let mut entries = tokio::fs::read_dir(current_dir).await?;
    while let Some(entry) = entries.next_entry().await? {
        let path = entry.path();
        if path.is_dir() {
            Box::pin(upload_files_recursive(op, model_id, base_dir, &path)).await?;
        } else {
            let relative = path
                .strip_prefix(base_dir)
                .map_err(|e| ResolveError::Storage(format!("path strip prefix failed: {e}")))?;
            let filename = relative.to_string_lossy();
            let storage_path = format_storage_path(model_id, &filename);

            let bytes = tokio::fs::read(&path).await.map_err(|e| {
                ResolveError::Storage(format!("failed to read {}: {e}", path.display()))
            })?;

            op.write(&storage_path, bytes).await.map_err(|e| {
                ResolveError::Storage(format!("write failed for {storage_path}: {e}"))
            })?;

            tracing::debug!(model_id, path = storage_path, "uploaded file to storage");
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    /// Create a memory-backed OpenDAL operator for testing.
    fn memory_operator() -> Operator {
        Operator::new(opendal::services::Memory::default()).unwrap()
    }

    // --- Constants tests ---

    #[test]
    fn constants_match_expected_filenames() {
        assert_eq!(MODEL_ONNX, "model.onnx");
        assert_eq!(TOKENIZER_JSON, "tokenizer.json");
        assert_eq!(CONFIG_JSON, "config.json");
        assert_eq!(ONNX_SUBDIR_MODEL, "onnx/model.onnx");
    }

    #[test]
    fn storage_cache_subdir_value() {
        assert_eq!(STORAGE_CACHE_SUBDIR, "hephaestus/storage-cache");
    }

    // --- Path formatting tests ---

    #[test]
    fn format_storage_path_basic() {
        assert_eq!(
            format_storage_path("org/model", "model.onnx"),
            "org/model/model.onnx"
        );
    }

    #[test]
    fn format_storage_path_preserves_slashes() {
        assert_eq!(
            format_storage_path("sentence-transformers/all-MiniLM-L6-v2", "config.json"),
            "sentence-transformers/all-MiniLM-L6-v2/config.json"
        );
    }

    // --- download_file tests ---

    #[tokio::test]
    async fn download_file_returns_none_on_miss() {
        let op = memory_operator();
        let result = download_file(&op, "nonexistent/file.onnx").await;
        assert!(matches!(result, Ok(None)));
    }

    #[tokio::test]
    async fn download_file_returns_bytes_on_hit() {
        let op = memory_operator();
        let data = b"test model data";
        op.write("test/model.onnx", data.to_vec()).await.unwrap();

        let result = download_file(&op, "test/model.onnx").await;
        assert!(result.is_ok());
        let bytes = result.unwrap();
        assert!(bytes.is_some());
        assert_eq!(bytes.unwrap(), data);
    }

    // --- download_model tests ---

    #[tokio::test]
    async fn download_model_returns_none_on_cache_miss() {
        let op = memory_operator();
        let cache_dir = TempDir::new().unwrap();
        let result = download_model(&op, "org/model", cache_dir.path()).await;
        assert!(matches!(result, Ok(None)));
    }

    #[tokio::test]
    async fn download_model_returns_path_on_hit() {
        let op = memory_operator();

        // Write all required model files to the memory operator.
        op.write("org/model/model.onnx", b"fake model".to_vec())
            .await
            .unwrap();
        op.write("org/model/tokenizer.json", b"{}".to_vec())
            .await
            .unwrap();
        op.write("org/model/config.json", b"{}".to_vec())
            .await
            .unwrap();

        let cache_dir = TempDir::new().unwrap();
        let result = download_model(&op, "org/model", cache_dir.path()).await;

        assert!(result.is_ok());
        let path = result.unwrap();
        assert!(path.is_some());
        let model_dir = path.unwrap();
        assert!(model_dir.join("model.onnx").exists());
        assert!(model_dir.join("tokenizer.json").exists());
        assert!(model_dir.join("config.json").exists());
    }

    #[tokio::test]
    async fn download_model_returns_existing_local_cache() {
        let op = memory_operator();
        let cache_dir = TempDir::new().unwrap();

        // Pre-create the final directory with a marker file.
        let model_dir = cache_dir
            .path()
            .join(STORAGE_CACHE_SUBDIR)
            .join("test/model");
        fs::create_dir_all(&model_dir).unwrap();
        fs::write(model_dir.join("model.onnx"), b"cached").unwrap();

        // Should return immediately without touching the operator.
        let result = download_model(&op, "test/model", cache_dir.path()).await;
        assert!(result.is_ok());
        let path = result.unwrap();
        assert!(path.is_some());
        assert_eq!(path.unwrap(), model_dir);
    }

    // --- upload_model tests ---

    #[tokio::test]
    async fn upload_model_writes_files() {
        let op = memory_operator();
        let dir = TempDir::new().unwrap();

        fs::write(dir.path().join("model.onnx"), b"model data").unwrap();
        fs::write(dir.path().join("tokenizer.json"), b"{}").unwrap();
        fs::write(dir.path().join("config.json"), b"{}").unwrap();

        let result = upload_model(&op, "org/model", dir.path()).await;
        assert!(result.is_ok());

        // Verify files are readable from the operator.
        let model_data = op.read("org/model/model.onnx").await.unwrap();
        assert_eq!(model_data.to_vec(), b"model data");

        let tokenizer_data = op.read("org/model/tokenizer.json").await.unwrap();
        assert_eq!(tokenizer_data.to_vec(), b"{}");

        let config_data = op.read("org/model/config.json").await.unwrap();
        assert_eq!(config_data.to_vec(), b"{}");
    }

    #[tokio::test]
    async fn upload_model_handles_subdirectories() {
        let op = memory_operator();
        let dir = TempDir::new().unwrap();

        fs::create_dir_all(dir.path().join("onnx")).unwrap();
        fs::write(dir.path().join("onnx/model.onnx"), b"model data").unwrap();
        fs::write(dir.path().join("tokenizer.json"), b"{}").unwrap();
        fs::write(dir.path().join("config.json"), b"{}").unwrap();

        let result = upload_model(&op, "org/model", dir.path()).await;
        assert!(result.is_ok());

        // Verify nested path is accessible.
        let data = op.read("org/model/onnx/model.onnx").await.unwrap();
        assert_eq!(data.to_vec(), b"model data");
    }

    // --- Atomic download pattern verification ---

    #[tokio::test]
    async fn atomic_download_creates_final_dir_via_rename() {
        let cache_dir = TempDir::new().unwrap();
        let final_dir = cache_dir
            .path()
            .join(STORAGE_CACHE_SUBDIR)
            .join("test-model");

        // Simulate the atomic pattern.
        let parent = final_dir.parent().unwrap();
        tokio::fs::create_dir_all(parent).await.unwrap();
        let temp = tempfile::TempDir::new_in(parent).unwrap();

        // Write test files.
        tokio::fs::write(temp.path().join("model.onnx"), b"fake model")
            .await
            .unwrap();
        tokio::fs::write(temp.path().join("tokenizer.json"), b"{}")
            .await
            .unwrap();
        tokio::fs::write(temp.path().join("config.json"), b"{}")
            .await
            .unwrap();

        // Atomic rename.
        tokio::fs::rename(temp.path(), &final_dir).await.unwrap();
        let _ = temp.keep();

        // Verify files exist at final location.
        assert!(final_dir.join("model.onnx").exists());
        assert!(final_dir.join("tokenizer.json").exists());
        assert!(final_dir.join("config.json").exists());
    }

    #[tokio::test]
    async fn keep_prevents_destructor_cleanup() {
        let cache_dir = TempDir::new().unwrap();
        let parent = cache_dir.path().join("test-parent");
        fs::create_dir_all(&parent).unwrap();

        let temp = tempfile::TempDir::new_in(&parent).unwrap();
        let temp_path = temp.path().to_path_buf();
        fs::write(temp_path.join("test.txt"), b"data").unwrap();

        // Rename to final location.
        let final_path = parent.join("final");
        fs::rename(&temp_path, &final_path).unwrap();

        // keep() prevents destructor from removing temp_path.
        let _ = temp.keep();

        // Final location should still have the file.
        assert!(final_path.join("test.txt").exists());
    }
}
