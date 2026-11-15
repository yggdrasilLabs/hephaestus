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

use crate::encoder_decoder::{
    DECODER_ONNX_CANDIDATES, ENCODER_ONNX, GENERATION_CONFIG_JSON, MERGED_DECODER_ONNX,
    ONNX_SUBDIR_PREFIX, is_whisper_config,
};
use crate::error::ResolveError;
use crate::external_data;

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

/// Bytes moved per request when an external data file is streamed to or
/// from storage. External data files run to several gigabytes, so they are
/// never held in memory whole.
const STREAM_CHUNK_BYTES: usize = 32 * 1024 * 1024;

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
    if final_dir.exists() && !discard_incomplete_local_copy(model_id, &final_dir).await {
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

    // Weights kept outside the graph file. A stored graph whose data is
    // not in storage cannot be loaded, so it counts as a miss and the
    // next tier fetches (and stores) the complete model.
    if !download_external_data(op, model_id, onnx_filename, temp_dir.path()).await? {
        return Ok(None);
    }

    // Atomic rename (same filesystem guarantees atomicity).
    tokio::fs::rename(temp_dir.path(), &final_dir).await?;
    // Prevent TempDir destructor from removing the now-renamed directory (D-12).
    let _ = temp_dir.keep();

    tracing::info!(model_id, path = %final_dir.display(), "downloaded model from storage cache");
    Ok(Some(final_dir))
}

/// Download an encoder-decoder (seq2seq) export from storage into the local cache.
///
/// The fallback for [`download_model`]: call it only after that lookup
/// returned `Ok(None)`. It looks for `encoder_model.onnx` plus one decoder
/// graph, first directly under `{model_id}/` and then under
/// `{model_id}/onnx/` (the order [`download_model`] uses), together with
/// `tokenizer.json`, `config.json` and, when present,
/// `generation_config.json`.
///
/// Files are written one at a time into a temporary directory on the same
/// filesystem as `cache_dir`, which is renamed to the final path only once
/// the set is complete (D-12). A partial set never becomes visible.
///
/// Returns `Ok(Some(path))` on cache hit, `Ok(None)` on cache miss (any
/// required file absent, or a Whisper model, whose resolution this lookup
/// leaves untouched), or `Err` on other errors.
pub(crate) async fn download_encoder_decoder_model(
    op: &Operator,
    model_id: &str,
    cache_dir: &Path,
) -> Result<Option<PathBuf>, ResolveError> {
    let final_dir = cache_dir.join(STORAGE_CACHE_SUBDIR).join(model_id);

    // Already cached locally -- skip download.
    if final_dir.exists() && !discard_incomplete_local_copy(model_id, &final_dir).await {
        tracing::debug!(model_id, path = %final_dir.display(), "model already cached locally");
        return Ok(Some(final_dir));
    }

    // The small files first: a miss costs nothing, and config.json tells
    // Whisper apart before any large graph is fetched.
    let config_path = format_storage_path(model_id, CONFIG_JSON);
    let Some(config_data) = download_file(op, &config_path).await? else {
        return Ok(None);
    };
    if is_whisper_config(&config_data) {
        tracing::debug!(
            model_id,
            "skipping encoder-decoder lookup for Whisper model"
        );
        return Ok(None);
    }
    let tokenizer_path = format_storage_path(model_id, TOKENIZER_JSON);
    let Some(tokenizer_data) = download_file(op, &tokenizer_path).await? else {
        return Ok(None);
    };

    // Atomic download: create temp dir on same filesystem as final path (D-12).
    let parent = final_dir.parent().unwrap_or(cache_dir);
    tokio::fs::create_dir_all(parent).await?;
    let temp_dir = tempfile::TempDir::new_in(parent)
        .map_err(|e| ResolveError::Storage(format!("failed to create temp dir: {e}")))?;

    // Encoder and decoder must sit in the same directory.
    let mut found = false;
    let mut graphs = Vec::new();
    for prefix in ["", ONNX_SUBDIR_PREFIX] {
        let encoder = format!("{prefix}{ENCODER_ONNX}");
        if !download_file_into(op, model_id, &encoder, temp_dir.path()).await? {
            continue;
        }
        for decoder in DECODER_ONNX_CANDIDATES {
            let path = format!("{prefix}{decoder}");
            if download_file_into(op, model_id, &path, temp_dir.path()).await? {
                found = true;
                graphs.push(encoder.clone());
                graphs.push(path);
                if decoder != MERGED_DECODER_ONNX {
                    // Optional: the graph for cached decoding.
                    let merged = format!("{prefix}{MERGED_DECODER_ONNX}");
                    if download_file_into(op, model_id, &merged, temp_dir.path()).await? {
                        graphs.push(merged);
                    }
                }
                break;
            }
        }
        if found {
            break;
        }
        // An encoder without a decoder is not a usable export here.
        tokio::fs::remove_file(temp_dir.path().join(&encoder)).await?;
    }
    if !found {
        // Dropping `temp_dir` removes anything written so far.
        return Ok(None);
    }
    // Weights kept outside the graph files; a graph without its data makes
    // the stored set incomplete, which is a miss.
    for graph in &graphs {
        if !download_external_data(op, model_id, graph, temp_dir.path()).await? {
            return Ok(None);
        }
    }

    tokio::fs::write(temp_dir.path().join(TOKENIZER_JSON), &tokenizer_data).await?;
    tokio::fs::write(temp_dir.path().join(CONFIG_JSON), &config_data).await?;
    download_file_into(op, model_id, GENERATION_CONFIG_JSON, temp_dir.path()).await?;

    // Atomic rename (same filesystem guarantees atomicity).
    tokio::fs::rename(temp_dir.path(), &final_dir).await?;
    // Prevent TempDir destructor from removing the now-renamed directory (D-12).
    let _ = temp_dir.keep();

    tracing::info!(
        model_id,
        path = %final_dir.display(),
        "downloaded encoder-decoder model from storage cache"
    );
    Ok(Some(final_dir))
}

/// Download `{model_id}/{filename}` from storage into `dir/{filename}`.
///
/// Returns `Ok(true)` when the file was written and `Ok(false)` when it
/// does not exist in storage. The bytes are dropped as soon as they are on
/// disk, so several large graphs are never held in memory together.
async fn download_file_into(
    op: &Operator,
    model_id: &str,
    filename: &str,
    dir: &Path,
) -> Result<bool, ResolveError> {
    let Some(data) = download_file(op, &format_storage_path(model_id, filename)).await? else {
        return Ok(false);
    };
    let target = dir.join(filename);
    if let Some(target_parent) = target.parent() {
        tokio::fs::create_dir_all(target_parent).await?;
    }
    tokio::fs::write(&target, &data).await?;
    Ok(true)
}

/// Download the external data files of a graph already written to `dir`.
///
/// `graph` is the graph's path relative to the model (`model.onnx`,
/// `onnx/decoder_model.onnx`, ...). The data files it names are fetched
/// from the same storage directory into the same local directory, in
/// pieces of [`STREAM_CHUNK_BYTES`]. A graph that carries its own weights
/// names none, and nothing is requested.
///
/// Returns `Ok(false)` when storage lacks one of the files: the stored
/// model is incomplete and must not be served.
async fn download_external_data(
    op: &Operator,
    model_id: &str,
    graph: &str,
    dir: &Path,
) -> Result<bool, ResolveError> {
    let locations = external_data::locations_of_graph(dir.join(graph)).await?;
    let directory = graph.rsplit_once('/').map_or("", |(directory, _)| directory);
    for location in locations {
        let filename = if directory.is_empty() {
            location
        } else {
            format!("{directory}/{location}")
        };
        if !download_file_streamed(op, model_id, &filename, dir).await? {
            tracing::warn!(
                model_id,
                graph,
                missing = %filename,
                "stored model lacks an external data file; treating it as a cache miss"
            );
            return Ok(false);
        }
        tracing::info!(model_id, graph, data_file = %filename, "downloaded ONNX external data from storage");
    }
    Ok(true)
}

/// Download `{model_id}/{filename}` into `dir/{filename}` piece by piece.
///
/// Returns `Ok(true)` when the file was written and `Ok(false)` when it
/// does not exist in storage.
async fn download_file_streamed(
    op: &Operator,
    model_id: &str,
    filename: &str,
    dir: &Path,
) -> Result<bool, ResolveError> {
    use tokio::io::AsyncWriteExt;

    let path = format_storage_path(model_id, filename);
    let size = match op.stat(&path).await {
        Ok(metadata) => metadata.content_length(),
        Err(e) if e.kind() == ErrorKind::NotFound => return Ok(false),
        Err(e) => return Err(ResolveError::Storage(format!("stat {path}: {e}"))),
    };
    let reader = op
        .reader(&path)
        .await
        .map_err(|e| ResolveError::Storage(format!("read {path}: {e}")))?;

    let target = dir.join(filename);
    if let Some(target_parent) = target.parent() {
        tokio::fs::create_dir_all(target_parent).await?;
    }
    let mut file = tokio::fs::File::create(&target).await?;
    let mut offset = 0_u64;
    while offset < size {
        let end = size.min(offset + STREAM_CHUNK_BYTES as u64);
        let piece = reader
            .read(offset..end)
            .await
            .map_err(|e| ResolveError::Storage(format!("read {path}: {e}")))?;
        if piece.is_empty() {
            return Err(ResolveError::Storage(format!(
                "read {path}: storage returned no data at byte {offset} of {size}"
            )));
        }
        offset += piece.len() as u64;
        file.write_all(&piece.to_bytes()).await?;
    }
    file.flush().await?;
    Ok(true)
}

/// Remove a locally cached model directory whose graphs lack external data.
///
/// Such a directory can only come from a build that did not know about
/// external data; ONNX Runtime cannot load it. Returns `true` when the
/// directory was removed, so the caller downloads the model afresh. A
/// directory without external data -- or one that cannot be inspected --
/// is left alone and used as before.
async fn discard_incomplete_local_copy(model_id: &str, dir: &Path) -> bool {
    let missing = match external_data::external_files_of_dir(dir.to_path_buf()).await {
        Ok(files) => files.into_iter().find(|file| !file.present),
        Err(_) => None,
    };
    let Some(missing) = missing else {
        return false;
    };
    tracing::warn!(
        model_id,
        path = %dir.display(),
        missing = %missing.relative_path.display(),
        "locally cached model lacks an external data file; downloading it again"
    );
    tokio::fs::remove_dir_all(dir).await.is_ok()
}

/// Upload a model directory to storage for cache-back.
///
/// Iterates files in `local_dir`, constructs storage paths as
/// `{model_id}/{filename}`, and uploads each via `Operator::write()`.
/// Upload is unconditional -- no existence check.
///
/// A graph may keep its weights in external data files. Nothing is
/// uploaded unless every such file is present, so storage never receives
/// a graph it cannot serve. The data files go first, piece by piece, and
/// the graphs after them: an upload that is cut short leaves no graph
/// without its data.
///
/// On failure, returns `Err` so the caller can handle retry or
/// log a warning.
pub(crate) async fn upload_model(
    op: &Operator,
    model_id: &str,
    local_dir: &Path,
) -> Result<(), ResolveError> {
    let external = external_data::external_files_of_dir(local_dir.to_path_buf())
        .await
        .map_err(|e| {
            ResolveError::Storage(format!(
                "refusing to upload model '{model_id}': cannot check its external data: {e}"
            ))
        })?;
    if let Some(missing) = external.iter().find(|file| !file.present) {
        return Err(ResolveError::Storage(format!(
            "refusing to upload incomplete model '{model_id}': external data file '{}' is missing",
            missing.relative_path.display(),
        )));
    }

    let external: Vec<PathBuf> = external.into_iter().map(|file| file.relative_path).collect();
    for relative in &external {
        upload_file_streamed(op, model_id, local_dir, relative).await?;
    }
    upload_files_recursive(op, model_id, local_dir, local_dir, &external).await
}

/// Upload `local_dir/relative` to `{model_id}/{relative}` piece by piece.
async fn upload_file_streamed(
    op: &Operator,
    model_id: &str,
    local_dir: &Path,
    relative: &Path,
) -> Result<(), ResolveError> {
    use tokio::io::AsyncReadExt;

    let storage_path = format_storage_path(model_id, &relative.to_string_lossy());
    let local_path = local_dir.join(relative);
    let failed =
        |e: &dyn std::fmt::Display| ResolveError::Storage(format!("write failed for {storage_path}: {e}"));

    let mut file = tokio::fs::File::open(&local_path).await.map_err(|e| {
        ResolveError::Storage(format!("failed to read {}: {e}", local_path.display()))
    })?;
    let mut writer = op
        .writer_with(&storage_path)
        .chunk(STREAM_CHUNK_BYTES)
        .await
        .map_err(|e| failed(&e))?;
    loop {
        let mut piece = Vec::with_capacity(STREAM_CHUNK_BYTES);
        let read = (&mut file)
            .take(STREAM_CHUNK_BYTES as u64)
            .read_to_end(&mut piece)
            .await
            .map_err(|e| {
                ResolveError::Storage(format!("failed to read {}: {e}", local_path.display()))
            })?;
        if read == 0 {
            break;
        }
        if let Err(e) = writer.write(piece).await {
            // Best effort: do not leave an unfinished multipart upload.
            let _ = writer.abort().await;
            return Err(failed(&e));
        }
    }
    writer.close().await.map_err(|e| failed(&e))?;
    tracing::debug!(model_id, path = storage_path, "uploaded external data file to storage");
    Ok(())
}

/// Recursively upload files from a directory to storage.
///
/// Files listed in `skip` (paths relative to `base_dir`) are left out;
/// the caller has uploaded them already.
async fn upload_files_recursive(
    op: &Operator,
    model_id: &str,
    base_dir: &Path,
    current_dir: &Path,
    skip: &[PathBuf],
) -> Result<(), ResolveError> {
    let mut entries = tokio::fs::read_dir(current_dir).await?;
    while let Some(entry) = entries.next_entry().await? {
        let path = entry.path();
        if path.is_dir() {
            Box::pin(upload_files_recursive(op, model_id, base_dir, &path, skip)).await?;
        } else {
            let relative = path
                .strip_prefix(base_dir)
                .map_err(|e| ResolveError::Storage(format!("path strip prefix failed: {e}")))?;
            if skip.iter().any(|skipped| skipped == relative) {
                continue;
            }
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

    // --- download_encoder_decoder_model tests ---

    const BART_CONFIG: &[u8] = br#"{"architectures": ["BartForConditionalGeneration"]}"#;

    /// Write `{model_id}/{name}` for every name, with the name as content.
    async fn write_files(op: &Operator, model_id: &str, names: &[&str]) {
        for name in names {
            let content = if name.ends_with("config.json") && !name.contains("generation") {
                BART_CONFIG.to_vec()
            } else {
                name.as_bytes().to_vec()
            };
            op.write(&format!("{model_id}/{name}"), content)
                .await
                .unwrap();
        }
    }

    /// Relative paths of every file under `dir`, sorted.
    fn list_files(dir: &Path) -> Vec<String> {
        fn walk(base: &Path, current: &Path, out: &mut Vec<String>) {
            for entry in fs::read_dir(current).unwrap() {
                let path = entry.unwrap().path();
                if path.is_dir() {
                    walk(base, &path, out);
                } else {
                    out.push(
                        path.strip_prefix(base)
                            .unwrap()
                            .to_string_lossy()
                            .into_owned(),
                    );
                }
            }
        }
        let mut files = Vec::new();
        walk(dir, dir, &mut files);
        files.sort();
        files
    }

    #[tokio::test]
    async fn encoder_decoder_download_returns_none_on_cache_miss() {
        let op = memory_operator();
        let cache_dir = TempDir::new().unwrap();

        let result = download_encoder_decoder_model(&op, "org/model", cache_dir.path()).await;

        assert!(matches!(result, Ok(None)));
    }

    #[tokio::test]
    async fn encoder_decoder_download_fetches_plain_decoder_and_generation_config() {
        // Arrange -- everything Forge uploads for a seq2seq model.
        let op = memory_operator();
        write_files(
            &op,
            "org/model",
            &[
                "encoder_model.onnx",
                "decoder_model.onnx",
                "decoder_model_merged.onnx",
                "decoder_with_past_model.onnx",
                "tokenizer.json",
                "config.json",
                "generation_config.json",
                "vocab.json",
            ],
        )
        .await;
        let cache_dir = TempDir::new().unwrap();

        // Act
        let model_dir = download_encoder_decoder_model(&op, "org/model", cache_dir.path())
            .await
            .unwrap()
            .expect("export should be found");

        // Assert -- both decoders the runtime can use; nothing it does not read.
        assert_eq!(
            list_files(&model_dir),
            vec![
                "config.json",
                "decoder_model.onnx",
                "decoder_model_merged.onnx",
                "encoder_model.onnx",
                "generation_config.json",
                "tokenizer.json",
            ]
        );
        assert_eq!(
            fs::read(model_dir.join("decoder_model.onnx")).unwrap(),
            b"decoder_model.onnx"
        );
    }

    #[tokio::test]
    async fn encoder_decoder_download_fetches_merged_decoder_from_onnx_subdirectory() {
        // Arrange
        let op = memory_operator();
        write_files(
            &op,
            "org/model",
            &[
                "onnx/encoder_model.onnx",
                "onnx/decoder_model.onnx",
                "onnx/decoder_model_merged.onnx",
                "tokenizer.json",
                "config.json",
            ],
        )
        .await;
        let cache_dir = TempDir::new().unwrap();

        // Act
        let model_dir = download_encoder_decoder_model(&op, "org/model", cache_dir.path())
            .await
            .unwrap()
            .expect("export should be found");

        // Assert -- the merged decoder lands next to the encoder.
        assert_eq!(
            list_files(&model_dir),
            vec![
                "config.json",
                "onnx/decoder_model.onnx",
                "onnx/decoder_model_merged.onnx",
                "onnx/encoder_model.onnx",
                "tokenizer.json",
            ]
        );
        assert_eq!(
            fs::read(model_dir.join("onnx/decoder_model_merged.onnx")).unwrap(),
            b"onnx/decoder_model_merged.onnx"
        );
    }

    #[tokio::test]
    async fn encoder_decoder_download_without_merged_decoder_is_unchanged() {
        // Arrange -- an export uploaded before the merged decoder was fetched.
        let op = memory_operator();
        write_files(
            &op,
            "org/model",
            &[
                "encoder_model.onnx",
                "decoder_model.onnx",
                "tokenizer.json",
                "config.json",
            ],
        )
        .await;
        let cache_dir = TempDir::new().unwrap();

        // Act
        let model_dir = download_encoder_decoder_model(&op, "org/model", cache_dir.path())
            .await
            .unwrap()
            .expect("export should be found");

        // Assert
        assert_eq!(
            list_files(&model_dir),
            vec![
                "config.json",
                "decoder_model.onnx",
                "encoder_model.onnx",
                "tokenizer.json",
            ]
        );
    }

    #[tokio::test]
    async fn encoder_decoder_download_falls_back_to_merged_decoder() {
        let op = memory_operator();
        write_files(
            &op,
            "org/model",
            &[
                "encoder_model.onnx",
                "decoder_model_merged.onnx",
                "tokenizer.json",
                "config.json",
            ],
        )
        .await;
        let cache_dir = TempDir::new().unwrap();

        let model_dir = download_encoder_decoder_model(&op, "org/model", cache_dir.path())
            .await
            .unwrap()
            .expect("export should be found");

        assert_eq!(
            list_files(&model_dir),
            vec![
                "config.json",
                "decoder_model_merged.onnx",
                "encoder_model.onnx",
                "tokenizer.json",
            ]
        );
    }

    #[tokio::test]
    async fn encoder_decoder_download_supports_onnx_subdirectory() {
        let op = memory_operator();
        write_files(
            &op,
            "org/model",
            &[
                "onnx/encoder_model.onnx",
                "onnx/decoder_model.onnx",
                "tokenizer.json",
                "config.json",
            ],
        )
        .await;
        let cache_dir = TempDir::new().unwrap();

        let model_dir = download_encoder_decoder_model(&op, "org/model", cache_dir.path())
            .await
            .unwrap()
            .expect("export should be found");

        assert_eq!(
            list_files(&model_dir),
            vec![
                "config.json",
                "onnx/decoder_model.onnx",
                "onnx/encoder_model.onnx",
                "tokenizer.json",
            ]
        );
    }

    #[tokio::test]
    async fn encoder_decoder_download_drops_stray_encoder_without_decoder() {
        // Arrange -- a lone flat encoder next to a complete export in onnx/.
        let op = memory_operator();
        write_files(
            &op,
            "org/model",
            &[
                "encoder_model.onnx",
                "onnx/encoder_model.onnx",
                "onnx/decoder_model.onnx",
                "tokenizer.json",
                "config.json",
            ],
        )
        .await;
        let cache_dir = TempDir::new().unwrap();

        // Act
        let model_dir = download_encoder_decoder_model(&op, "org/model", cache_dir.path())
            .await
            .unwrap()
            .expect("export should be found");

        // Assert
        assert_eq!(
            list_files(&model_dir),
            vec![
                "config.json",
                "onnx/decoder_model.onnx",
                "onnx/encoder_model.onnx",
                "tokenizer.json",
            ]
        );
    }

    #[tokio::test]
    async fn encoder_decoder_download_misses_on_incomplete_set() {
        // Each set lacks one required file; none may produce a model directory.
        let incomplete: [&[&str]; 3] = [
            &["encoder_model.onnx", "tokenizer.json", "config.json"],
            &["encoder_model.onnx", "decoder_model.onnx", "config.json"],
            &["encoder_model.onnx", "decoder_model.onnx", "tokenizer.json"],
        ];
        for names in incomplete {
            let op = memory_operator();
            write_files(&op, "org/model", names).await;
            let cache_dir = TempDir::new().unwrap();

            let result = download_encoder_decoder_model(&op, "org/model", cache_dir.path()).await;

            assert!(matches!(result, Ok(None)), "expected a miss for {names:?}");
            // Atomicity: neither the final directory nor a temp directory is left behind.
            let final_dir = cache_dir
                .path()
                .join(STORAGE_CACHE_SUBDIR)
                .join("org/model");
            assert!(
                !final_dir.exists(),
                "partial set must not become visible: {names:?}"
            );
            let parent = final_dir.parent().unwrap();
            assert!(
                !parent.exists() || fs::read_dir(parent).unwrap().next().is_none(),
                "temp directory must be removed: {names:?}"
            );
        }
    }

    #[tokio::test]
    async fn encoder_decoder_download_skips_whisper_model() {
        // Arrange -- Whisper ships the same file layout.
        let op = memory_operator();
        write_files(
            &op,
            "org/whisper",
            &[
                "onnx/encoder_model.onnx",
                "onnx/decoder_model.onnx",
                "tokenizer.json",
            ],
        )
        .await;
        op.write(
            "org/whisper/config.json",
            br#"{"architectures": ["WhisperForConditionalGeneration"]}"#.to_vec(),
        )
        .await
        .unwrap();
        let cache_dir = TempDir::new().unwrap();

        // Act
        let result = download_encoder_decoder_model(&op, "org/whisper", cache_dir.path()).await;

        // Assert -- still a miss, as before this lookup existed.
        assert!(matches!(result, Ok(None)));
    }

    #[tokio::test]
    async fn single_file_download_ignores_encoder_decoder_files() {
        // Arrange -- a model stored in both layouts.
        let op = memory_operator();
        write_files(
            &op,
            "org/model",
            &[
                "model.onnx",
                "encoder_model.onnx",
                "decoder_model.onnx",
                "generation_config.json",
                "tokenizer.json",
                "config.json",
            ],
        )
        .await;
        let cache_dir = TempDir::new().unwrap();

        // Act -- the existing lookup, unchanged.
        let model_dir = download_model(&op, "org/model", cache_dir.path())
            .await
            .unwrap()
            .expect("single-file model should be found");

        // Assert -- exactly the three files it has always fetched.
        assert_eq!(
            list_files(&model_dir),
            vec!["config.json", "model.onnx", "tokenizer.json"]
        );
    }

    // --- ONNX external data ---

    use crate::external_data::test_support::model_with_external_data;

    /// Names of every object stored under `prefix`, sorted.
    async fn stored_files(op: &Operator, prefix: &str) -> Vec<String> {
        let mut names: Vec<String> = op
            .list_with(prefix)
            .recursive(true)
            .await
            .unwrap()
            .into_iter()
            .filter(|entry| entry.metadata().is_file())
            .map(|entry| entry.path().to_string())
            .collect();
        names.sort();
        names
    }

    #[tokio::test]
    async fn single_file_download_fetches_external_data() {
        // Arrange -- the layout of an Optimum export above 2 GB.
        let op = memory_operator();
        let graph = model_with_external_data("model.onnx_data");
        op.write("org/model/onnx/model.onnx", graph.clone()).await.unwrap();
        op.write("org/model/onnx/model.onnx_data", b"weights".to_vec())
            .await
            .unwrap();
        write_files(&op, "org/model", &["tokenizer.json", "config.json"]).await;
        let cache_dir = TempDir::new().unwrap();

        // Act
        let model_dir = download_model(&op, "org/model", cache_dir.path())
            .await
            .unwrap()
            .expect("model should be found");

        // Assert -- the data file sits next to its graph.
        assert_eq!(
            list_files(&model_dir),
            vec![
                "config.json",
                "onnx/model.onnx",
                "onnx/model.onnx_data",
                "tokenizer.json",
            ]
        );
        assert_eq!(fs::read(model_dir.join("onnx/model.onnx")).unwrap(), graph);
        assert_eq!(
            fs::read(model_dir.join("onnx/model.onnx_data")).unwrap(),
            b"weights"
        );
    }

    #[tokio::test]
    async fn single_file_download_misses_when_external_data_is_not_stored() {
        // Arrange -- what a build without external data support uploaded:
        // the graph alone.
        let op = memory_operator();
        op.write(
            "org/model/onnx/model.onnx",
            model_with_external_data("model.onnx_data"),
        )
        .await
        .unwrap();
        write_files(&op, "org/model", &["tokenizer.json", "config.json"]).await;
        let cache_dir = TempDir::new().unwrap();

        // Act
        let result = download_model(&op, "org/model", cache_dir.path()).await;

        // Assert -- a miss, and nothing left behind in the local cache.
        assert!(matches!(result, Ok(None)));
        assert!(
            !cache_dir
                .path()
                .join(STORAGE_CACHE_SUBDIR)
                .join("org/model")
                .exists()
        );
    }

    #[tokio::test]
    async fn encoder_decoder_download_fetches_external_data() {
        // Arrange
        let op = memory_operator();
        write_files(
            &op,
            "org/model",
            &["config.json", "tokenizer.json", "encoder_model.onnx"],
        )
        .await;
        op.write(
            "org/model/decoder_model.onnx",
            model_with_external_data("decoder_model.onnx_data"),
        )
        .await
        .unwrap();
        op.write("org/model/decoder_model.onnx_data", b"weights".to_vec())
            .await
            .unwrap();
        let cache_dir = TempDir::new().unwrap();

        // Act
        let model_dir = download_encoder_decoder_model(&op, "org/model", cache_dir.path())
            .await
            .unwrap()
            .expect("export should be found");

        // Assert
        assert_eq!(
            list_files(&model_dir),
            vec![
                "config.json",
                "decoder_model.onnx",
                "decoder_model.onnx_data",
                "encoder_model.onnx",
                "tokenizer.json",
            ]
        );
    }

    #[tokio::test]
    async fn encoder_decoder_download_misses_when_external_data_is_not_stored() {
        // Arrange
        let op = memory_operator();
        write_files(
            &op,
            "org/model",
            &["config.json", "tokenizer.json", "encoder_model.onnx"],
        )
        .await;
        op.write(
            "org/model/decoder_model.onnx",
            model_with_external_data("decoder_model.onnx_data"),
        )
        .await
        .unwrap();
        let cache_dir = TempDir::new().unwrap();

        // Act
        let result = download_encoder_decoder_model(&op, "org/model", cache_dir.path()).await;

        // Assert
        assert!(matches!(result, Ok(None)));
    }

    #[tokio::test]
    async fn incomplete_local_copy_is_downloaded_again() {
        // Arrange -- storage is complete, but the local cache holds the
        // graph without its data.
        let op = memory_operator();
        let graph = model_with_external_data("model.onnx_data");
        op.write("org/model/model.onnx", graph.clone()).await.unwrap();
        op.write("org/model/model.onnx_data", b"weights".to_vec())
            .await
            .unwrap();
        write_files(&op, "org/model", &["tokenizer.json", "config.json"]).await;
        let cache_dir = TempDir::new().unwrap();
        let stale = cache_dir.path().join(STORAGE_CACHE_SUBDIR).join("org/model");
        fs::create_dir_all(&stale).unwrap();
        fs::write(stale.join("model.onnx"), &graph).unwrap();

        // Act
        let model_dir = download_model(&op, "org/model", cache_dir.path())
            .await
            .unwrap()
            .expect("model should be found");

        // Assert
        assert_eq!(model_dir, stale);
        assert_eq!(fs::read(model_dir.join("model.onnx_data")).unwrap(), b"weights");
    }

    #[tokio::test]
    async fn upload_model_uploads_external_data_with_its_graph() {
        // Arrange -- a data file larger than one streamed piece would be
        // too slow here; a few bytes take the same code path.
        let op = memory_operator();
        let dir = TempDir::new().unwrap();
        fs::create_dir_all(dir.path().join("onnx")).unwrap();
        let graph = model_with_external_data("model.onnx_data");
        fs::write(dir.path().join("onnx/model.onnx"), &graph).unwrap();
        fs::write(dir.path().join("onnx/model.onnx_data"), b"weights").unwrap();
        fs::write(dir.path().join("tokenizer.json"), b"{}").unwrap();
        fs::write(dir.path().join("config.json"), b"{}").unwrap();

        // Act
        upload_model(&op, "org/model", dir.path())
            .await
            .expect("upload should succeed");

        // Assert -- every file once, bytes intact.
        assert_eq!(
            stored_files(&op, "org/model/").await,
            vec![
                "org/model/config.json",
                "org/model/onnx/model.onnx",
                "org/model/onnx/model.onnx_data",
                "org/model/tokenizer.json",
            ]
        );
        assert_eq!(
            op.read("org/model/onnx/model.onnx_data").await.unwrap().to_vec(),
            b"weights"
        );
        assert_eq!(op.read("org/model/onnx/model.onnx").await.unwrap().to_vec(), graph);
    }

    #[tokio::test]
    async fn upload_model_refuses_incomplete_model() {
        // Arrange -- the graph without the data file it names: the state
        // that reached storage in production.
        let op = memory_operator();
        let dir = TempDir::new().unwrap();
        fs::create_dir_all(dir.path().join("onnx")).unwrap();
        fs::write(
            dir.path().join("onnx/model.onnx"),
            model_with_external_data("model.onnx_data"),
        )
        .unwrap();
        fs::write(dir.path().join("tokenizer.json"), b"{}").unwrap();
        fs::write(dir.path().join("config.json"), b"{}").unwrap();

        // Act
        let err = upload_model(&op, "org/model", dir.path())
            .await
            .expect_err("incomplete model must not be uploaded");

        // Assert -- named in the error, and not one object written.
        assert!(err.to_string().contains("onnx/model.onnx_data"), "unexpected error: {err}");
        assert!(stored_files(&op, "org/model/").await.is_empty());
    }

    #[tokio::test]
    async fn uploaded_model_with_external_data_downloads_complete() {
        // Arrange -- upload, then fetch into an empty cache, as a second
        // pod would.
        let op = memory_operator();
        let dir = TempDir::new().unwrap();
        let graph = model_with_external_data("model.onnx_data");
        let weights: Vec<u8> = (0..=255).cycle().take(100_000).collect();
        fs::write(dir.path().join("model.onnx"), &graph).unwrap();
        fs::write(dir.path().join("model.onnx_data"), &weights).unwrap();
        fs::write(dir.path().join("tokenizer.json"), b"{}").unwrap();
        fs::write(dir.path().join("config.json"), b"{}").unwrap();
        upload_model(&op, "org/model", dir.path()).await.unwrap();
        let cache_dir = TempDir::new().unwrap();

        // Act
        let model_dir = download_model(&op, "org/model", cache_dir.path())
            .await
            .unwrap()
            .expect("model should be found");

        // Assert
        assert_eq!(fs::read(model_dir.join("model.onnx")).unwrap(), graph);
        assert_eq!(fs::read(model_dir.join("model.onnx_data")).unwrap(), weights);
    }
}
