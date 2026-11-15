//! HuggingFace model download operations.
//!
//! Downloads ONNX model files from the HuggingFace Hub using the `hf-hub`
//! crate. Handles the `onnx/model.onnx` vs `model.onnx` layout convention
//! and returns the snapshot root directory for `ClassifierPipeline::new()`.

use std::future::Future;
use std::path::PathBuf;

use hf_hub::{HFClient, HFError};

use crate::encoder_decoder::{
    DECODER_ONNX_CANDIDATES, ENCODER_ONNX, GENERATION_CONFIG_JSON, ONNX_SUBDIR_PREFIX,
    is_whisper_config,
};
use crate::error::ResolveError;

/// Split a combined model ID into `(owner, name)` components.
///
/// If the model ID contains a `/`, splits on the first occurrence:
/// `"Xenova/distilbert"` becomes `("Xenova", "distilbert")`.
///
/// If no `/` is present, returns `(model_id, model_id)` as both owner
/// and name (Pitfall 6 from RESEARCH.md).
pub(crate) fn split_model_id(model_id: &str) -> (String, String) {
    match model_id.split_once('/') {
        Some((owner, name)) => (owner.to_string(), name.to_string()),
        None => (model_id.to_string(), model_id.to_string()),
    }
}

/// Download a model's ONNX files from HuggingFace Hub.
///
/// Tries `onnx/model.onnx` first, falls back to `model.onnx`. If neither
/// exists, returns [`ResolveError::NoOnnxExport`] per D-04. Also downloads
/// `tokenizer.json` and `config.json`.
///
/// Returns the snapshot root directory containing all model files.
pub(crate) async fn download_from_hf(
    model_id: &str,
) -> Result<PathBuf, ResolveError> {
    let client = HFClient::new()
        .map_err(|e| ResolveError::HuggingFace(format!("failed to create HFClient: {e}")))?;

    let (owner, name) = split_model_id(model_id);
    let repo = client.model(&owner, &name);

    // Try onnx/model.onnx first, then model.onnx (D-04, Pitfall 4).
    let onnx_path = match repo.download_file().filename("onnx/model.onnx").send().await {
        Ok(path) => path,
        Err(HFError::EntryNotFound { .. }) => {
            // Try flat layout.
            match repo.download_file().filename("model.onnx").send().await {
                Ok(path) => path,
                Err(HFError::EntryNotFound { .. }) => {
                    return Err(ResolveError::NoOnnxExport {
                        model_id: model_id.to_string(),
                    });
                }
                Err(e) => {
                    return Err(ResolveError::HuggingFace(e.to_string()));
                }
            }
        }
        Err(e) => {
            return Err(ResolveError::HuggingFace(e.to_string()));
        }
    };

    // Download supporting files.
    let _tokenizer = repo
        .download_file()
        .filename("tokenizer.json")
        .send()
        .await
        .map_err(|e| {
            ResolveError::HuggingFace(format!("failed to download tokenizer.json: {e}"))
        })?;

    let _config = repo
        .download_file()
        .filename("config.json")
        .send()
        .await
        .map_err(|e| {
            ResolveError::HuggingFace(format!("failed to download config.json: {e}"))
        })?;

    // Optional: vocab.json (required by CTC ASR models like wav2vec2).
    match repo.download_file().filename("vocab.json").send().await {
        Ok(_) => {}
        Err(HFError::EntryNotFound { .. }) => {}
        Err(e) => {
            tracing::warn!(model_id, error = %e, "failed to download optional vocab.json");
        }
    }

    // Navigate to snapshot root from the ONNX file path.
    // onnx_path is {snapshot_root}/onnx/model.onnx or {snapshot_root}/model.onnx
    let snapshot_root =
        if onnx_path.parent().and_then(|p| p.file_name()) == Some(std::ffi::OsStr::new("onnx")) {
            onnx_path
                .parent()
                .and_then(|p| p.parent())
                .ok_or_else(|| ResolveError::HuggingFace(
                    format!("unexpected ONNX path structure: {}", onnx_path.display()),
                ))?
                .to_path_buf()
        } else {
            onnx_path
                .parent()
                .ok_or_else(|| ResolveError::HuggingFace(
                    format!("ONNX file has no parent directory: {}", onnx_path.display()),
                ))?
                .to_path_buf()
        };

    Ok(snapshot_root)
}

/// Download a model's encoder-decoder (seq2seq) ONNX export from HuggingFace Hub.
///
/// The fallback for [`download_from_hf`]: call it only after that lookup
/// returned [`ResolveError::NoOnnxExport`]. See
/// [`fetch_encoder_decoder_export`] for the files it fetches.
///
/// Returns the snapshot root directory containing all model files.
pub(crate) async fn download_encoder_decoder_from_hf(
    model_id: &str,
) -> Result<PathBuf, ResolveError> {
    let client = HFClient::new()
        .map_err(|e| ResolveError::HuggingFace(format!("failed to create HFClient: {e}")))?;

    let (owner, name) = split_model_id(model_id);
    let repo = client.model(&owner, &name);
    let repo = &repo;

    fetch_encoder_decoder_export(model_id, |filename| async move {
        match repo.download_file().filename(filename).send().await {
            Ok(path) => Ok(Some(path)),
            Err(HFError::EntryNotFound { .. }) => Ok(None),
            Err(e) => Err(ResolveError::HuggingFace(e.to_string())),
        }
    })
    .await
}

/// Fetch the files of an encoder-decoder export through `fetch`.
///
/// `fetch` downloads one repository file and returns its local path, or
/// `None` when the repository does not have it. Files are requested in
/// this order, stopping at the first gap:
///
/// 1. `config.json` -- a Whisper model ends the lookup here, so its
///    resolution stays as it was before this lookup existed.
/// 2. `encoder_model.onnx`, under `onnx/` first and then at the repository
///    root (the order [`download_from_hf`] uses).
/// 3. One decoder next to the encoder: `decoder_model.onnx`, else
///    `decoder_model_merged.onnx`.
/// 4. `tokenizer.json` (required) and `generation_config.json` (optional).
///
/// Returns the snapshot root directory, or [`ResolveError::NoOnnxExport`]
/// when the repository holds no complete encoder-decoder export.
async fn fetch_encoder_decoder_export<F, Fut>(
    model_id: &str,
    mut fetch: F,
) -> Result<PathBuf, ResolveError>
where
    F: FnMut(String) -> Fut,
    Fut: Future<Output = Result<Option<PathBuf>, ResolveError>>,
{
    let no_onnx_export = || ResolveError::NoOnnxExport {
        model_id: model_id.to_string(),
    };

    let Some(config_path) = fetch("config.json".to_string()).await? else {
        return Err(no_onnx_export());
    };
    let config_data = tokio::fs::read(&config_path).await?;
    if is_whisper_config(&config_data) {
        tracing::debug!(
            model_id,
            "skipping encoder-decoder lookup for Whisper model"
        );
        return Err(no_onnx_export());
    }

    for prefix in [ONNX_SUBDIR_PREFIX, ""] {
        let Some(encoder_path) = fetch(format!("{prefix}{ENCODER_ONNX}")).await? else {
            continue;
        };
        let mut has_decoder = false;
        for decoder in DECODER_ONNX_CANDIDATES {
            if fetch(format!("{prefix}{decoder}")).await?.is_some() {
                has_decoder = true;
                break;
            }
        }
        if !has_decoder {
            continue;
        }

        if fetch("tokenizer.json".to_string()).await?.is_none() {
            return Err(ResolveError::HuggingFace(
                "failed to download tokenizer.json: entry not found".to_string(),
            ));
        }
        // Optional: generation settings may live in config.json instead.
        if let Err(e) = fetch(GENERATION_CONFIG_JSON.to_string()).await {
            tracing::warn!(model_id, error = %e, "failed to download optional generation_config.json");
        }

        // encoder_path is {snapshot_root}/onnx/encoder_model.onnx or
        // {snapshot_root}/encoder_model.onnx.
        let graph_dir = encoder_path.parent();
        let snapshot_root = if prefix.is_empty() {
            graph_dir
        } else {
            graph_dir.and_then(|dir| dir.parent())
        };
        return snapshot_root.map(PathBuf::from).ok_or_else(|| {
            ResolveError::HuggingFace(format!(
                "unexpected ONNX path structure: {}",
                encoder_path.display()
            ))
        });
    }

    Err(no_onnx_export())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_model_id_with_org() {
        let (owner, name) = split_model_id("Xenova/distilbert");
        assert_eq!(owner, "Xenova");
        assert_eq!(name, "distilbert");
    }

    #[test]
    fn split_model_id_without_org() {
        let (owner, name) = split_model_id("bert-base");
        assert_eq!(owner, "bert-base");
        assert_eq!(name, "bert-base");
    }

    #[test]
    fn split_model_id_with_multiple_slashes() {
        // Only split on the first slash
        let (owner, name) = split_model_id("org/sub/model");
        assert_eq!(owner, "org");
        assert_eq!(name, "sub/model");
    }

    // --- fetch_encoder_decoder_export tests ---

    use std::cell::RefCell;

    use tempfile::TempDir;

    /// A fake repository snapshot on disk plus a log of requested files.
    struct FakeRepo {
        root: TempDir,
        requested: RefCell<Vec<String>>,
    }

    impl FakeRepo {
        fn with_files(files: &[&str]) -> Self {
            let root = TempDir::new().unwrap();
            for file in files {
                let path = root.path().join(file);
                std::fs::create_dir_all(path.parent().unwrap()).unwrap();
                let content: &[u8] = if *file == "config.json" {
                    br#"{"architectures": ["BartForConditionalGeneration"]}"#
                } else {
                    b"data"
                };
                std::fs::write(path, content).unwrap();
            }
            Self {
                root,
                requested: RefCell::new(Vec::new()),
            }
        }

        async fn resolve(&self) -> Result<PathBuf, ResolveError> {
            fetch_encoder_decoder_export("org/model", |filename| async move {
                self.requested.borrow_mut().push(filename.clone());
                let path = self.root.path().join(&filename);
                Ok(path.exists().then_some(path))
            })
            .await
        }

        fn requested(&self) -> Vec<String> {
            self.requested.borrow().clone()
        }
    }

    #[tokio::test]
    async fn encoder_decoder_export_in_onnx_subdirectory_resolves_to_snapshot_root() {
        // Arrange -- the layout Xenova-style repositories publish.
        let repo = FakeRepo::with_files(&[
            "config.json",
            "generation_config.json",
            "tokenizer.json",
            "onnx/encoder_model.onnx",
            "onnx/decoder_model.onnx",
            "onnx/decoder_model_merged.onnx",
        ]);

        // Act
        let root = repo.resolve().await.expect("export should resolve");

        // Assert -- one decoder fetched, and it is the plain one.
        assert_eq!(root, repo.root.path());
        assert_eq!(
            repo.requested(),
            vec![
                "config.json",
                "onnx/encoder_model.onnx",
                "onnx/decoder_model.onnx",
                "tokenizer.json",
                "generation_config.json",
            ]
        );
    }

    #[tokio::test]
    async fn encoder_decoder_export_at_repository_root_resolves_to_snapshot_root() {
        // Arrange -- merged decoder only, no generation_config.json.
        let repo = FakeRepo::with_files(&[
            "config.json",
            "tokenizer.json",
            "encoder_model.onnx",
            "decoder_model_merged.onnx",
        ]);

        // Act
        let root = repo.resolve().await.expect("export should resolve");

        // Assert
        assert_eq!(root, repo.root.path());
        assert_eq!(
            repo.requested(),
            vec![
                "config.json",
                "onnx/encoder_model.onnx",
                "encoder_model.onnx",
                "decoder_model.onnx",
                "decoder_model_merged.onnx",
                "tokenizer.json",
                "generation_config.json",
            ]
        );
    }

    #[tokio::test]
    async fn repository_without_encoder_reports_no_onnx_export() {
        // Arrange -- a PyTorch-only repository.
        let repo = FakeRepo::with_files(&["config.json", "tokenizer.json"]);

        // Act
        let result = repo.resolve().await;

        // Assert -- the error that sends the resolver on to Forge.
        assert!(matches!(result, Err(ResolveError::NoOnnxExport { .. })));
    }

    #[tokio::test]
    async fn encoder_without_decoder_reports_no_onnx_export() {
        // Arrange
        let repo =
            FakeRepo::with_files(&["config.json", "tokenizer.json", "onnx/encoder_model.onnx"]);

        // Act
        let result = repo.resolve().await;

        // Assert
        assert!(matches!(result, Err(ResolveError::NoOnnxExport { .. })));
    }

    #[tokio::test]
    async fn whisper_repository_reports_no_onnx_export_without_fetching_graphs() {
        // Arrange -- Whisper ships the same file layout.
        let repo = FakeRepo::with_files(&[
            "tokenizer.json",
            "onnx/encoder_model.onnx",
            "onnx/decoder_model.onnx",
        ]);
        std::fs::write(
            repo.root.path().join("config.json"),
            br#"{"architectures": ["WhisperForConditionalGeneration"]}"#,
        )
        .unwrap();

        // Act
        let result = repo.resolve().await;

        // Assert -- as before this lookup existed, and nothing large fetched.
        assert!(matches!(result, Err(ResolveError::NoOnnxExport { .. })));
        assert_eq!(repo.requested(), vec!["config.json"]);
    }

    #[tokio::test]
    async fn export_without_tokenizer_is_an_error() {
        // Arrange
        let repo = FakeRepo::with_files(&[
            "config.json",
            "onnx/encoder_model.onnx",
            "onnx/decoder_model.onnx",
        ]);

        // Act
        let result = repo.resolve().await;

        // Assert -- same class of error as a single-file model without one.
        let err = result.expect_err("missing tokenizer should fail");
        assert!(
            matches!(err, ResolveError::HuggingFace(_)),
            "unexpected error: {err:?}"
        );
        assert!(
            err.to_string().contains("tokenizer.json"),
            "unexpected error: {err}"
        );
    }

    #[tokio::test]
    async fn fetch_failure_is_propagated() {
        // Arrange / Act
        let result = fetch_encoder_decoder_export("org/model", |_| async {
            Err(ResolveError::HuggingFace(
                "connection timed out".to_string(),
            ))
        })
        .await;

        // Assert
        assert!(matches!(result, Err(ResolveError::HuggingFace(_))));
    }
}
