//! Error types for the core inference pipeline.

use thiserror::Error;

/// Errors produced by the core inference pipeline.
#[derive(Error, Debug)]
pub enum CoreError {
    /// Failed to tokenize input text.
    #[error("tokenization failed: {0}")]
    Tokenization(String),

    /// Failed to run ONNX inference.
    #[error("inference failed: {0}")]
    Inference(String),

    /// Failed to load an ONNX model file.
    #[error("model load failed: {0}")]
    ModelLoad(String),

    /// Model validation failed (e.g., tokenizer-model input mismatch).
    #[error("model validation failed: {0}")]
    ModelValidation(String),

    /// Invalid input type for the current pipeline profile.
    #[error("invalid input: {0}")]
    InvalidInput(String),

    /// Configuration error.
    #[error("config error: {0}")]
    Config(String),

    /// Inference ran past its time budget and was abandoned.
    #[error("inference timed out: {0}")]
    Timeout(String),

    /// The caller cancelled a generation between two token steps.
    #[error("inference cancelled: {0}")]
    Cancelled(String),

    /// The prompt has more tokens than the pipeline accepts.
    #[error("input too long: the prompt has {tokens} tokens, the limit is {limit}")]
    InputTooLong {
        /// Tokens in the prompt, template and system message included.
        tokens: usize,
        /// The pipeline's cap on prompt tokens.
        limit: usize,
    },

    /// I/O error reading model files.
    #[error("i/o error")]
    Io(#[from] std::io::Error),

    /// Failed to parse JSON configuration (e.g., config.json).
    #[error("json parse error")]
    JsonParse(#[from] serde_json::Error),
}
