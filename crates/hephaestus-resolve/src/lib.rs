//! Model resolution for the Hephaestus ONNX inference runtime.
//!
//! This crate implements the 3-tier model resolution chain: storage cache,
//! HuggingFace Hub, and Forge conversion. Callers interact only through
//! [`ModelResolver::resolve()`] -- all download, caching, and retry
//! details are hidden behind this single method (RSLV-05).

pub mod error;
pub mod forge;
pub(crate) mod hf;
pub mod resolver;
pub(crate) mod storage;

pub use error::ResolveError;
pub use forge::{ConversionMetadata, ForgeClient, ForgeResponse, HttpForgeClient, StubForgeClient};
pub use resolver::ModelResolver;
