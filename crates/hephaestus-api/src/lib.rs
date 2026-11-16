//! HTTP serving layer for the Hephaestus unified ONNX inference runtime.
//!
//! This crate provides the REST API surface, health probes,
//! graceful shutdown, and error mapping for Hephaestus.
//! Handlers access the inference pipeline through shared [`AppState`]
//! and the router is constructed via [`build_router`].

pub mod batcher;
pub mod entitlement;
pub mod error;
pub mod grpc;
pub mod handlers;
pub mod metrics;
pub mod routes;
pub mod state;
pub mod telemetry;
pub mod worker;
pub mod ws;

pub use batcher::{Batcher, batcher_loop};
pub use metrics::{StageTimer, install_recorder};
pub use routes::build_router;
pub use state::AppState;
pub use worker::{CancelFlag, DEFAULT_QUEUE_CAPACITY, InferenceTicket};
