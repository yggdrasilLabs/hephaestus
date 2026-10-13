//! axum router construction for the Hephaestus HTTP API.
//!
//! The [`build_router`] function mounts all HTTP endpoints and
//! attaches shared [`AppState`] as axum state.

use std::sync::Arc;

use axum::routing::{get, post};
use axum::Router;
use tower_http::limit::RequestBodyLimitLayer;
use tower_http::trace::TraceLayer;

use crate::handlers;
use crate::metrics;
use crate::state::AppState;
use crate::ws;

/// Maximum request body size (1 MB).
///
/// Protects against oversized payloads that could exhaust memory.
/// Text inference requests should be well under this limit.
const MAX_BODY_SIZE: usize = 1024 * 1024;

/// Construct the axum router with all HTTP endpoints.
///
/// Mounts:
/// - `POST /infer` -- text classification inference (D-01)
/// - `GET /ws/transcribe` -- WebSocket streaming transcription (D-05)
/// - `GET /healthz/live` -- liveness probe (D-05)
/// - `GET /healthz/ready` -- readiness probe (D-05)
/// - `GET /metrics` -- Prometheus scrape endpoint (OBSV-01)
///
/// Applies a 1 MB request body size limit to prevent oversized payloads.
/// WebSocket upgrade requests are GET requests with no body, so the limit
/// layer does not interfere with the upgrade handshake.
pub fn build_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/infer", post(handlers::infer))
        .route("/ws/transcribe", get(ws::ws_transcribe))
        .route("/healthz/live", get(handlers::liveness))
        .route("/healthz/ready", get(handlers::readiness))
        .route("/metrics", get(metrics::metrics_handler))
        .layer(RequestBodyLimitLayer::new(MAX_BODY_SIZE))
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}
