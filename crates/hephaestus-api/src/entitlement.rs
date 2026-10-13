//! Scylla entitlement guard for the inference endpoints.
//!
//! Every REST, WebSocket and gRPC inference request is checked against
//! scylla before it reaches a handler. The tenant is read from the
//! [`TENANT_HEADER`] header (gRPC metadata are the same headers), which
//! the calling service sets. Health, metrics and reflection are never
//! guarded.

use axum::Json;
use axum::http::{Request, Response, StatusCode};
use axum::response::IntoResponse;
use scylla_client::{EntitlementLayer, Refusal, ScyllaClient, ServiceName, TenantId};

/// Header carrying the calling tenant's ID.
pub const TENANT_HEADER: &str = "x-tenant-id";

/// Entitlement layer for the REST and WebSocket inference routes.
pub type HttpGuard = EntitlementLayer<
    fn(&Request<axum::body::Body>) -> Option<TenantId>,
    fn(Refusal) -> Response<axum::body::Body>,
>;

/// Entitlement layer for the gRPC `InferenceService`.
pub type GrpcGuard = EntitlementLayer<
    fn(&Request<tonic::body::Body>) -> Option<TenantId>,
    fn(Refusal) -> Response<tonic::body::Body>,
>;

/// Build the guard for the REST and WebSocket inference routes.
#[must_use]
pub fn http_guard(client: ScyllaClient, service: ServiceName) -> HttpGuard {
    EntitlementLayer::new(
        client,
        service,
        tenant_id::<axum::body::Body> as fn(&Request<axum::body::Body>) -> Option<TenantId>,
        http_refusal as fn(Refusal) -> Response<axum::body::Body>,
    )
}

/// Build the guard for the gRPC `InferenceService`.
#[must_use]
pub fn grpc_guard(client: ScyllaClient, service: ServiceName) -> GrpcGuard {
    EntitlementLayer::new(
        client,
        service,
        tenant_id::<tonic::body::Body> as fn(&Request<tonic::body::Body>) -> Option<TenantId>,
        grpc_refusal as fn(Refusal) -> Response<tonic::body::Body>,
    )
}

/// Read the tenant from [`TENANT_HEADER`]; absent or unparseable is `None`.
fn tenant_id<B>(req: &Request<B>) -> Option<TenantId> {
    let value = req.headers().get(TENANT_HEADER)?.to_str().ok()?;
    TenantId::parse(value).ok()
}

/// Map a refusal to the API's JSON error shape. Never echoes the deny
/// reason or meter: they are internal.
fn http_refusal(refusal: Refusal) -> Response<axum::body::Body> {
    let (status, code, message) = match refusal {
        Refusal::Denied { .. } => (
            StatusCode::FORBIDDEN,
            "NOT_ENTITLED",
            "tenant is not entitled to this service",
        ),
        Refusal::NoTenant => (
            StatusCode::UNAUTHORIZED,
            "MISSING_TENANT",
            "missing or invalid x-tenant-id header",
        ),
        Refusal::NoBanList => (
            StatusCode::SERVICE_UNAVAILABLE,
            "NOT_READY",
            "service not ready",
        ),
    };
    let body = serde_json::json!({
        "error": {
            "code": code,
            "message": message,
        }
    });
    (status, Json(body)).into_response()
}

/// Map a refusal to a gRPC status. Never echoes the deny reason or meter:
/// they are internal.
fn grpc_refusal(refusal: Refusal) -> Response<tonic::body::Body> {
    let status = match refusal {
        Refusal::Denied { .. } => {
            tonic::Status::permission_denied("tenant is not entitled to this service")
        }
        Refusal::NoTenant => {
            tonic::Status::unauthenticated("missing or invalid x-tenant-id metadata")
        }
        Refusal::NoBanList => tonic::Status::unavailable("service not ready"),
    };
    status.into_http()
}
