//! Scylla entitlement guard for the whole application.
//!
//! [`guard`] wraps the complete router, REST and gRPC alike, so every
//! request is checked against scylla by default, including any route or
//! gRPC service added later. Only [`EXEMPT_PATHS`] skip the check. The
//! tenant is read from the [`TENANT_HEADER`] header (gRPC metadata are the
//! same headers), which the calling service sets.
//!
//! Refusals are plain HTTP statuses (401, 403, 503). gRPC clients map
//! those to `UNAUTHENTICATED`, `PERMISSION_DENIED` and `UNAVAILABLE` per
//! the gRPC HTTP-status mapping, so one guard serves both protocols.

use axum::body::Body;
use axum::http::{Request, Response, StatusCode};
use axum::response::IntoResponse;
use axum::{Json, Router};
use scylla_client::{EntitlementLayer, Refusal, ScyllaClient, ServiceName, TenantId};

/// Header carrying the calling tenant's ID.
pub const TENANT_HEADER: &str = "x-tenant-id";

/// Paths that skip the entitlement check: the k8s probes, the Prometheus
/// scrape and the gRPC health service. None of them carry a tenant.
pub const EXEMPT_PATHS: &[&str] = &[
    "/healthz/live",
    "/healthz/ready",
    "/metrics",
    "/grpc.health.v1.Health/Check",
    "/grpc.health.v1.Health/Watch",
];

/// Wrap `app` so every request except [`EXEMPT_PATHS`] passes the scylla
/// entitlement check for `service` before reaching its handler.
#[must_use]
pub fn guard(app: Router, client: ScyllaClient, service: ServiceName) -> Router {
    let guarded = app
        .clone()
        .layer(EntitlementLayer::new(client, service, tenant_id, refusal_response));
    EXEMPT_PATHS.iter().fold(
        Router::new().fallback_service(guarded.into_service::<Body>()),
        |router, path| router.route_service(path, app.clone().into_service::<Body>()),
    )
}

/// Read the tenant from [`TENANT_HEADER`]; absent or unparseable is `None`.
fn tenant_id(req: &Request<Body>) -> Option<TenantId> {
    let value = req.headers().get(TENANT_HEADER)?.to_str().ok()?;
    TenantId::parse(value).ok()
}

/// Map a refusal to the API's JSON error shape. Never echoes the deny
/// reason or meter: they are internal.
fn refusal_response(refusal: Refusal) -> Response<Body> {
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
