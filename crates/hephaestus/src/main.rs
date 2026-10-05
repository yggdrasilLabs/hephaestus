//! Hephaestus binary entry point.
//!
//! Startup sequence: load config from env vars, initialize tracing,
//! detect model profile, construct the appropriate pipeline, run a
//! warmup inference pass, flip readiness, and start the multiplexed
//! gRPC + HTTP/REST server with graceful shutdown.

mod config;

use std::sync::Arc;
use std::time::Duration;

use anyhow::Context;
use hephaestus_api::{AppState, Batcher, batcher_loop, build_router};
use hephaestus_api::grpc::GrpcInferenceService;
use hephaestus_core::{
    AsrPipeline, ClassifierPipeline, EmbeddingsPipeline, ExecutionProvider, ModelProfile,
    PipelineKind, Seq2SeqPipeline, TokenClassifierPipeline, detect_profile,
};
use hephaestus_resolve::{HttpForgeClient, ModelResolver};

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    // 1. Load typed configuration from environment variables.
    //    Config must be loaded before tracing init so we can use LOG_LEVEL.
    let config = config::Config::from_env()?;
    config.validate()?;
    let ep: ExecutionProvider = config.parsed_execution_provider()?;

    // 2. Initialize telemetry: structured JSON logging + conditional OTel export (D-11).
    //    Must be called inside the tokio runtime (after #[tokio::main]) because the
    //    OTel batch span processor spawns a background tokio task (Pitfall 1).
    hephaestus_api::telemetry::init(
        &config.log_level,
        config.otel_exporter_otlp_endpoint.as_deref(),
    )?;
    tracing::info!(
        model_id = %config.model_id,
        execution_provider = %ep,
        port = config.port,
        request_timeout_secs = config.request_timeout_secs,
        shutdown_timeout_secs = config.shutdown_timeout_secs,
        storage_type = %config.storage_type,
        storage_bucket = ?config.storage_bucket,
        storage_prefix = ?config.storage_prefix,
        storage_credential_path = ?config.storage_credential_path,
        forge_url = ?config.forge_url,
        forge_timeout_secs = config.forge_timeout_secs,
        model_profile = ?config.model_profile,
        feature_extractor = %config.feature_extractor,
        chunking_strategy = %config.chunking_strategy,
        window_size_secs = config.window_size_secs,
        overlap_secs = config.overlap_secs,
        "configuration loaded"
    );

    // 2b. Install Prometheus metrics recorder (OBSV-01).
    let metrics_handle = hephaestus_api::install_recorder()?;
    tracing::info!("prometheus metrics recorder installed");

    // 2c. Build OpenDAL storage operator from config (D-01, D-02, D-05).
    let operator = config.storage_operator()?;
    tracing::info!(storage_type = %config.storage_type, "storage operator constructed");

    // 3. Resolve model directory: local override (MODEL_PATH) or automatic resolution.
    let model_dir = if config.model_path.is_some() {
        // Local path override -- preserves backward compatibility.
        config.model_dir()?
    } else {
        // Automatic resolution: storage cache -> HuggingFace -> Forge (RSLV-05).
        // When FORGE_URL is set, use HttpForgeClient; otherwise StubForgeClient.
        // The two branches produce different generic types, so we resolve
        // inside each branch and return the PathBuf.
        if let Some(ref forge_url) = config.forge_url {
            let forge_client = HttpForgeClient::new(forge_url, config.forge_timeout_secs)
                .context("failed to create Forge HTTP client")?;
            let resolver = ModelResolver::new_with_client(
                operator.clone(),
                forge_client,
            )
            .await
            .context("failed to construct model resolver")?;

            resolver
                .resolve(&config.model_id)
                .await
                .context("failed to resolve model")?
        } else {
            let resolver = ModelResolver::new_with_stub(
                operator.clone(),
            )
            .await
            .context("failed to construct model resolver")?;

            resolver
                .resolve(&config.model_id)
                .await
                .context("failed to resolve model")?
        }
    };
    tracing::info!(
        model_id = %config.model_id,
        model_dir = %model_dir.display(),
        "model directory resolved"
    );

    // 3b. Detect model profile from config.json (D-01, D-02).
    let config_json_path = model_dir.join("config.json");
    let config_json_text = std::fs::read_to_string(&config_json_path)
        .context("failed to read config.json from model directory")?;
    let model_config: serde_json::Value = serde_json::from_str(&config_json_text)
        .context("failed to parse config.json")?;
    let profile = detect_profile(&model_config, config.model_profile.as_deref())
        .context("failed to detect model profile")?;
    tracing::info!(
        model_id = %config.model_id,
        profile = ?profile,
        "model profile detected"
    );

    // 4. Construct the appropriate pipeline based on detected profile (D-03).
    let pipeline_kind = match profile {
        ModelProfile::Classifier => {
            let pipeline = ClassifierPipeline::new(&model_dir, &ep)
                .context("failed to construct classifier pipeline")?;
            tracing::info!("classifier pipeline constructed");
            PipelineKind::Classifier(pipeline)
        }
        ModelProfile::Embeddings => {
            let pipeline = EmbeddingsPipeline::new(&model_dir, &ep)
                .context("failed to construct embeddings pipeline")?;
            tracing::info!("embeddings pipeline constructed");
            PipelineKind::Embeddings(pipeline)
        }
        ModelProfile::Seq2Seq => {
            let pipeline = Seq2SeqPipeline::new(&model_dir, &ep)
                .context("failed to construct seq2seq pipeline")?;
            tracing::info!("seq2seq pipeline constructed");
            PipelineKind::Seq2Seq(pipeline)
        }
        ModelProfile::TokenClassifier => {
            let pipeline = TokenClassifierPipeline::new(&model_dir, &ep)
                .context("failed to construct token classifier pipeline")?;
            tracing::info!("token classifier pipeline constructed");
            PipelineKind::TokenClassifier(pipeline)
        }
        ModelProfile::Asr => {
            let pipeline = AsrPipeline::new(&model_dir, &ep, &config.feature_extractor)
                .context("failed to construct ASR pipeline")?;
            tracing::info!(
                feature_extractor = %config.feature_extractor,
                chunking_strategy = %config.chunking_strategy,
                "asr pipeline constructed"
            );
            PipelineKind::Asr(pipeline)
        }
    };

    // 5. Build shared state with optional batcher (D-07).
    let batcher_handle = if config.batch_enabled {
        let (batcher, receiver) = Batcher::new(config.batch_max_size as usize);
        Some((batcher, receiver))
    } else {
        None
    };

    let (batcher_opt, batcher_rx) = match batcher_handle {
        Some((batcher, receiver)) => (Some(batcher), Some(receiver)),
        None => (None, None),
    };

    let state = Arc::new(AppState::new(
        pipeline_kind,
        config.model_id.clone(),
        Duration::from_secs(config.request_timeout_secs),
        metrics_handle,
        batcher_opt,
        config.window_size_secs,
        config.overlap_secs,
    ));

    // 5b. Spawn batcher background task if batching is enabled (D-06).
    if let Some(receiver) = batcher_rx {
        let batcher_state = state.clone();
        let max_batch_size = config.batch_max_size as usize;
        let max_wait = Duration::from_millis(config.batch_max_wait_ms);
        tokio::spawn(batcher_loop(receiver, batcher_state, max_batch_size, max_wait));
        tracing::info!(
            batch_max_size = config.batch_max_size,
            batch_max_wait_ms = config.batch_max_wait_ms,
            "dynamic batching enabled"
        );
    } else {
        tracing::info!("dynamic batching disabled");
    }

    // 5c. Create gRPC health reporter (SC-03).
    //     HealthReporter defaults "" to SERVING; override to NOT_SERVING
    //     until warmup completes so gRPC health probes are accurate.
    let (health_reporter, health_service) = tonic_health::server::health_reporter();
    health_reporter
        .set_service_status("", tonic_health::ServingStatus::NotServing)
        .await;
    health_reporter
        .set_service_status(
            "hephaestus.v1.InferenceService",
            tonic_health::ServingStatus::NotServing,
        )
        .await;

    // 6. Run warmup inference pass (CORE-03), then flip readiness.
    //    Warmup is a performance optimization (pre-warms caches), not a
    //    correctness gate. Failure logs a warning but does not crash the pod.
    //    ASR models require audio input for warmup, which is not available
    //    at startup -- skip warmup for ASR profiles.
    if profile == ModelProfile::Asr {
        tracing::info!(
            model_id = %config.model_id,
            "skipping warmup for ASR profile (no audio warmup implemented)"
        );
    } else {
        let warmup_text = config
            .warmup_input
            .as_deref()
            .unwrap_or("This is a warmup inference pass.");
        // Read lock for prepare (tokenization), write lock for execute (inference).
        // Mirrors the handler's read/write split pattern (SC-02).
        let prepared = {
            let pipeline = state.read_pipeline().await;
            pipeline.prepare(warmup_text.to_string())
        };
        match prepared {
            Ok(prepared) => {
                let mut pipeline = state.write_pipeline().await;
                match pipeline.execute(prepared) {
                    Ok(_output) => {
                        tracing::info!(
                            model_id = %config.model_id,
                            "warmup inference complete"
                        );
                    }
                    Err(e) => {
                        tracing::warn!(
                            model_id = %config.model_id,
                            error = %e,
                            "warmup inference failed, continuing without warmup"
                        );
                    }
                }
            }
            Err(e) => {
                tracing::warn!(
                    model_id = %config.model_id,
                    error = %e,
                    "warmup prepare failed, continuing without warmup"
                );
            }
        }
    }
    state.set_ready(true);
    health_reporter
        .set_service_status("", tonic_health::ServingStatus::Serving)
        .await;
    health_reporter
        .set_service_status(
            "hephaestus.v1.InferenceService",
            tonic_health::ServingStatus::Serving,
        )
        .await;
    tracing::info!("warmup complete, readiness enabled");

    // 7. Start HTTP server with graceful shutdown.
    let addr = format!("0.0.0.0:{}", config.port);
    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .context("failed to bind TCP listener")?;
    tracing::info!(%addr, "listening");

    // 7a. Build gRPC router: InferenceService, health, and reflection (SC-01..SC-04).
    let reflection_service = tonic_reflection::server::Builder::configure()
        .register_encoded_file_descriptor_set(hephaestus_proto::FILE_DESCRIPTOR_SET)
        .register_encoded_file_descriptor_set(tonic_health::pb::FILE_DESCRIPTOR_SET)
        .build_v1()
        .context("failed to build gRPC reflection service")?;

    let inference_service =
        hephaestus_proto::v1::inference_service_server::InferenceServiceServer::new(
            GrpcInferenceService::new(state.clone()),
        );

    let grpc_router = tonic::service::Routes::new(inference_service)
        .add_service(health_service)
        .add_service(reflection_service)
        .into_axum_router();

    // 7b. Build REST router and merge with gRPC (SC-06: existing REST unchanged).
    let rest_router = build_router(state.clone());
    let app = rest_router.merge(grpc_router);

    // Drain-timeout watchdog (D-13).
    //
    // The serve future uses `with_graceful_shutdown` to begin draining
    // on SIGTERM/Ctrl-C.  A `select!` on the serve future itself acts
    // as the hard-kill safety net: if in-flight connections do not
    // close within `shutdown_timeout` after readiness flips to false,
    // the watchdog branch completes and the serve future is cancelled.
    let shutdown_timeout = Duration::from_secs(config.shutdown_timeout_secs);
    let server_state = state.clone();
    let watchdog_state = state.clone();
    let serve_fut = axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal(server_state, health_reporter));

    tokio::select! {
        result = serve_fut => {
            result.context("HTTP server error")?;
        }
        () = async {
            // Poll until readiness is flipped to false (shutdown signal received).
            loop {
                if !watchdog_state.is_ready() { break; }
                tokio::time::sleep(Duration::from_millis(200)).await;
            }
            // Grace period for in-flight requests to drain.
            tokio::time::sleep(shutdown_timeout).await;
            tracing::warn!(
                timeout_secs = shutdown_timeout.as_secs(),
                "drain timeout exceeded, forcing shutdown"
            );
        } => {
            // serve_fut is cancelled -- server stops immediately.
        }
    }

    tracing::info!("server shut down");

    // 8. Flush pending OTel spans before exit.
    hephaestus_api::telemetry::shutdown();

    Ok(())
}

/// Wait for a shutdown signal (Ctrl-C or SIGTERM).
///
/// On signal receipt, flips readiness to false so the k8s readiness
/// probe returns 503, updates the gRPC health reporter to NOT_SERVING,
/// and the load balancer stops routing new traffic while in-flight
/// requests drain (D-07, SC-03).
async fn shutdown_signal(
    state: Arc<AppState>,
    health_reporter: tonic_health::server::HealthReporter,
) {
    let ctrl_c = async {
        tokio::signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("failed to install SIGTERM handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        () = ctrl_c => {},
        () = terminate => {},
    }

    tracing::info!("shutdown signal received, draining connections");
    state.set_ready(false);
    health_reporter
        .set_service_status("", tonic_health::ServingStatus::NotServing)
        .await;
    health_reporter
        .set_service_status(
            "hephaestus.v1.InferenceService",
            tonic_health::ServingStatus::NotServing,
        )
        .await;
}
