//! Integration test for batched text generation through the public API
//! and the real router: responses, and the batch and token metrics, over
//! the batch fixture graph of `hephaestus-core`.

use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use hephaestus_api::{AppState, build_router};
use hephaestus_core::{
    ExecutionProvider, PipelineKind, TextGenerationOptions, TextGenerationPipeline,
};
use metrics_exporter_prometheus::PrometheusHandle;
use tower::ServiceExt;

const TEXTS: [&str; 8] = [
    "h i",
    "a",
    "l k j i h g",
    "a b c d e f g h i j k l a b c",
    "k",
    "b b",
    "e f g h i j k l",
    "j k l a",
];

/// Value of the sample of `metric` whose line contains every `label`.
fn sample(exposition: &str, metric: &str, labels: &[&str]) -> Option<f64> {
    exposition
        .lines()
        .filter(|line| !line.starts_with('#'))
        .filter(|line| line.starts_with(metric))
        .filter(|line| labels.iter().all(|label| line.contains(label)))
        .find_map(|line| line.rsplit(' ').next()?.parse().ok())
}

/// A ready server around the batch fixture graph, `width` generations
/// wide.
fn server(
    width: usize,
    model_id: &str,
    metrics_handle: PrometheusHandle,
) -> (Arc<AppState>, tempfile::TempDir) {
    let fixtures = Path::new(env!("CARGO_MANIFEST_DIR")).join("../hephaestus-core/tests/fixtures");
    let dir = tempfile::tempdir().expect("temp dir");
    for (from, file, to) in [
        ("text_generation", "model_batch.onnx", "model.onnx"),
        (
            "text_generation",
            "tokenizer_config.json",
            "tokenizer_config.json",
        ),
        ("seq2seq", "tokenizer.json", "tokenizer.json"),
    ] {
        std::fs::copy(fixtures.join(from).join(file), dir.path().join(to)).expect("copy fixture");
    }
    let config = serde_json::json!({
        "architectures": ["LlamaForCausalLM"],
        "bos_token_id": 0,
        "eos_token_id": 2,
        "max_position_embeddings": 512,
        "vocab_size": 16
    });
    std::fs::write(dir.path().join("config.json"), config.to_string()).expect("write config");
    let options = TextGenerationOptions::new()
        .with_max_new_tokens(40)
        .with_max_batch(width);
    let pipeline =
        TextGenerationPipeline::with_options(dir.path(), &ExecutionProvider::Cpu, &options)
            .expect("pipeline should load");
    let state = Arc::new(AppState::new(
        PipelineKind::TextGeneration(pipeline),
        model_id.to_string(),
        Duration::from_secs(30),
        metrics_handle,
        None,
        30.0,
        5.0,
    ));
    state.set_ready(true);
    (state, dir)
}

/// The `/infer` response bodies for `TEXTS`, sent all at once, without
/// their latencies.
async fn infer_all(state: &Arc<AppState>) -> Vec<serde_json::Value> {
    let app = build_router(Arc::clone(state));
    let requests: Vec<_> = TEXTS
        .iter()
        .map(|text| {
            let request = Request::post("/infer")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::json!({ "text": text }).to_string()))
                .expect("request should build");
            tokio::spawn(app.clone().oneshot(request))
        })
        .collect();
    let mut bodies = Vec::new();
    for request in requests {
        let response = request
            .await
            .expect("task should join")
            .expect("infallible");
        assert_eq!(response.status(), StatusCode::OK);
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("body");
        let mut body: serde_json::Value = serde_json::from_slice(&bytes).expect("json");
        let fields = body.as_object_mut().expect("object");
        fields.remove("latency_ms");
        fields.remove("model_id");
        bodies.push(body);
    }
    bodies
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn batched_server_answers_like_an_unbatched_one_and_counts_its_passes() {
    // Arrange -- the global recorder (one per test binary), a server that
    // generates one request at a time and one that runs four together.
    let metrics_handle = hephaestus_api::install_recorder().expect("install recorder");
    let (solo, _solo_dir) = server(1, "fixture/solo", metrics_handle.clone());
    let (batched, _batched_dir) = server(4, "fixture/batched", metrics_handle);
    let model = "model_id=\"fixture/batched\"";

    // Act
    let expected = infer_all(&solo).await;
    let bodies = infer_all(&batched).await;
    let metrics = batched.render_metrics();

    // Assert -- the same text and token counts, request by request.
    assert_eq!(bodies, expected);
    let count = |body: &serde_json::Value, field: &str| body[field].as_u64().expect("count") as f64;
    let prompt_tokens: f64 = bodies.iter().map(|body| count(body, "prompt_tokens")).sum();
    let completion_tokens: f64 = bodies
        .iter()
        .map(|body| count(body, "completion_tokens"))
        .sum();
    let requests = TEXTS.len() as f64;
    assert_eq!(
        sample(&metrics, "hephaestus_prompt_tokens_total", &[model]),
        Some(prompt_tokens),
        "{metrics}"
    );
    assert_eq!(
        sample(&metrics, "hephaestus_generated_tokens_total", &[model]),
        Some(completion_tokens),
        "{metrics}"
    );
    // Every prompt is read in exactly one prompt pass, which also yields
    // the request's first token; every later token comes from one row of
    // one decode pass.
    let passes = "hephaestus_generation_batch_sequences";
    assert_eq!(
        sample(
            &metrics,
            &format!("{passes}_sum"),
            &[model, "pass=\"prompt\""]
        ),
        Some(requests),
        "{metrics}"
    );
    assert_eq!(
        sample(
            &metrics,
            &format!("{passes}_sum"),
            &[model, "pass=\"decode\""]
        ),
        Some(completion_tokens - requests),
        "{metrics}"
    );
    // No pass held more than the width.
    for pass in ["pass=\"prompt\"", "pass=\"decode\""] {
        let widest = sample(&metrics, passes, &[model, pass, "quantile=\"1\""]).expect("quantile");
        assert!((1.0..=4.0).contains(&widest), "{pass}: {widest}");
    }
    assert_eq!(
        sample(&metrics, "hephaestus_generation_active_sequences", &[model]),
        Some(0.0),
        "{metrics}"
    );
    assert_eq!(
        sample(&metrics, "hephaestus_inference_queue_depth", &[model]),
        Some(0.0),
        "{metrics}"
    );
    // The unbatched server recorded no batch passes at all.
    assert_eq!(
        sample(
            &metrics,
            &format!("{passes}_count"),
            &["model_id=\"fixture/solo\""]
        ),
        None,
        "{metrics}"
    );
}
