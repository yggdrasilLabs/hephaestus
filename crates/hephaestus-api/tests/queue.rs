//! Integration tests for the inference queue: admission, the 503 for a
//! full queue, and the queue and token metrics, through the public API and
//! the real router, over the tiny seq2seq fixture graphs of
//! `hephaestus-core`.

use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use hephaestus_api::{AppState, build_router};
use hephaestus_core::{ExecutionProvider, PipelineKind, Seq2SeqPipeline};
use tower::ServiceExt;

fn infer_request(text: &str) -> Request<Body> {
    Request::post("/infer")
        .header("content-type", "application/json")
        .body(Body::from(serde_json::json!({ "text": text }).to_string()))
        .expect("request should build")
}

/// Value of the sample of `metric` whose line contains every `label`.
fn sample(exposition: &str, metric: &str, labels: &[&str]) -> Option<f64> {
    exposition
        .lines()
        .filter(|line| !line.starts_with('#'))
        .filter(|line| line.starts_with(metric))
        .filter(|line| labels.iter().all(|label| line.contains(label)))
        .find_map(|line| line.rsplit(' ').next()?.parse().ok())
}

#[tokio::test]
async fn queue_metrics_follow_admission_rejection_and_runs() {
    // Arrange -- the global recorder (one per test binary), and a state
    // with room for two waiting requests.
    let metrics_handle = hephaestus_api::install_recorder().expect("install recorder");
    let fixtures =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../hephaestus-core/tests/fixtures/seq2seq");
    let dir = tempfile::tempdir().expect("temp dir");
    for file in ["tokenizer.json", "encoder_model.onnx", "decoder_model.onnx"] {
        std::fs::copy(fixtures.join(file), dir.path().join(file)).expect("copy fixture");
    }
    let config = serde_json::json!({
        "architectures": ["BartForConditionalGeneration"],
        "bos_token_id": 0,
        "decoder_start_token_id": 2,
        "eos_token_id": 2,
        "max_length": 32,
        "max_position_embeddings": 64,
        "pad_token_id": 1,
        "vocab_size": 16
    });
    std::fs::write(dir.path().join("config.json"), config.to_string()).expect("write config");
    let pipeline =
        Seq2SeqPipeline::new(dir.path(), &ExecutionProvider::Cpu).expect("pipeline should load");
    let state = Arc::new(
        AppState::new(
            PipelineKind::Seq2Seq(pipeline),
            "fixture/queue".to_string(),
            Duration::from_secs(30),
            metrics_handle,
            None,
            30.0,
            5.0,
        )
        .with_inference_queue_capacity(2),
    );
    state.set_ready(true);
    let app = build_router(Arc::clone(&state));
    let model = "model_id=\"fixture/queue\"";

    // Act -- two callers hold the two places; a request finds none.
    let first = state.admit(None).expect("first place");
    let second = state.admit(None).expect("second place");
    let refused = app
        .clone()
        .oneshot(infer_request("a b c"))
        .await
        .expect("infallible");
    let while_full = state.render_metrics();
    // The places are given back; the next request runs.
    drop(first);
    drop(second);
    let served = app
        .clone()
        .oneshot(infer_request("a b c"))
        .await
        .expect("infallible");
    let afterwards = state.render_metrics();

    // Assert -- the refusal.
    assert_eq!(refused.status(), StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(refused.headers()["retry-after"], "1");
    assert_eq!(
        sample(&while_full, "hephaestus_inference_queue_depth", &[model]),
        Some(2.0),
        "{while_full}"
    );
    assert_eq!(
        sample(
            &while_full,
            "hephaestus_inference_jobs_rejected_total",
            &[model, "reason=\"queue_full\""]
        ),
        Some(1.0),
        "{while_full}"
    );
    // Nothing had run yet, so no queue wait was recorded.
    assert_eq!(
        sample(
            &while_full,
            "hephaestus_inference_queue_wait_seconds_count",
            &[model]
        ),
        None,
        "{while_full}"
    );

    // Assert -- the served request.
    assert_eq!(served.status(), StatusCode::OK);
    assert_eq!(
        sample(&afterwards, "hephaestus_inference_queue_depth", &[model]),
        Some(0.0),
        "{afterwards}"
    );
    assert_eq!(
        sample(
            &afterwards,
            "hephaestus_inference_queue_wait_seconds_count",
            &[model]
        ),
        Some(1.0),
        "{afterwards}"
    );
    assert_eq!(
        sample(
            &afterwards,
            "hephaestus_inference_jobs_rejected_total",
            &[model, "reason=\"queue_full\""]
        ),
        Some(1.0),
        "{afterwards}"
    );

    // Assert -- its tokens: <s> a b c </s> read, "a b c" </s> generated.
    let body = axum::body::to_bytes(served.into_body(), usize::MAX)
        .await
        .expect("body should read");
    let body: serde_json::Value = serde_json::from_slice(&body).expect("JSON body");
    assert_eq!(body["prompt_tokens"], 5);
    assert_eq!(body["completion_tokens"], 4);
    assert_eq!(
        sample(&afterwards, "hephaestus_prompt_tokens_total", &[model]),
        Some(5.0),
        "{afterwards}"
    );
    assert_eq!(
        sample(&afterwards, "hephaestus_generated_tokens_total", &[model]),
        Some(4.0),
        "{afterwards}"
    );
    assert_eq!(
        sample(
            &afterwards,
            "hephaestus_generation_tokens_per_second_count",
            &[model]
        ),
        Some(1.0),
        "{afterwards}"
    );
    let rate = sample(
        &afterwards,
        "hephaestus_generation_tokens_per_second_sum",
        &[model],
    )
    .expect("token rate recorded");
    assert!(rate > 0.0, "{afterwards}");
}
