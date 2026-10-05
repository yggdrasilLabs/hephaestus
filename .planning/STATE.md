---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
current_phase: 12
status: verifying
stopped_at: Phase 11 context gathered
last_updated: "2026-08-29T14:47:41.280Z"
last_activity: 2026-08-29
last_activity_desc: Phase 12 complete
progress:
  total_phases: 12
  completed_phases: 11
  total_plans: 29
  completed_plans: 29
  percent: 92
current_phase_name: hot-path-performance-optimization
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-08-26)

**Core value:** A single Rust binary that takes a model name, resolves it to ONNX files, and serves inference with full pre/post-processing -- replacing every per-model Python runtime in the cluster.
**Current focus:** Phase 12 — hot-path-performance-optimization

## Current Position

Phase: 12
Plan: Not started
Status: Phase complete — ready for verification
Last activity: 2026-10-05 - Completed quick task 261005-h8p: Add GCS storage support via OpenDAL and drop azblob

Progress: [████████████████████] 9/9 plans (100%)

## Performance Metrics

**Velocity:**

- Total plans completed: 11
- Average duration: -
- Total execution time: 0 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 01 | 3 | - | - |
| 03 | 2 | - | - |
| 05 | 2 | - | - |
| 10 | 2 | - | - |
| 12 | 2 | - | - |

**Recent Trend:**

- Last 5 plans: -
- Trend: -

*Updated after each plan completion*
| Phase 01 P01 | 7min | 2 tasks | 15 files |
| Phase 01 P02 | 4min | 2 tasks | 4 files |
| Phase 01 P03 | 3min | 2 tasks | 5 files |
| Phase 02 P00 | 2min | 2 tasks | 7 files |
| Phase 02 P01 | 7min | 2 tasks | 15 files |
| Phase 02 P02 | 8min | 2 tasks | 12 files |
| Phase 02 P03 | 2min | 2 tasks | 4 files |
| Phase 04 P01 | 8min | 2 tasks | 9 files |
| Phase 04 P02 | 7min | 2 tasks | 4 files |
| Phase 04 P04 | 19min | 3 tasks | 5 files |
| Phase 05 P02 | 8min | 2 tasks | 6 files |
| Phase 06 P03 | 3min | 2 tasks | 8 files |
| Phase 06 P02 | 4min | 2 tasks | 4 files |
| Phase 07 P01 | 8min | 4 tasks | 6 files |
| Phase 08 P01 | 2min | 2 tasks | 2 files |
| Phase 08 P02 | 2min | 2 tasks | 4 files |
| Phase 08 P03 | 2min | 2 tasks | 2 files |
| Phase 10 P01 | 2min | 2 tasks | 5 files |
| Phase 10 P02 | 6min | 2 tasks | 7 files |
| Phase 11 P01 | 5min | 2 tasks | 7 files |
| Phase 11 P02 | 3min | 2 tasks | 5 files |
| Phase 12 P01 | 2min | 2 tasks | 1 files |
| Phase 12 P02 | 1min | 1 tasks | 3 files |

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- [Roadmap]: 5 phases derived from 34 v1 requirements; classifier profile is first vertical slice
- [Roadmap]: Forge service (Python) is Phase 5, independent of Rust runtime phases
- [Roadmap]: Model resolution (Phase 3) implements Forge client stub; Forge server built separately
- [Phase ?]: PreparedInput made pub (fields pub(crate)) -- Pipeline trait associated type must match trait visibility
- [Phase ?]: execute() takes &mut self -- ort Session::run() requires mutability, deviating from D-06
- [Phase ?]: ndarray 0.17 (not 0.16) -- ort 2.0.0-rc.13 depends on ^0.17
- [Phase ?]: ort v2 API uses methods (inputs(), name()) not fields; inputs! returns Vec not Result; try_extract_tensor returns (Shape, &[T]) tuple
- [Phase ?]: Tests co-committed with implementation in same source files per Rust cfg(test) convention
- [Phase ?]: LOG_LEVEL uses EnvFilter fallback: RUST_LOG takes precedence, LOG_LEVEL used when RUST_LOG unset
- [Phase ?]: Config loaded before tracing init so log_level is available for env filter
- [Phase ?]: model_dir() validates path existence in addition to absolute + no-traversal checks (T-01-01)
- [Phase ?]: Minimal hephaestus-api crate with no production deps; 02-01 adds axum/tonic/tower
- [Phase ?]: tokio::time::timeout at handler level for D-14 structured 504 (not tower-http TimeoutLayer)
- [Phase ?]: Drain watchdog as background tokio task; force-exits after SHUTDOWN_TIMEOUT_SECS
- [Phase ?]: Integration tests require model files; unit tests cover logic inline
- [Phase ?]: OTel v0.32 removed global shutdown_tracer_provider; store SdkTracerProvider in OnceLock for clean shutdown
- [Phase ?]: Deep-module StageTimer hides all metrics crate interaction; handlers never touch metrics macros
- [Phase ?]: Conditional OTel layer via Option in subscriber registry; None passes through with zero overhead
- [Phase ?]: Per-request tracing events with model_id/latency_ms/status on all handler exit paths (OBSV-02)
- [Phase 03]: Used std::env::var("HOME") instead of dirs crate for HF cache directory
- [Phase 03]: Vec<u8> for S3 file content instead of bytes::Bytes to avoid adding direct bytes dependency
- [Phase 03]: Concrete StubForgeClient field in ModelResolver -- Phase 5 will generalize to trait object
- [Phase 03]: tempfile::TempDir::keep() for atomic download pattern (into_path deprecated)
- [Phase ?]: Extracted shared load_session_and_tokenizer and tokenize_text helpers across pipeline types
- [Phase ?]: PipelineKind returns serde_json::Value -- handler inserts model_id/latency_ms dynamically (D-05)
- [Phase ?]: InferResponse struct removed; model-determined JSON output replaces fixed response type
- [Phase ?]: Entity struct in pipeline.rs for public trait visibility
- [Phase ?]: PreparedInput optional encoding field avoids re-tokenization
- [Phase ?]: Seq2Seq tries i64 first, falls back to f32 with rounding
- [Phase ?]: check_outputs_nonempty() inline guard for SessionOutputs
- [Phase ?]: Result returns for softmax/argmax per err-result-over-panic.md
- [Phase ?]: pytest-asyncio with auto mode for async test support
- [Phase ?]: Manual app.state setup in test fixtures (ASGITransport does not trigger lifespan)
- [Phase ?]: uv.lock committed for reproducible builds per D-14
- [Phase ?]: [Phase 05]: Generic ModelResolver with static dispatch over ForgeClient trait
- [Phase ?]: [Phase 05]: ForgeResponse carries s3_paths + ConversionMetadata for observability
- [Phase ?]: opendal.Operator is synchronous in Python; asyncio.to_thread wrapping preserved for non-blocking upload
- [Phase ?]: Operator root absorbs storage prefix so callers use model_id/filename paths only
- [Phase ?]: STORAGE_PREFIX maps to OpenDAL root with leading slash for cloud backends, joined with STORAGE_ROOT for fs
- [Phase ?]: Config validation runs before Operator construction -- startup fails fast on invalid storage_type or missing fs root
- [Phase 07]: Conditional token_type_ids via session.inputs() check -- backward compatible with DistilBERT
- [Phase 07]: Transient trait for retry classification instead of string matching in with_retry
- [Phase 07]: tokio::sync::Notify for shutdown watchdog instead of process::exit
- [Phase ?]: Retained argmax_per_token with allow(dead_code) for future raw-logit use
- [Phase ?]: tokio::sync::RwLock for pipeline: read lock for prepare (concurrent tokenization), write lock for execute (exclusive ONNX session access)
- [Phase ?]: Seq2seq integration test excluded; no reliable fused ONNX model available
- [Phase ?]: Feature-gated integration tests use cfg(feature = integration) to avoid model downloads in default cargo test
- [Phase ?]: InferResponse uses opaque bytes result_json -- new model profiles never require proto changes
- [Phase ?]: tonic-prost runtime dependency required for generated ProstCodec references
- [Phase 10]: HealthReporter initialized to NOT_SERVING, set to SERVING only after warmup completes
- [Phase 10]: tonic::service::Routes merged into axum Router via into_axum_router for same-port multiplexing
- [Phase ?]: PipelineKind::prepare accepts impl Into<InferenceInput> for backward-compatible String callers
- [Phase ?]: PipelineOutput::to_json() replaces serde_json::json! macros for typed output
- [Phase ?]: PreparedAudio fields marked allow(dead_code) until ASR pipeline (Plan 11-03)
- [Phase ?]: AudioBuffer caps at 2x window_samples for memory exhaustion prevention (T-11-04)
- [Phase ?]: 30-second idle timeout on WebSocket recv (T-11-05)
- [Phase ?]: TranscriptMessage.text empty until Plan 11-03 wires ASR
- [Phase ?]: [Phase 12]: Separate session lifetime from prepared input lifetime for zero-copy tensor access
- [Phase ?]: Arc<str> over Cow<str> for StageTimer model_id

### Roadmap Evolution

- Phase 8 added: Inference Quality and Concurrency (NER score bug, mutex split, integration tests)
- Phase 9 removed: GPU Acceleration & TensorRT Engine Pipeline (dropped — image size too large for production benefit)
- Phase 10 added: gRPC Inference API (tonic gRPC layer multiplexed with HTTP/REST, health checking, reflection, all model profiles)
- Phase 11 added: WebSocket Streaming & ASR Pipeline (WebSocket endpoint, CTC/Whisper ASR, mel spectrogram, audio buffering)
- Phase 12 added: Hot-Path Performance Optimization (eliminate Vec clones in tensor construction, Arc<str> metrics, dead field removal)

### Pending Todos

None yet.

### Blockers/Concerns

- ort crate is pre-release (v2.0.0-rc.13); API may change. Pin exact version.
- Research flags Phase 1 (core engine) for deeper investigation of ort v2 Session builder patterns.

### Quick Tasks Completed

| # | Description | Date | Commit | Directory |
|---|-------------|------|--------|-----------|
| 260826-ren | Wire EXECUTION_PROVIDER config to ort session builder | 2026-08-26 | d8a9e30 | [260826-ren-wire-execution-provider-config-to-ort-se](./quick/260826-ren-wire-execution-provider-config-to-ort-se/) |
| 261005-h8p | Add GCS storage support via OpenDAL and drop azblob | 2026-10-05 | 8153e45 | [261005-h8p-add-gcs-storage-support-via-opendal-and-](./quick/261005-h8p-add-gcs-storage-support-via-opendal-and-/) |

## Deferred Items

Items acknowledged and carried forward from previous milestone close:

| Category | Item | Status | Deferred At |
|----------|------|--------|-------------|
| *(none)* | | | |

## Session Continuity

Last session: 2026-08-29T14:35:04.467Z
Stopped at: Phase 11 context gathered
Resume file: .planning/phases/11-websocket-streaming-asr-pipeline/11-CONTEXT.md
