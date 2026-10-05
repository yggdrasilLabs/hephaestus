# Hephaestus

A unified ONNX model inference runtime in Rust — one container that loads, serves, and manages any ONNX-compatible model, replacing per-model Python services and cloud API calls.

## Features

- **Single binary, any model** — classifiers, embeddings, seq2seq, token classifiers via ONNX Runtime
- **3-tier model resolution** — storage cache → HuggingFace Hub → Forge conversion, fully automatic
- **Backend-agnostic storage** — S3, GCS, local filesystem via [Apache OpenDAL](https://github.com/apache/opendal)
- **Forge conversion service** — companion Python service converts non-ONNX HuggingFace models to ONNX format
- **One model per pod** — configured via environment variables, scales with Kubernetes natively
- **CPU + GPU support** — toggle execution provider via env var; compile with `--features cuda` for GPU acceleration
- **Atomic model caching** — temp-dir-then-rename pattern prevents partial downloads from being served
- **Full pre/post-processing** — tokenization, inference, and output decoding in one binary (callers send text, get predictions)
- **Observability built in** — Prometheus metrics, OpenTelemetry tracing, structured JSON logs

## Tech Stack

### Rust Runtime (Hephaestus)

| Component | Technology |
|-----------|------------|
| **Language** | Rust 2024 edition, workspace resolver 3 |
| **ONNX inference** | `ort` 2.0.0-rc.13 (ONNX Runtime 1.28 bindings) |
| **Tokenization** | `tokenizers` 0.23 (HuggingFace Rust-native) |
| **HTTP server** | `axum` 0.8 |
| **Model downloads** | `hf-hub` 1.0 (HuggingFace Hub client) |
| **Storage abstraction** | `opendal` 0.58 (S3, GCS, filesystem) |
| **Tensors** | `ndarray` 0.17 |
| **Config** | `envy` (environment variables, no CLI parser) |
| **Metrics** | `metrics` + `metrics-exporter-prometheus` |
| **Tracing** | `tracing` + `tracing-subscriber` + `tracing-opentelemetry` |

### Python Conversion Service (Forge)

| Component | Technology |
|-----------|------------|
| **Language** | Python 3.11+ |
| **API** | FastAPI + Uvicorn |
| **ONNX conversion** | HuggingFace Optimum + ONNX Runtime |
| **Storage** | `opendal` Python bindings (same abstraction as Rust) |
| **Config** | Pydantic Settings (env vars) |

## Quick Start

### Prerequisites

- **Rust** 1.85+ (2024 edition support)
- **Python** 3.11+ with [uv](https://docs.astral.sh/uv/) (for Forge)
- **ONNX Runtime** shared libraries (auto-downloaded by `ort` crate)

### Build and test the Rust runtime

```bash
git clone https://github.com/sean-mca/hephaestus.git
cd hephaestus

# Build the full workspace
cargo build --workspace

# Run all tests
cargo test --workspace
```

### Build with GPU support

```bash
# NVIDIA CUDA
cargo build --workspace --features cuda

# NVIDIA TensorRT (optimized inference)
cargo build --workspace --features tensorrt

# Apple CoreML (macOS)
cargo build --workspace --features coreml
```

GPU execution providers fall back to CPU gracefully at runtime if the hardware isn't available. Requesting a GPU provider without the matching cargo feature produces a clear error at startup.

### Run inference

```bash
# Minimal — CPU inference, no storage cache
MODEL_ID=distilbert/distilbert-base-uncased-finetuned-sst-2-english \
STORAGE_TYPE=none \
cargo run -p hephaestus
```

### Set up the Forge (optional — only needed for non-ONNX models)

```bash
cd forge
uv sync
uv run python -m forge.main
```

### Run Forge tests

```bash
cd forge
uv run pytest tests/ -v
```

## Architecture

```
hephaestus/
├── Cargo.toml                          # Workspace root
├── crates/
│   ├── hephaestus/                     # Binary crate — config, startup, wiring
│   │   └── src/
│   │       ├── main.rs                 # Entry point, Operator construction, resolver init
│   │       └── config.rs               # Env var config (MODEL_ID, STORAGE_*, etc.)
│   ├── hephaestus-core/                # Inference engine — pipelines, profiles, post-processing
│   │   └── src/
│   │       ├── pipeline.rs             # Pipeline trait + all profile implementations
│   │       ├── profile.rs              # Model type detection from config.json
│   │       ├── ep.rs                   # Execution provider enum (CPU, CUDA, TensorRT, CoreML)
│   │       └── postprocess.rs          # Softmax, argmax, mean pooling, BIO span merging
│   ├── hephaestus-api/                 # HTTP server — routes, handlers, health, metrics
│   │   ├── src/
│   │   │   ├── routes.rs               # Axum router (/infer, /health, /metrics)
│   │   │   ├── handlers.rs             # Request → inference → response
│   │   │   ├── telemetry.rs            # OTel + Prometheus setup
│   │   │   └── batcher.rs              # Dynamic batching (configurable)
│   │   └── tests/                      # API integration tests
│   ├── hephaestus-resolve/             # Model resolution — storage, HuggingFace, Forge
│   │   └── src/
│   │       ├── resolver.rs             # 3-tier resolution chain (Storage → HF → Forge)
│   │       ├── storage.rs              # OpenDAL download/upload (backend-agnostic)
│   │       ├── hf.rs                   # HuggingFace Hub downloads
│   │       └── forge.rs                # Forge client (HTTP, triggers ONNX conversion)
│   └── hephaestus-proto/               # Protobuf definitions (gRPC, future)
└── forge/                              # Python ONNX conversion service
    ├── Dockerfile
    ├── pyproject.toml
    └── src/forge/
        ├── api.py                      # FastAPI endpoints
        ├── converter.py                # Optimum ONNX export + validation
        ├── queue.py                    # Conversion queue with semaphore
        ├── storage.py                  # OpenDAL upload (matches Rust storage)
        └── config.py                   # Pydantic settings (STORAGE_*, FORGE_*)
```

## Usage / Examples

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| **`MODEL_ID`** | *(required)* | HuggingFace model identifier (e.g., `org/model-name`) |
| **`STORAGE_TYPE`** | `s3` | Storage backend: `s3`, `fs`, `gcs`, `none` |
| **`STORAGE_BUCKET`** | — | Bucket name (required for S3/GCS) |
| **`STORAGE_PREFIX`** | — | Path prefix applied across all backends |
| **`STORAGE_ROOT`** | — | Root directory (required when `STORAGE_TYPE=fs`) |
| **`STORAGE_REGION`** | — | AWS region (S3 only) |
| **`STORAGE_CREDENTIAL_PATH`** | — | Path to a GCS service-account JSON key (GCS only). When unset, falls back to `GOOGLE_APPLICATION_CREDENTIALS`, then GKE Workload Identity via the metadata server. Read by Hephaestus and Forge. |
| **`EXECUTION_PROVIDER`** | `cpu` | ONNX execution provider: `cpu`, `cuda`, `tensorrt`, `coreml` (GPU providers require matching cargo feature) |
| **`PORT`** | `8080` | HTTP listen port |
| **`LOG_LEVEL`** | `info` | Log verbosity |
| **`FORGE_URL`** | — | Forge service URL (enables Forge conversion tier) |
| **`BATCH_ENABLED`** | `false` | Enable dynamic request batching |
| **`BATCH_MAX_SIZE`** | `8` | Max requests per batch (1-64) |
| **`BATCH_MAX_WAIT_MS`** | `50` | Max wait time before executing a partial batch |
| **`REQUEST_TIMEOUT_SECS`** | `30` | Per-request inference timeout |
| **`OTEL_EXPORTER_OTLP_ENDPOINT`** | — | OTel collector endpoint (enables distributed tracing) |

### Run with S3 storage cache

```bash
MODEL_ID=sentence-transformers/all-MiniLM-L6-v2 \
STORAGE_TYPE=s3 \
STORAGE_BUCKET=my-model-cache \
STORAGE_REGION=us-east-1 \
cargo run -p hephaestus
```

### Run with GCS storage cache

```bash
MODEL_ID=sentence-transformers/all-MiniLM-L6-v2 \
STORAGE_TYPE=gcs \
STORAGE_BUCKET=my-model-cache \
cargo run -p hephaestus
```

On GKE with Workload Identity no extra variables are needed; elsewhere, set `STORAGE_CREDENTIAL_PATH` to a mounted service-account key file.

### Run with local filesystem cache

```bash
MODEL_ID=distilbert/distilbert-base-uncased-finetuned-sst-2-english \
STORAGE_TYPE=fs \
STORAGE_ROOT=/tmp/models \
cargo run -p hephaestus
```

### Run without any storage cache (HuggingFace-only)

```bash
MODEL_ID=distilbert/distilbert-base-uncased-finetuned-sst-2-english \
STORAGE_TYPE=none \
cargo run -p hephaestus
```

### Call the inference endpoint

```bash
curl -X POST http://localhost:8080/infer \
  -H "Content-Type: application/json" \
  -d '{"text": "This product is amazing"}'
```

### Kubernetes deployment (one model per pod)

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: sentiment-classifier
spec:
  replicas: 3
  template:
    spec:
      containers:
        - name: hephaestus
          image: hephaestus:latest
          env:
            - name: MODEL_ID
              value: "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
            - name: STORAGE_TYPE
              value: "s3"
            - name: STORAGE_BUCKET
              value: "model-cache"
```

Deploying a different model is the same image with a different `MODEL_ID`. For GPU inference, use the `cuda`-enabled image and set `EXECUTION_PROVIDER`:

```yaml
- name: MODEL_ID
  value: "sentence-transformers/all-MiniLM-L6-v2"
- name: EXECUTION_PROVIDER
  value: "cuda"
```

## License

MIT
