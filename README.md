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
| **`SEQ2SEQ_MAX_INPUT_TOKENS`** | — | Seq2seq only. Input token cap, 8-16384; an encoder-decoder model also rejects a value above its `max_position_embeddings`. Unset: 512, as for every other profile. |
| **`SEQ2SEQ_MAX_NEW_TOKENS`** | — | Seq2seq encoder-decoder models only. Most tokens generated per request, 1-4096. Unset: taken from the model's generation settings. |
| **`OTEL_EXPORTER_OTLP_ENDPOINT`** | — | OTel collector endpoint (enables distributed tracing) |

### Seq2seq models

The seq2seq profile serves two ONNX layouts through the same `/infer` request and the same `{"generated_text": ...}` response. The files in the model directory pick the layout; there is no setting for it.

| Layout | Files (in the model directory or its `onnx/` subdirectory) | How it runs |
|--------|-------------------------------------------------------------|-------------|
| **Fused** | `model.onnx` | One forward pass. The graph returns the output token IDs (or logits). |
| **Encoder-decoder** (standard Optimum export) | `encoder_model.onnx` plus `decoder_model.onnx` or `decoder_model_merged.onnx` | The encoder runs once, then the decoder runs once per generated token. |

Rules:

- `model.onnx` always wins. A directory that holds both layouts runs as a fused model, exactly as before the encoder-decoder layout existed.
- `decoder_model.onnx` is preferred over `decoder_model_merged.onnx`. The merged graph is used only when it is the only decoder, with its cache branch switched off. `decoder_with_past_model.onnx` is never used.
- Both layouts also need `tokenizer.json` and `config.json`. `generation_config.json` is read when present.
- Whisper models are not handled here; they stay on the `asr` profile.

Encoder-decoder generation is greedy (one candidate, most likely token each step) and does not reuse decoder state between steps, so every step re-reads the whole output so far. Cost grows with the square of the output length. Each replica generates one summary at a time; with `BATCH_ENABLED=true` the items of a batch run one after another and each caller gets its own result.

Generation settings come from `generation_config.json`, then `config.json`:

| Setting | Effect |
|---------|--------|
| `decoder_start_token_id` (else `bos_token_id`) | First decoder token |
| `eos_token_id` | Stops generation |
| `forced_bos_token_id` | Fixes the first generated token |
| `forced_eos_token_id` | Fixes the last token when the limit is reached |
| `no_repeat_ngram_size` | Never repeats a token sequence of that length |
| `max_new_tokens`, else `max_length` minus one, else 19 | Token limit, unless `SEQ2SEQ_MAX_NEW_TOKENS` is set |

Not implemented: `num_beams`, `length_penalty`, `early_stopping`, `min_length`, `repetition_penalty`, sampling settings, `prefix` and `task_specific_params`. A model that declares them is served greedy, so its output can differ from the same model run with Hugging Face defaults.

The token limit is always bounded: never more than 4096 and never more than the model's `max_position_embeddings` minus one. Generation also stops with HTTP 504 once `REQUEST_TIMEOUT_SECS` has passed.

The input cap stays at 512 tokens unless `SEQ2SEQ_MAX_INPUT_TOKENS` is set. Longer input is truncated, so raise it for long documents (for example `1024` for BART models).

```bash
# A local Optimum export (encoder_model.onnx + decoder_model.onnx)
MODEL_ID=philschmid/distilbart-cnn-12-6-samsum \
MODEL_PATH=/models/distilbart-samsum \
STORAGE_TYPE=none \
SEQ2SEQ_MAX_INPUT_TOKENS=1024 \
cargo run -p hephaestus
```

Model resolution looks for `model.onnx` first in every tier, unchanged. Only when a tier has no `model.onnx` does it look for the encoder-decoder files: flat or under `onnx/` in the storage cache, `onnx/` then the repository root on the Hugging Face Hub. Forge accepts an encoder-decoder export as a valid conversion and uploads every exported file under `{model_id}/`, where the storage tier finds it.

The tiny models under `crates/hephaestus-core/tests/fixtures/seq2seq/` are generated by `make_fixtures.py` in that directory. To compare token IDs against a Hugging Face reference on a real model, see the ignored test `encoder_decoder_token_ids_match_reference_cases` in `crates/hephaestus-core/src/pipeline/seq2seq.rs`.

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
