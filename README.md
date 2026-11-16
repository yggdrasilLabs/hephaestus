# Hephaestus

A unified ONNX model inference runtime in Rust — one container that loads, serves, and manages any ONNX-compatible model, replacing per-model Python services and cloud API calls.

## Features

- **Single binary, any model** — classifiers, embeddings, seq2seq, text generation, token classifiers via ONNX Runtime
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
│   │   │   ├── worker.rs               # Inference worker thread + bounded job queue + generation batch
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
| **`REQUEST_TIMEOUT_SECS`** | `30` | Per-request inference timeout, measured from the arrival of the request. A request still waiting for the model at that moment is dropped without running. |
| **`INFERENCE_QUEUE_CAPACITY`** | `1024` | Most requests that may wait for the model at one time, 1-65536. One more is refused at once with HTTP 503, `Retry-After: 1` and error code `QUEUE_FULL` (gRPC `UNAVAILABLE`). The default is far above what a single-pass model queues under normal load; set a small value (for example `4`) for seq2seq and text generation models, where each waiting request takes seconds. Not applied with `BATCH_ENABLED=true`, where requests wait in the batcher's own bounded channel. |
| **`SEQ2SEQ_MAX_INPUT_TOKENS`** | — | Seq2seq only. Input token cap, 8-16384; an encoder-decoder model also rejects a value above its `max_position_embeddings`. Unset: 512, as for every other profile. |
| **`SEQ2SEQ_MAX_NEW_TOKENS`** | — | Seq2seq encoder-decoder models only. Most tokens generated per request, 1-4096. Unset: taken from the model's generation settings. |
| **`TEXT_GENERATION_MAX_INPUT_TOKENS`** | — | Text generation only. Cap on prompt tokens (chat template, system message and user text together), 16-32768; startup fails if it plus the new-token limit exceeds the model's context. Unset: 4096, or less when the model's context minus the new-token limit is smaller. A request above the cap is refused with HTTP 413; nothing is truncated. |
| **`TEXT_GENERATION_MAX_NEW_TOKENS`** | — | Text generation only. Most tokens generated per request, 1-4096. Unset: `max_new_tokens` from `generation_config.json`, else 256. |
| **`TEXT_GENERATION_SYSTEM_PROMPT`** | — | Text generation only. System message used when a request sends no `system` field. Unset: no system message. |
| **`TEXT_GENERATION_MAX_BATCH`** | `1` | Text generation only. How many requests may generate together, sharing each forward pass, 1-64. `1` generates one request at a time, exactly as without the variable. Above 1 see [Batched generation](#batched-generation); `BATCH_ENABLED` is then ignored. Needs a graph with a key/value cache and an `attention_mask` input, else the width stays 1 (logged at startup). Cache memory grows with the width. |
| **`MODEL_ONNX_FILE`** | — | Text generation only. File name of the graph to serve instead of `model.onnx` (for example `model_q4.onnx`). Only that graph and its external data files are fetched. Startup fails if it is set for another profile. |
| **`OTEL_EXPORTER_OTLP_ENDPOINT`** | — | OTel collector endpoint (enables distributed tracing) |

### Seq2seq models

The seq2seq profile serves two ONNX layouts through the same `/infer` request and the same `{"generated_text": ...}` response. The files in the model directory pick the layout; there is no setting for it.

| Layout | Files (in the model directory or its `onnx/` subdirectory) | How it runs |
|--------|-------------------------------------------------------------|-------------|
| **Fused** | `model.onnx` | One forward pass. The graph returns the output token IDs (or logits). |
| **Encoder-decoder** (standard Optimum export) | `encoder_model.onnx` plus `decoder_model_merged.onnx` and/or `decoder_model.onnx` | The encoder runs once, then the decoder runs once per generated token, with a key/value cache when `decoder_model_merged.onnx` is there. |

Rules:

- `model.onnx` always wins. A directory that holds both layouts runs as a fused model, exactly as before the encoder-decoder layout existed.
- `decoder_model_merged.onnx` is preferred when present and runs with a key/value cache: the first step runs its uncached branch, every later step feeds only the newest token plus the cached tensors. Without that file, `decoder_model.onnx` runs without a cache, as before. If both files exist and the merged one cannot be loaded or cannot feed its cache back, a warning is logged and `decoder_model.onnx` runs. `decoder_with_past_model.onnx` is never used.
- The model resolver (storage and Hugging Face tiers) downloads `decoder_model.onnx` and, when the export has it, `decoder_model_merged.onnx` as well. Models already in the local cache directory are not re-fetched.
- Only one decoder graph is loaded. The startup log line `seq2seq encoder-decoder mode loaded` reports the file and `kv_cache`.
- There is no environment variable to switch the cache off; remove `decoder_model_merged.onnx` from the model directory to get the uncached loop (library users: `Seq2SeqOptions::with_kv_cache(false)`).
- Both layouts also need `tokenizer.json` and `config.json`. `generation_config.json` is read when present.
- Whisper models are not handled here; they stay on the `asr` profile.

Encoder-decoder generation is greedy (one candidate, most likely token each step). With the cache each step costs about the same; without it every step re-reads the whole output so far, and cost grows with the square of the output length. Both loops generate the same tokens. Each replica generates one summary at a time; with `BATCH_ENABLED=true` the items of a batch run one after another and each caller gets its own result.

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

An encoder-decoder response also carries `prompt_tokens` (encoder input tokens, after truncation) and `completion_tokens` (generated tokens, the end token included). A fused model's response has neither.

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

### Text generation models (decoder-only)

The `text_generation` profile serves instruction-tuned decoder-only models (Qwen, Llama, SmolLM, Phi, ...) exported the way Optimum does it: one graph that takes `input_ids`, `attention_mask`, optionally `position_ids` and `past_key_values.*`, and returns `logits` and `present.*`. It is detected from `pipeline_tag: text-generation` or an architecture ending in `ForCausalLM`, after every other rule, so no model that loaded before changes profile. `MODEL_PROFILE=text_generation` forces it (needed for names such as `GPT2LMHeadModel`).

```bash
curl -X POST http://localhost:8080/infer \
  -H "Content-Type: application/json" -H "x-tenant-id: <tenant>" \
  -d '{"text": "<call transcript>", "system": "Summarize the call in two sentences."}'
# {"generated_text": "...", "prompt_tokens": 412, "completion_tokens": 58, "latency_ms": 1234, "model_id": "..."}
```

- `text` becomes the user message. `system` is optional: a string sets the system message for that request, `""` sends none, absent or `null` uses `TEXT_GENERATION_SYSTEM_PROMPT`. Any other JSON type is a 400. Every other profile ignores the field, whatever its type, like any unknown field.
- The prompt is built with the model's own chat template: `chat_template.jinja` when the model has it, else `chat_template` in `tokenizer_config.json`. A model without a template does not start. Templates are rendered with minijinja with `add_generation_prompt=true` and no tools. Not supported: the `{% generation %}` tag (such a model fails at startup), tool definitions, and Python-exact `tojson` formatting.
- Decoding is greedy, with the key/value cache when the graph has one (float32 cache tensors only; fp16 cache graphs are rejected at startup). It stops at any `eos_token_id` of `generation_config.json` (else `config.json`), at the new-token limit, or with HTTP 504 after `REQUEST_TIMEOUT_SECS`. `repetition_penalty` and `no_repeat_ngram_size` are applied. Sampling settings (`do_sample`, `temperature`, `top_p`, `top_k`), beams and streaming are not implemented.
- A prompt above the input cap (`TEXT_GENERATION_MAX_INPUT_TOKENS`) is refused with HTTP 413 and `{"error": {"code": "INPUT_TOO_LONG", "message": "input too long: the prompt has 5120 tokens, the limit is 4096"}}` (gRPC `INVALID_ARGUMENT`). The count covers the template, the system message and the user text. Nothing is truncated: shorten the text and send it again.
- `prompt_tokens` and `completion_tokens` count the prompt as the model read it and the tokens it generated (the end token included).
- One request generates at a time per replica unless `TEXT_GENERATION_MAX_BATCH` is above 1 (see [Batched generation](#batched-generation)); with `BATCH_ENABLED=true` and a width of 1 the items of a batch run one after another. gRPC `Infer` works with the default system prompt only (the proto has no `system` field). Warmup generates one token; `WARMUP_INPUT` is not used.
- The resolver additionally fetches `tokenizer_config.json`, `generation_config.json` and `chat_template.jinja` when `config.json` names a causal LM. With `MODEL_ONNX_FILE` set, only that graph is looked up (`onnx/` then the repository root); there is no fallback to another variant or to Forge.

ONNX external data: for every profile, the data files a graph names (`model.onnx_data`, `model_q4.onnx_data_1`, ...) are read from the graph file itself and fetched, cached and uploaded with it. A model whose data files are missing is never uploaded to storage, and an incomplete model in storage is treated as a cache miss.

```bash
MODEL_ID=onnx-community/Qwen3-4B-Instruct-2507-ONNX \
MODEL_ONNX_FILE=model_q4.onnx \
TEXT_GENERATION_SYSTEM_PROMPT="Summarize the call in two sentences." \
TEXT_GENERATION_MAX_NEW_TOKENS=200 \
REQUEST_TIMEOUT_SECS=120 \
STORAGE_TYPE=none \
cargo run -p hephaestus
```

To compare prompts and generated token IDs with a Hugging Face reference, see the ignored tests `text_generation_matches_reference_cases` (`crates/hephaestus-core/src/pipeline/text_generation.rs`) and `chat_templates_match_reference_cases` (`.../chat_template.rs`).

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

### Inference queue

One thread per replica runs the model, one job at a time (text generation with `TEXT_GENERATION_MAX_BATCH` above 1: one batch of generations, see [Batched generation](#batched-generation)). `/infer`, gRPC `Infer` and the WebSocket endpoint put jobs on a bounded queue (`INFERENCE_QUEUE_CAPACITY`) and wait for the result; tokenization happens before queueing and never waits for the model.

- Queue full: the request is refused at once with HTTP 503, `Retry-After: 1` and `{"error": {"code": "QUEUE_FULL", ...}}` (gRPC `UNAVAILABLE`). The readiness probe is not affected.
- A job whose `REQUEST_TIMEOUT_SECS` deadline passes while it is queued, or whose client has gone, is removed without running.
- On SIGTERM the job that is running finishes (every generation of a running batch); queued jobs and new requests get 503 `NOT_READY`.
- With `BATCH_ENABLED=true` requests wait in the batcher's channel instead (4 × `BATCH_MAX_SIZE`, senders wait rather than being refused); each batch is one job.

A request that times out or whose client disconnects while its generation is running (text generation, encoder-decoder seq2seq) stops that generation before the next token, so the next job starts at once. This holds for HTTP, gRPC and batched requests. A single forward pass (every other profile) cannot be interrupted and runs to its end.

Metrics, all labelled `model_id`:

| Metric | Type | Meaning |
|--------|------|---------|
| `hephaestus_inference_queue_depth` | gauge | Requests admitted and not yet started |
| `hephaestus_inference_queue_wait_seconds` | histogram | Time from admission to the start of the run |
| `hephaestus_inference_jobs_rejected_total` | counter | Requests refused; `reason` is `queue_full` or `shutting_down` |
| `hephaestus_inference_jobs_dropped_total` | counter | Queued requests that never ran; `reason` is `abandoned` (caller timed out or disconnected), `expired` (deadline passed, caller still waiting) or `shutdown` |
| `hephaestus_inference_jobs_cancelled_total` | counter | Running generations stopped because the caller timed out or disconnected |
| `hephaestus_prompt_tokens_total` | counter | Prompt tokens read by finished generations |
| `hephaestus_generated_tokens_total` | counter | Tokens generated by finished generations |
| `hephaestus_generation_tokens_per_second` | histogram | Generated tokens divided by the time of the whole run (prompt processing included), per finished generation |
| `hephaestus_generation_batch_sequences` | histogram | Batched generation only. Sequences in one forward pass; `pass` is `decode` (a token step of the running batch) or `prompt` (the pass that reads the prompts of joining requests) |
| `hephaestus_generation_active_sequences` | gauge | Batched generation only. Generations running together right now |

### Batched generation

With `TEXT_GENERATION_MAX_BATCH` above 1 the worker runs up to that many text generation requests together: one forward pass per token step for all of them.

- **Joining.** Between two token steps the worker takes waiting requests off the queue while the batch has a free place. Their prompts are read in one extra forward pass over the new prompts only (the running generations wait for that pass), and they decode with the others from the next step on.
- **Leaving.** A request is answered in the step it ends: end token, new-token limit, its deadline (504), or its caller going away. The others go on and its place is free at once. A failed forward pass fails every request that was in it.
- **Same output as alone.** Rows are padded on the left and the padding is masked, so each request gets the tokens it gets alone. On a real model the logits of a padded row differ from the solo run in the last bits (other positions, other tensor shapes), so a near tie between two tokens can fall the other way; the test graphs compute in whole numbers and match exactly.
- **Which requests run together.** A graph without a `position_ids` input counts positions by column, so a row can never be moved once it has started, and a request can only join a batch at least as wide (in tokens) as its prompt. A request with a longer prompt waits until the batch has grown that far or has emptied, and requests behind it wait with it (arrival order is kept). A request is also not put into a batch where its row would be more than half padding at its longest (padding above prompt + new-token limit), nor into one already wider than `TEXT_GENERATION_MAX_INPUT_TOKENS`. Prompts of similar length batch best.
- **`BATCH_ENABLED`.** Ignored for a text generation model with a width above 1 (a warning is logged): the dynamic batcher hands over closed batches, which would make each request wait for the batch before it. Requests then use the inference queue, so `INFERENCE_QUEUE_CAPACITY` applies and a full queue is a 503. Other profiles, and text generation with a width of 1, batch as before.
- **Queue and metrics.** `INFERENCE_QUEUE_CAPACITY` still counts waiting requests only; the width is on top. `hephaestus_stage_duration_seconds{stage="inference"}` and `hephaestus_generation_tokens_per_second` cover a request from joining to leaving, so per-request token rates fall as the width grows while the total rises.
- **Memory.** The cache holds `width × columns` positions, padding included, and columns never exceed max input tokens + max new tokens. So each place in the batch can cost what one worst-case request costs alone, and twice that for a moment while the cache is copied because a request joined or left.

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
