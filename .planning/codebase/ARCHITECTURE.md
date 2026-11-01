---
last_mapped_commit: ad046b308fac4b7670543698064c366af75a3015
---
<!-- refreshed: 2026-11-01 -->
# Architecture (Forge service)

**Scope:** This document covers the Forge service (`forge/`) only, not the Rust runtime.

**Analysis Date:** 2026-11-01

## System Overview

```text
Hephaestus (HttpForgeClient)  --POST /convert {model_id}-->
┌─────────────────────────────────────────────────────────────┐
│ HTTP layer  `forge/src/forge/main.py`, `forge/src/forge/api.py`│
│ FastAPI app factory, lifespan, /health, POST /convert        │
└───────────────────────────┬─────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ Orchestration  `forge/src/forge/queue.py` (ConversionQueue)  │
│ per-model lock, global Semaphore(1), LRU result cache        │
└──────┬──────────────────┬───────────────────┬───────────────┘
       ▼                  ▼                   ▼
 convert_model()    validate_model()    build_operator() + upload_to_storage()
 `converter.py`     `converter.py`      `storage.py` (OpenDAL)
                                              ▼
                                  S3 / GCS / fs / memory(tests)
```

## Component Responsibilities

| Component | Responsibility | File |
|-----------|----------------|------|
| App factory / lifespan | Builds FastAPI app, configures structlog JSON logging, creates `ForgeSettings` and `ConversionQueue` on `app.state`, `/health` | `forge/src/forge/main.py` |
| Router | `POST /convert`, maps exceptions to HTTP 500 | `forge/src/forge/api.py` |
| Queue | Serialises conversions, dedupes, caches, drives pipeline | `forge/src/forge/queue.py` |
| Converter | HF -> ONNX via optimum `main_export`, tokenizer save, two-stage validation | `forge/src/forge/converter.py` |
| Storage | Settings -> OpenDAL options, operator construction, recursive upload | `forge/src/forge/storage.py` |
| Settings | Env-driven config (`pydantic-settings`) | `forge/src/forge/config.py` |
| Models | `ConvertRequest` (model_id validator), `ConversionMetadata`, `ConvertResponse` | `forge/src/forge/models.py` |

## Pattern Overview

**Overall:** Small layered FastAPI service; flat functional modules with one stateful class (`ConversionQueue`). Blocking work runs via `asyncio.to_thread`.

**Key Characteristics:**
- App state (`settings`, `queue`) held on `app.state`, read by the router via `request.app.state`.
- Storage is abstracted by OpenDAL, not by an in-repo interface; backend is chosen by passing `settings.storage_type` as the OpenDAL scheme string.
- One conversion at a time, process-local state (not shared across replicas).

## Data Flow

### Primary request path (request -> conversion -> upload)

1. Client `POST /convert` with `{"model_id": "org/name"}` (`api.py:18`). `ConvertRequest.validate_model_id` rejects empty, `..`, and chars outside `[A-Za-z0-9\-_./]` (422) (`models.py:20`).
2. Router calls `queue.convert(body.model_id, settings)` (`api.py:30`).
3. `ConversionQueue.convert` (`queue.py:39`): get/create per-model `asyncio.Lock`; return cached `ConvertResponse` if present (LRU, `MAX_CACHED=256`); else acquire `Semaphore(1)`, `tempfile.mkdtemp(prefix="forge-")`.
4. `_do_convert` (`queue.py:76`), each step in `asyncio.to_thread`:
   - `convert_model(model_id, output_dir)` -> `main_export(task="auto")`, saves tokenizer, returns `ConversionMetadata` (`converter.py:22`).
   - `validate_model(output_dir)` -> `onnx.checker.check_model(path)`, dummy `ort.InferenceSession` run, checks `tokenizer.json` and `config.json` parse (`converter.py:56`). Raises `ConversionError`.
   - `build_operator(settings)` then `upload_to_storage(op, model_id, output_dir)` (`queue.py:91-97`).
5. Temp dir removed on success (`queue.py:105`) or on any exception (`queue.py:73`). Result cached.
6. Response `ConvertResponse{storage_paths, metadata}`. Errors (`ConversionError`, `asyncio.TimeoutError`, any `Exception`) -> HTTP 500 with detail (`api.py:29-49`).

### Storage abstraction (precise)

- **Interface:** none of Forge's own; the abstraction is `opendal.Operator` (`opendal>=0.47`). Forge exposes three functions in `forge/src/forge/storage.py`:
  - `storage_options(settings) -> dict[str, str]` (line 12): pure mapping of settings to OpenDAL config keys.
    - `bucket` <- `storage_bucket` if set (all backends).
    - `region` <- `storage_region` only when `storage_type == "s3"`.
    - `credential_path` <- `storage_credential_path` only when `storage_type == "gcs"` and non-empty (empty => ADC / `GOOGLE_APPLICATION_CREDENTIALS` / Workload Identity).
    - `root`: for `fs`, `storage_root` (+ `/storage_prefix`) if root set; for all others `"/" + storage_prefix` if prefix set.
  - `build_operator(settings) -> opendal.Operator` (line 45): `opendal.Operator(settings.storage_type, **storage_options(settings))`. This is the factory/selection logic: no if/else on backend; `storage_type` string is the OpenDAL scheme (`s3`, `gcs`, `fs`, `memory` in tests). Unknown values fail inside OpenDAL.
  - `upload_to_storage(op, model_id, local_dir) -> list[str]` (line 54): `os.walk`, writes each file with `op.write(f"{model_id}/{relative}", bytes)`, returns paths. Reads whole file into memory.
- **Call sites:** only `forge/src/forge/queue.py:91-97` (`build_operator`, `upload_to_storage`). Operator is rebuilt on every conversion. Tests: `forge/tests/test_storage.py`, `forge/tests/conftest.py`.
- **Config:** `ForgeSettings` fields `storage_type` (default `"s3"`), `storage_bucket`, `storage_prefix`, `storage_root`, `storage_region`, `storage_credential_path` (`config.py`). The GCS path is already wired and parameterised-tested (`test_build_operator_supports_storage_type` over `s3`, `gcs`, `fs`; credential_path scoping test). Not covered: any real GCS write, and the Dockerfile/`forge` has no GCS-specific dependency beyond OpenDAL's built-in `gcs` service.

**State Management:** `ForgeSettings` created once in lifespan; `ConversionQueue` in-memory (`_locks`, `_results` OrderedDict, semaphore). No persistence.

## Key Abstractions

- `ConversionQueue` (`queue.py`): lock+semaphore+LRU cache around the pipeline.
- `ConversionError` (`converter.py:18`): domain error for conversion/validation failures.
- `ForgeSettings` (`config.py`): single config surface.
- `opendal.Operator`: storage handle, root already includes prefix.

## Entry Points

- `forge.main:create_app` (factory): run by `uvicorn forge.main:create_app --factory` (`forge/Dockerfile` CMD) or `python -m forge.main` (`main.py:62`).
- Routes: `GET /health`, `POST /convert`.

## Architectural Constraints

- **Threading:** asyncio event loop; blocking work in default thread pool via `to_thread`. Semaphore(1) bounds concurrency to one conversion.
- **Global state:** `app.state.queue`, `app.state.settings`; module-level `router` in `api.py`; structlog configured globally in lifespan.
- **Timeouts:** `conversion_timeout_secs` setting exists but is not enforced server-side (`queue.py:57-63` comment; client enforces).
- **Contract with Rust:** `_MODEL_ID_PATTERN` mirrors Rust `validate_model_id`; storage path layout `{prefix}/{model_id}/{relative}` is consumed by the runtime.
- **Circular imports:** none.

## Anti-Patterns

### Blocking whole-file reads on upload
**What happens:** `upload_to_storage` does `op.write(path, f.read())`.
**Why it's wrong:** large `model.onnx` files are fully buffered in memory.
**Do this instead:** stream via `op.writer(path)` in chunks if models grow (`forge/src/forge/storage.py:75`).

### Unused timeout setting
**What happens:** `conversion_timeout_secs` is declared but never read.
**Do this instead:** remove it or enforce at the HTTP layer; don't assume server-side timeout.

## Error Handling

**Strategy:** Domain `ConversionError` plus catch-all in the router; all failures -> 500 with message. Temp dir cleaned on failure. Storage errors propagate as generic exceptions (hit the catch-all). Failures are not cached.

## Cross-Cutting Concerns

**Logging:** structlog JSON to stdout, level from `LOG_LEVEL`; event-name style (`conversion_start`, `conversion_uploaded`).
**Validation:** Pydantic field validator on `model_id`.
**Authentication:** None on the Forge API; HF via `HF_TOKEN` env read by transformers; cloud creds via OpenDAL/env.

---

*Architecture analysis: 2026-11-01*
