---
last_mapped_commit: ad046b308fac4b7670543698064c366af75a3015
---
# External Integrations

**Scope:** This document covers the Forge service (`forge/`) only, not the Rust runtime.

**Analysis Date:** 2026-11-01

## APIs & External Services

**HuggingFace Hub:**
- Model + tokenizer download during conversion
  - Client: `optimum.exporters.onnx.main_export`, `transformers.AutoTokenizer.from_pretrained`, `AutoConfig.from_pretrained` in `forge/src/forge/converter.py` (`convert_model`)
  - Auth: `HF_TOKEN` env var, read by the libraries directly (no `ForgeSettings` field)

**Hephaestus (caller):**
- The Rust runtime calls `POST /convert` over HTTP (HttpForgeClient; timeout enforced client-side via `FORGE_TIMEOUT_SECS`, per comment in `forge/src/forge/queue.py`). Forge does not call out to Hephaestus.

**HTTP endpoints (`forge/src/forge/api.py`, `forge/src/forge/main.py`):**
- `POST /convert` body `ConvertRequest{model_id}` -> `ConvertResponse{storage_paths, metadata}` (`forge/src/forge/models.py`); errors -> HTTP 500
- `GET /health` -> `{"status":"ok"}`

## Storage Backends (focus: GCS)

### Backends that exist today
Storage is Apache OpenDAL (`opendal` Python 0.47.6). The scheme string is `STORAGE_TYPE` passed verbatim to `opendal.Operator(...)`. Supported/tested: `s3` (default), `gcs`, `fs`. `memory` is used only in tests. There is no `none` handling in Forge (README lists `none` for the Rust side only); `STORAGE_TYPE=none` would fail at `opendal.Operator("none")`.

**GCS support is already implemented in code and covered by offline tests**; remaining work is likely validation/ops (real-bucket test, deployment wiring), not a new code path.

### Selection and invocation chain
1. `ForgeSettings()` created in lifespan, `forge/src/forge/main.py` `lifespan()` -> `app.state.settings`.
2. `POST /convert` handler `convert()` in `forge/src/forge/api.py` passes `request.app.state.settings` to `ConversionQueue.convert(model_id, settings)`.
3. `ConversionQueue._do_convert()` in `forge/src/forge/queue.py` (~line 91) calls `build_operator(settings)` then `asyncio.to_thread(upload_to_storage, op, model_id, output_dir)`. A new Operator is built per conversion.
4. `build_operator()` in `forge/src/forge/storage.py`: `opendal.Operator(settings.storage_type, **storage_options(settings))`.
5. `upload_to_storage(op, model_id, local_dir)` in `forge/src/forge/storage.py` walks `local_dir` recursively and `op.write(f"{model_id}/{relative}", bytes)`; returns the list of storage paths (the whole file is read into memory per write).

### Option mapping: `storage_options()` in `forge/src/forge/storage.py`
| Setting (env var) | OpenDAL key | Applies to |
|---|---|---|
| `storage_bucket` (`STORAGE_BUCKET`) | `bucket` | any backend when non-empty |
| `storage_region` (`STORAGE_REGION`) | `region` | `s3` only |
| `storage_credential_path` (`STORAGE_CREDENTIAL_PATH`) | `credential_path` | `gcs` only, when non-empty |
| `storage_root` + `storage_prefix` | `root` | `fs` only: `root[/prefix]` |
| `storage_prefix` | `root` = `/{prefix}` | `s3`, `gcs` (non-fs) |

Defaults in `forge/src/forge/config.py` `ForgeSettings`: `storage_type="s3"`, all other storage fields `""`.

### Authentication
- **S3:** no explicit key fields; OpenDAL's default AWS credential chain (env vars, IRSA, IMDS). No access key settings exist in `ForgeSettings`.
- **GCS:** `STORAGE_CREDENTIAL_PATH` -> `credential_path` (service-account JSON key). When unset, OpenDAL falls back to `GOOGLE_APPLICATION_CREDENTIALS`, then GKE Workload Identity / metadata server (documented in `forge/src/forge/config.py` docstring and root `README.md` lines ~152-182, "Run with GCS storage cache"). No `endpoint`, `scope`, `service_account`, or `credential` (inline JSON) option is plumbed; adding one means a new `ForgeSettings` field plus a `storage_options()` branch.
- **fs:** no auth; `STORAGE_ROOT` required.

### Tests (`forge/tests/test_storage.py`, `forge/tests/conftest.py`)
- `test_build_operator_supports_storage_type` parametrized over `s3`, `gcs`, `fs` (offline Operator construction only).
- `test_storage_options_passes_region_only_to_s3`, `test_storage_options_passes_credential_path_only_to_gcs` assert per-backend option scoping.
- Upload tests use `memory_operator` fixture (`opendal.Operator("memory")`). No test performs real GCS I/O or uses a GCS emulator.

### Touchpoints if extending GCS support
- `forge/src/forge/config.py` (new fields), `forge/src/forge/storage.py` `storage_options()` (key mapping), `forge/tests/test_storage.py` (scoping tests), `forge/pyproject.toml`/`forge/uv.lock` if opendal needs bumping (README says opendal 0.58 for Rust docs; Forge locks 0.47.6), root `README.md` env table. `queue.py` and `api.py` need no change (backend-agnostic).

## Data Storage

**Databases:** None.

**File Storage:** Converted ONNX artifacts written to the configured OpenDAL backend (above). Conversion uses a local temp dir (`tempfile.mkdtemp(prefix="forge-")` in `forge/src/forge/queue.py`); it is removed only on failure.

**Caching:** In-process LRU of `ConvertResponse` per `model_id` (`ConversionQueue.MAX_CACHED = 256`, `forge/src/forge/queue.py`); lost on restart. One conversion at a time (`asyncio.Semaphore(1)`).

## Authentication & Identity

**Auth Provider:** None on the Forge API (`/convert` and `/health` are unauthenticated; rely on network policy).

## Monitoring & Observability

**Error Tracking:** None.
**Logs:** structlog JSON to stdout, level from `LOG_LEVEL` (`forge/src/forge/main.py`).

## CI/CD & Deployment

**Hosting:** Container (`forge/Dockerfile`), port 8080, Docker HEALTHCHECK on `/health`.
**CI Pipeline:** Not detected within `forge/`.

## Environment Configuration

**Required env vars:** `STORAGE_TYPE` (default `s3`), `STORAGE_BUCKET` (s3/gcs), `STORAGE_ROOT` (fs). Optional: `STORAGE_PREFIX`, `STORAGE_REGION` (s3), `STORAGE_CREDENTIAL_PATH` (gcs), `HF_TOKEN`, `CONVERSION_TIMEOUT_SECS`, `LOG_LEVEL`, `HOST`, `PORT`. Ambient: `GOOGLE_APPLICATION_CREDENTIALS`, AWS credential env vars.

**Secrets location:** Environment / mounted key file; none in `forge/`.

## Webhooks & Callbacks

**Incoming:** None. **Outgoing:** None.

---

*Integration audit: 2026-11-01*
