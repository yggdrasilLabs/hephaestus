<!-- GSD:project-start source:PROJECT.md -->
## Project

**Hephaestus** (formerly Blacksmith)

A unified ONNX model inference runtime in Rust. Single container that loads, serves, and manages any ONNX-compatible model — classifiers, embeddings, seq2seq, token classifiers. Pairs with a Python Forge service for converting non-ONNX HuggingFace models.

**Core Value:** A single Rust binary that takes a model name, resolves it to ONNX files (S3 cache → HuggingFace → Forge conversion), and serves inference with full pre/post-processing — replacing every per-model Python runtime in the Minerva cluster.

### Constraints

- **Language**: Rust only, 2024 edition & workspace resolver 3
- **Rules compliance**: Every file must adhere to all rules in `rules/`.
- **No AI attribution**: Never include Co-Authored-By lines, AI-generated mentions, or any reference to Claude/AI in commits, tags, release notes, code comments, or any other artifact in this repo.
- **Code Convention**: Traits must follow John Ousterhout's "deep module" principle — expose a minimal interface (1-3 methods) that hides significant implementation complexity. Callers should never need to understand internals. Prefer one `process()` over separate `tokenize()`, `infer()`, `decode()`. Prefer one `resolve()` over separate `check_s3()`, `check_hf()`, `call_forge()`. If a trait has more than 3 required methods, it is probably too shallow — reconsider the abstraction boundary.

<!-- GSD:project-end -->

<!-- GSD:stack-start source:STACK.md -->
## Technology Stack

Technology stack not yet documented. Will populate after codebase mapping or first phase.
<!-- GSD:stack-end -->

<!-- GSD:conventions-start source:CONVENTIONS.md -->
## Conventions

Conventions not yet established. Will populate as patterns emerge during development.
<!-- GSD:conventions-end -->

<!-- GSD:architecture-start source:ARCHITECTURE.md -->
## Architecture

Architecture not yet mapped. Follow existing patterns found in the codebase.
<!-- GSD:architecture-end -->

<!-- GSD:skills-start source:skills/ -->
## Project Skills

No project skills found. Add skills to any of: `.claude/skills/`, `.agents/skills/`, `.cursor/skills/`, `.github/skills/`, or `.codex/skills/` with a `SKILL.md` index file.
<!-- GSD:skills-end -->

## Local Testing

### Build

```bash
cargo build --workspace --release
cargo test --workspace
```

### Run inference (sentiment classifier)

```bash
MODEL_ID=Xenova/distilbert-base-uncased-finetuned-sst-2-english \
STORAGE_TYPE=none \
PORT=8090 \
./target/release/hephaestus
```

```bash
curl -s -X POST http://localhost:8090/infer \
  -H "Content-Type: application/json" \
  -d '{"text": "This product is amazing"}'
# → {"label":"POSITIVE","score":0.9998,"model_id":"...","latency_ms":2}
```

### Run inference (NER — BERT-based, requires token_type_ids)

```bash
MODEL_ID=Xenova/bert-base-NER \
STORAGE_TYPE=none \
PORT=8090 \
./target/release/hephaestus
```

```bash
curl -s -X POST http://localhost:8090/infer \
  -H "Content-Type: application/json" \
  -d '{"text": "John Smith works at Google in Mountain View, California."}'
# → {"entities":[{"word":"John Smith","entity":"PER",...},{"word":"Google","entity":"ORG",...}],...}
```

### Run inference (sentence embeddings)

```bash
MODEL_ID=Xenova/multi-qa-distilbert-cos-v1 \
STORAGE_TYPE=none \
PORT=8090 \
./target/release/hephaestus
```

```bash
curl -s -X POST http://localhost:8090/infer \
  -H "Content-Type: application/json" \
  -d '{"text": "How do I reset my password?"}'
# → {"embedding":[0.038,...768 floats...],"model_id":"...","latency_ms":14}
```

### Forge tests

```bash
cd forge && uv run pytest tests/ -v
```

### Health and metrics

- `GET /healthz/live` — liveness probe
- `GET /healthz/ready` — readiness probe (200 after warmup)
- `GET /metrics` — Prometheus scrape endpoint

### Key env vars

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_ID` | *(required)* | HuggingFace model identifier |
| `STORAGE_TYPE` | `s3` | `s3`, `fs`, `gcs`, `none` |
| `STORAGE_BUCKET` | — | Bucket name (required for cloud backends) |
| `STORAGE_ROOT` | — | Root directory (required for `fs`) |
| `STORAGE_CREDENTIAL_PATH` | — | GCS key file path (gcs only; unset = ADC / Workload Identity) |
| `PORT` | `8080` | HTTP listen port |
| `EXECUTION_PROVIDER` | `cpu` | `cpu`, `cuda`, `tensorrt`, `coreml` |
| `FORGE_URL` | — | Forge service URL (enables conversion tier) |
| `MODEL_PROFILE` | *(auto)* | Override: `classifier`, `embeddings`, `seq2seq`, `token_classifier` |

<!-- GSD:workflow-start source:GSD defaults -->
## GSD Workflow Enforcement

Before using Edit, Write, or other file-changing tools, start work through a GSD command so planning artifacts and execution context stay in sync.

Use these entry points:
- `/gsd-quick` for small fixes, doc updates, and ad-hoc tasks
- `/gsd-debug` for investigation and bug fixing
- `/gsd-execute-phase` for planned phase work

Do not make direct repo edits outside a GSD workflow unless the user explicitly asks to bypass it.
<!-- GSD:workflow-end -->
