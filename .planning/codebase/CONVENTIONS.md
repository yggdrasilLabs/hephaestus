---
last_mapped_commit: ad046b308fac4b7670543698064c366af75a3015
---
# Coding Conventions (Forge service)

**Scope:** This document covers the Python Forge service (`forge/`) only, not the Rust runtime.
**Analysis Date:** 2026-11-01

## Applicability of `rules/`

The `rules/*.md` files are Rust-specific (ownership, tokio, thiserror, rustfmt). No Python rules exist. Language-agnostic intents that carry over to Forge: descriptive test names and arrange/act/assert (`rules/test-descriptive-names.md`, `rules/test-arrange-act-assert.md`), lowercase error messages (`rules/err-lowercase-msg.md`), no empty catch (`rules/anti-empty-catch.md`), doc on all public items (`rules/doc-all-public.md`). The Ousterhout deep-module guidance from `CLAUDE.md` applies: keep module interfaces small (e.g. `build_operator`, `upload_to_storage`).

## Naming Patterns

**Files:** lowercase snake_case single-word modules in `forge/src/forge/` (`storage.py`, `queue.py`, `converter.py`). Tests are `forge/tests/test_<module>.py`.
**Functions:** snake_case verbs (`build_operator`, `upload_to_storage`, `validate_model`); private helpers prefixed `_` (`_do_convert`, `_settings`, `_make_minimal_onnx`).
**Classes:** PascalCase (`ForgeSettings`, `ConversionQueue`, `ConvertRequest`); exceptions end in `Error` (`ConversionError` in `forge/src/forge/converter.py`).
**Constants:** SCREAMING_SNAKE (`MAX_CACHED`, `TEST_PREFIX`); module-private use `_` prefix (`_MODEL_ID_PATTERN`).
**Log events:** snake_case event names (`conversion_start`, `conversion_cache_hit`, `conversion_failed`).

## Code Style

- No formatter or linter config detected (no ruff/black/flake8 in `forge/pyproject.toml`). Follow existing style: 4-space indent, double quotes, ~88 col lines, trailing commas in multiline calls.
- Python `>=3.11`; every module starts with a docstring then `from __future__ import annotations`.
- Full type hints on all signatures, using builtin generics (`list[str]`, `dict[str, str]`); `Iterator`/`AsyncIterator` from `typing`.
- Build backend hatchling; deps managed with `uv` (`forge/uv.lock`).

## Import Organization

1. Module docstring, `from __future__ import annotations`
2. stdlib (`asyncio`, `os`, `shutil`)
3. third party (`structlog`, `opendal`, `fastapi`, `pydantic`)
4. first party absolute imports (`from forge.config import ForgeSettings`)

Absolute `forge.*` imports only, no relative imports, no aliases, no barrel files (`forge/src/forge/__init__.py` is minimal). Lazy import with `# noqa` is used once (`import optimum` in `converter.py`).

## Configuration

Settings live in one `pydantic_settings.BaseSettings` subclass, `ForgeSettings` (`forge/src/forge/config.py`). Fields are lowercase with defaults, mapped to uppercase env vars (`STORAGE_TYPE`, `STORAGE_BUCKET`, `STORAGE_PREFIX`, `STORAGE_ROOT`, `STORAGE_REGION`, `STORAGE_CREDENTIAL_PATH`, `CONVERSION_TIMEOUT_SECS`, `LOG_LEVEL`, `HOST`, `PORT`). Optional string settings default to `""` (not `None`) and are treated as unset when falsy. Settings are instantiated once in the lifespan (`forge/src/forge/main.py`) and stored on `app.state.settings`; handlers read them from `request.app.state`. Backend-specific options are gated on `storage_type` in `storage_options()` (`forge/src/forge/storage.py`) so a setting never leaks to the wrong backend (region only for `s3`, credential_path only for `gcs`). Document each new env var in the `ForgeSettings` docstring. Never read `.env` files.

## Storage Pattern

Storage goes through Apache OpenDAL (`opendal.Operator(scheme, **options)`), not cloud SDKs. Adding a backend means: a new `storage_type` value, backend-gated option keys in `storage_options()`, and no changes to `upload_to_storage`. Objects are written as `{model_id}/{relative_path}` with the prefix carried by the Operator `root` (cloud: `/{prefix}`; fs: `{root}/{prefix}`).

## Error Handling

- Domain failures raise `ConversionError` with a lowercase, contextual message and `from exc` chaining (`converter.py`).
- Third-party exceptions are caught broadly at stage boundaries and re-raised as `ConversionError`.
- API layer (`forge/src/forge/api.py`) maps `ConversionError`, `asyncio.TimeoutError`, and generic `Exception` to `HTTPException(500, detail=...)`, logging first, raising `from exc`.
- Request validation is done by Pydantic `field_validator` (`ConvertRequest.validate_model_id` in `forge/src/forge/models.py`), yielding 422. Keep the pattern in sync with the Rust `validate_model_id`.
- Cleanup on failure with `shutil.rmtree(..., ignore_errors=True)` in `queue.py`.

## Logging

`structlog` with JSON rendering, configured in the lifespan (`main.py`). Modules use `logger = structlog.get_logger()` at module level and structured key-value logging: `logger.info("conversion_start", model_id=model_id)`. Never use f-strings in event names; pass fields as kwargs. Level from `LOG_LEVEL`.

## Comments

Google-style-free prose docstrings (reST double-backticks for identifiers, `:class:` refs) on every public function/class. Inline comments reference design decisions by ID (`D-07`, `D-08`, `D-10`) and "pitfall N"; keep that convention when a decision is documented in planning.

## Function / Module Design

- Sync blocking work (conversion, validation, upload) runs via `asyncio.to_thread` from async code (`queue.py`).
- Pydantic `BaseModel` for API payloads (`models.py`); return typed models from handlers with `response_model`.
- App built by `create_app()` factory with a `lifespan` context manager (`main.py`); routes in an `APIRouter` (`api.py`).
- Functions are small and take explicit args (`upload_to_storage(op, model_id, local_dir)`), return plain lists/models.

---

*Convention analysis: 2026-11-01*
