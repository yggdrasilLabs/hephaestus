---
last_mapped_commit: ad046b308fac4b7670543698064c366af75a3015
---
# Technology Stack

**Scope:** This document covers the Forge service (`forge/`) only, not the Rust runtime.

**Analysis Date:** 2026-11-01

## Languages

**Primary:**
- Python >=3.11 (`forge/pyproject.toml` `requires-python`); container runs 3.13 (`forge/Dockerfile`)

**Secondary:**
- None in `forge/`

## Runtime

**Environment:**
- CPython 3.13 (`python:3.13-slim` base image in `forge/Dockerfile`), non-root user `forge`
- ASGI server: uvicorn (`forge.main:create_app --factory`, port 8080)

**Package Manager:**
- uv (copied from `ghcr.io/astral-sh/uv:latest` in the Dockerfile)
- Lockfile: present, `forge/uv.lock` (lock format version 1). Dockerfile uses `uv sync --frozen --no-dev`, falling back to unfrozen sync.

## Frameworks

**Core:**
- FastAPI >=0.115 (locked 0.141.1) - HTTP API (`forge/src/forge/main.py`, `forge/src/forge/api.py`)
- pydantic >=2.0 (locked 2.13.4) + pydantic-settings >=2.0 - request models (`forge/src/forge/models.py`) and env config (`forge/src/forge/config.py`)
- optimum[onnxruntime] >=2.0 (locked 2.1.0) - `main_export` ONNX export (`forge/src/forge/converter.py`)
- transformers (locked 4.57.6, transitive) - `AutoConfig`, `AutoTokenizer`
- structlog >=24.0 (locked 26.1.0) - JSON logging

**Testing:**
- pytest >=8.0 (locked 9.1.1), pytest-asyncio >=0.24 (locked 1.4.0, `asyncio_mode = "auto"`), httpx >=0.27 (locked 0.28.1). Dev dependency group `[dependency-groups] dev`.

**Build/Dev:**
- hatchling build backend; wheel packages `src/forge`

## Key Dependencies

**Critical:**
- opendal >=0.47 (locked 0.47.6) - storage abstraction for S3/GCS/fs (`forge/src/forge/storage.py`)
- onnx >=1.16, onnxruntime >=1.28 (locked 1.29.0) - export validation (`validate_model` in `forge/src/forge/converter.py`)
- torch >=2.13 (override, locked 2.13.0 / 2.13.0+cpu) - via `override-dependencies`; CPU wheels from `https://download.pytorch.org/whl/cpu` (`extra-index-url`)
- numpy - used in `forge/src/forge/converter.py` (transitive)

**Infrastructure:**
- uvicorn[standard] >=0.30 (locked 0.52.4)

## Configuration

**Environment:**
- `pydantic_settings.BaseSettings` subclass `ForgeSettings` in `forge/src/forge/config.py`; fields map case-insensitively to env vars. No `.env` file loading configured.
- Fields: `STORAGE_TYPE` (default `s3`), `STORAGE_BUCKET`, `STORAGE_PREFIX`, `STORAGE_ROOT`, `STORAGE_REGION`, `STORAGE_CREDENTIAL_PATH`, `CONVERSION_TIMEOUT_SECS` (540), `LOG_level` -> `LOG_LEVEL` (info), `HOST` (0.0.0.0), `PORT` (8080)
- `HF_TOKEN` read directly by transformers/optimum (no config field)
- Settings are instantiated in the FastAPI lifespan (`forge/src/forge/main.py`) and stored on `app.state.settings`.

**Build:**
- `forge/pyproject.toml`, `forge/uv.lock`, `forge/Dockerfile` (two-stage: builder with uv, slim runtime; `PYTHONPATH=/app/src`, `HEALTHCHECK` on `/health`), `forge/.dockerignore`

## Platform Requirements

**Development:**
- uv; run tests with `cd forge && uv run pytest tests/ -v` (testpaths `tests`)

**Production:**
- Container image from `forge/Dockerfile`, exposes 8080; single conversion at a time (memory-heavy)

---

*Stack analysis: 2026-11-01*
