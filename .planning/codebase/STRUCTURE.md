---
last_mapped_commit: ad046b308fac4b7670543698064c366af75a3015
---
# Codebase Structure (Forge service)

**Scope:** This document covers the Forge service (`forge/`) only, not the Rust runtime.

**Analysis Date:** 2026-11-01

## Directory Layout

```
forge/
├── pyproject.toml      # deps, hatchling build, pytest config (asyncio_mode=auto)
├── uv.lock             # uv lockfile
├── Dockerfile          # multi-stage python:3.13-slim, uv sync, uvicorn factory
├── .dockerignore
├── .gitignore
├── src/forge/          # package (src layout)
│   ├── __init__.py     # __version__
│   ├── main.py         # create_app, lifespan, /health, __main__ runner
│   ├── api.py          # APIRouter: POST /convert
│   ├── queue.py        # ConversionQueue (lock/semaphore/LRU), pipeline driver
│   ├── converter.py    # convert_model, validate_model, ConversionError
│   ├── storage.py      # storage_options, build_operator, upload_to_storage
│   ├── config.py       # ForgeSettings (env vars)
│   └── models.py       # Pydantic request/response models
└── tests/
    ├── conftest.py     # fixtures: test_settings, memory_operator, tmp_output_dir, populated_output_dir
    ├── test_api.py     # httpx AsyncClient + patch.object on queue.convert
    ├── test_converter.py  # validate_model cases
    └── test_storage.py # upload + operator/options per backend
```

## Key File Locations

**Entry Points:** `forge/src/forge/main.py` (`create_app`); container CMD in `forge/Dockerfile`.
**Configuration:** `forge/src/forge/config.py` (`ForgeSettings`), `forge/pyproject.toml`.
**Core Logic:** `forge/src/forge/queue.py`, `forge/src/forge/converter.py`.
**Storage:** `forge/src/forge/storage.py`.
**Testing:** `forge/tests/`; run with `cd forge && uv run pytest tests/ -v`.

## Naming Conventions

- Files: lowercase single-word modules (`storage.py`); tests `test_<module>.py`.
- Functions snake_case; classes PascalCase; constants UPPER_SNAKE (`MAX_CACHED`) or leading underscore for private (`_MODEL_ID_PATTERN`).
- Every module starts with a docstring and `from __future__ import annotations`.
- Env vars map to lowercase settings fields (`STORAGE_TYPE` -> `storage_type`).
- Log events are snake_case names with keyword context.

## Where to Add New Code

**New/extended storage backend (e.g. GCS):**
- Backend selection is data-driven: `STORAGE_TYPE` is passed straight to `opendal.Operator` in `build_operator` (`forge/src/forge/storage.py:45`). GCS (`gcs`) already works through this path with `STORAGE_BUCKET`, `STORAGE_PREFIX`, `STORAGE_CREDENTIAL_PATH`.
- New backend-specific option: add a field to `ForgeSettings` in `forge/src/forge/config.py` (document it in the class docstring), then add a gated branch in `storage_options` in `forge/src/forge/storage.py:12` (follow the `if settings.storage_type == "gcs"` pattern so options never reach other backends).
- Do not change call sites: `forge/src/forge/queue.py:91-97` is the only consumer.
- Tests: extend `forge/tests/test_storage.py` (add the scheme to the parametrized `test_build_operator_supports_storage_type`, add an options-scoping test; `_settings()` helper populates all fields). Use `memory`/`fs` operators for upload tests via `memory_operator` fixture in `forge/tests/conftest.py`.
- Deployment: env var docs live in the repo-root CLAUDE.md table; container in `forge/Dockerfile` needs no change unless a non-OpenDAL client is added.

**New endpoint:** add route in `forge/src/forge/api.py` (or a new router module included in `create_app`, `main.py:53`); models in `forge/src/forge/models.py`.
**New conversion/validation step:** `forge/src/forge/converter.py`, wired in `ConversionQueue._do_convert` (`queue.py:76`) via `asyncio.to_thread`.
**New setting:** `forge/src/forge/config.py`.
**New tests:** `forge/tests/test_<module>.py`, shared fixtures in `conftest.py`.

## Special Directories

- `forge/.venv/`: uv virtualenv; generated, not committed (ignored).
- `forge/tests/__pycache__`, `.pytest_cache`: generated.

---

*Structure analysis: 2026-11-01*
