---
last_mapped_commit: ad046b308fac4b7670543698064c366af75a3015
---
# Testing Patterns (Forge service)

**Scope:** This document covers the Python Forge service (`forge/`) only, not the Rust runtime.
**Analysis Date:** 2026-11-01

## Test Framework

- pytest `>=8.0`, pytest-asyncio `>=0.24`, httpx `>=0.27` (dev group in `forge/pyproject.toml`)
- Config: `[tool.pytest.ini_options]` with `testpaths = ["tests"]`, `asyncio_mode = "auto"` (async tests need no marker)
- Assertions: plain `assert`, `pytest.raises(..., match=...)`

```bash
cd forge && uv run pytest tests/ -v          # all tests
cd forge && uv run pytest tests/test_storage.py -v
```

No coverage tooling or threshold configured.

## Layout

```
forge/tests/
├── __init__.py
├── conftest.py        # shared fixtures
├── test_api.py        # FastAPI endpoint contract (httpx ASGITransport)
├── test_converter.py  # validate_model with generated ONNX models
└── test_storage.py    # OpenDAL upload + option mapping
```

Separate `tests/` dir (not co-located), one file per source module, `test_*` function names that describe behavior, docstring on each test. Style is mixed: module-level functions in `test_storage.py`, `Test*` classes in `test_api.py` and `test_converter.py`.

## Fixtures (`forge/tests/conftest.py`)

- `test_settings` - `ForgeSettings(storage_type="memory", storage_prefix="models", ...)`
- `memory_operator` - `opendal.Operator("memory")`, the in-process fake backend
- `tmp_output_dir` - `tempfile.mkdtemp` with rmtree teardown (yield fixture)
- `populated_output_dir` - writes placeholder `model.onnx`, `tokenizer.json`, `config.json`
- `test_api.py` local fixtures: `app` (calls `create_app()` and sets `app.state.settings`/`app.state.queue` manually because ASGITransport does not run lifespan) and async `client` (`AsyncClient(transport=ASGITransport(app=app, raise_app_exceptions=False), base_url="http://test")`).

## Mocking Strategy

- **Storage is not mocked with `unittest.mock`.** OpenDAL's built-in `memory` scheme is the fake: `opendal.Operator("memory")` is passed directly to `upload_to_storage`, then results are verified with `memory_operator.read(path)`.
- **Cloud clients are never contacted.** Backend construction is tested offline: `build_operator()` with `s3`, `gcs`, `fs` succeeds without credentials or network because OpenDAL builds operators lazily (the GCS test uses a nonexistent `credential_path` of `/nonexistent/key.json`).
- **Option mapping is tested as pure data**: call `storage_options(settings)` and assert on the returned dict, including negative assertions (key absent for other backends).
- **Queue/conversion is mocked at the API boundary**: `patch.object(ConversionQueue, "convert", new_callable=AsyncMock, return_value=canned | side_effect=Exc)`.
- Heavy dependencies (optimum export, HF downloads) are not run; `convert_model` is untested. `validate_model` uses a real tiny ONNX graph built with `onnx.helper` (`_make_minimal_onnx` in `test_converter.py`).

## Pattern for a new GCS (or other) backend's tests

Copy `forge/tests/test_storage.py`:

1. Add the backend to the `@pytest.mark.parametrize("storage_type", ["s3", "gcs", "fs"])` in `test_build_operator_supports_storage_type` (use `tmp_path` for `fs` root; GCS needs bucket plus fake `credential_path`).
2. Use the `_settings(storage_type, *, storage_root=..., storage_credential_path=...)` helper that populates every storage field, so leakage between backends is detected.
3. Write one `test_storage_options_passes_<option>_only_to_<backend>` per backend-specific option: assert the value is present for the target backend, absent for others, and absent when the setting is empty (ADC / Workload Identity fallback), with failure messages like `f"... {opts}"`.
4. Test root/prefix composition for cloud backends (`root == "/{prefix}"`) and `bucket` pass-through; these are not yet covered.
5. Keep Arrange / Act / Assert comments and a docstring per test.
6. Exercise upload behavior against `memory_operator`, not a real bucket; do not add network tests.

## Common Patterns

```python
# Async endpoint test (asyncio_mode=auto, no decorator)
async def test_convert_conversion_error_returns_500(self, client):
    with patch.object(ConversionQueue, "convert", new_callable=AsyncMock,
                      side_effect=ConversionError("model.onnx not found")):
        resp = await client.post("/convert", json={"model_id": "org/bad-model"})
    assert resp.status_code == 500

# Error testing
with pytest.raises(ConversionError, match="tokenizer.json not found"):
    validate_model(tmp_output_dir)
```

## Coverage Gaps

- `convert_model` (`forge/src/forge/converter.py`): no test; needs mocked `main_export`/`AutoTokenizer`/`AutoConfig`. Priority: Medium.
- `ConversionQueue` (`forge/src/forge/queue.py`): no direct tests of dedup, semaphore serialization, LRU eviction at `MAX_CACHED`, or temp-dir cleanup on failure. Priority: High.
- `build_operator` only asserts the type; no real GCS/S3 read/write, root-prefix composition (`storage_prefix`) for cloud backends, or `storage_options` for `memory`. Priority: Medium.
- `lifespan` / structlog configuration in `forge/src/forge/main.py` untested (tests bypass lifespan); invalid `LOG_LEVEL` would raise KeyError.
- `ForgeSettings` env-var loading (`monkeypatch.setenv`) untested.
- `validate_model` dummy-inference branch with non-int64 inputs and `onnx/model.onnx` subdirectory path untested.
- `test_settings` fixture in `conftest.py` is unused by `test_api.py`, which duplicates settings construction.

---

*Testing analysis: 2026-11-01*
