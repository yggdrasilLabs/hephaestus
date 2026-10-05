---
phase: quick
plan: 261005-h8p
subsystem: storage
status: complete
tags: [storage, opendal, gcs, forge, config]
requires: []
provides:
  - "Config::storage_operator() -- single entry point for OpenDAL operator construction"
  - "STORAGE_CREDENTIAL_PATH env var (gcs-only credential_path) in Hephaestus and Forge"
  - "Forge /convert wire field storage_paths"
affects:
  - crates/hephaestus
  - crates/hephaestus-resolve
  - forge
tech-stack:
  added:
    - "opendal services-gcs feature (opendal-service-gcs 0.58.2, reqsign-google, rsa 0.9.10 + RustCrypto support crates)"
  patterns:
    - "Allowlist const drives both validation error text and a compiled-feature test"
    - "Backend-specific OpenDAL keys scoped per storage type (region -> s3, credential_path -> gcs)"
key-files:
  created: []
  modified:
    - Cargo.toml
    - Cargo.lock
    - crates/hephaestus/src/config.rs
    - crates/hephaestus/src/main.rs
    - crates/hephaestus-resolve/src/storage.rs
    - crates/hephaestus-resolve/src/forge.rs
    - crates/hephaestus-resolve/src/resolver.rs
    - crates/hephaestus-resolve/src/lib.rs
    - forge/src/forge/models.py
    - forge/src/forge/queue.py
    - forge/src/forge/api.py
    - forge/src/forge/config.py
    - forge/src/forge/storage.py
    - forge/tests/test_api.py
    - forge/tests/test_storage.py
    - README.md
    - Claude.md
decisions:
  - "ALLOWED_STORAGE_TYPES hoisted to a module const (s3, fs, gcs, none); validate() error text is derived from it"
  - "Operator construction lives in Config::storage_operator(); main.rs no longer knows OpenDAL option keys"
  - "region is passed only to s3 and credential_path only to gcs, in both Rust and Forge"
  - "Forge wire field renamed s3_paths -> storage_paths with no compat alias (both services ship together)"
  - "TDD tasks committed as one buildable commit each rather than separate RED/GREEN commits; RED was verified before implementation"
metrics:
  duration: "~15min"
  completed: 2026-10-05
  tasks: 3
  files: 17
requirements: [GCS-01, GCS-02, GCS-03, GCS-04, GCS-05, GCS-06, GCS-07]
---

# Quick Task 261005-h8p: Add GCS storage support via OpenDAL and drop azblob Summary

GCS is now a working OpenDAL storage backend in both Hephaestus and Forge. A new gcs-only `STORAGE_CREDENTIAL_PATH` option falls back to ADC / Workload Identity when unset. Region is passed to s3 only, azblob is removed, and the Forge wire field is now the backend-neutral `storage_paths`. A test fails CI if any allowed storage type lacks its compiled opendal feature.

## Tasks

| # | Task | Commit | Key files |
|---|------|--------|-----------|
| 1 | Enable opendal gcs, drop azblob, extract tested `Config::storage_operator()` | 93d4f0d | Cargo.toml, Cargo.lock, config.rs, main.rs, hephaestus-resolve/src/storage.rs |
| 2 | Rename Forge wire field to `storage_paths` (Python + Rust), neutral wording | deba629 | models.py, queue.py, api.py, test_api.py, forge.rs, resolver.rs, lib.rs |
| 3 | Forge gcs-aware `storage_options()` + README/Claude.md docs | 8153e45 | config.py, storage.py, test_storage.py, README.md, Claude.md |

## What was built

- **Rust config (`crates/hephaestus/src/config.rs`)**: module-level `ALLOWED_STORAGE_TYPES = ["s3", "fs", "gcs", "none"]`; `validate()` error lists values from the const. New `storage_credential_path` field. Private `storage_options()` maps env config to OpenDAL keys (bucket always; region s3-only; credential_path gcs-only; root mapping unchanged). Public `storage_operator()` returns `Ok(None)` for `none`, otherwise builds via `Operator::via_iter` with a 3-retry `RetryLayer`.
- **main.rs**: step 2c is now `let operator = config.storage_operator()?;`. The unused `HashMap` import was removed, and `storage_credential_path` (the path only) is added to the startup config log.
- **Forge**: `storage_options(settings)` mirrors the Rust mapping; `build_operator` delegates to it; `ForgeSettings.storage_credential_path` added.
- **Wire contract**: `ConvertResponse.storage_paths` / `ForgeResponse.storage_paths`. Resolver logs and docs now say "storage" instead of "S3".
- **Docs**: README features/stack/env table updated, new `STORAGE_CREDENTIAL_PATH` row, new "Run with GCS storage cache" section; Claude.md key env vars table updated.

## Verification

- RED confirmed for Task 1: config tests failed to compile before implementation (missing field, method, const).
- Feature-drift guard checked by hand: temporarily removing `services-gcs` makes `storage_operator_builds_for_every_allowed_storage_type` fail with `scheme: gcs ... scheme is not registered`. The feature was restored afterwards.
- RED confirmed for Task 3: `ImportError: cannot import name 'storage_options'`.
- `cargo build --workspace`: Finished, no warnings.
- `cargo test --workspace`: all suites ok. Totals: hephaestus 31, hephaestus-api 38 + integration (1 metrics, 2 tracing, others ignored as before), hephaestus-core 69, hephaestus-proto 3, hephaestus-resolve 49. 0 failed.
- `cd forge && uv run pytest tests/ -v`: 22 passed.
- `cargo clippy -p hephaestus --all-targets`: no findings in config.rs/main.rs.
- Cargo.lock: `opendal-service-gcs` `version = "0.58.2"`, `source = "registry+https://github.com/rust-lang/crates.io-index"`. No existing package versions changed (0 removed `version`/`checksum` lines); opendal was not bumped.
- No `s3_paths` references remain under crates/ or forge/src, forge/tests; no azblob/Azure in README.md or Claude.md.
- Commit messages contain no Co-Authored-By trailer or AI attribution.

## Deviations from Plan

### Process adjustments

**1. Single commit per TDD task instead of separate test/feat commits**
- **Reason:** The orchestrator required one atomic commit per task. A separate RED commit would have left a non-compiling commit in history (the Rust tests reference a field and methods that do not exist yet).
- **Mitigation:** RED was still run and observed before each implementation (see Verification).

**2. Verify commands run against the worktree path**
- The plan's `<verify>` blocks hard-code the main repo path. They were run with the same checks from the worktree root.

Otherwise the plan was executed as written.

## Known Stubs

None.

## Threat Flags

None. All new surface (the credential path option and the new transitive crates) is covered by T-h8p-01..T-h8p-SC in the plan's threat model. The mitigations are implemented: the allowlist const plus the every-type test, option scoping with unit tests on both sides, only the path is logged, and the lockfile provenance was checked.

## Self-Check: PASSED

- FOUND: crates/hephaestus/src/config.rs (`pub fn storage_operator`, `storage_operator_builds_for_every_allowed_storage_type`)
- FOUND: crates/hephaestus-resolve/src/forge.rs (`pub storage_paths: Vec<String>`)
- FOUND: forge/src/forge/storage.py (`credential_path`)
- FOUND: forge/src/forge/models.py (`storage_paths: list[str]`)
- FOUND commits: 93d4f0d, deba629, 8153e45
