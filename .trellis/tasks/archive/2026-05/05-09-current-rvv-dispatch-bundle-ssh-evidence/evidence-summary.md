# Evidence Summary

## Current HEAD

- Full SHA: `8a05665315e7c57bf9085f3344299cdfb6536806`
- Short SHA: `8a05665`
- Subject: `fix: bind direct callable abi by role`

## Local Validation

- `git diff --check`: passed.
- CMake configure:
  `cmake -S . -B artifacts/tmp/tianchenrv-build -G Ninja -DLLVM_DIR=/usr/lib/llvm-20/lib/cmake/llvm -DMLIR_DIR=/usr/lib/llvm-20/lib/cmake/mlir`
  passed.
- `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`
  passed with 132 discovered tests, 132 passed.

## Dry-Run Bundle Evidence

- Artifact directory:
  `artifacts/tmp/tianchenrv-rvv-dispatch-bundle-e2e/role-bound-bundle-dry-run-8a05665`
- Evidence JSON:
  `artifacts/tmp/tianchenrv-rvv-dispatch-bundle-e2e/role-bound-bundle-dry-run-8a05665/evidence.json`
- `git_sha`: `8a05665315e7c57bf9085f3344299cdfb6536806`
- `mode`: `dry-run`
- `status`: `success`
- `bundle_export_mode`: `plan-and-export-target-artifact-bundle`
- Claim scope: local dry-run verifies bundle export, index parsing, source /
  header / object file discovery, runtime ABI signature parsing, and generated
  external caller construction only.

## Real ssh rvv Bundle Evidence

- Artifact directory:
  `artifacts/tmp/tianchenrv-rvv-dispatch-bundle-e2e/role-bound-bundle-ssh-8a05665`
- Evidence JSON:
  `artifacts/tmp/tianchenrv-rvv-dispatch-bundle-e2e/role-bound-bundle-ssh-8a05665/evidence.json`
- `git_sha`: `8a05665315e7c57bf9085f3344299cdfb6536806`
- `mode`: `ssh`
- `status`: `success`
- `bundle_export_mode`: `plan-and-export-target-artifact-bundle`
- Success marker:
  `tcrv_rvv_scalar_i32_vadd_bundle_external_abi_ok`
- Host/toolchain summary:
  - architecture: `riscv64`
  - clang: `Ubuntu clang version 18.1.3 (1ubuntu1)`
  - uname: `Linux ubuntu 6.12.23 #1 SMP Thu Apr 17 11:46:50 EDT 2025 riscv64 riscv64 riscv64 GNU/Linux`
- Selected compile flags: `-O2 -march=rv64gcv`
- Source-built caller result:
  - `source_run_exit_code`: `0`
  - `source_stdout_marker_observed`: `true`
- Bundle-object caller result:
  - `bundle_object_run_exit_code`: `0`
  - `bundle_object_stdout_marker_observed`: `true`

## Bundle Contract Observed

- Selected bundle records: `source`, `header`, `object`.
- Shared `component_group`:
  `rvv-scalar-i32-vadd-dispatch-external-abi.v1`
- Shared `external_abi_name`:
  `rvv-scalar-i32-vadd-dispatch-runtime-callable-c-function.v1`
- Component roles: `source`, `header`, `object`.
- Runtime ABI parameter roles:
  - `lhs-input-buffer`
  - `rhs-input-buffer`
  - `output-buffer`
  - `runtime-element-count`
  - `dispatch-availability-guard`
- Runtime caller inputs exercised:
  - `rvv_available = 0`
  - `rvv_available = 1`
  - runtime element counts `7` and `16`

## Bounded Claim Scope

This evidence proves only that the current compiler-generated RVV+scalar
i32-vadd dispatch target-artifact bundle external ABI handoff compiled, linked,
and ran on `ssh rvv` for the generated source/header/object artifacts and the
explicit runtime `n = 7` / `n = 16` caller inputs across both dispatch guard
branches.

It does not claim generic RVV lowering, arbitrary-kernel correctness, dynamic
runtime integration, broad RVV support, hardware probing integration,
performance, throughput, latency, or speedup.

## Artifact Hygiene

- Generated evidence, command logs, generated source/header/object files,
  copied remote inputs, hashes, and build outputs remain under `artifacts/tmp`
  and are not committed.
- Sanitization spot-check found no raw credential, private key, URL, token, or
  password-like strings in the dry-run or ssh evidence directories.
