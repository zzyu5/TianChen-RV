# Journal - codex (Part 1)

> AI development session journal
> Started: 2026-05-06

---



## Session 1: Initialize TianChen-RV Trellis specs

**Date**: 2026-05-06
**Task**: Initialize TianChen-RV Trellis specs

### Summary

Initialized Trellis for TianChen-RV MLIR and replaced default web specs with long-term capability-driven RISC-V execution layer specs.

### Main Changes

- Added `--arithmetic-family=i32-vadd|i32-vsub` and `--lower-linalg-frontend`
  to `scripts/rvv_scalar_dispatch_e2e.py`.
- Added vsub-specific dispatch self-check translate routes:
  `--tcrv-export-rvv-scalar-i32-vsub-dispatch-self-check-c` and
  `--tcrv-export-rvv-scalar-i32-vsub-dispatch-self-check-object`.
- Extended focused lit coverage for vsub direct self-check, vsub bundle
  external caller generation, and stale vadd semantics rejection in runner
  checks.
- Captured fresh `ssh rvv` target-artifact bundle evidence for generated
  i32-vsub source/header/object plus external caller.

### Git Commits

(No commits - planning session)

### Testing

- [OK] `python3 scripts/rvv_scalar_dispatch_e2e.py --self-test`
- [OK] `python3 -m py_compile scripts/rvv_scalar_dispatch_e2e.py`
- [OK] `git diff --check`
- [OK] CMake configure with LLVM/MLIR 20 paths
- [OK] focused lit filter:
  `rvv-scalar-dispatch-e2e|rvv-scalar-dispatch-bundle-e2e|rvv-scalar-i32-vsub-dispatch-generic-route`
- [OK] `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`
- [OK] `python3 scripts/rvv_scalar_dispatch_e2e.py --use-target-artifact-bundle --use-plan-and-export-bundle-front-door --arithmetic-family=i32-vsub --run-id vsub-dispatch-ssh-rvv-20260509 --overwrite --tcrv-translate artifacts/tmp/tianchenrv-build/bin/tcrv-translate --input test/Target/TargetArtifactBundleExport/plan-linalg-i32-vsub-and-export-target-artifact-bundle.mlir`
- [OK] Archived task validation:
  `.trellis/tasks/archive/2026-05/05-09-rvv-scalar-vsub-dispatch-ssh-rvv-runtime-evidence`

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 5: RVV i32 binary family descriptor registry

**Date**: 2026-05-09
**Task**: RVV i32 binary family descriptor registry
**Branch**: `main`

### Summary

Created a bounded C++ i32 binary family descriptor registry for the existing
`i32-vadd` and `i32-vsub` microkernel families, then migrated real target/export
consumers to read descriptor-backed route, ABI, function-stem, intrinsic,
scalar operator, dispatch, and self-check metadata.

### Main Changes

- Added `include/TianChenRV/Target/I32BinaryFamilyRegistry.h`.
- Migrated `RVVScalarDispatch.cpp`, `RVVMicrokernel.cpp`, and
  `ScalarMicrokernel.cpp` to remove local duplicate add/sub family string
  tables from active target/export code.
- Added a focused C++/lit registry test proving both descriptors are present,
  stale add/sub identities remain distinct, and registered RVV/scalar/dispatch
  exporter routes match descriptor values.

### Testing

- [OK] `git diff --check`
- [OK] `cmake -S . -B artifacts/tmp/tianchenrv-build -DLLVM_DIR=/usr/lib/llvm-20/lib/cmake/llvm -DMLIR_DIR=/usr/lib/llvm-20/lib/cmake/mlir`
- [OK] `cmake --build artifacts/tmp/tianchenrv-build --target tianchenrv-i32-binary-family-registry-test -j2`
- [OK] `artifacts/tmp/tianchenrv-build/bin/tianchenrv-i32-binary-family-registry-test`
- [OK] focused lit filter for registry, add/sub dispatch routes, bundle export, and dispatch scripts
- [OK] `python3 scripts/rvv_scalar_dispatch_e2e.py --self-test`
- [OK] `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`

### Status

[OK] **Completed**

### Next Steps

- Optional continuation: migrate plugin-local RVV/scalar proposal and
  materialization family structs to consume the registry where that does not
  obscure plugin-specific supported-message and materialization wording.


## Session 2: Optimize Hermes Codex module-sized Trellis workflow

**Date**: 2026-05-09
**Task**: Optimize Hermes Codex module-sized Trellis workflow
**Branch**: `main`

### Summary

Shortened the Codex base prompt, changed Hermes reviews to emit module-sized task briefs appended under the base prompt, documented anti-stall/module-owner workflow, validated prompt rendering and no-exec evidence packaging.

### Main Changes

(Add details)

### Git Commits

| Hash | Message |
|------|---------|
| `this commit` | chore(supervisor): use module-sized Trellis task briefs |

### Testing

- [OK] (Add test results)

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 3: RVV/scalar add-sub dispatch artifact path

**Date**: 2026-05-09
**Task**: RVV/scalar add-sub dispatch artifact path
**Branch**: `main`

### Summary

Added family-aware RVV/scalar dispatch artifact exports for bounded i32 vadd/vsub, including vsub composite routes, stale family mismatch rejection, focused lit coverage, and full check-tianchenrv validation.

### Main Changes

### Main Changes

- Made `RVVScalarDispatch.cpp` select add/sub dispatch families from validated RVV and scalar callable artifact metadata.
- Registered vsub-specific RVV/scalar dispatch source/header/object composite routes with vsub component group and ABI identity.
- Added focused frontend-lowered vsub dispatch artifact and target artifact bundle tests, including subtract semantics and stale vadd mismatch rejection.
- Updated target artifact exporter registry tests and legacy vadd diagnostic checks for the expanded add/sub dispatch surface.

### Testing

- [OK] `git diff --check`
- [OK] `cmake -S . -B artifacts/tmp/tianchenrv-build -DLLVM_DIR=/usr/lib/llvm-20/lib/cmake/llvm -DMLIR_DIR=/usr/lib/llvm-20/lib/cmake/mlir -DLLVM_EXTERNAL_LIT=/usr/lib/llvm-20/build/utils/lit/lit.py -DTIANCHENRV_LLVM_LIT=/usr/lib/llvm-20/build/utils/lit/lit.py`
- [OK] `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`
- [OK] `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-09-rvv-scalar-add-sub-dispatch-artifact-path`

### Status

[OK] **Completed**

### Next Steps

- None - task complete


### Git Commits

| Hash | Message |
|------|---------|
| `this commit` | (see git log) |

### Testing

- [OK] (Add test results)

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 4: RVV/scalar vsub dispatch ssh rvv runtime evidence

**Date**: 2026-05-09
**Task**: RVV/scalar vsub dispatch ssh rvv runtime evidence
**Branch**: `main`

### Summary

Added i32-vsub support to the RVV/scalar dispatch evidence runner and direct self-check translate route; captured ssh rvv bundle external ABI correctness evidence for both dispatch guard paths.

### Main Changes

(Add details)

### Git Commits

| Hash | Message |
|------|---------|
| `this commit` | (see git log) |

### Testing

- [OK] (Add test results)

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 5: Plugin i32 binary family descriptor consumption

**Date**: 2026-05-09
**Task**: Plugin i32 binary family descriptor consumption
**Branch**: `main`

### Summary

Migrated RVV and scalar plugin-local i32 add/sub proposal, materialization, readiness, and emission-plan consumers to use the bounded i32 binary family descriptor registry for shared family facts while preserving plugin-local wording, legality, RVV policy/capacity decisions, scalar fallback semantics, and generated behavior.

### Main Changes

(Add details)

### Git Commits

| Hash | Message |
|------|---------|
| `this commit` | (see git log) |

### Testing

- [OK] (Add test results)

### Status

[OK] **Completed**

### Next Steps

- None - task complete
