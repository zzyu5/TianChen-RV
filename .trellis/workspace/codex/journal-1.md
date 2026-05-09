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


## Session 10: RVV i32 binary microkernel e2e family evidence bridge

**Date**: 2026-05-09
**Task**: RVV i32 binary microkernel e2e family evidence bridge
**Branch**: `main`

### Summary

Generalized `scripts/rvv_microkernel_e2e.py` from a vadd-only evidence bridge
to a bounded direct RVV i32 add/sub/mul family evidence runner. The bridge now
selects `i32-vadd`, `i32-vsub`, or `i32-vmul` via `--arithmetic-family`,
validates family-specific compiler-emitted handoff and bundle metadata, rejects
stale-family artifacts, generates family-correct external callers, and records
family-aware sanitized evidence. A real non-add `ssh rvv` vsub target-artifact
bundle external caller run succeeded.

### Main Changes

- Added a direct RVV microkernel family table covering default input, route ids,
  runtime ABI fields, component group, external ABI name, microkernel op,
  arithmetic op, intrinsic, result vector, success markers, and caller
  arithmetic.
- Generalized manifest, source/header, bundle-index, external-caller,
  command-summary, evidence JSON, and claim-scope validation to the selected
  family.
- Added focused lit coverage for vsub/vmul dry-runs, stale vadd rejection, and
  target-artifact bundle / plan-and-export bundle consumption.
- Updated README and the testing code-spec for the finite add/sub/mul direct
  RVV microkernel evidence bridge.
- Produced real vsub evidence under
  `artifacts/tmp/rvv_microkernel_bundle_e2e/20260509T-rvv-microkernel-vsub-bundle-ssh`.

### Git Commits

| Hash | Message |
|------|---------|
| `this commit` | (see git log) |

### Testing

- [OK] `python3 -m py_compile scripts/rvv_microkernel_e2e.py`
- [OK] `python3 scripts/rvv_microkernel_e2e.py --self-test`
- [OK] focused vadd/vsub/vmul dry-runs and vsub/vmul bundle dry-runs
- [OK] real `ssh rvv` vsub bundle external caller run with source-built and
  bundle-object marker observations
- [OK] focused lit filters `rvv-microkernel` and
  `rvv-microkernel|TargetArtifactBundleExport`
- [OK] `git diff --check`
- [OK] `cmake -S . -B artifacts/tmp/tianchenrv-build -DLLVM_DIR=/usr/lib/llvm-20/lib/cmake/llvm -DMLIR_DIR=/usr/lib/llvm-20/lib/cmake/mlir`
- [OK] `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2` (163/163 tests passed)

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 10: Descriptor-backed i32 binary linalg frontend lowering contract

**Date**: 2026-05-09
**Task**: Descriptor-backed i32 binary linalg frontend lowering contract
**Branch**: `main`

### Summary

Replaced the active bounded linalg frontend lowering owner with the i32 binary
family-named pass surface. The public pass is now
`--tcrv-lower-linalg-i32-binary-to-exec`, with
`createLowerLinalgI32BinaryToExecPass` as the active factory. The old
vadd-named pass remains only as a deprecated compatibility alias that delegates
to the same implementation.

### Main Changes

- Renamed the transform source owner to `LowerLinalgI32BinaryToExec.cpp` and
  added the family-named TableGen pass/factory while preserving the finite
  accepted frontend family markers `i32-vadd`, `i32-vsub`, and `i32-vmul`.
- Updated `tcrv-opt`, `tcrv-translate` plan-and-export, and
  `scripts/rvv_scalar_dispatch_e2e.py` to use the family-named frontend
  lowering surface.
- Migrated active frontend lowering tests to
  `--tcrv-lower-linalg-i32-binary-to-exec` and added one focused compatibility
  alias test for the old vadd-named option.
- Updated README and Trellis specs to describe the bounded i32 binary
  add/sub/mul frontend contract and compatibility boundary.
- Archived the Trellis task with completed PRD/context notes.

### Git Commits

| Hash | Message |
|------|---------|
| `this commit` | feat(transforms): add i32 binary linalg lowering pass |

### Testing

- [OK] `git diff --check`
- [OK] `cmake -S . -B artifacts/tmp/tianchenrv-build -DLLVM_DIR=/usr/lib/llvm-20/lib/cmake/llvm -DMLIR_DIR=/usr/lib/llvm-20/lib/cmake/mlir`
- [OK] `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2` (162/162 tests passed)
- [OK] focused lit from `artifacts/tmp/tianchenrv-build/test` over
  `Transforms/LinalgToExec`, `Target/RVVScalarDispatch`,
  `Target/TargetArtifactBundleExport`, and the two
  `Scripts/rvv-scalar-dispatch*` tests (26/26 tests passed)
- [OK] archived task context validation

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 9: Capability-backed RVV i32m1 policy legality

**Date**: 2026-05-09
**Task**: Capability-backed RVV i32m1 policy legality
**Branch**: `main`

### Summary

Completed the RVV first-slice capability legality module for the existing i32
add/sub/mul RVV microkernel families. The compiler now models SEW=32,
LMUL=m1, tail agnostic, and mask agnostic as stable RVV capability ids and
requires them through RVV proposal, materialized-variant legality, selected
lowering-boundary validation, probe replay, and focused tests.

### Main Changes

- Added plugin-local first-slice capability ids/symbols for
  `rvv.i32_m1.sew32`, `rvv.i32_m1.lmul_m1`,
  `rvv.i32_m1.tail_policy.agnostic`, and
  `rvv.i32_m1.mask_policy.agnostic`.
- Updated the RVV capability profile, plugin proposal, variant legality, and
  selected lowering-boundary validation so RVV i32m1 selected paths fail closed
  before artifact support when required config/policy capabilities are missing,
  disabled, or mismatched.
- Extended `rvv_remote_probe.py` and `rvv_probe_to_mlir.py` to preserve/replay
  first-slice config facts as evidence only; compiler decisions remain in
  C++/MLIR.
- Migrated RVV add/sub/mul, RVV+scalar dispatch, target/export, lowering, and
  emission fixtures to provide the new first-slice capability requirements.
- Updated RVV and capability code-specs with the executable capability/config
  contract and validation matrix.

### Git Commits

| Hash | Message |
|------|---------|
| `this commit` | (see git log) |

### Testing

- [OK] `git diff --check`
- [OK] `cmake -S . -B artifacts/tmp/tianchenrv-build -DLLVM_DIR=/usr/lib/llvm-20/lib/cmake/llvm -DMLIR_DIR=/usr/lib/llvm-20/lib/cmake/mlir`
- [OK] `python3 scripts/rvv_remote_probe.py --self-test`
- [OK] `python3 scripts/rvv_probe_to_mlir.py --self-test`
- [OK] `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`

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

- [OK] `git diff --check`
- [OK] CMake configure with `/usr/lib/llvm-20`
- [OK] focused `tcrv-opt`, `tcrv-translate`, registry, and target artifact
  exporter build targets
- [OK] focused registry and target artifact exporter C++ tests
- [OK] focused lit filter for vmul dispatch, script dry-runs, bundle export,
  and touched add/sub dispatch regressions
- [OK] `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`
- [OK] fresh `ssh rvv` vmul bundle external ABI evidence:
  `artifacts/tmp/tianchenrv-rvv-dispatch-bundle-e2e/vmul-dispatch-ssh-rvv-20260509/evidence.json`

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


## Session 6: Descriptor-backed i32-vmul standalone artifacts

**Date**: 2026-05-09
**Task**: Descriptor-backed i32-vmul standalone artifacts
**Branch**: `main`

### Summary

Implemented descriptor-backed i32-vmul through registry, RVV/scalar plugin materialization, frontend lowering, and standalone RVV/scalar source artifact export; focused checks and check-tianchenrv pass.

### Main Changes

- Added descriptor-backed `i32-vmul` registry facts with distinct RVV/scalar
  op, route, ABI, intrinsic, and scalar operator identities.
- Added RVV `i32_mul` / `i32_vmul_microkernel`, scalar
  `i32_vmul_microkernel`, frontend `arith.muli` lowering, plugin
  proposal/materialization/readiness, and RVV/scalar standalone source exports.
- Added focused C++ and lit coverage for registry, plugins, dialects,
  frontend lowering, target artifact export, and add/sub regression surfaces.

### Git Commits

| Hash | Message |
|------|---------|
| `this commit` | (see git log) |

### Testing

- [OK] `git diff --check`
- [OK] CMake configure with `/usr/lib/llvm-20`
- [OK] focused registry/RVV plugin/scalar plugin/target exporter C++ binaries
- [OK] focused lit filter for vmul plus touched add/sub artifact tests
- [OK] `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 7: i32-vmul RVV scalar dispatch bundle ssh rvv evidence

**Date**: 2026-05-09
**Task**: i32-vmul RVV scalar dispatch bundle ssh rvv evidence
**Branch**: `main`

### Summary

Extended descriptor-backed i32-vmul into the RVV-primary plus scalar-fallback dispatch bundle path, added local route/script/frontend coverage, and captured fresh ssh rvv bundle external ABI evidence for both scalar and RVV guard cases.

### Main Changes

- Registered descriptor-backed vmul RVV+scalar dispatch source/header/object bundle routes from the i32 binary family registry, with distinct route ids, ABI names, component group, function stems, intrinsic, scalar operator, and self-check marker.
- Added vmul direct self-check source/object translate routes and extended the dispatch evidence runner so `--arithmetic-family=i32-vmul` drives existing compiler tools instead of Python compiler semantics.
- Added bounded linalg/frontend-to-dispatch bundle tests, script dry-run coverage, target/export C++ coverage, add/sub regression expectations, and spec wording for the finite add/sub/vmul family.
- Captured fresh `ssh rvv` bundle external ABI evidence under `artifacts/tmp/tianchenrv-rvv-dispatch-bundle-e2e/vmul-dispatch-ssh-rvv-20260509` with source-built and bundle-object paths passing.


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


## Session 8: Descriptor-driven i32 binary dispatch route manifest

**Date**: 2026-05-09
**Task**: Descriptor-driven i32 binary dispatch route manifest
**Branch**: `main`

### Summary

Completed the RVV+scalar i32 add/sub/mul dispatch route manifest, migrated target exporter and tcrv-translate registration to manifest-backed APIs, added stale-family fail-closed coverage, updated the lowering-runtime code-spec, and archived the Trellis task after focused and full checks.

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


## Session 9: Descriptor-backed i32 binary runtime ABI contract

**Date**: 2026-05-09
**Task**: Descriptor-backed i32 binary runtime ABI contract
**Branch**: `main`

### Summary

Completed the descriptor-backed i32 binary runtime ABI boundary. The support
layer now exposes family-aware `I32BinaryRuntimeABIContract` and
`I32BinaryCallableABIPlan` APIs, with descriptor-derived RVV, scalar, and
RVV+scalar dispatch ABI identities for add/sub/mul. Active RVV, scalar,
offload descriptor, dispatch, frontend/runtime-boundary, and target artifact
paths were migrated off vadd-named ABI ownership.

### Main Changes

- Added descriptor-backed i32 binary support APIs for callable parameters,
  role requirements, mem_window/runtime_param specs, dispatch guard parameters,
  callable-plan building, and ABI metadata mirror validation.
- Kept the old `I32VAdd*` support symbols only as explicit temporary wrappers
  around the new i32 binary API.
- Migrated RVV/scalar emission planning and target export preflight,
  RVV+scalar dispatch ABI planning, offload descriptor ABI validation, and
  frontend ABI boundary materialization to `I32Binary*` entry points.
- Updated C++ and lit/FileCheck coverage so add/sub/mul expose distinct
  descriptor-derived ABI identities, stale family/runtime ABI names fail
  closed, and vsub/vmul paths carry family-correct ABI metadata without
  vadd-only sed substitutions.
- Updated durable lowering-runtime/offload specs for descriptor-backed i32
  binary ABI ownership.

### Git Commits

| Hash | Message |
|------|---------|
| `this commit` | (see git log) |

### Testing

- [OK] `git diff --check`
- [OK] `cmake -S . -B artifacts/tmp/tianchenrv-build -DLLVM_DIR=/usr/lib/llvm-20/lib/cmake/llvm -DMLIR_DIR=/usr/lib/llvm-20/lib/cmake/mlir`
- [OK] `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2` (161/161 tests passed)

### Status

[OK] **Completed**

### Next Steps

- None - task complete
