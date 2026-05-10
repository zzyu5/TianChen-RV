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


## Session 13: RVV i64 binary sub/mul frontend artifact ssh evidence

**Date**: 2026-05-10
**Task**: RVV i64 binary sub/mul frontend artifact and ssh evidence
**Branch**: `main`

### Summary

Completed bounded `i64-vsub` and `i64-vmul` RVV binary family support through
the descriptor-backed compiler path and collected real `ssh rvv` correctness
evidence for compiler-generated linalg frontend artifacts.

### Main Changes

- Created Trellis task `05-10-rvv-i64-binary-sub-mul-frontend-artifact-ssh-evidence`.
- Added `i64-vsub` and `i64-vmul` descriptors to the RVV-owned binary family
  registry with family ids, frontend lowering strings, lowering descriptors,
  route ids, header/object routes, runtime ABI names, glue roles, component
  groups, C ABI metadata, intrinsic prefixes, and function/header stems.
- Added descriptor-backed i64m1 intrinsic helpers for `__riscv_vsub_vv_i64m1`
  and `__riscv_vmul_vv_i64m1`.
- Added finite RVV dialect ops and verifiers for `tcrv_rvv.i64_sub`,
  `tcrv_rvv.i64_mul`, `tcrv_rvv.i64_vsub_microkernel`, and
  `tcrv_rvv.i64_vmul_microkernel`.
- Generalized RVV plugin and target microkernel/export logic so i64 add/sub/mul
  materialization, readiness, route validation, and emission plans are selected
  from the RVV family descriptor instead of i64-vadd-only branches.
- Added linalg frontend artifact lit tests, explicit microkernel lit fixtures,
  a missing-i64m1 fail-closed test, C++ registry/plugin/exporter coverage, and
  script dry-run coverage for the new frontend evidence modes.

### Evidence

- `ssh rvv` probe passed: host `ubuntu`, arch `riscv64`, compiler `/usr/bin/clang`.
- `i64-vsub` real frontend evidence passed:
  `python3 scripts/rvv_microkernel_e2e.py --arithmetic-family=i64-vsub --lower-linalg-frontend --expect-selected-kernel frontend_i64_vsub --run-id codex-i64-vsub-frontend-ssh --overwrite`
  Evidence dir: `artifacts/tmp/rvv_microkernel_e2e/codex-i64-vsub-frontend-ssh`.
- `i64-vmul` real frontend evidence passed:
  `python3 scripts/rvv_microkernel_e2e.py --arithmetic-family=i64-vmul --lower-linalg-frontend --expect-selected-kernel frontend_i64_vmul --run-id codex-i64-vmul-frontend-ssh --overwrite`
  Evidence dir: `artifacts/tmp/rvv_microkernel_e2e/codex-i64-vmul-frontend-ssh`.

### Testing

- [OK] `git diff --check`
- [OK] `cmake --build artifacts/tmp/tianchenrv-build --target tcrv-opt tcrv-translate -j2`
- [OK] `artifacts/tmp/tianchenrv-build/bin/tianchenrv-i32-binary-family-registry-test`
- [OK] `artifacts/tmp/tianchenrv-build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `artifacts/tmp/tianchenrv-build/bin/tianchenrv-target-artifact-export-test`
- [OK] focused lit filter for new i64-vsub/i64-vmul, existing i64-vadd/i32,
  and `rvv-microkernel-e2e` tests passed 37/37 selected tests.
- [OK] `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2` passed 184/184.

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 13: RVV binary family registry owner

**Date**: 2026-05-10
**Task**: Promote RVV binary family ownership out of i32 legacy registry
**Branch**: `main`

### Summary

Moved RVV binary family ownership into a target-local RVV registry for the four
currently supported RVV binary families: i32-vadd, i32-vsub, i32-vmul, and
i64-vadd. Frontend lowering, RVV plugin emission/proposal paths, RVV target
microkernel export, and target artifact export tests now consume RVV-owned
descriptor metadata instead of combining the legacy i32 registry with an
i64-vadd special case.

### Main Changes

- Added `include/TianChenRV/Target/RVV/RVVBinaryFamilyRegistry.h` with finite
  family descriptors, lookup by family id/frontend lowering/lowering descriptor,
  dtype/arithmetic metadata, route IDs, runtime ABI names, scalar C types, and
  selected shape hooks.
- Refactored `RVVBinaryDescriptor.h` so it no longer includes
  `I32BinaryFamilyRegistry.h`; generic RVV binary descriptors now derive
  callable ABI, buffer window, intrinsic, shape, and runtime parameter metadata
  from the RVV family descriptor.
- Updated `LowerLinalgI32BinaryToExec.cpp`, `RVVExtensionPlugin.cpp`, and
  `RVVMicrokernel.cpp` to consume the RVV-owned family interface for RVV
  frontend markers and microkernel/export metadata while keeping the legacy i32
  registry only where scalar/dispatch compatibility still owns behavior.
- Preserved existing artifact spellings, including i32 selected-shape comments
  and role-bound runtime `c_name` behavior; repaired stale emission-plan
  diagnostics to keep the prior callable-plan error wording.
- No `ssh rvv` evidence was run or claimed because this round changed ownership
  and metadata consumption only, not generated runtime behavior or evidence
  runner semantics.

### Git Commits

| Hash | Message |
|------|---------|
| `this commit` | (see git log) |

### Testing

- [OK] `git diff --check`
- [OK] `cmake --build artifacts/tmp/tianchenrv-build --target tcrv-opt tcrv-translate -j2`
- [OK] `artifacts/tmp/tianchenrv-build/bin/tianchenrv-i32-binary-family-registry-test`
- [OK] `artifacts/tmp/tianchenrv-build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `artifacts/tmp/tianchenrv-build/bin/tianchenrv-target-artifact-export-test`
- [OK] focused lit filter
  `linalg-i32-vadd-to-exec|linalg-i64-vadd-to-rvv-artifact|linalg-i32-vsub-to-rvv-artifact`
- [OK] focused self-repair lit filter
  `rvv-microkernel-runtime-abi-role-binding|rvv-scalar-i32-vadd-dispatch-runtime-abi-role-binding`
- [OK] `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2` passed 179/179
- [OK] `python3 .trellis/scripts/task.py validate .trellis/tasks/05-10-rvv-binary-family-registry-owner`

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 14: RVV capability-driven vector-shape proposal and selection

**Date**: 2026-05-10
**Task**: RVV capability-driven vector-shape proposal and selection
**Branch**: `main`

### Summary

Moved bounded RVV i32 binary vector-shape choice further upstream into plugin
proposal/legality. The RVV plugin can now consume a structured selected-shape
capability selector when both finite i32m1 and i32m2 config facts are present,
validate that selector against the target-owned `RVVVectorShape` descriptors,
and emit descriptor-derived selected shape capability requirements.

### Main Changes

- Added finite selector capability contract
  `rvv.i32_binary.selected_vector_shape` with bounded `shape = "i32m1" |
  "i32m2"` semantics in the RVV spec and C++ target/RVV helper.
- Updated `RVVExtensionPlugin` proposal selection so an available selector
  drives the selected finite config; selector-requested incomplete i32m2 facts
  now decline/fail closed instead of silently falling back to i32m1.
- Kept no-selector behavior compatible: existing i32m1 fixtures still select
  the deterministic default when only i32m1 config facts are available.
- Routed proposal required capability ids through the composed
  `RVVI32BinaryIntrinsicDescriptor` so add/sub/mul proposal requirements stay
  aligned with descriptor-owned selected shape metadata.
- Added C++ and lit coverage for selector-driven i32m2 with both shapes
  present, default i32m1 without selector, planned IR/lowering-boundary
  metadata agreement, and missing i32m2 capability rejection.
- No new `ssh rvv` evidence was run; this changed compiler planning metadata
  selection and preserved existing runtime/export routes without making a new
  runtime correctness or performance claim.

### Git Commits

| Hash | Message |
|------|---------|
| `this commit` | (see git log) |

### Testing

- [OK] `git diff --check`
- [OK] `cmake --build artifacts/tmp/tianchenrv-build --target tcrv-opt tcrv-translate tianchenrv-rvv-extension-plugin-test -j2`
- [OK] `artifacts/tmp/tianchenrv-build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] focused lit filter `plugin-variant-materialization-rvv-selected-shape|linalg-i32-vsub-to-rvv-artifact|plan-linalg-i32m2-vmul|plan-linalg-i32m2-vsub` passed 5/5 selected tests
- [OK] `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2` passed 175/175
- [NOTE] `clang-format` was not installed in this environment; formatting was kept manually and `git diff --check` passed.

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 13: RVV i32 binary descriptor registry across family and shape

**Date**: 2026-05-10
**Task**: RVV i32 binary intrinsic descriptor registry across family and vector shape
**Branch**: `main`

### Summary

Completed the descriptor-structured RVV i32 add/sub/mul family x vector-shape
path. The compiler/exporter now has a target/RVV-owned
`RVVI32BinaryIntrinsicDescriptor` that composes the existing
`I32BinaryFamilyRegistry` and `RVVVectorShape` truth instead of letting the
microkernel and dispatch exporters independently rebuild full intrinsic names,
shape comments, or self-check arithmetic.

### Main Changes

- Added `include/TianChenRV/Target/RVV/RVVI32BinaryDescriptor.h`.
- Routed RVV microkernel source/self-check emission through descriptor-derived
  `__riscv_vsetvl_*`, load/store, full arithmetic intrinsic, selected-shape
  comments, and arithmetic check expressions.
- Routed RVV+scalar dispatch embedded RVV source validation and self-check
  arithmetic through the same descriptor projection.
- Extended C++ descriptor coverage to prove all add/sub/mul x i32m1/i32m2
  combinations and the specific `i32-vmul` + `i32m2` full intrinsic
  `__riscv_vmul_vv_i32m2`.
- Added `plan-linalg-i32m2-vmul-and-export-target-artifact-bundle.mlir` as the
  non-vsub selected-shape front-door proof.
- Extended `scripts/rvv_scalar_dispatch_e2e.py` default fixture routing for
  `--arithmetic-family=i32-vmul --vector-shape=i32m2`.
- Produced bounded real `ssh rvv` source/object external ABI evidence under
  `artifacts/tmp/tianchenrv-rvv-dispatch-bundle-e2e/codex-i32m2-vmul-ssh-20260510T0001/`.
- No `.trellis/spec/` update was needed; the existing specs already describe
  descriptor-derived i32 binary ABI identity, selected vector-shape metadata,
  focused plan-and-export coverage, and bounded RVV evidence.

### Git Commits

| Hash | Message |
|------|---------|
| `this commit` | (see git log) |

### Testing

- [OK] `git diff --check`
- [OK] `cmake --build artifacts/tmp/tianchenrv-build --target tcrv-opt tcrv-translate tianchenrv-i32-binary-family-registry-test -j2`
- [OK] `artifacts/tmp/tianchenrv-build/bin/tianchenrv-i32-binary-family-registry-test`
- [OK] `python3 -m py_compile scripts/rvv_scalar_dispatch_e2e.py`
- [OK] focused lit filter `I32BinaryFamilyRegistry|plan-linalg-i32m2-vmul|plan-linalg-i32m2-vsub|rvv-scalar-dispatch-bundle-e2e` passed 4/4 selected tests
- [OK] focused lit filter `rvv-microkernel-family-mul|rvv-microkernel-i32m2-family-sub|rvv-microkernel-i32m2-object|rvv-scalar-i32-vmul-dispatch-generic-route|rvv-scalar-i32-vsub-dispatch-i32m2-generic-route` passed 6/6 selected tests
- [OK] real ssh evidence command:
  `python3 scripts/rvv_scalar_dispatch_e2e.py --use-target-artifact-bundle --use-plan-and-export-bundle-front-door --arithmetic-family=i32-vmul --vector-shape=i32m2 --run-id codex-i32m2-vmul-ssh-20260510T0001 --overwrite --timeout 120`
- [OK] read-only evidence JSON assertion for runtime success, source/object
  external ABI success markers, branch/count coverage, selected family/shape,
  required intrinsic, and obvious secret-like text absence
- [OK] `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2` passed 173/173

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 12: Bounded linalg i32-vsub i32m2 frontend bundle proof

**Date**: 2026-05-10
**Task**: Bounded linalg i32 binary frontend to selected RVV scalar artifact pipeline
**Branch**: `main`

### Summary

Created and completed Trellis task `bounded-linalg-i32-binary-rvv-scalar-artifact-pipeline` for internal case `switch module`. The full one-command `tcrv-plan-and-export-target-artifact-bundle` route already lowered bounded linalg, ran execution planning, selected RVV+scalar dispatch, and exported the bundle, so this round tightened the lit proof around the existing compiler-owned path instead of adding duplicate plumbing.

### Main Changes

- Strengthened `plan-linalg-i32m2-vsub-and-export-target-artifact-bundle.mlir` so the positive bundle test now checks selected RVV `i32m2` shape metadata, selected-plan metadata, runtime ABI parameter roles, dispatch mem-window roles, dispatch guard linkage, RVV capability continuity, scalar fallback selected boundary, and source/header/object bundle records.
- Added `plan-linalg-i32-vsub-marker-mismatch-no-bundle.mlir` to prove the one-command front door rejects an `i32-vsub` marker with an `arith.addi` body during bounded linalg lowering, before bundle index or selected RVV+scalar artifact output is produced.
- No new runtime-visible generated source/object/ABI behavior was introduced, so no new `ssh rvv` claim was made; `5db3128` remains the latest runtime evidence for the affected `i32-vsub i32m2` dispatch path.

### Validation

- [OK] `git diff --check`
- [OK] `python3 ./.trellis/scripts/task.py validate 05-10-bounded-linalg-i32-binary-rvv-scalar-artifact-pipeline`
- [OK] `cmake --build artifacts/tmp/tianchenrv-build --target tcrv-opt tcrv-translate -j2`
- [OK] Focused lit filter `plan-linalg-i32m2-vsub-and-export-target-artifact-bundle|plan-linalg-i32-vsub-marker-mismatch-no-bundle` passed 2/2 from `artifacts/tmp/tianchenrv-build/test`
- [OK] `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2` passed 172/172

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 14: RVV i32m2 vsub dispatch artifact path

**Date**: 2026-05-10
**Task**: RVV i32m2 vsub dispatch artifact path
**Branch**: `main`

### Summary

Extended the RVV+scalar dispatch evidence bridge so frontend-generated
`i32-vsub` dispatch can be validated and executed with the selected RVV
`i32m2` shape through both direct self-check and target-artifact bundle paths.

### Main Changes

- Added explicit `--vector-shape={i32m1,i32m2}` support to
  `scripts/rvv_scalar_dispatch_e2e.py`, including shape-aware intrinsic
  validation, selected-shape metadata validation, and evidence-level
  `rvv_config`.
- Added checked-in `i32m2` linalg-origin dispatch and plan-and-export bundle
  fixtures for `i32-vsub`.
- Extended script lit coverage for positive `i32m2` dry-runs, negative
  `i32m2` over `i32m1` rejection, and bundle front-door evidence.
- Archived Trellis task
  `.trellis/tasks/archive/2026-05/05-10-rvv-i32m2-vsub-dispatch-artifact-path/`.

### Validation

- [OK] `python3 -m py_compile scripts/rvv_scalar_dispatch_e2e.py`
- [OK] `python3 scripts/rvv_scalar_dispatch_e2e.py --self-test`
- [OK] direct local dry-run for `--arithmetic-family=i32-vsub --vector-shape=i32m2`
- [OK] negative local dry-run rejected `--vector-shape=i32m2` over the `i32m1` input
- [OK] bundle front-door local dry-run for `i32-vsub i32m2`
- [OK] focused lit filter `rvv-scalar-dispatch-e2e|rvv-scalar-dispatch-bundle-e2e|rvv-scalar-i32-vsub-dispatch-i32m2|plan-linalg-i32m2-vsub` passed 4/4
- [OK] real `ssh rvv` direct dispatch self-check:
  `artifacts/tmp/rvv_scalar_dispatch_e2e/20260510-rvv-scalar-vsub-i32m2-dispatch-ssh/evidence.json`
- [OK] real `ssh rvv` bundle external caller:
  `artifacts/tmp/tianchenrv-rvv-dispatch-bundle-e2e/20260510-rvv-scalar-vsub-i32m2-dispatch-bundle-ssh/evidence.json`
- [OK] `python3 .trellis/scripts/task.py validate 05-10-rvv-i32m2-vsub-dispatch-artifact-path`
- [OK] `git diff --check`
- [OK] `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2` passed 171/171

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 13: Selected RVV plus scalar dispatch ABI artifact boundary

**Date**: 2026-05-10
**Task**: Selected RVV plus scalar dispatch ABI artifact boundary
**Branch**: `main`

### Summary

Completed the bounded `i32-vsub` RVV-preferred plus scalar-fallback dispatch
artifact boundary. The target-owned dispatch exporter now preserves structured
selected-plan metadata from the RVV callable candidate and scalar fallback
metadata from the selected `tcrv.exec.fallback` op at the outer dispatch C
artifact surface.

### Main Changes

- Created Trellis task
  `.trellis/tasks/05-10-selected-rvv-scalar-dispatch-abi-artifact-boundary/`
  with PRD and curated implement/check spec context.
- Extended `lib/Target/Builtin/RVVScalarDispatch.cpp` so dispatch source
  comments include each callable candidate's `selectedPlanMetadata`.
- Captured scalar fallback `origin` and `fallback_role` from the resolved
  dispatch IR link and emitted them as dispatch fallback metadata.
- Strengthened the `i32-vsub` focused route fixture to check RVV selected
  vector-shape metadata, base RVV capacity metadata, scalar fallback metadata,
  callable symbols, and a malformed selected-shape fail-closed case before
  source banner emission.

### Validation

- [OK] `cmake --build artifacts/tmp/tianchenrv-build --target tcrv-translate tcrv-opt -j2`
- [OK] focused lit filter
  `rvv-scalar-i32-vsub-dispatch-generic-route` passed 1/1
- [OK] `artifacts/tmp/tianchenrv-build/bin/tianchenrv-target-artifact-export-test`
- [OK] `git diff --check`
- [OK] `python3 ./.trellis/scripts/task.py validate 05-10-selected-rvv-scalar-dispatch-abi-artifact-boundary`
- [OK] `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`
  passed 169/169

No new `ssh rvv` evidence was required or run; this round changed compiler
artifact metadata/export validation coverage, not generated runtime dispatch
object behavior or runtime correctness claims.

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 11: RVV i32 family intrinsic prefix contract

**Date**: 2026-05-10
**Task**: RVV i32 family intrinsic prefix contract
**Branch**: `main`

### Summary

Separated RVV i32 arithmetic family identity from selected vector-shape suffix
ownership. The family registry now records suffix-free RVV arithmetic intrinsic
prefixes, and RVV target source emission composes the final intrinsic from the
family prefix plus the validated selected vector-shape suffix.

### Main Changes

- Replaced RVV family descriptor `intrinsicName` strings such as
  `__riscv_vadd_vv_i32m1` with suffix-free `arithmeticIntrinsicPrefix` values.
- Removed the local arithmetic intrinsic prefix switch in
  `lib/Target/RVV/RVVMicrokernel.cpp`; emission now consumes the registry-owned
  family prefix and the selected vector-shape suffix from validated RVV config.
- Strengthened `tianchenrv-i32-binary-family-registry-test` so add/sub/mul
  prefixes are distinct, end at the suffix boundary, and do not bake in
  `i32m1` or `i32m2`.
- Updated RVV plugin and lowering/runtime specs to state that full intrinsic
  spellings are emission results, not family descriptor fields.
- Completed and archived Trellis task
  `.trellis/tasks/archive/2026-05/05-10-rvv-i32-family-intrinsic-prefix-contract/`.

Validation:
- [OK] `git diff --check`
- [OK] `cmake --build artifacts/tmp/tianchenrv-build --target tianchenrv-i32-binary-family-registry-test -j2`
- [OK] `artifacts/tmp/tianchenrv-build/bin/tianchenrv-i32-binary-family-registry-test`
- [OK] Focused lit on linalg i32 vadd/vsub/vmul and RVV i32m2 source export (4/4)
- [OK] `python3 ./.trellis/scripts/task.py validate 05-10-rvv-i32-family-intrinsic-prefix-contract`
- No new `ssh rvv` run was collected; emitted source semantics and runtime
  claims did not change.

### Git Commits

| Hash | Message |
|------|---------|
| `this commit` | (see git log) |

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 11: RVV selected vector-shape config boundary cleanup

**Date**: 2026-05-10
**Task**: RVV selected vector-shape config boundary cleanup
**Branch**: `main`

### Summary

Completed the RVV selected vector-shape boundary cleanup for existing i32m1
and i32m2 i32 binary flows. The selected compile-time shape is now explicit
plugin/target-owned metadata, while base i32 M1 lane capacity remains a
separate hardware/profile fact.

### Main Changes

- Added shared `RVVI32VectorShapeConfig` for the finite i32m1/i32m2 RVV
  selected shape contract.
- Threaded selected-shape metadata through RVV proposals, materialized
  variants, `tcrv_rvv.lowering_boundary`, i32 add/sub/mul microkernel ops,
  emission-plan selected metadata, target source comments, bundle metadata,
  and evidence validation.
- Renamed plugin-local capacity metadata from selected-looking
  `i32_m1_lanes` wording to `base_i32_m1_lanes`, while preserving
  `rvv.i32_m1_lane_count` as the hardware/profile capacity fact.
- Added i32m2 positive FileCheck coverage and a stale selected i32m1 metadata
  guard that fails before target source output.
- Updated capability/RVV/emission specs to lock the parameter-layer boundary.

### Testing

- [OK] `git diff --check`
- [OK] `artifacts/tmp/tianchenrv-build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `artifacts/tmp/tianchenrv-build/bin/tianchenrv-emission-readiness-test`
- [OK] focused lit filter covering linalg i32m1/i32m2 planning/export, stale
  selected-shape guard, RVV microkernel i32m2 export, probe, and bundle tests
  (11/11)
- [OK] `python3 scripts/rvv_microkernel_e2e.py --self-test`
- [OK] focused lit filter `rvv-microkernel-bundle-e2e` (1/1)
- [OK] `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`
  (169/169 tests passed)
- [OK] real `ssh rvv` linalg-origin i32-vsub i32m2 bundle evidence:
  `artifacts/tmp/rvv_microkernel_bundle_e2e/codex-shape-boundary-linalg-vsub-i32m2/evidence.json`

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 12: Linalg i32m2 vsub SSH RVV evidence handoff

**Date**: 2026-05-10
**Task**: Linalg i32m2 vsub SSH RVV evidence handoff
**Branch**: `main`

### Summary

Extended the RVV microkernel evidence runner so frontend-generated linalg-origin
bundle runs can fail closed on an expected selected kernel before evidence is
accepted. Collected real `ssh rvv` external-ABI evidence for the
`linalg-i32-vsub-to-rvv-artifact.mlir` i32m2 path and recorded bounded
compiler-path context in evidence JSON.

### Main Changes

- Added `--expect-selected-kernel` to `scripts/rvv_microkernel_e2e.py` and
  validated the generated source `selected_kernel` comment before accepting
  evidence.
- Recorded compiler-path context in evidence JSON:
  selected kernel, selected variant, selected role, lowering boundary, active
  route, and callable ABI source.
- Added focused lit coverage for the linalg-origin i32-vsub i32m2 bundle dry
  run plus a negative selected-kernel mismatch case.
- Collected real `ssh rvv` evidence for the frontend-generated
  `linalg.generic` i32-vsub -> RVV i32m2 bundle path and persisted sanitized
  artifacts under `artifacts/tmp/rvv_microkernel_bundle_e2e/20260510-linalg-i32m2-vsub-ssh/`.

### Validation

- [OK] `python3 ./.trellis/scripts/task.py validate 05-10-linalg-i32m2-vsub-ssh-rvv-evidence-handoff`
- [OK] `python3 scripts/rvv_microkernel_e2e.py --self-test`
- [OK] focused dry-run for linalg-origin i32-vsub i32m2 bundle evidence
- [OK] negative selected-kernel mismatch dry-run rejected as expected
- [OK] focused lit filter `rvv-microkernel-bundle-e2e` passed 1/1
- [OK] real `ssh rvv` bundle external-ABI evidence run
- [OK] `git diff --check`
- [OK] `python3 -m py_compile scripts/rvv_microkernel_e2e.py`
- [OK] `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2` (169/169 tests passed)

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 11: Linalg i32-vsub to RVV i32m2 artifact handoff

**Date**: 2026-05-10
**Task**: Linalg i32-vsub to RVV i32m2 execution-planning artifact handoff
**Branch**: `main`

### Summary

Completed the linalg-origin i32-vsub RVV i32m2 artifact handoff by upgrading the
focused frontend fixture to consume finite i32m2 target capability metadata
through the normal lower-linalg plus execution-planning pipeline, and by adding
fail-closed coverage for an incomplete m2 capability provider.

### Main Changes

- Created Trellis task
  `.trellis/tasks/05-10-linalg-i32-vsub-rvv-i32m2-artifact-handoff/` with PRD
  and curated implement/check context.
- Updated `test/Transforms/LinalgToExec/linalg-i32-vsub-to-rvv-artifact.mlir`
  so the linalg-origin target profile provides the finite i32m2 config shape.
- Extended that fixture to check selected m2 profile metadata, `setvl` /
  `with_vl` LMUL m2, `!tcrv_rvv.i32m2` dataflow, i32m2 source intrinsics,
  declaration-only header export, and RISC-V relocatable object export.
- Added
  `test/Transforms/LinalgToExec/linalg-i32-vsub-rvv-i32m2-missing-config-fails.mlir`
  to prove missing finite m2 config providers fail before microkernel or
  emission-plan materialization.

### Validation

- [OK] `python3 ./.trellis/scripts/task.py validate 05-10-linalg-i32-vsub-rvv-i32m2-artifact-handoff`
- [OK] focused lit filter
  `linalg-i32-vsub-(to-rvv-artifact|rvv-i32m2-missing-config-fails)` passed 2/2
- [OK] `cmake --build artifacts/tmp/tianchenrv-build --target tcrv-opt tcrv-translate -j2`
- [OK] `git diff --check`
- [OK] focused lit filter `LinalgToExec` passed 8/8
- [OK] focused lit filter
  `rvv-microkernel-i32m2-(family-sub|object|body-mismatch-fails)` passed 3/3
- [OK] `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2` passed 169/169

No new `ssh rvv` evidence was collected, so this session makes no new runtime,
correctness, or performance claim.

### Status

[OK] **Completed**

### Next Steps

- Optional continuation: collect real `ssh rvv` evidence for the linalg-origin
  generated i32-vsub i32m2 source/header/object artifacts, tying hashes and
  marker observations to this frontend-generated path.


## Session 11: RVV i32m2 typed microkernel SSH evidence handoff

**Date**: 2026-05-10
**Task**: RVV i32m2 typed microkernel SSH evidence handoff
**Branch**: `main`

### Summary

Extended the bounded RVV microkernel evidence runner with an explicit typed
`i32m2` vector-shape mode for the existing i32-vsub compiler/export path. The
runner now selects the typed m2 MLIR fixture by default, uses the normal
execution-planning pipeline, validates compiler-emitted m2 intrinsic metadata
before any runtime claim, records boolean SSH evidence plus detailed remote
compile/link/run metadata, and fails closed when an m1 artifact is supplied for
an m2 claim.

### Main Changes

- Added `--vector-shape=i32m2` to `scripts/rvv_microkernel_e2e.py`, with m2
  fixture selection for `i32-vsub` and m2-specific source metadata validation.
- Recorded SEW=32, LMUL=m2, tail/mask policy, required m2 intrinsics, route ids,
  artifact paths, hashes, expected marker, observed marker, and bounded claim
  scope in evidence JSON.
- Preserved existing m1 add/sub/mul behavior and kept Python limited to
  orchestration/evidence collection.
- Added focused lit coverage for m2 dry-run evidence/source metadata and the
  m2-request/m1-artifact fail-closed path.
- Collected real `ssh rvv` evidence under
  `artifacts/tmp/rvv_microkernel_e2e/20260510-rvv-i32m2-vsub-ssh/`.

### Testing

- [OK] `python3 -m py_compile scripts/rvv_microkernel_e2e.py`
- [OK] `python3 scripts/rvv_microkernel_e2e.py --self-test`
- [OK] i32-vsub m2 dry-run:
  `python3 scripts/rvv_microkernel_e2e.py --dry-run --arithmetic-family=i32-vsub --vector-shape=i32m2 --run-id codex-i32m2-vsub-dry --overwrite`
- [OK] m2-request/m1-artifact fail-closed probe expectedly rejected stale
  i32m1 vector-shape metadata.
- [OK] focused lit filter `rvv-microkernel-e2e` passed 1/1.
- [OK] focused lit filter `rvv-microkernel-i32m2-(family-sub|object)` passed
  2/2.
- [OK] real `ssh rvv` i32-vsub m2 run:
  `python3 scripts/rvv_microkernel_e2e.py --arithmetic-family=i32-vsub --vector-shape=i32m2 --run-id 20260510-rvv-i32m2-vsub-ssh --overwrite --timeout 120 --ssh-target rvv`
- [OK] evidence marker:
  `tcrv_rvv_i32_vsub_microkernel_external_abi_ok`,
  `ssh_evidence=true`, `stdout_marker_observed=true`.
- [OK] `git diff --check`
- [OK] `cmake --build artifacts/tmp/tianchenrv-build --target tcrv-translate -j2`
- [OK] `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`
  (168/168 tests passed)

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 10: RVV i32 family direct helper SSH evidence handoff

**Date**: 2026-05-09
**Task**: RVV i32 family direct helper SSH evidence handoff
**Branch**: `main`

### Summary

Extended the bounded RVV microkernel evidence runner so direct non-bundle
evidence for `i32-vsub` and `i32-vmul` calls the family-specific
source/header/object `tcrv-translate` helper routes and records ssh-backed
external caller evidence under `artifacts/tmp`.

### Main Changes

- Routed non-bundle direct source/header/object evidence through the active
  family route ids instead of the generic target artifact front doors.
- Recorded `direct_helper_routes`, produced direct helper artifacts, hashes,
  sanitized command summaries, and bounded claim scope in evidence JSON.
- Kept Python limited to evidence orchestration; compiler route registration,
  artifact emission, and ABI truth remain in the C++/MLIR target/export stack.
- Updated script self-test and lit coverage for vsub/vmul direct helper route
  evidence and header artifact dry-run output.

### Git Commits

| Hash | Message |
|------|---------|
| `this commit` | (see git log) |

### Testing

- [OK] `python3 -m py_compile scripts/rvv_microkernel_e2e.py`
- [OK] `python3 scripts/rvv_microkernel_e2e.py --self-test`
- [OK] vsub direct dry-run: `python3 scripts/rvv_microkernel_e2e.py --dry-run --arithmetic-family=i32-vsub --run-id codex-direct-vsub-dry --overwrite`
- [OK] vmul direct dry-run: `python3 scripts/rvv_microkernel_e2e.py --dry-run --arithmetic-family=i32-vmul --run-id codex-direct-vmul-dry --overwrite`
- [OK] `python3 scripts/rvv_microkernel_e2e.py --arithmetic-family=i32-vsub --run-id 20260509-rvv-i32-vsub-direct-helper-ssh --overwrite --timeout 120 --ssh-target rvv`
- [OK] `python3 scripts/rvv_microkernel_e2e.py --arithmetic-family=i32-vmul --run-id 20260509-rvv-i32-vmul-direct-helper-ssh --overwrite --timeout 120 --ssh-target rvv`
- [OK] focused lit filter `rvv-microkernel-e2e|rvv-microkernel-bundle-e2e|rvv-microkernel-family-(sub|mul)` (6/6)
- [OK] `git diff --check`
- [OK] `python3 ./.trellis/scripts/task.py validate 05-09-rvv-i32-family-direct-helper-ssh-evidence-handoff`
- [OK] `cmake --build artifacts/tmp/tianchenrv-build --target tcrv-translate -j2`
- [OK] `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2` (164/164 tests passed)

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


## Session 9: RVV i32 family direct microkernel route helpers

**Date**: 2026-05-09
**Task**: RVV i32 family direct microkernel route helpers
**Branch**: `main`

### Summary

Added route-id-aligned tcrv-translate helpers for bounded RVV i32 vsub/vmul microkernel source/header/object exports, with route-specific stale-family validation, direct-helper metadata coverage, focused lit object/header/source checks, and full check-tianchenrv validation.

### Main Changes

- Added route-specific RVV microkernel target APIs for bounded i32 family
  source/header/object export, with stale selected-family diagnostics before
  artifact bytes are emitted.
- Registered `tcrv-translate` helpers from the finite i32 binary descriptor
  registry for vsub/vmul source/header/object route ids while preserving the
  legacy generic vadd helper route.
- Marked RVV vsub/vmul source/header/object routes as direct helper routes in
  built-in exporter metadata and extended registry/exporter C++ coverage.
- Added lit/FileCheck coverage for vsub/vmul direct C/header helpers,
  stale-family negative routes, help registration, and direct object export.

### Git Commits

| Hash | Message |
|------|---------|
| `this commit` | (see git log) |

### Testing

- [OK] `git diff --check`
- [OK] `python3 .trellis/scripts/task.py validate 05-09-rvv-i32-family-direct-microkernel-route-helpers`
- [OK] `cmake --build artifacts/tmp/tianchenrv-build --target tcrv-translate -j2`
- [OK] `cmake --build artifacts/tmp/tianchenrv-build --target tianchenrv-target-artifact-export-test tianchenrv-i32-binary-family-registry-test tcrv-translate -j2`
- [OK] `artifacts/tmp/tianchenrv-build/bin/tianchenrv-target-artifact-export-test`
- [OK] `artifacts/tmp/tianchenrv-build/bin/tianchenrv-i32-binary-family-registry-test`
- [OK] focused lit on RVV family source/header/object/helper tests (6/6)
- [OK] `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2` (164/164 tests passed)
- [OK] archived task context validation

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 10: RVV i32 LMUL policy intrinsic emission contract

**Date**: 2026-05-09
**Task**: RVV i32 LMUL policy intrinsic emission contract
**Branch**: `main`

### Summary

Implemented body-driven e32/m1 RVV i32 microkernel intrinsic emission from validated setvl/with_vl metadata; archived the Trellis task after full check-tianchenrv validation.

### Main Changes

- Added `RVVIntrinsicConfig` in RVV target/export code so the C emitter derives vector type, intrinsic suffixes, setvl/load/arithmetic/store intrinsic names, and emitted policy metadata from validated `tcrv_rvv.setvl` / `tcrv_rvv.with_vl` config.
- Preserved existing e32/m1 add/sub/mul behavior while moving hardcoded C intrinsic spellings behind the validated config object.
- Strengthened policy/config mismatch diagnostics to name the active route and selected RVV microkernel family before source/header/object emission.
- Added focused FileCheck coverage for generated intrinsic config metadata and route/family policy mismatch diagnostics.
- Completed Trellis task `.trellis/tasks/archive/2026-05/05-09-rvv-i32-lmul-policy-intrinsic-emission-contract/`; e32/m2 is explicitly left as the next continuation because the current dialect/plugin slice is still `i32m1`-only.

Validation:
- [OK] `git diff --check`
- [OK] `cmake --build artifacts/tmp/tianchenrv-build --target tcrv-translate tianchenrv-target-artifact-export-test tianchenrv-i32-binary-family-registry-test -j2`
- [OK] `artifacts/tmp/tianchenrv-build/bin/tianchenrv-target-artifact-export-test`
- [OK] `artifacts/tmp/tianchenrv-build/bin/tianchenrv-i32-binary-family-registry-test`
- [OK] Focused lit filter `rvv-microkernel-(auto-materialization|family-sub|family-mul|control-body-policy-mismatch-fails)` passed 6/6
- [OK] Focused lit filter `(Dialect/RVV|Target/RVVMicrokernel)` passed 29/29
- [OK] `python3 ./.trellis/scripts/task.py validate 05-09-rvv-i32-lmul-policy-intrinsic-emission-contract`
- [OK] `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2` passed 164/164
- No new `ssh rvv` run was collected; no new RVV runtime/correctness/performance claim is made.


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


## Session 11: RVV selected-shape descriptor ownership

**Date**: 2026-05-10
**Task**: RVV selected-shape descriptor ownership
**Branch**: `main`

### Summary

Moved finite RVV i32 selected-shape truth into C++ target-owned descriptor/API, added dispatch exporter fail-closed validation, reduced Python evidence script to generated-artifact consumption, and passed focused plus full TianChen-RV checks.

### Main Changes

- Created and archived Trellis task `rvv-selected-shape-descriptor-ownership-cxx-target-backend` for internal case `switch module`.
- Extended `RVVVectorShape.h` with bounded selected-plan metadata descriptor helpers for `i32m1` and `i32m2`.
- Routed RVV plugin selected-plan metadata generation and RVV+scalar dispatch source/header validation through the C++ descriptor.
- Added fail-closed checks for selected variant capability IDs, selected-plan metadata values/roles/notes, emitted shape comments, vector type, vsetvl suffix, and RVV intrinsic spelling.
- Reduced `scripts/rvv_scalar_dispatch_e2e.py` shape ownership to CLI/default-fixture routing plus generated artifact parsing; parser self-test fixtures remain local parser inputs only.
- Validation: `git diff --check`; `tcrv-translate tcrv-opt`; descriptor/exporter C++ tests; script py_compile/self-test; focused direct and bundle dry-runs for `i32-vsub i32m2`; selected-plan mismatch negative probe; `check-tianchenrv` 171/171.
- No new `ssh rvv` evidence was produced; commit `5db3128` remains the latest runtime evidence for the affected path.


### Git Commits

| Hash | Message |
|------|---------|
| `this commit` | (see git log) |

### Testing

- [OK] `git diff --check`
- [OK] `cmake --build artifacts/tmp/tianchenrv-build --target tcrv-translate tcrv-opt -j2`
- [OK] `python3 -m py_compile scripts/rvv_scalar_dispatch_e2e.py`
- [OK] `artifacts/tmp/tianchenrv-build/bin/tianchenrv-i32-binary-family-registry-test`
- [OK] `artifacts/tmp/tianchenrv-build/bin/tianchenrv-target-artifact-export-test`
- [OK] `python3 scripts/rvv_scalar_dispatch_e2e.py --self-test`
- [OK] Focused direct dry-run for `i32-vsub --vector-shape=i32m2`
- [OK] Focused bundle dry-run for `i32-vsub --vector-shape=i32m2`
- [OK] Manual selected-plan metadata mismatch probe rejected stale vector suffix before artifact emission
- [OK] `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2` passed 171/171

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 12: Linalg i32m2 vsub ssh external ABI evidence

**Date**: 2026-05-10
**Task**: Linalg i32m2 vsub ssh external ABI evidence
**Branch**: `main`

### Summary

Collected bounded ssh rvv source/object external ABI evidence for the compiler-produced linalg i32-vsub i32m2 RVV+scalar dispatch bundle and added explicit runtime_success fields to real ssh evidence JSON.

### Main Changes

- Created and archived Trellis task `05-10-linalg-i32m2-vsub-rvv-scalar-ssh-external-abi-evidence` for internal case `switch module`.
- Drove the compiler-produced linalg/front-door route through `scripts/rvv_scalar_dispatch_e2e.py --use-target-artifact-bundle --use-plan-and-export-bundle-front-door --arithmetic-family=i32-vsub --vector-shape=i32m2`.
- Produced bounded untracked evidence under `artifacts/tmp/tianchenrv-rvv-dispatch-bundle-e2e/codex-i32m2-vsub-ssh-20260510T0001/`.
- Confirmed the generated external caller compiles/links/runs on `ssh rvv` against both the generated dispatch source path and generated bundle object path.
- Added explicit `runtime_success: true` to real ssh evidence JSON after successful remote compile/link/run; dry-run evidence remains claim-free.
- No `.trellis/spec/` update was needed because the existing specs already require bounded real RVV evidence and sanitized evidence fields.

### Git Commits

| Hash | Message |
|------|---------|
| `this commit` | (see git log) |

### Testing

- [OK] `git diff --check`
- [OK] `python3 -m py_compile scripts/rvv_scalar_dispatch_e2e.py`
- [OK] `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-10-05-10-linalg-i32m2-vsub-rvv-scalar-ssh-external-abi-evidence`
- [OK] `cmake --build artifacts/tmp/tianchenrv-build --target tcrv-opt tcrv-translate -j2`
- [OK] `python3 scripts/rvv_scalar_dispatch_e2e.py --self-test`
- [OK] focused dry-run for `i32-vsub --vector-shape=i32m2`
- [OK] focused lit filter `rvv-scalar-dispatch-bundle-e2e` passed 1/1 selected test
- [OK] real ssh evidence command passed:
  `python3 scripts/rvv_scalar_dispatch_e2e.py --use-target-artifact-bundle --use-plan-and-export-bundle-front-door --arithmetic-family=i32-vsub --vector-shape=i32m2 --run-id codex-i32m2-vsub-ssh-20260510T0001 --overwrite --timeout 120`
- [OK] read-only `evidence.json` assertion for runtime success, branch/count coverage, selected family/shape, remote arch, and secret-like text absence
- [OK] `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2` passed 172/172

### Status

[OK] **Completed**

### Next Steps

- None - task complete
