# rvv-i64-binary-descriptor-exit-closure

## Goal

Complete one structural descriptor-exit closure for the bounded i64 finite
binary default compiler routes `i64-vsub` and `i64-vmul`. The default route
must derive source family, selected RVV/scalar family operations, selected-plan
metadata, callable ABI, common EmitC route metadata, and target source/header/
object or RVV+scalar dispatch artifacts from typed source/body facts and
selected extension-family surfaces, not from legacy lowering descriptors as
compute, C source, or callable ABI authority.

## What I Already Know

- Repository root is `/home/kingdom/phdworks/TianchenRV`.
- Task start HEAD is `e6234c0 test(frontend): validate i64 vadd descriptorless route`.
- Worktree was clean at task start and `.trellis/.current-task` was absent.
- The previous task
  `.trellis/tasks/archive/2026-05/05-12-rvv-i64-vadd-default-emitc-route/`
  is finished and archived and must not be reopened.
- The previous i64-vadd round closed with test/task/journal changes only
  because the production path was already migrated in current HEAD. This round
  must not repeat that coverage-only stall pattern.
- Current specs define the default route as extension family ops -> common
  EmitC route -> generated C/C++ artifact surfaces. Descriptor-driven
  computation and descriptor-to-C authority remain invalid long-term
  architecture.
- Recent descriptor-exit tasks already removed descriptor authority from RVV
  selected emission planning, scalar selected emission planning, RVV+scalar
  dispatch composite identity, and descriptor-shaped registry helper names.
- The module owner for this task is the bounded `i64-vsub` and `i64-vmul`
  default route from source linalg/body identity through RVV/scalar selected
  family ops, selected-plan metadata, common EmitC route, and direct or
  dispatch target export.

## Requirements

- Audit the default `i64-vsub` and `i64-vmul` routes for any remaining
  production acceptance of `tcrv_rvv.lowering_descriptor`,
  `tcrv_scalar.lowering_descriptor`,
  `tcrv_rvv.selected_lowering_descriptor`,
  `tcrv_scalar.selected_lowering_descriptor`,
  `selected_lowering_descriptor`, or legacy descriptor-family lookup.
- Keep frontend selection bounded to marked linalg/body identity. The source
  memref element type, region argument/result type, arithmetic op, and yielded
  value infer `i64-vsub` or `i64-vmul`; stale descriptor metadata must fail
  before an exec kernel is created.
- Ensure RVV selected paths for `i64-vsub` and `i64-vmul` are descriptorless
  by default and derive production authority from typed
  `tcrv_rvv.i64_vsub_microkernel` / `tcrv_rvv.i64_vmul_microkernel` bodies,
  selected vector config metadata, selected-plan metadata, common EmitC route
  provenance, and exec-IR-backed callable ABI plans.
- Ensure scalar selected paths for `i64-vsub` and `i64-vmul` are descriptorless
  by default and derive production authority from typed
  `tcrv_scalar.i64_vsub_microkernel` /
  `tcrv_scalar.i64_vmul_microkernel` bodies, selected-plan metadata, common
  EmitC route provenance, and exec-IR-backed callable ABI plans.
- Ensure target source/header/object and RVV+scalar dispatch export validate
  selected component plans, runtime AVL/ABI parameters, selected vector config,
  common EmitC route metadata, and typed operation metadata before source or
  bundle output.
- Delete, bypass, or explicitly quarantine any obsolete descriptor-driven
  production branch discovered by the audit. Legacy descriptor metadata may
  remain only as an optional mirror or fail-closed cross-check after typed plan
  construction.
- Do not close this task with only route metadata assertions, extra coverage,
  task notes, or journal updates. If the audit proves no production code gap
  exists and no obsolete code can be safely removed, leave the task open and
  record the exact audited continuation point.

## Non-Goals

- No new dtype, arithmetic family, LMUL family, tuning, generic RVV lowering,
  performance benchmark, or runtime scheduler.
- No IME, AME, Sophgo/offload, TensorExt, Template, Toy, or unrelated plugin
  work.
- No descriptor-to-C exporter and no descriptor-owned computation semantics.
- No compiler core, dialect, pass, plugin registry, capability model, lowering,
  or emission implementation in Python.
- No broad smoke matrix, helper-only work, status/report-only work, or
  standalone RVV ssh evidence package.
- No RVV runtime, correctness, or performance claim unless a fresh real
  `ssh rvv` run is performed.

## Acceptance Criteria

- [x] The default `i64-vsub` frontend-to-planning-to-export route succeeds from
      source body identity, typed RVV/scalar selected family ops, selected-plan
      metadata, common EmitC route metadata, and exec-IR-backed ABI parameters.
- [x] The default `i64-vmul` frontend-to-planning-to-export route succeeds from
      source body identity, typed RVV/scalar selected family ops, selected-plan
      metadata, common EmitC route metadata, and exec-IR-backed ABI parameters.
- [x] At least one real production/default path change, obsolete
      descriptor-driven branch removal/quarantine, or typed-family-only route
      hardening is implemented for the bounded `i64-vsub` / `i64-vmul` module.
- [x] Descriptor-only RVV, scalar, or RVV+scalar dispatch paths for these
      families fail closed before source/header/object/bundle output.
- [x] Stale descriptor mirror metadata beside valid selected typed plans is
      rejected or validated only as non-authoritative mirror data; it cannot
      alter family, selected vector config, runtime ABI parameters, C function
      signature, generated C body, emitted call target, route id, artifact
      kind, component group, or external ABI metadata.
- [x] Focused lit/FileCheck checks cover the changed `i64-vsub` and
      `i64-vmul` default route behavior through `tcrv-opt` / `tcrv-translate`.
- [x] Relevant C++ tests cover changed behavior in RVV planning, selected
      lowering boundary, selected emission planning, scalar plugin behavior,
      target artifact export, or dispatch export.
- [x] `git diff --check` and `git diff --cached --check` pass.
- [x] `python3 ./.trellis/scripts/task.py validate` passes for the active task
      path, and for the archived path if the task is finished.
- [x] `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`
      runs if the existing build tree is usable; otherwise the exact blocker
      is recorded and focused available targets pass.

## Technical Notes

- Specs read:
  - `.trellis/spec/index.md`
  - `.trellis/spec/architecture/unified-riscv-mlir.md`
  - `.trellis/spec/extension-plugins/rvv-plugin.md`
  - `.trellis/spec/extension-plugins/scalar-fallback-plugin.md`
  - `.trellis/spec/lowering-runtime/emitc-route.md`
  - `.trellis/spec/lowering-runtime/emission-runtime-contract.md`
  - `.trellis/spec/variant-pipeline/generation-selection-tuning.md`
  - `.trellis/spec/testing/mlir-testing-contract.md`
- Archived PRDs read:
  - `.trellis/tasks/archive/2026-05/05-12-rvv-i64-vadd-default-emitc-route/prd.md`
  - `.trellis/tasks/archive/2026-05/05-12-rvv-selected-emission-descriptor-exit/prd.md`
  - `.trellis/tasks/archive/2026-05/05-12-scalar-selected-emission-descriptor-exit/prd.md`
  - `.trellis/tasks/archive/2026-05/05-12-rvv-scalar-dispatch-descriptor-exit/prd.md`
  - `.trellis/tasks/archive/2026-05/05-12-rvv-scalar-descriptor-registry-quarantine/prd.md`
  - `.trellis/tasks/archive/2026-05/05-11-finite-binary-frontend-lowering-contract/prd.md`
- Source entry points to audit:
  - `lib/Transforms/LowerLinalgRVVBinaryToExec.cpp`
  - `include/TianChenRV/Support/FiniteBinaryFrontendLowering.h`
  - `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`
  - `include/TianChenRV/Dialect/Scalar/IR/ScalarOps.td`
  - `lib/Plugin/RVV/RVVBinaryPlanning.cpp`
  - `lib/Plugin/RVV/RVVBinarySelectedLoweringBoundary.cpp`
  - `lib/Plugin/RVV/RVVBinarySelectedEmissionPlanning.cpp`
  - `lib/Plugin/RVV/RVVBinaryMicrokernelMaterialization.cpp`
  - `lib/Plugin/Scalar/ScalarExtensionPlugin.cpp`
  - `lib/Target/RVV/RVVMicrokernel.cpp`
  - `lib/Target/Scalar/ScalarMicrokernel.cpp`
  - `lib/Target/Builtin/RVVScalarDispatch.cpp`
  - `lib/Target/TargetArtifactExport.cpp`
- Focused tests to inspect are under `test/Transforms/LinalgToExec/`,
  `test/Target/RVVMicrokernel/`, `test/Target/RVVScalarDispatch/`,
  `test/Target/TargetArtifactBundleExport/`, and `test/Plugin/`.

## Definition Of Done

The task is done only when the bounded `i64-vsub` and `i64-vmul`
production/default routes have been audited and one structural route closure is
implemented, focused checks pass, Trellis validation passes, the task status is
truthful, the task is finished/archived only if complete, and one coherent
commit records the completed module. If unfinished, leave the task open with
the exact descriptor-authority file/function/route that remains and the next
smallest structural continuation point.

## Task Status Notes

- Created this task because no current Trellis task existed and the user brief
  explicitly requested `rvv-i64-binary-descriptor-exit-closure`.
- PRD was written before source edits. Initial constraints are structural:
  production/default behavior must change, an obsolete descriptor-driven branch
  must be removed or quarantined, or the task must remain open with the exact
  audited continuation point rather than closing as coverage-only work.
- Removed the RVV descriptor-to-selected-plan/materialization helper surface:
  `buildRVVBinarySelectedPlanFromVariant`,
  `buildRVVBinaryMicrokernelMaterializationPlanFromVariant`, and the public
  `buildRVVBinarySelectedMicrokernelMaterializationPlan` entry point.
- Rewired RVV selected lowering-boundary materialization so a variant carrying
  `tcrv_rvv.lowering_descriptor` and no typed microkernel attachment now fails
  closed immediately as legacy-registration-only; descriptorless typed-family
  materialization remains the only default path that may synthesize i32/i64
  RVV microkernel bodies.
- Replaced descriptor-derived materialization validation in RVV variant
  legality with `validateLegacyRVVBinarySelectedDescriptorMetadata`, a
  fail-closed mirror validator that checks bounded descriptor syntax,
  registered family, selected vector-shape consistency, element-count range,
  and required march metadata without returning a materialization plan.
- Added lit coverage proving descriptor-only `i64-vsub` and `i64-vmul`
  selected lowering-boundary materialization is legacy-quarantined and demands
  typed `tcrv_rvv.i64_vsub_microkernel` /
  `tcrv_rvv.i64_vmul_microkernel` authority.
- Focused C++ checks passed:
  `tianchenrv-rvv-binary-planning-test`,
  `tianchenrv-rvv-selected-lowering-boundary-test`,
  `tianchenrv-rvv-binary-variant-legality-test`,
  `tianchenrv-rvv-extension-plugin-test`,
  `tianchenrv-scalar-extension-plugin-test`,
  `tianchenrv-target-artifact-export-test`, and
  `tianchenrv-runtime-abi-callable-plan-test`.
- Focused lit passed from `artifacts/tmp/tianchenrv-build/test` with filter
  `rvv-i64-vsub-descriptor-only-legacy-quarantine|rvv-i64-vmul-descriptor-only-legacy-quarantine|linalg-i64-vsub-to-rvv-artifact|linalg-i64-vmul-to-rvv-artifact|rvv-microkernel-i64-vsub|rvv-microkernel-i64-vmul|rvv-scalar-i64-vsub-dispatch-generic-route|rvv-scalar-i64-vmul-dispatch-generic-route|plan-linalg-i64-vsub-and-export-target-artifact-bundle|plan-linalg-i64-vmul-and-export-target-artifact-bundle`: 10/10 passed.
- `git diff --check`, `git diff --cached --check`, and
  `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`
  passed; full lit reported 209/209 tests passing.
- No real `ssh rvv` run was performed, so this task makes no RVV runtime,
  correctness, or performance claim.
