# rvv-i64-vadd-default-emitc-route

## Goal

Make the bounded `i64-vadd` production/default route follow the same migrated
architecture as the descriptor-exit i32 path: source body and typed operands
select the finite family, RVV/scalar selected plans use typed extension-family
ops plus selected-plan metadata, and target source/header/object or dispatch
artifacts consume the common EmitC route rather than descriptor strings as
compute authority.

## What I Already Know

- Repository root is `/home/kingdom/phdworks/TianchenRV`.
- Task start HEAD is `ba11408 feat(abi): remove i32-vadd compatibility wrappers`.
- Worktree was clean at task start and `.trellis/.current-task` was absent.
- The previous archived task
  `.trellis/tasks/archive/2026-05/05-12-remove-i32-vadd-runtime-abi-compat-wrappers/prd.md`
  removed obsolete i32-vadd ABI compatibility wrappers after the codebase had
  already moved production owners to family-aware ABI helpers.
- Current specs require extension family ops -> common EmitC route -> target
  source/artifact export. Descriptor-driven computation and descriptor-to-C
  authority remain invalid long-term architecture.
- Current source inspection found bounded i64-vadd support already present in:
  frontend finite binary contracts, RVV and scalar ODS ops, RVV/scalar family
  registries, RVV selected lowering-boundary materialization, scalar selected
  lowering-boundary materialization, RVV/scalar emission planning, RVV and
  scalar target exporters, and RVV+scalar dispatch route registration.
- Current focused tests already include i64-vadd frontend-to-planning/export,
  direct RVV i64-vadd source/header/object surfaces, scalar i64-vadd fallback
  routing, stale descriptor mismatch failures, and i64-vadd RVV+scalar dispatch
  generic route coverage.

## Requirements

- Keep `i64-vadd` frontend lowering bounded to marked linalg.generic source
  bodies whose typed memrefs, region arguments, arithmetic op, and yield infer
  `i64-vadd`; reject stale RVV/scalar lowering descriptor metadata on the
  frontend input before creating an exec kernel.
- Ensure selected RVV i64-vadd variants are descriptorless by default while
  still carrying typed selected vector-shape metadata, required march/policy,
  bounded element-count mirror metadata, and selected-plan metadata naming the
  typed RVV family op and generated EmitC lowerable op interface.
- Ensure selected scalar i64-vadd fallback variants are descriptorless by
  default while materializing a typed `tcrv_scalar.i64_vadd_microkernel` and
  selected-plan metadata naming the typed scalar source op and generated EmitC
  lowerable op interface.
- Ensure target source/export paths for direct RVV, direct scalar, and existing
  RVV+scalar dispatch support consume the selected component plans and
  IR-backed callable ABI boundary rather than finite descriptor records as
  source/compute authority.
- Keep legacy descriptor metadata only as optional mirror/cross-check metadata.
  Stale descriptor family/operator/dtype mismatches must fail closed before C
  source output.
- Do not add new dtypes, operations, LMUL families, tuning, performance work,
  offload/Sophgo/IME/AME changes, broad matrix coverage, or Python compiler
  internals.
- Do not make RVV runtime, correctness, or performance claims in this round
  unless a real `ssh rvv` run is performed. This task targets compiler route
  structure and local artifact export behavior only.

## Acceptance Criteria

- [x] A marked i64 add linalg source lowers to a `tcrv.exec.kernel` with
      `tcrv_frontend_lowering = "i64-vadd"`, `int64_t` callable ABI mem_window
      mirrors, and `size_t` runtime element-count parameter.
- [x] The full planning pipeline creates descriptorless RVV and scalar selected
      variants for i64-vadd; default selected RVV/scalar plans do not require
      `tcrv_rvv.lowering_descriptor` or `tcrv_scalar.lowering_descriptor` as
      production authority.
- [x] RVV selected-boundary materialization creates a typed
      `tcrv_rvv.i64_vadd_microkernel` body with `setvl`, `with_vl`, i64 load,
      i64 add, and i64 store ops, and fails closed on family/body/dtype/config
      mismatches.
- [x] Scalar selected-boundary materialization creates a typed
      `tcrv_scalar.i64_vadd_microkernel`, emits selected-plan metadata for the
      typed scalar source op/interface, and fails closed on stale descriptor
      family/operator/dtype mismatch.
- [x] Generic target source export for i64-vadd emits either the direct
      selected RVV/scalar callable source or, when dispatch support is present,
      the RVV+scalar dispatch source from the selected component plans.
- [x] Generated source contains common EmitC route metadata and route-authored
      function bodies; it does not reconstruct production computation from
      descriptor strings.
- [x] Focused negative tests prove stale frontend descriptor metadata, missing
      i64 RVV config, stale RVV descriptor mirror, stale scalar descriptor
      mirror, and stale callable ABI surfaces fail before source output.
- [x] `git diff --check` passes.
- [x] Focused lit tests for i64-vadd frontend/planning/export and direct
      RVV/scalar/dispatch artifact surfaces pass.
- [x] `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`
      passes if the existing build tree is usable; otherwise the exact blocker
      is recorded and the focused available targets are run.
- [x] Trellis task validation passes before finish/archive.

## Definition Of Done

- The active i64-vadd production/default compiler route is descriptorless for
  selected RVV/scalar family authority.
- All modified or audited behavior is covered by focused lit/C++ checks.
- Task status notes state whether descriptor authority was removed from the
  default i64-vadd path, which checks were run, and whether any RVV hardware
  claim was intentionally not made.
- If the module is complete, archive the task and create one coherent commit.
  If incomplete, leave the task open with the exact remaining descriptor-owned
  route and next continuation point.

## Out Of Scope

- Generic linalg lowering beyond the bounded finite RVV binary add/sub/mul
  slice.
- New RVV shapes, new element types, new arithmetic families, or performance
  tuning.
- Descriptor-to-C emission, descriptor-owned compute semantics, or restoring
  obsolete descriptor registries as production authority.
- Runtime execution, RVV correctness evidence, or performance claims without
  explicit `ssh rvv` evidence.

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
- Source entry points inspected:
  - `lib/Transforms/LowerLinalgRVVBinaryToExec.cpp`
  - `include/TianChenRV/Support/FiniteBinaryFrontendLowering.h`
  - `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`
  - `include/TianChenRV/Dialect/Scalar/IR/ScalarOps.td`
  - `include/TianChenRV/Target/RVV/RVVBinaryFamilyRegistry.h`
  - `include/TianChenRV/Target/RVVScalarBinaryFamily.h`
  - `lib/Plugin/RVV/RVVBinaryPlanning.cpp`
  - `lib/Plugin/RVV/RVVBinarySelectedLoweringBoundary.cpp`
  - `lib/Plugin/RVV/RVVBinarySelectedEmissionPlanning.cpp`
  - `lib/Plugin/RVV/RVVBinaryMicrokernelMaterialization.cpp`
  - `lib/Plugin/Scalar/ScalarExtensionPlugin.cpp`
  - `lib/Target/RVV/RVVMicrokernel.cpp`
  - `lib/Target/Scalar/ScalarMicrokernel.cpp`
  - `lib/Target/Builtin/RVVScalarDispatch.cpp`
  - `lib/Target/TargetArtifactExport.cpp`
- Focused tests identified:
  - `test/Transforms/LinalgToExec/linalg-i64-vadd-to-rvv-artifact.mlir`
  - `test/Transforms/LinalgToExec/linalg-i64-vadd-rvv-i64m1-missing-config-fails.mlir`
  - `test/Target/RVVMicrokernel/rvv-microkernel-i64-vadd.mlir`
  - `test/Target/RVVScalarDispatch/rvv-scalar-i64-vadd-dispatch-generic-route.mlir`
  - related scalar/RVV plugin C++ tests under `test/Plugin` and
    `test/Target/TargetArtifactExportTest.cpp`.

## Task Status Notes

- Created this task because no current Trellis task existed and the user brief
  explicitly requested a new Trellis task/PRD for
  `rvv-i64-vadd-default-emitc-route`.
- Initial source audit shows much of the i64-vadd route already exists in the
  current HEAD; implementation work will focus on proving whether any default
  route still treats descriptor metadata as production authority and patching
  only concrete gaps found by focused checks.
- Source audit plus focused verification showed the default i64-vadd production
  route was already structurally migrated in current HEAD: typed frontend/body
  identity drives selected RVV/scalar family ops, selected-plan metadata feeds
  the common EmitC/export route, and legacy descriptors remain only optional
  mirror/cross-check metadata.
- This round added explicit i64-vadd frontend negative coverage for stale
  `tcrv_rvv.lowering_descriptor`, `tcrv_scalar.lowering_descriptor`, and
  `selected_lowering_descriptor` metadata on the input linalg op.
- Focused checks run:
  `artifacts/tmp/tianchenrv-build/bin/tcrv-opt test/Transforms/LinalgToExec/linalg-finite-binary-frontend-invalid.mlir --split-input-file --verify-diagnostics --tcrv-lower-linalg-rvv-binary-to-exec`,
  `tianchenrv-rvv-binary-planning-test`,
  `tianchenrv-rvv-selected-lowering-boundary-test`,
  `tianchenrv-scalar-extension-plugin-test`, and
  `tianchenrv-target-artifact-export-test`.
- Full validation passed:
  `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`
  reported 207/207 tests passing.
- No `ssh rvv` run was performed and no RVV runtime, correctness, or
  performance claim is made by this task.
