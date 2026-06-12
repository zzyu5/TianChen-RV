# Selected RVV plus scalar dispatch ABI artifact boundary

## Goal

Make the selected RVV preferred path plus scalar fallback path visible as one
compiler-owned dispatch ABI artifact boundary for the bounded i32 binary path.
This round uses `i32-vsub` as the acceptance fixture, while keeping family
selection routed through the existing i32 add/sub/mul registry and plugin/target
interfaces.

## What I Already Know

- Repository root is `/home/kingdom/phdworks/TianchenRV`.
- Initial inspected state for this round is a clean worktree at HEAD `8db8609`.
- No active `.trellis/.current-task` existed at session start.
- The previous task
  `.trellis/tasks/archive/2026-05/05-10-rvv-i32-family-intrinsic-prefix-contract/`
  has `task.json` status `completed` and is archived.
- That previous task moved RVV arithmetic intrinsic spelling to
  family-prefix plus selected vector-shape suffix ownership.
- Current code already has bounded RVV/scalar dispatch route infrastructure in
  `lib/Target/Builtin/RVVScalarDispatch.cpp`, positive vadd/vsub/vmul family
  registry data, plugin-local RVV/scalar lowering-boundary materialization, and
  existing `i32-vsub` generic route tests.
- The remaining module boundary for this round is not more evidence reporting:
  the target-owned dispatch artifact must explicitly preserve the structured
  dispatch plan metadata it consumes, especially selected vector-shape metadata
  from the RVV callable candidate and fallback/route metadata from the scalar
  candidate.

## Requirements

- Keep compiler implementation in C++ / MLIR / LLVM / TableGen / CMake / lit /
  FileCheck.
- Do not implement compiler core, dialects, passes, plugin registry,
  capability model, lowering, emission, or artifact route selection in Python.
- Do not add new dtypes, arithmetic families, LMULs, benchmarks, performance
  claims, or broad evidence matrices.
- Reuse the existing i32 family registry for add/sub/mul route facts; do not
  hard-code vsub-only logic into core passes or generic artifact routing.
- Preserve plugin locality: RVV and scalar semantic checks remain in
  plugin/target-owned code, while core passes only route through registry and
  generic selected-path/coherence interfaces.
- The selected dispatch/fallback plan must fail closed before export when
  selected variant refs, fallback refs, origin/plugin roles, required
  capabilities, runtime ABI roles, mem-window roles, runtime guard linkage,
  selected RVV shape metadata, or scalar fallback route metadata are stale or
  contradictory.
- The target-owned dispatch exporter must consume the selected RVV callable
  candidate and selected scalar dispatch fallback candidate as one coherent
  ABI boundary, including:
  - selected kernel;
  - selected RVV and scalar variants;
  - selected roles (`dispatch case`, `dispatch fallback`);
  - source/export route ids;
  - runtime ABI kind/name/signature;
  - mem-window and runtime-param roles;
  - runtime dispatch guard linkage;
  - RVV selected vector-shape metadata;
  - scalar fallback route/family metadata.

## Acceptance Criteria

- The `i32-vsub` fixture goes from bounded linalg frontend through
  `--tcrv-execution-planning-pipeline` into RVV/scalar selected lowering
  boundaries, supported emission plans, and a single dispatch source/header
  artifact path.
- The emitted dispatch source for `i32-vsub` explicitly preserves selected
  kernel, selected RVV/scalar variants, selected roles, source/export route ids,
  runtime ABI signature, mem-window/runtime-param roles, runtime guard linkage,
  RVV selected vector-shape metadata, and scalar fallback metadata.
- A malformed dispatch boundary fails closed before artifact output. At least
  one negative case must cover stale/missing dispatch ABI linkage or malformed
  runtime/mem-window role metadata.
- Focused C++ or lit checks cover any changed artifact exporter or verifier
  behavior.
- `git diff --check` passes.
- Focused lit/FileCheck tests for the positive `i32-vsub` dispatch artifact
  path and negative fail-closed case pass.
- `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`
  passes before archive.
- No new `ssh rvv` evidence is required unless this round changes generated
  runtime-dispatch source/object behavior or makes a new runtime correctness
  claim.

## Out of Scope

- New dtype, new family, new LMUL, broad route matrix, benchmark, or performance
  evidence work.
- Python-defined compiler semantics or Python-owned route selection.
- Core-pass branches on RVV/scalar family semantics.
- Adding computation semantics to `tcrv.exec`.
- Treating helper-only, metadata-only, prompt/report-only, or smoke-only work
  as completion.

## Technical Approach

- Keep the existing dispatch pair collector and route manifest as the target
  owner for RVV+scalar i32 dispatch artifacts.
- Strengthen dispatch artifact metadata emission so selected callable
  candidates print their structured `selected_plan_metadata` entries. This
  makes selected RVV vector-shape config and related plugin-selected metadata
  visible at the dispatch artifact boundary, not only inside the embedded RVV
  callable source or bundle records.
- Reuse existing candidate preflight:
  `validateRegisteredCallableRouteMetadata`,
  `validateRVVMicrokernelSourceCandidate`,
  `validateScalarMicrokernelSourceCandidate`, dispatch IR fallback/guard
  resolution, and `buildDispatchABIPlan`.
- Extend the `i32-vsub` focused lit fixture to assert the dispatch artifact
  preserves selected vector-shape metadata, runtime ABI role metadata, and
  fallback route metadata.
- Add or extend a focused negative test only where the current fail-closed
  matrix does not already prove the modified boundary.

## Technical Notes

- Specs read:
  - `.trellis/spec/index.md`
  - `.trellis/spec/core-dialect/tcrv-exec-contract.md`
  - `.trellis/spec/variant-pipeline/generation-selection-tuning.md`
  - `.trellis/spec/lowering-runtime/emission-runtime-contract.md`
  - `.trellis/spec/extension-plugins/rvv-plugin.md`
  - `.trellis/spec/extension-plugins/scalar-fallback-plugin.md`
  - `.trellis/spec/plugin-protocol/locality-contract.md`
- Archived PRDs read:
  - `.trellis/tasks/archive/2026-05/05-10-rvv-selected-vector-shape-config-boundary-cleanup/prd.md`
  - `.trellis/tasks/archive/2026-05/05-10-rvv-i32-family-intrinsic-prefix-contract/prd.md`
- Primary implementation surfaces inspected:
  - `include/TianChenRV/Dialect/Exec/IR/ExecOps.td`
  - `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`
  - `include/TianChenRV/Dialect/Scalar/IR/ScalarOps.td`
  - `include/TianChenRV/Target/I32BinaryFamilyRegistry.h`
  - `include/TianChenRV/Target/RVVScalarDispatch.h`
  - `lib/Transforms/VariantSelection.cpp`
  - `lib/Transforms/VariantDispatchSynthesis.cpp`
  - `lib/Transforms/DispatchRuntimeGuard.cpp`
  - `lib/Transforms/LoweringBoundary.cpp`
  - `lib/Transforms/ExecutionPlanCoherence.cpp`
  - `lib/Plugin/RVV/RVVExtensionPlugin.cpp`
  - `lib/Plugin/Scalar/ScalarExtensionPlugin.cpp`
  - `lib/Target/Builtin/RVVScalarDispatch.cpp`
  - `lib/Target/TargetArtifactExport.cpp`
  - `test/Target/RVVScalarDispatch/`

## Validation Plan

- Run focused lit for `test/Target/RVVScalarDispatch/rvv-scalar-i32-vsub-dispatch-generic-route.mlir`.
- Run focused lit for any negative dispatch ABI test touched.
- Run focused C++ tests only if exporter registry or public API behavior
  changes.
- Run `git diff --check`.
- Run `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`.

## Current Status

Completed.

## Completion Notes

- `lib/Target/Builtin/RVVScalarDispatch.cpp` now emits each selected callable
  candidate's structured `selectedPlanMetadata` at the outer RVV+scalar
  dispatch artifact boundary. The dispatch source therefore preserves the RVV
  selected vector-shape config and related plugin-selected metadata next to
  the selected kernel, selected variants, ABI, mem-window, and route comments.
- The dispatch IR resolver now captures scalar fallback `origin` and
  `fallback_role` metadata from the selected `tcrv.exec.fallback` operation
  and prints it as target-owned dispatch fallback metadata before callable
  export.
- The focused `i32-vsub` route fixture now checks the positive end-to-end
  dispatch artifact surface for selected RVV vector-shape metadata, base RVV
  capacity metadata, scalar fallback route metadata, and callable symbols.
- The same fixture now includes a malformed selected vector-shape boundary
  case that fails closed before the RVV+scalar dispatch C source banner can be
  emitted.

## Validation Results

- [OK] `cmake --build artifacts/tmp/tianchenrv-build --target tcrv-translate tcrv-opt -j2`
- [OK] Focused lit filter
  `rvv-scalar-i32-vsub-dispatch-generic-route` passed 1/1.
- [OK] `artifacts/tmp/tianchenrv-build/bin/tianchenrv-target-artifact-export-test`
- [OK] `git diff --check`
- [OK] `python3 ./.trellis/scripts/task.py validate 05-10-selected-rvv-scalar-dispatch-abi-artifact-boundary`
- [OK] `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`
  passed 169/169 tests.

No new `ssh rvv` evidence was required or run. This round changes compiler
artifact boundary metadata and fail-closed export validation coverage, not
generated runtime dispatch source/object behavior or runtime correctness
claims.
