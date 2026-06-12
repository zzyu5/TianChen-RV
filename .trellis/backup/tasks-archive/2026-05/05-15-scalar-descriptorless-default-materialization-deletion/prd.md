# Scalar descriptorless default materialization deletion

## Goal

Delete scalar fallback descriptorless default microkernel materialization as
compute authority. A selected scalar fallback path with no explicit typed
scalar body must not auto-create `tcrv_scalar.i32_vadd_microkernel`, derive ABI
windows/params from RVV-scalar bridge metadata, or otherwise turn absent body
state into executable scalar compute semantics. Future scalar executable
behavior must be rebuilt from explicit `tcrv_scalar` extension-family ops
through the common EmitC/artifact route; this round does not rebuild that route.

This is a Wrong Logic Deletion Campaign round. Deletion before rebuild is the
rule.

## What I Already Know

- Repository root is `/home/kingdom/phdworks/TianchenRV`.
- Initial repo state for this round was clean at HEAD `64e6513`.
- `.trellis/.current-task` did not exist at session start; this task was
  created from the Hermes Direction Brief.
- The previous archived task
  `.trellis/tasks/archive/2026-05/05-15-scalar-frontend-lowering-authority-deletion/prd.md`
  deleted scalar `tcrv_frontend_lowering` family authority but explicitly left
  descriptorless no-marker default `i32_vadd` materialization as remaining
  bounded behavior.
- Current journal Session 71 records the same remaining scalar bridge residue:
  descriptorless no-marker fallback may still materialize the default i32 vadd
  typed op, and the next deletion owner should not rebuild scalar EmitC in a
  deletion-only round.
- The brief identifies
  `lib/Plugin/Scalar/ScalarExtensionPlugin.cpp` as the owner of
  descriptorless default scalar typed materialization and selected-boundary
  microkernel auto-insertion.
- The brief identifies
  `include/TianChenRV/Target/RVVScalarBinaryFamily.h` as the RVV-derived scalar
  bridge metadata source for finite family/ABI details.

## Requirements

- Remove `buildDescriptorlessDefaultScalarTypedMaterializationPlan` as a
  scalar compute authority, or make the descriptorless default path fail
  closed before creating typed scalar compute.
- Remove the selected lowering-boundary materialization path that inserts a
  scalar microkernel when no explicit scalar body is present.
- Ensure scalar proposals, selected boundaries, emission plans, artifact
  fixtures, and tests cannot rely on absent body plus default finite family to
  create scalar compute semantics.
- Preserve explicit pre-existing typed scalar microkernel handling only when
  the body is already present as typed extension-family input and is not
  selected from metadata/defaults.
- Preserve neutral scalar fallback capability/proposal metadata, conservative
  fallback role, metadata-only emission diagnostics, and fail-closed direct
  route deletion behavior where they do not synthesize computation.
- Delete or rewrite tests/fixtures that protect default `i32_vadd`
  materialization from an otherwise empty selected scalar path.
- Record any newly exposed scalar executable gap as a rebuild gap rather than
  adding Common EmitC, route/exporter code, wrappers, selectors, legacy modes,
  or compatibility materialization.

## Acceptance Criteria

- [x] No scalar proposal, boundary, emission plan, artifact fixture, or test
      relies on absent body plus descriptorless default finite family to create
      scalar compute semantics.
- [x] `buildDescriptorlessDefaultScalarTypedMaterializationPlan` is removed or
      fail-closed so it no longer creates `i32_vadd` or any other finite scalar
      microkernel from absence of an explicit body.
- [x] `materializeSelectedLoweringBoundary` no longer auto-inserts a scalar
      microkernel for selected scalar fallback variants with no explicit typed
      scalar body.
- [x] `selectedPathHasCallableMicrokernel`, `microkernelPlan` auto-insertion,
      `getI32VAddFamilySpec` default use, `scalar.element_count`
      metadata-alone paths, and related tests are ref-scanned and reported
      truthfully.
- [x] Explicit typed scalar microkernel handling remains valid only for
      pre-existing typed extension-family input.
- [x] No Common EmitC implementation, new scalar lowering, new scalar extension
      op, runtime ABI rebuild, route/exporter code, compatibility selector,
      wrapper, or legacy mode is added.
- [x] Focused scalar plugin build/test and affected artifact/lit checks pass,
      or failures are recorded as expected deletion gaps without restoring the
      deleted default path.
- [x] `git diff --check`, Trellis validation, finish/archive, final clean
      `git status --short`, and one coherent commit are produced if complete.

## Non-goals

- No Common EmitC implementation.
- No new scalar lowering.
- No new scalar extension ops.
- No new runtime ABI modeling.
- No new route/exporter code.
- No compatibility selectors, wrappers, or legacy modes.
- No broad scalar feature coverage.
- No restoration of `tcrv_frontend_lowering`, default-family selection, or
  descriptor-driven compute.
- No deletion of neutral scalar dialect declarations or common runtime ABI
  helpers unless directly tied to descriptorless default materialization.
- No RVV runtime/correctness/performance claim; this round is deletion-focused
  and does not require `ssh rvv` evidence unless a changed claim depends on it.

## Minimal Evidence

- Bounded ref-scan for:
  `buildDescriptorlessDefaultScalarTypedMaterializationPlan`,
  descriptorless typed default materialization,
  `getI32VAddFamilySpec`,
  `microkernelPlan` auto-insertion,
  `selectedPathHasCallableMicrokernel`,
  `scalar.element_count` metadata-alone paths,
  and tests expecting no-body scalar materialization.
- Focused build/run of `tianchenrv-scalar-extension-plugin-test`.
- Affected artifact export or lit/FileCheck tests that depended on no-body
  scalar materialization.
- Full `check-tianchenrv` if practical after focused validation.
- `git diff --check` and `git diff --cached --check`.
- `python3 ./.trellis/scripts/task.py validate` before finish/archive.

## Technical Notes

- `.trellis/spec/index.md` states descriptor-driven computation is invalid as
  long-term architecture and the primary compiler stack is
  C++/MLIR/LLVM/TableGen/CMake/lit/FileCheck.
- `.trellis/spec/architecture/design-boundaries.md` rejects
  descriptor-driven microkernel/exporter frameworks and independent backend
  dialect framing.
- `.trellis/spec/core-dialect/tcrv-exec-contract.md` keeps computation and
  hardware semantics out of `tcrv.exec`; scalar fallback must not be inferred
  by core orchestration from metadata/defaults.
- `.trellis/spec/extension-plugins/scalar-fallback-plugin.md` is directly
  relevant and currently still documents descriptorless default
  `i32_vadd_microkernel` materialization as allowed first-slice behavior; this
  round should update that obsolete protection.
- `.trellis/spec/lowering-runtime/emitc-route.md` states the correct future
  route is typed extension family ops -> EmitC -> C/C++ emitter, not
  descriptor/default-selected computation.
- Archived PRD
  `.trellis/tasks/archive/2026-05/05-15-scalar-frontend-lowering-authority-deletion/prd.md`
  is the immediate baseline for the previous deletion round and identifies
  descriptorless default scalar materialization as intentionally left behind.

## Completion Evidence

- Removed `buildDescriptorlessDefaultScalarTypedMaterializationPlan`,
  `ScalarMicrokernelMaterializationPlan`, `materializeScalarMicrokernelOp`,
  `microkernelPlan`, the default element count constant, and the selected
  lowering-boundary microkernel auto-insertion path from
  `lib/Plugin/Scalar/ScalarExtensionPlugin.cpp`.
- `materializeSelectedLoweringBoundary` now preserves explicit typed scalar
  microkernel bodies and may materialize only a metadata-only scalar boundary
  for no-body fallback paths; it no longer creates `i32_vadd` from absent body
  state.
- `tcrv_scalar.element_count` metadata without an explicit typed scalar body
  still fails closed before `tcrv_scalar.lowering_boundary` creation.
- Rewrote `test/Plugin/ScalarExtensionPluginTest.cpp` so no-body scalar
  fallback asserts no typed microkernel is synthesized and receives a
  metadata-only emission plan; added explicit typed microkernel preservation
  coverage and metadata-alone rejection coverage.
- Updated RVV+scalar dispatch and rvv-probe lit expectations so no-body scalar
  fallback emits metadata-only diagnostics instead of the deleted direct-C route
  diagnostic that only applies to explicit typed scalar microkernel bodies.
- Updated scalar fallback, lowering-runtime, and MLIR testing specs to remove
  descriptorless default scalar microkernel materialization authority.
- Focused ref-scan found no remaining
  `buildDescriptorlessDefaultScalarTypedMaterializationPlan`,
  `microkernelPlan`, `materializeScalarMicrokernelOp`,
  `descriptorless typed default materialization`, or
  `descriptorless default materialization` occurrences in active code/spec/tests.
- Remaining `getI32VAddFamilySpec` use is explicit typed-op family mapping, not
  default materialization. Remaining `selectedPathHasCallableMicrokernel`
  branches gate explicit typed-body ABI boundary setup only. Remaining
  `tcrv_scalar.element_count` occurrences are bounded metadata validation,
  metadata-alone rejection tests/specs, or dialect tests. Remaining scalar
  direct-C deletion diagnostics are explicit typed microkernel fail-closed
  checks.
- Checks passed:
  `cmake --build build --target tianchenrv-scalar-extension-plugin-test tcrv-opt tcrv-translate -j2`;
  `./build/bin/tianchenrv-scalar-extension-plugin-test`;
  `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv Target/RVVScalarDispatch/rvv-scalar-i32-vadd-dispatch-generic-route.mlir Target/ArtifactExport/scalar-target-source-artifact-routes.test Scripts/rvv-probe-to-mlir.test`
  from `build/test`;
  `cmake --build build --target tianchenrv-target-artifact-export-test -j2`;
  `./build/bin/tianchenrv-target-artifact-export-test`;
  `cmake --build build --target check-tianchenrv -j2` with 114/114 passing.
- Missing new-architecture gap: no scalar executable output exists for no-body
  fallback. That is intentional for this deletion round; rebuild must come
  later from explicit `tcrv_scalar` extension-family ops through the common
  EmitC/artifact route.
