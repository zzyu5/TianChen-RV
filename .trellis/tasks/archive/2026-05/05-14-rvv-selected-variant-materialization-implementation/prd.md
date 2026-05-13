# RVV selected variant materialization implementation

## Goal

Convert the previous test-only selected-variant materialization coverage into
one production/default compiler route improvement for the bounded RVV
`i32-vadd` / `i32m1` slice. The upstream marked `linalg.generic` route must
still lower through the existing `tcrv.exec` envelope into RVV plugin planning,
but this round must change active RVV dialect/plugin/target code so the
materialized `tcrv_rvv.*_microkernel` op carries the same typed frontend
source identity that downstream artifact export validates.

## What I Already Know

* Repository root is `/home/kingdom/phdworks/TianchenRV`.
* Initial real repo state for this round: clean worktree, HEAD
  `e6f0486 test(rvv): cover selected variant materialization route`.
* No `.trellis/.current-task` existed, so this task was created from the
  Hermes Direction Brief.
* The archived `05-14-rvv-selected-variant-materialization-route` task expanded
  `linalg-i32-vadd-to-exec.mlir` evidence, but its implementation summary says
  the changed source behavior was test-only and no `lib/Transforms`,
  `include/TianChenRV/Dialect/Exec`, or `lib/Dialect/Exec` production path
  changed.
* Current code already routes marked `linalg.generic` through
  `--tcrv-lower-linalg-rvv-binary-to-exec`, RVV proposal/planning, selected
  lowering-boundary materialization, typed RVV microkernel materialization, and
  RVV source/header/object artifact export.
* Current boundary op carries typed source identity, but the materialized
  executable microkernel op itself does not carry a complete
  `selected_binary_*` source identity group. That leaves the previous proof
  stronger in tests than in the op-owned production surface.

## Requirements

* Keep the migrated slice bounded to the existing RVV binary
  `i32-vadd` / `i32m1`, SEW 32, LMUL m1, tail agnostic, mask agnostic route
  covered by `test/Transforms/LinalgToExec/linalg-i32-vadd-to-exec.mlir`.
* Change active production code in the RVV dialect/plugin/materialization or
  target validation owner. Test-only changes do not satisfy this task.
* Extend the materialized `tcrv_rvv.*_microkernel` source surface so the op can
  carry typed source identity:
  `selected_binary_source_kind`, `selected_binary_dtype`,
  `selected_binary_family`, `selected_binary_operator`,
  `selected_binary_microkernel_op`, `emitc_source_op`, and
  `emitc_lowerable_op_interface`.
* Have RVV plugin materialization populate that identity for the compiler
  produced `frontend-lowering` route, using selected family/config authority,
  not descriptor text.
* Have target artifact validation cross-check the op-owned identity against the
  selected boundary and typed microkernel body for the `frontend-lowering`
  route before source/header/object output.
* Preserve existing selected SEW/LMUL/tail/mask config, runtime AVL/VL dataflow,
  runtime element-count ABI role data, and selected-config/runtime-length
  artifact contracts.
* Descriptor-only or stale descriptor-driven production attempts must remain
  rejected or quarantined for the migrated slice.
* Core `tcrv.exec` remains orchestration-only; generic transforms must not gain
  RVV semantic branches.
* Compiler implementation stays in C++ / MLIR / LLVM / TableGen / CMake / lit /
  FileCheck. Python may only be used for existing runners/checks.

## Acceptance Criteria

* [x] `tcrv_rvv` microkernel ODS/dialect verification admits and validates the
      bounded typed source identity attributes without accepting arbitrary
      unknown microkernel attributes.
* [x] RVV selected microkernel materialization writes op-owned source identity
      for the selected `frontend-lowering` route.
* [x] RVV target artifact validation requires the op-owned source identity for
      the `frontend-lowering` route and rejects missing or stale op identity
      before generated source/header/object output.
* [x] The linalg `i32-vadd` route still materializes
      `tcrv_rvv.lowering_boundary`, `tcrv_rvv.i32_vadd_microkernel`,
      selected `i32m1` config, `tcrv_rvv.setvl`, `tcrv_rvv.with_vl`,
      load/add/store body, runtime element-count role metadata, and generated
      RVV artifact output.
* [x] Focused fail-closed tests cover removing or staling the microkernel
      op-owned source identity from the compiler-produced route.
* [x] Existing focused regressions for selected variant materialization,
      selected config consumption, runtime-length consumption, op-owned object
      artifact evidence, RVV microkernel export, and scalar dispatch still
      pass.
* [x] A bounded diff/scan shows no core `tcrv.exec` or generic transform RVV
      semantic branch was added.
* [x] If generated source/object bytes materially change, focused `ssh rvv`
      compile/run evidence is collected through the existing e2e path. If only
      IR metadata and validation change, no RVV runtime/correctness claim is
      made.
* [x] `git diff --check`, staged diff check, Trellis validation before finish
      and after archive, final archive, and one coherent commit complete the
      round if finished.

## Definition Of Done

The default compiler-produced `linalg.generic i32-vadd -> RVV selected path`
does not merely prove selected variant materialization through tests. The
production RVV materializer creates an executable family op with op-owned
typed source identity, the artifact exporter consumes and checks that identity
for the frontend-lowering route, and descriptor-only authority remains
quarantined.

## Out Of Scope

* New RVV family, new extension family, i64 expansion, dtype matrix, LMUL
  matrix, broad smoke matrix, or performance tuning as the main result.
* Descriptor-to-C production export or descriptor element count/vector shape as
  authoritative compute, selected config, or runtime control.
* Moving computation semantics into `tcrv.exec`.
* RVV semantic branches in generic core orchestration.
* GCC/vendor compiler as the default route.
* Template, Toy, TensorExtLite, IME, or Offload changes except narrow shared
  regressions caused by existing route validation.
* Runtime, correctness, or performance claims beyond focused evidence actually
  run.

## Technical Approach

Add the bounded source-identity attrs to the RVV microkernel ODS definitions
for the finite binary family and update dialect verification to treat them as
known, complete, bounded, and family-consistent when present. Update
`materializeRVVBinaryMicrokernelOp()` so plugin-created selected microkernels
copy the selected source kind and typed family identity from
`RVVBinarySelectedPlan`. Update RVV target microkernel record construction so
`frontend-lowering` records require matching op-owned identity before
artifact output. Extend the existing linalg i32-vadd artifact test with
positive checks and a focused negative mutation that removes or stales the
microkernel identity.

## Implementation Summary

Completed in this round:

* Added optional op-owned selected source identity attributes to all finite
  RVV binary microkernel ops in `RVVOps.td`.
* Tightened RVV dialect verification so any partial op-owned identity must be
  complete, bounded, recognized, and consistent with the concrete
  `tcrv_rvv.i32/i64_*_microkernel` family, arithmetic op, and EmitC interface.
* Extended `RVVBinaryMicrokernelMaterializationPlan` with `sourceKind` and made
  the plugin materializer write the complete op-owned identity onto generated
  RVV microkernel ops.
* Corrected selected microkernel materialization to derive source kind from the
  selected variant/boundary identity instead of re-guessing from kernel-level
  fallback resolution. This preserves existing direct typed fixtures while
  still carrying `frontend-lowering` for compiler-produced frontend routes.
* Added RVV target artifact preflight validation that requires matching
  op-owned identity for `frontend-lowering` before source/header/object export.
  Direct/default legacy fixture routes remain allowed only after boundary
  identity and existing selected-plan metadata checks.
* Updated Linalg, Vector, RVV microkernel, scalar dispatch, target bundle, and
  artifact export tests so positive checks see the op-owned identity and
  negative checks fail before artifact output when identity is missing or stale.

No core `tcrv.exec` dialect or generic orchestration semantic branch was added.
The route remains extension family ops to common EmitC/artifact export.

## Validation Summary

Focused build:

* `cmake --build build --target TianChenRVRVVDialect TianChenRVRVVPlugin TianChenRVRVVTarget tcrv-opt tcrv-translate -j2`
* `cmake --build build --target TianChenRVRVVDialect TianChenRVRVVPlugin TianChenRVRVVTarget TianChenRVScalarTarget tianchenrv-rvv-extension-plugin-test tianchenrv-rvv-selected-lowering-boundary-test tianchenrv-rvv-binary-planning-test tianchenrv-rvv-binary-variant-legality-test tianchenrv-target-artifact-export-test tcrv-opt tcrv-translate -j2`

Focused C++ tests:

* `./build/bin/tianchenrv-rvv-extension-plugin-test`
* `./build/bin/tianchenrv-rvv-selected-lowering-boundary-test`
* `./build/bin/tianchenrv-rvv-binary-planning-test`
* `./build/bin/tianchenrv-rvv-binary-variant-legality-test`
* `./build/bin/tianchenrv-target-artifact-export-test`

Focused lit:

* `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv Dialect/RVV`
* `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv Transforms/LinalgToExec`
* `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv Transforms/LoweringBoundary`
* `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv Target/RVVMicrokernel`
* `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv Target/RVVScalarDispatch`
* `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv Target/TargetArtifactBundleExport`
* `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv Target/ArtifactExport`
* `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv Target`
* `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv Transforms`

Self-repair performed:

* The first `Target/RVVMicrokernel` run exposed a source-kind regression where
  direct typed fixtures were mislabeled as `frontend-lowering`. The production
  materialization source-kind derivation was fixed to use selected
  variant/boundary identity.
* Several tests had FileCheck anchors that matched the lowering boundary's
  string-valued microkernel op name instead of the executable microkernel op
  line after new attrs were added. The checks were retargeted to
  `tcrv_rvv.*_microkernel attributes`.
* Older artifact-export fixture inputs were repaired to satisfy current
  selected source identity and `tcrv_rvv.descriptor_element_count` preflight
  before exercising their intended negative cases.

`git diff --check` passed. `git diff --cached --check` and archive validation
are run during finish/commit staging.

No `ssh rvv` compile/run was run because this round changes IR metadata,
verification, and artifact preflight only. The generated RVV C/source runtime
semantics were not changed and no runtime/correctness/performance claim is made.

## Spec Update Judgment

No Trellis spec update is needed for this round. The implementation tightens
already-specified RVV selected-source, selected-config, runtime-length, and
EmitC-route boundaries; it does not introduce a new architectural rule.

## Technical Notes

Specs read before implementation:

* `.trellis/spec/index.md`
* `.trellis/spec/architecture/design-boundaries.md`
* `.trellis/spec/core-dialect/tcrv-exec-contract.md`
* `.trellis/spec/extension-plugins/rvv-plugin.md`
* `.trellis/spec/plugin-protocol/interfaces-and-registry.md`
* `.trellis/spec/variant-pipeline/generation-selection-tuning.md`
* `.trellis/spec/capability-model/capability-contract.md`
* `.trellis/spec/lowering-runtime/emission-runtime-contract.md`
* `.trellis/spec/lowering-runtime/emitc-route.md`
* `.trellis/spec/testing/mlir-testing-contract.md`
* `.trellis/spec/guides/capability-first-design-guide.md`
* `.trellis/spec/guides/plugin-locality-review-guide.md`
* `.trellis/spec/guides/compute-boundary-review-guide.md`

Prior task context read:

* `.trellis/tasks/archive/2026-05/05-14-rvv-selected-variant-materialization-route/prd.md`
* `.trellis/tasks/archive/2026-05/05-13-rvv-selected-config-artifact-consumption/prd.md`
* `.trellis/workspace/codex/journal-5.md` Session 58

Likely implementation surface:

* `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`
* `lib/Dialect/RVV/IR/RVVDialect.cpp`
* `lib/Plugin/RVV/RVVBinaryMicrokernelMaterialization.cpp`
* `lib/Target/RVV/RVVMicrokernel.cpp`
* `test/Transforms/LinalgToExec/linalg-i32-vadd-to-exec.mlir`
