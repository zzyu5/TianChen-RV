# RVV selected variant materialization route

## Goal

Make one bounded upstream compiler route materialize the RVV
extension-family op/state that the hardened RVV artifact path already
consumes. The first slice is the existing finite i32 RVV binary route with
the best downstream coverage. The result must prove that compiler-produced IR,
not only hand-authored target fixtures, reaches a `tcrv_rvv` typed
microkernel, selected lowering boundary, plugin-owned planning, and generated
RVV source/header artifact output while preserving op-owned identity, selected
config, and runtime length authority.

## What I Already Know

* Repository root is `/home/kingdom/phdworks/TianchenRV`.
* Initial repo state for this round: clean worktree, HEAD
  `a5b700d feat(rvv): consume selected config in artifact export`.
* No `.trellis/.current-task` existed, so this task was created from the
  Hermes Direction Brief.
* The immediately previous RVV artifact rounds completed downstream
  consumption of op-owned source identity, object artifact evidence, runtime
  AVL/VL role data, and selected SEW/LMUL/tail/mask config.
* The current bottleneck is upstream structural migration: a supported compiler
  route must produce the selected RVV extension-family op/state consumed by
  that artifact path instead of relying on hand-authored or descriptor-adjacent
  target fixtures.
* The likely bounded route is the current i32 RVV binary path, preferably
  `i32-vadd` / `i32m1`, because it has source/header/object, scalar dispatch,
  runtime-length, selected-config, and e2e evidence coverage.

## Requirements

* Keep the migration bounded to one existing supported compiler-produced route
  and one finite RVV binary config slice, preferably `i32-vadd` with selected
  `i32m1`, SEW 32, LMUL m1, tail agnostic, mask agnostic.
* Materialize concrete `tcrv_rvv` selected-state for that route:
  `tcrv_rvv.lowering_boundary`, typed RVV binary microkernel op, source
  identity fields, selected SEW/LMUL/tail/mask config, and runtime
  AVL/VL role data.
* Ensure RVV plugin legality/planning consumes the compiler-produced op/state
  without making descriptor-owned compute semantics authoritative.
* Keep the downstream artifact exporter consuming op-owned identity, selected
  config, and runtime length contracts.
* Fail closed or quarantine descriptor-only, stale descriptor-driven, missing
  source-identity, stale selected-config, and stale runtime-length attempts
  for the migrated production slice.
* Prove that generated RVV source/header artifacts can be reached from the
  upstream compiler route, not only from hand-authored selected-path target
  tests.
* Keep `tcrv.exec` orchestration-only. Generic transforms and core passes must
  not gain RVV semantic branches.
* Compiler production logic must stay in C++ / MLIR / LLVM / TableGen / CMake /
  lit / FileCheck. Python may only remain runner or evidence tooling.

## Acceptance Criteria

* [x] A compiler-produced upstream selected boundary or high-level lowering
      route materializes a matching `tcrv_rvv` typed microkernel plus
      `tcrv_rvv.lowering_boundary` for the migrated slice.
* [x] The materialized boundary carries op-owned source identity, including the
      selected binary source kind, dtype, family, operator, microkernel op,
      EmitC source op, and generated interface identity expected by the target
      artifact preflight.
* [x] Selected config metadata carried into the route includes selected
      SEW/LMUL/tail/mask plus selected vector type/suffix and setvl suffix
      consumed by the downstream selected-config contract.
* [x] Runtime length data carried into the route includes the IR-backed runtime
      element-count role and the `tcrv_rvv.setvl` / `tcrv_rvv.with_vl`
      runtime AVL/VL control surface consumed by the runtime-length contract.
* [x] RVV plugin legality/planning validates and consumes the
      compiler-produced op/state and still rejects missing, stale, duplicate,
      or descriptor-only production authority.
* [x] A focused lit/FileCheck path starts from the upstream route and reaches
      generated RVV source/header artifact output through existing public
      planning/export front doors.
* [x] Fail-closed tests cover missing/wrong selected boundary, missing RVV op
      identity, stale config/runtime-length role data, and descriptor-only
      production attempts for the migrated slice.
* [x] Existing focused regressions for op-owned object artifact evidence,
      runtime-length consumption, selected-config consumption, RVV microkernel
      export, and scalar dispatch still pass.
* [x] A bounded scan confirms descriptor-only compute/config/runtime control is
      not the default production route for the migrated slice and no core
      `tcrv.exec` or generic transform RVV semantic branch was added.
* [x] If generated source/object behavior changes materially, focused
      `ssh rvv` compile/run evidence is collected through the existing e2e
      path. If object execution is staged, the blocker is reported explicitly.
      No generated RVV source/object behavior changed in this round, so no new
      `ssh rvv` runtime claim was made.
* [x] `git diff --check`, staged diff check, Trellis validation before finish
      and after archive, final archive, and one coherent commit complete the
      round if finished.

## Definition Of Done

* The default migrated route is no longer hand-authored target fixture first:
  a supported upstream compiler path produces the RVV selected op/state that
  planning and artifact export consume.
* Descriptor metadata remains legacy mirror or bounded fixture metadata only
  after typed source authority, selected config authority, and runtime-length
  authority have been established.
* The final report states the task id/title, phase, migrated route/family/config
  slice, upstream selected boundary used, `tcrv_rvv` op/source identity
  materialized, selected config and runtime length data carried, plugin
  planning consumed, generated artifact outputs, descriptor quarantine,
  fail-closed cases, core-pass neutrality evidence, local checks, `ssh rvv`
  evidence or exact blocker, archive status, and commit hash.

## Out Of Scope

* New RVV family, new extension family, i64 expansion, dtype matrix, LMUL
  matrix, broad smoke matrix, or performance tuning as the main result.
* Another artifact-side validation micro-round unless it is the single blocker
  for compiler-produced materialization.
* Descriptor-to-C production export or descriptor element count/vector shape as
  authoritative compute, selected config, or runtime control.
* Moving computation semantics into `tcrv.exec`.
* RVV semantic branches in generic core passes.
* GCC or vendor compiler as the default route.
* Template, Toy, or TensorExtLite work except narrow regressions caused by
  shared interfaces.
* Runtime, correctness, or performance claims beyond focused evidence actually
  run.

## Technical Approach

Start from the existing finite RVV binary frontend/planning path and identify
the shortest compiler-produced route that already reaches a selected
`rvv-plugin` variant. Rewire the production path, not only tests, so that the
selected RVV route materializes the same typed microkernel body and boundary
source identity that the target artifact exporter requires. Preserve the
downstream `RVVBinarySelectedConfigContract` and `RVVRuntimeLengthContract`
surfaces as consumers, and add focused fail-closed tests around the migrated
route.

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

* `.trellis/tasks/archive/2026-05/05-13-rvv-selected-config-artifact-consumption/prd.md`
* `.trellis/tasks/archive/2026-05/05-13-rvv-runtime-length-artifact-consumption/prd.md`
* `.trellis/workspace/codex/journal-5.md`

Likely implementation surface to inspect:

* `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`
* `lib/Dialect/RVV/IR/RVVDialect.cpp`
* `include/TianChenRV/Plugin/RVV/`
* `lib/Plugin/RVV/`
* `lib/Transforms/LowerSourceRVVBinaryToExec.cpp`
* `include/TianChenRV/Target/RVV/`
* `lib/Target/RVV/`
* `test/Transforms/LoweringBoundary/`
* `test/Transforms/LinalgToExec/`
* `test/Target/RVVMicrokernel/`
* `test/Target/RVVScalarDispatch/`

## Implementation Summary

Current-code audit found that the production compiler route was already
materializing the selected RVV op/state through:

```text
marked linalg.generic i32-vadd
  -> --tcrv-lower-linalg-rvv-binary-to-exec
  -> tcrv.exec.kernel with target profile and IR-backed callable ABI boundary
  -> --tcrv-execution-planning-pipeline
  -> rvv-plugin proposal/legality/selection
  -> tcrv_rvv.lowering_boundary with frontend-lowering source identity
  -> tcrv_rvv.i32_vadd_microkernel with setvl/with_vl/load/add/store body
  -> target source/header/object artifact front doors
```

The missing piece for this bounded task was executable evidence on the default
`i32-vadd` / `i32m1` upstream route. I extended
`test/Transforms/LinalgToExec/linalg-i32-vadd-to-exec.mlir` so the same
compiler-produced input now proves:

* selected variant metadata carries `tcrv_rvv.selected_binary_*`,
  selected vector-shape config, runtime AVL/VL metadata, and runtime
  element-count C-name metadata;
* `tcrv_rvv.lowering_boundary` carries `selected_binary_source_kind =
  "frontend-lowering"` plus typed source identity;
* `tcrv_rvv.i32_vadd_microkernel` is materialized from the compiler route with
  `tcrv_rvv.setvl`, `tcrv_rvv.with_vl`, two `tcrv_rvv.i32_load` ops,
  `tcrv_rvv.i32_add`, and `tcrv_rvv.i32_store`;
* generated RVV source contains selected-config authority, runtime-length
  authority, IR-backed callable ABI evidence, and `__riscv_vsetvl_e32m1`,
  `__riscv_vle32_v_i32m1`, `__riscv_vadd_vv_i32m1`, and
  `__riscv_vse32_v_i32m1`;
* generated header declares the same IR-backed callable ABI; and
* generic artifact export produces a RISC-V ELF relocatable object whose symbol
  is `tcrv_rvv_i32_vadd_microkernel_frontend_i32_vadd_rvv_first_slice`.

The same test now also fails closed when the compiler-produced route is
tampered to remove the typed RVV microkernel body, remove boundary source
identity fields, stale the selected vector suffix, or stale runtime AVL role
metadata from runtime-element-count to descriptor-element-count.

During broader focused validation, `Transforms/LinalgToExec` exposed one stale
check in `linalg-i32-vmul-to-rvv-artifact.mlir`: it still expected the old
`intrinsic_config_source` wording after selected-config contract consumption.
I updated that FileCheck line to the current
`RVVBinarySelectedConfigContract cross-checked...` source authority wording.

## Evidence

Local checks run:

* `cmake --build build --target TianChenRVRVVDialect TianChenRVRVVPlugin TianChenRVRVVTarget TianChenRVScalarTarget tianchenrv-rvv-extension-plugin-test tianchenrv-rvv-selected-lowering-boundary-test tianchenrv-rvv-binary-planning-test tianchenrv-rvv-binary-variant-legality-test tianchenrv-target-artifact-export-test tcrv-opt tcrv-translate -j2`
* `./build/bin/tianchenrv-rvv-extension-plugin-test`
* `./build/bin/tianchenrv-rvv-selected-lowering-boundary-test`
* `./build/bin/tianchenrv-rvv-binary-planning-test`
* `./build/bin/tianchenrv-rvv-binary-variant-legality-test`
* `./build/bin/tianchenrv-target-artifact-export-test`
* `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv Transforms/LinalgToExec/linalg-i32-vadd-to-exec.mlir` from `build/test`
* `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv Transforms/LinalgToExec` from `build/test`
* `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv Target/RVVMicrokernel` from `build/test`
* `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv Target/RVVScalarDispatch` from `build/test`
* `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv Transforms/LoweringBoundary` from `build/test`
* `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-14-rvv-selected-variant-materialization-route`
* `python3 ./.trellis/scripts/task.py validate .trellis/tasks/archive/2026-05/05-14-rvv-selected-variant-materialization-route`
* `git diff --check`
* `git diff --cached --check`

Self-repair:

* The first `Transforms/LinalgToExec` run found stale FileCheck text in
  `linalg-i32-vmul-to-rvv-artifact.mlir`; after updating it, the full
  `Transforms/LinalgToExec` directory passed.

Core neutrality / descriptor quarantine:

* `git diff -- lib/Transforms include/TianChenRV/Dialect/Exec lib/Dialect/Exec`
  is empty.
* Changed source behavior is test-only; no generic transform, core
  `tcrv.exec`, or RVV production C++ path was modified.
* The new default `i32-vadd` upstream route checks explicitly reject stale
  runtime AVL role metadata and missing frontend-lowering source identity
  before source output.

Spec update judgment:

* No `.trellis/spec/` update was made. This round introduced no new API,
  command, pass signature, payload field, or long-term contract; it added
  executable coverage for existing selected-boundary, selected-config, and
  runtime-length contracts.
