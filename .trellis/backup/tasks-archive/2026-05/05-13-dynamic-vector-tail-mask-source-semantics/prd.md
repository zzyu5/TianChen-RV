# Dynamic vector tail/mask source semantics boundary

## Goal

Make the bounded dynamic vector i32-vadd front door's source-tail semantics
explicit and fail-closed from MLIR source active-lane authority through
`tcrv.exec` runtime ABI, selected RVV metadata, materialized `tcrv_rvv`
setvl/load/add/store body, EmitC artifacts, dispatch bundle artifacts, and
fresh `ssh rvv` evidence for runtime counts 7, 16, and 23.

This task preserves the neutral production owner:

```text
marked dynamic vector/SCF i32-vadd source wrapper
  -> --tcrv-lower-source-rvv-binary-to-exec
  -> tcrv.exec runtime ABI plus source active-lane authority
  -> RVV plugin selected finite i32-vadd variant/config
  -> materialized tcrv_rvv body with setvl(n)
  -> common EmitC artifacts and RVV+scalar dispatch bundle
  -> ssh rvv evidence
```

## What I Already Know

* Initial repository state for this task: `pwd` is
  `/home/kingdom/phdworks/TianchenRV`, the worktree was clean, and HEAD was
  `f366fe0 refactor(frontend): split source lowering owner`.
* No `.trellis/.current-task` existed at session start. This task was created
  from the Hermes Direction Brief before source edits and started as current.
* The prior ownership split moved production source lowering to
  `--tcrv-lower-source-rvv-binary-to-exec`; this round must not weaken that
  route or reintroduce a linalg/vector-specific default path.
* The prior dynamic vector task proved `%n` reaches runtime AVL and produced
  fresh RVV evidence for counts 7, 16, and 23.
* Current inspection shows the checked-in dynamic vector source fixture uses
  `vector.transfer_read` and `vector.transfer_write` with
  `in_bounds = [true]`. That is not a truthful source-tail authority for
  runtime counts that are not multiples of 16, because tail iterations like
  7 and 23 need bounded out-of-bounds lanes to be padded/ignored instead of
  asserted in bounds.
* Existing production metadata already distinguishes source `%n` runtime AVL
  from selected RVV vector shape and descriptor-local `element_count`, but it
  does not record the MLIR transfer tail/active-lane authority explicitly.

## Requirements

* Keep compiler implementation in C++ / MLIR / LLVM / TableGen / CMake / lit /
  FileCheck. Python remains only for tooling/evidence.
* Keep the task bounded to the dynamic vector i32-vadd submodule. Do not add
  another op, dtype, generic vector backend, scalable-vector maturity claim, or
  performance claim.
* Preserve the neutral source-frontdoor owner. Direct lowering and
  plan-and-export bundle must use the same
  `--tcrv-lower-source-rvv-binary-to-exec` production route.
* Reject dynamic vector source wrappers that assert full-vector in-bounds
  transfer semantics for tail-capable `%n` loops. The accepted dynamic source
  pattern must expose active-lane/tail behavior with MLIR transfer tail
  semantics: `scf.for` from zero to `%n` in step 16, vector transfer reads with
  i32 padding, vector transfer write at the induction variable, and no
  `in_bounds = [true]` assertion on those transfer ops.
* Materialize the dynamic source active-lane/tail authority on both
  `tcrv.exec.kernel` and `@abi_runtime_element_count` so stale, missing, or
  inconsistent metadata fails before artifact output.
* Carry source active-lane/tail metadata into RVV selected-plan metadata
  separately from compile-time selected vector shape, selected RVV tail/mask
  policy, and descriptor-local `element_count`.
* Generated direct source/header/object artifacts and RVV+scalar dispatch
  source/header/object bundle artifacts must expose the source active-lane
  authority, selected RVV tail/mask policy, and `setvl(n)` runtime AVL
  boundary.
* Fixed-vector `n == 16` enforcement must remain intact and must not inherit
  the dynamic tail contract.
* Descriptor-driven computation, descriptor-to-C export, prompt/report/helper
  only closeout, and generic broad smoke matrices are out of scope.

## Acceptance Criteria

* [ ] Trellis PRD, implement context, and check context exist before source
      edits, and the task is current.
* [ ] Dynamic vector lowering rejects `vector.transfer_read` or
      `vector.transfer_write` with `in_bounds = [true]` under the runtime
      `%n`/SCF wrapper.
* [ ] The accepted dynamic vector fixture uses MLIR transfer tail semantics
      instead of in-bounds assertions, and direct lowering records source
      active-lane/tail authority on the kernel and runtime-element-count
      runtime_param.
* [ ] Selected-plan metadata includes source active-lane/tail authority entries
      distinct from `tcrv_rvv.selected_tail_policy`,
      `tcrv_rvv.selected_mask_policy`, and `tcrv_rvv.descriptor_element_count`.
* [ ] RVV direct source/header/object export rejects stale, missing, or
      inconsistent active-lane/tail metadata before output.
* [ ] Dynamic RVV direct artifacts expose the source active-lane authority,
      selected RVV tail/mask policy, descriptor-local `element_count = 16`,
      runtime `n`, and `tcrv_rvv.setvl` authority without a fixed `n == 16`
      trap.
* [ ] Dynamic plan-and-export bundle artifacts expose the same source
      active-lane contract through the neutral production route.
* [ ] Fixed-vector direct and bundle tests still show fixed source extent 16
      and `n == 16` fail-closed behavior.
* [ ] Focused build covers `TianChenRVTransforms`, `TianChenRVTarget`,
      `tcrv-opt`, `tcrv-translate`, and
      `tianchenrv-target-artifact-export-test` as applicable.
* [ ] Focused lit covers dynamic `VectorToExec`, fixed-vector regression,
      dynamic vector plan-and-export bundle, `RVVMicrokernel`,
      `TargetArtifactBundleExport`, `EmissionManifest`, and representative
      `LinalgToExec`.
* [ ] Exact direct and bundle artifact commands for the dynamic vector input
      are recorded.
* [ ] Fresh `ssh rvv` evidence proves counts 7, 16, and 23 under the repaired
      source-tail contract.
* [ ] `git diff --check`, `git diff --cached --check`, and Trellis validation
      pass before finish/archive.
* [ ] The task is finished/archived and one coherent commit records the round
      if complete.

## Definition Of Done

* Dynamic source `%n` is not merely runtime AVL; it is tied to explicit MLIR
  transfer-tail active-lane authority for non-multiple counts.
* Selected RVV tail/mask policy remains compile-time vector config, not source
  active-lane semantics.
* Generated artifacts make the distinction readable and reject stale metadata.
* Hardware claims are bounded to fresh named `ssh rvv` evidence paths and do
  not claim performance or generic vector support.

## Out Of Scope

* Generic MLIR vector backend, scalable-vector maturity, new dtypes, new ops,
  operation/dtype matrices, performance evidence, LLVM/RISC-V backend work,
  inline assembly, or new hardware targets.
* Descriptor-driven computation, direct descriptor-to-C export, or moving
  arithmetic semantics into `tcrv.exec`.
* Report-only, prompt-only, helper-only, or broad smoke-test closeout.

## Technical Notes

* Specs read: `.trellis/spec/index.md`,
  `.trellis/spec/core-dialect/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  `.trellis/spec/lowering-runtime/emission-runtime-contract.md`, and
  `.trellis/spec/testing/mlir-testing-contract.md`.
* Prior PRDs read:
  `.trellis/tasks/archive/2026-05/05-13-05-13-source-frontend-lowering-ownership-split/prd.md`
  and
  `.trellis/tasks/archive/2026-05/05-13-dynamic-vector-i32-vadd-runtime-avl-front-door/prd.md`.
* Initial code/tests inspected:
  `include/TianChenRV/Support/FiniteBinaryFrontendLowering.h`,
  `include/TianChenRV/Target/RVV/RVVSelectedConfigContract.h`,
  `include/TianChenRV/Target/RVV/RVVVectorShape.h`,
  `include/TianChenRV/Transforms/Passes.td`,
  `lib/Transforms/LowerSourceRVVBinaryToExec.cpp`,
  `lib/Plugin/RVV/RVVBinarySelectedEmissionPlanning.cpp`,
  `lib/Target/RVV/RVVMicrokernel.cpp`,
  `lib/Target/Builtin/RVVScalarDispatch.cpp`,
  `lib/Conversion/EmitC/TCRVEmitCLowerableMaterializer.cpp`,
  `test/Transforms/VectorToExec/vector-dynamic-i32-vadd-to-exec.mlir`, and
  `test/Target/TargetArtifactBundleExport/plan-vector-dynamic-i32-vadd-and-export-target-artifact-bundle.mlir`.
