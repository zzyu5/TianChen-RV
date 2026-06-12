# Stage2 RVV memory-family runtime AVL/VL sole-authority cleanup

## Goal

Make promoted RVV memory-family target artifact validation visibly treat
`RVVRuntimeAVLVLSelectedBoundaryContract` as the sole acceptance authority for
runtime `n` / AVL / VL facts. Existing route-description runtime/control copies
may remain only as generated mirrors checked after the shared selected-boundary
contract has matched the rebuilt provider route.

The bounded production path is:

```text
tcrv.exec runtime_param n
  -> selected typed tcrv_rvv memory body
  -> RVV plugin-owned memory route-family facts
  -> RVVRuntimeAVLVLSelectedBoundaryContract
  -> provider-built memory route validation contract
  -> target artifact validation consumes the runtime contract first
  -> route-local runtime/control mirror consistency checks
  -> route payload/header/type/ABI/statement/mirror validation
```

## Direction Source

Hermes Direction Brief:

`Stage2 RVV memory-family runtime AVL/VL sole-authority cleanup`

## Initial Repository State

* Repository root: `/home/kingdom/phdworks/TianchenRV`.
* Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
* Initial `git status --short`: clean through RTK.
* Initial `git log --oneline -8` starts at
  `240ada25 rvv: consume runtime AVL VL contract for computed strided memory`.
* `.trellis/.current-task` was absent, so this task was created from the
  supplied Direction Brief before source edits.
* Serial worker constraint from the brief: one Codex worker; no spawned
  subagents or multi-agent workflow.

## What I Already Know

* `.trellis/spec/index.md` keeps route authority in the selected `tcrv.exec`
  envelope, typed low-level `tcrv_rvv` body, RVV plugin-owned legality /
  realization / provider, provider-built `TCRVEmitCLowerableRoute`, neutral
  common EmitC materialization, and target artifact validation.
* `.trellis/spec/extension-plugins/rvv-plugin.md` requires runtime AVL/VL,
  SEW/LMUL, policy, ABI order, and selected capability facts to flow through
  RVV-owned provider plans; target artifacts and generated metadata may only
  mirror provider facts after route construction.
* `.trellis/spec/lowering-runtime/emitc-route.md` defines
  `RVVRuntimeAVLVLSelectedBoundaryContract` as the provider-owned selected
  boundary authority for runtime AVL source, runtime VL contract, selected
  `with_vl` scope, `setvl`, VL C type, full-chunk VL, loop VL, loop induction,
  runtime `n` ABI parameter, remaining AVL metadata, pointer advancement,
  bounded slice, and multi-VL facts.
* Recent archived memory tasks completed per-family promotion:
  * base memory embeds and consumes the runtime contract;
  * unit-stride masked memory embeds and consumes the runtime contract;
  * computed-mask indexed memory embeds and consumes the runtime contract;
  * computed-mask strided memory embeds and consumes the runtime contract;
  * segment2 memory already embeds the runtime contract and later removed
    target-side raw segment2 fact reconstruction.
* Live header inspection shows all five memory-family validation contracts
  already embed `runtimeAVLVLContract`, but they still expose route-local fields
  such as `runtimeControlPlanID`, `runtimeABIOrder`,
  `emitCFullChunkVLName`, `emitCLoopVLName`, and
  `emitCLoopInductionName`.
* Live target inspection shows memory validators call
  `validateRVVRuntimeAVLVLSelectedBoundaryContract(...)` before most
  route-local checks, then still compare the route-local runtime/control fields
  against the rebuilt route description.

## Requirements

* Cover the promoted memory-family target consumers in one bounded cleanup:
  base memory movement, unit-stride masked memory, computed-mask indexed
  memory, computed-mask strided memory, and segment2 memory.
* Keep `RVVRuntimeAVLVLSelectedBoundaryContract` as the first runtime
  acceptance check in each memory-family target payload validator.
* Rename, demote, or explicitly document route-local runtime/control fields as
  mirrors only where they still need consistency checks against rebuilt route
  descriptions or candidate metadata.
* Stale, missing, or mismatched route-local runtime/control mirrors must fail
  closed after the embedded runtime contract has already been validated. They
  must not become substitute acceptance authority.
* Avoid target-local reconstruction of runtime `n`, AVL, VL, loop induction,
  remaining AVL, pointer advancement, ABI order, or setvl facts from route ids,
  artifact names, C snippets, descriptors, test names, or metadata mirrors.
* Preserve emitted C, runtime ABI order, runtime counts, statement ordering,
  memory behavior, correctness behavior, and performance behavior.
* Keep common EmitC/export neutral and avoid moving RVV semantics into common
  code.

## Acceptance Criteria

* [x] Memory-family provider/target code makes route-local runtime/control
      copies explicit mirrors, either by field names, helper names, or local
      comments at the consumption boundary.
* [x] Base memory, unit-stride masked memory, computed-mask indexed memory,
      computed-mask strided memory, and segment2 target validators call the
      shared runtime AVL/VL selected-boundary validator before route-local
      runtime/control mirror checks.
* [x] Focused target C++ coverage proves stale route-description runtime
      mirrors cannot override the embedded selected-boundary contract for the
      touched memory consumers.
* [x] Existing fail-closed diagnostics for stale provider mirrors and stale
      candidate metadata remain intact.
* [x] `tianchenrv-target-artifact-export-test` and
      `tianchenrv-rvv-extension-plugin-test` pass if provider/target code
      changes.
* [x] Focused memory-family lit or generated-bundle dry-run checks pass for the
      touched route slices where local tooling is available.
* [x] Added-line old-authority scan over touched files finds no new positive
      dependency on `RVVI32M1`, `rvv-i32m1`, `tcrv_rvv.i32_*`,
      `!tcrv_rvv.i32m*`, descriptor/direct-C/source-export/source-front-door,
      common-EmitC, route-id/artifact-name, exact `__riscv_*_i32m1`, or
      mirror-only authority.
* [x] `rtk git diff --check` passes.
* [x] Trellis task status, context, journal/archive state, and commit state are
      truthful.

## Completion Notes

* `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp` now has a shared
  route-local runtime AVL/VL mirror helper. The five promoted memory consumers
  validate `runtimeAVLVLContract` first, then compare retained local runtime
  control plan, runtime ABI order, setvl callee, VL C type, full-chunk VL,
  loop VL, and loop induction copies as mirrors only.
* `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h` documents those
  memory-family validation contract copies as target-side consistency mirrors.
* `test/Target/TargetArtifactExportTest.cpp` asserts the retained route-local
  runtime mirrors equal the embedded selected-boundary contract for base
  memory, unit-stride masked memory, computed-mask indexed memory,
  computed-mask strided memory, and segment2 memory.
* `.trellis/spec/lowering-runtime/emitc-route.md` records the promoted
  memory-family selected-boundary consumption order.
* No `ssh rvv` run was needed because emitted C/C++, runtime ABI order/counts,
  statement ordering, memory behavior, correctness behavior, and performance
  behavior were not changed.

## Technical Approach

1. Inspect current provider contract builders and target payload validators for
   base memory, unit-stride masked memory, computed-mask indexed memory,
   computed-mask strided memory, and segment2 memory.
2. Use the smallest production diff that makes route-local runtime/control
   copies explicitly mirror-only. Prefer a shared target helper and targeted
   field renames for validation contracts if the rename blast radius remains
   bounded.
3. Ensure each memory-family target path validates
   `runtimeAVLVLContract` before any mirror consistency checks.
4. Add or update focused target C++ mutations for stale route-local runtime
   mirrors where current tests do not already prove the ordering.
5. Run focused build/test/lit checks, old-authority scan, `git diff --check`,
   Trellis validation, archive, and commit if complete.

## Non-Goals

* No new memory operations or route coverage.
* No reduction, elementwise, conversion, MAcc, widening-dot, scalar, IME,
  offload, TensorExt, source-front-door, or high-level frontend work.
* No broad smoke matrix or dashboard/report-only evidence.
* No `ssh rvv` unless emitted C, runtime ABI order, runtime counts, statement
  ordering, memory behavior, correctness, or performance behavior changes.

## Evidence Plan

* Build focused tests:
  `rtk cmake --build build --target tianchenrv-target-artifact-export-test tianchenrv-rvv-extension-plugin-test -j 16`
* Run focused C++ tests:
  `rtk build/bin/tianchenrv-target-artifact-export-test`
  `rtk build/bin/tianchenrv-rvv-extension-plugin-test`
* Run focused memory-family lit/generated-bundle dry-run filters after final
  touched route slices are known.
* Run bounded scans:
  * direct target raw-fact scan for removed/forbidden target-local runtime
    authority;
  * added-line old-authority scan over touched files.
* Run `rtk git diff --check`.
* Run Trellis validation:
  `rtk python3 ./.trellis/scripts/task.py validate .trellis/tasks/06-04-stage2-rvv-memory-runtime-avl-vl-sole-authority`

## SSH RVV Rationale

Do not run `ssh rvv` if the implementation remains a provider/target
validation naming/order/test cleanup. The expected diff should not change
emitted C/C++, runtime ABI order, runtime counts, statement ordering, memory
behavior, correctness behavior, or performance behavior.

## Definition Of Done

Promoted RVV memory-family target artifact validation accepts runtime `n` /
AVL / VL facts only through the embedded
`RVVRuntimeAVLVLSelectedBoundaryContract`; route-local runtime/control fields
are visibly mirror-only consistency checks. Focused checks and scans pass, the
task is finished/archived, and a coherent commit records the production diff
and evidence.
