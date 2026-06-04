# Stage2 RVV memory-family runtime AVL/VL mirror closure

## Goal

Close residual RVV memory-family runtime-boundary consumers so target artifact
validation accepts runtime `n` / AVL / VL facts only through the provider-built
`RVVRuntimeAVLVLSelectedBoundaryContract`. Route-local fields such as
`runtimeControlPlanID`, `runtimeABIOrder`, setvl/VL/loop names, and candidate
metadata labels may remain only as consistency mirrors checked after the shared
selected-boundary contract is accepted.

The bounded production path is:

```text
tcrv.exec runtime_param n
  -> selected typed tcrv_rvv memory body
  -> RVV plugin-owned runtime AVL/VL selected-boundary contract
  -> provider-built memory-family TCRVEmitCLowerableRoute
  -> common EmitC materialization
  -> target artifact validation consumes runtime facts through the contract
  -> route-local and candidate metadata runtime mirrors are checked only as mirrors
```

## Direction Source

Hermes Direction Brief:

`Stage2 RVV memory-family runtime AVL/VL mirror closure`

## Initial Repository State

* Repository root: `/home/kingdom/phdworks/TianchenRV`.
* Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
* Initial `git status --short`: clean through RTK.
* Initial `git log --oneline -8` starts at
  `fdb36920 rvv: make conversion runtime mirrors non-authoritative`.
* `.trellis/.current-task` was absent, so this task was created from the
  supplied Direction Brief before source edits.
* Current task directory:
  `.trellis/tasks/06-04-stage2-rvv-memory-family-runtime-avl-vl-mirror-closure`.
* Serial worker constraint from the brief: one Codex worker; no spawned
  subagents or multi-agent workflow.

## What I Already Know

* `.trellis/spec/index.md` keeps route authority in the selected `tcrv.exec`
  envelope, typed low-level `tcrv_rvv` body, RVV plugin-owned legality /
  realization / provider, provider-built `TCRVEmitCLowerableRoute`, neutral
  common EmitC materialization, and target artifact validation.
* `.trellis/spec/extension-plugins/rvv-plugin.md` requires base memory,
  computed-mask memory, and segment2 memory statement/route facts to come from
  typed body/config/runtime facts and RVV-owned provider plans, not common
  EmitC, descriptors, route ids, artifact names, or metadata mirrors.
* `.trellis/spec/lowering-runtime/emitc-route.md` defines
  `RVVRuntimeAVLVLSelectedBoundaryContract` as the provider-owned selected
  boundary authority for runtime AVL source, runtime VL contract, selected
  `with_vl` scope, `setvl`, VL C type, full-chunk VL, loop VL, loop induction,
  runtime `n` ABI parameter, remaining AVL metadata, pointer advancement,
  bounded slice, and multi-VL facts.
* Recent archived tasks promoted base memory, unit-stride masked memory,
  computed-mask indexed memory, computed-mask strided memory, segment2 memory,
  compare/select, reduction, contraction, scalar/elementwise, and conversion
  consumers toward selected-boundary-first runtime AVL/VL validation.
* The current brief is a bounded cleanup after those promotions: inspect the
  live memory-family planning metadata and target validation/candidate metadata
  labels and close any residual cases where runtime-control or ABI-order fields
  read like independent authority instead of route-local mirrors.

## Requirements

* Cover memory-family runtime-boundary consumers only:
  base memory movement, computed-mask memory, and segment2 memory target
  metadata/validation.
* Keep `RVVRuntimeAVLVLSelectedBoundaryContract` as the sole runtime `n` /
  AVL / VL acceptance authority before route-local runtime checks.
* Ensure retained `runtimeControlPlanID`, `runtimeABIOrder`, setvl/VL/loop-name
  fields are documented or validated as route-local runtime AVL/VL mirrors only.
* Ensure memory and segment2 candidate metadata runtime labels are mirror-only.
* Stale or missing route-local runtime metadata/mirrors must fail closed with
  diagnostics that name `route-local runtime AVL/VL mirror`.
* Retained runtime ABI parameter-list checks may validate ABI binding only
  after the selected-boundary contract is accepted; `runtimeABIOrder` must not
  become a separate acceptance authority.
* Preserve emitted C/C++, runtime ABI order/counts, statement ordering, memory
  behavior, correctness behavior, and performance behavior unless repository
  evidence proves a runtime bug.
* Keep common EmitC/export neutral and avoid moving RVV semantics into common
  code.

## Acceptance Criteria

* [x] Base memory, computed-mask memory, and segment2 target validators accept
      runtime facts through `RVVRuntimeAVLVLSelectedBoundaryContract` before
      route-local runtime mirror checks.
* [x] Retained route-local runtime/control fields in the touched memory-family
      contracts or validation helpers are explicitly named, commented, or
      labeled as mirrors.
* [x] Candidate metadata checks for memory and segment2 runtime-control /
      runtime-ABI fields use route-local runtime AVL/VL mirror labels.
* [x] Stale or missing route-local runtime metadata/mirrors fail closed with
      diagnostics containing `route-local runtime AVL/VL mirror`.
* [x] Any retained runtime ABI parameter-list checks run after selected-boundary
      validation and validate ABI binding, not independent runtime-order
      authority.
* [x] Focused target/export tests prove stale route-local runtime mirrors or
      candidate mirrors cannot override the embedded selected-boundary contract.
* [x] `tianchenrv-target-artifact-export-test` passes.
* [x] `tianchenrv-rvv-extension-plugin-test` passes if provider/planning code
      changes.
* [x] Focused memory/segment2 lit or export checks pass for touched slices.
* [x] Before/after bounded grep records memory and segment2 runtime-control /
      runtime-ABI labels in touched validation/planning paths.
* [x] Added-line old-authority scan over touched files finds no new positive
      dependency on `RVVI32M1`, `rvv-i32m1`, `tcrv_rvv.i32_*`,
      `!tcrv_rvv.i32m*`, descriptor/direct-C/source-export/source-front-door,
      common-EmitC, route-id/artifact-name, exact `__riscv_*_i32m1`, or
      mirror-only authority.
* [x] `rtk git diff --check` passes.
* [x] Trellis task status, context, journal/archive state, commit state, and
      final report are truthful.

## Completion Notes

* `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp` now labels retained
  `tcrv_rvv.runtime_control_plan` and `tcrv_rvv.runtime_abi_order` candidate
  metadata mirrors as `route-local runtime AVL/VL control plan mirror` and
  `route-local runtime AVL/VL ABI order mirror` for:
  * unit-stride masked memory;
  * computed-mask indexed memory;
  * computed-mask strided memory;
  * plain segment2 memory;
  * computed-mask segment2 memory.
* `test/Target/TargetArtifactExportTest.cpp` now asserts the memory-family
  metadata mirror contract labels for unit-stride masked, computed-mask
  indexed, computed-mask strided, and segment2 route families.
* Segment2 generated-artifact lit checks now expect the mirror-only runtime ABI
  diagnostic label.
* Self-repair: the first focused lit run failed because `tcrv-translate` had
  not been rebuilt after the planning label change and still emitted the old
  segment2 ABI label. Rebuilt `tcrv-opt` and `tcrv-translate`, then reran the
  same lit filter successfully.
* No `ssh rvv` run was needed because this round changed metadata labels and
  tests only. It did not change emitted C/C++, runtime ABI order/counts,
  statement ordering, memory behavior, correctness behavior, or performance
  behavior.

## Checks Run

* `rtk cmake --build build --target tianchenrv-target-artifact-export-test tianchenrv-rvv-extension-plugin-test -j 16`
* `rtk build/bin/tianchenrv-target-artifact-export-test`
* `rtk build/bin/tianchenrv-rvv-extension-plugin-test`
* `rtk cmake --build build --target tcrv-opt tcrv-translate -j 16`
* `rtk python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'masked-unit|computed-masked|computed_masked|segment2'` from `build/test`: 71 passed, 406 excluded.
* `rtk git diff --check`
* Bounded old-label grep over touched memory/segment2 planning, target, and
  test paths: no old selected-typed runtime-control/runtime-ABI label matches.
* Bounded route-local label grep over touched planning/target/test paths:
  updated route-local runtime AVL/VL mirror labels present.
* Added-line old-authority scan over touched files: no matches.
* `rtk python3 ./.trellis/scripts/task.py validate .trellis/tasks/06-04-stage2-rvv-memory-family-runtime-avl-vl-mirror-closure`

## Technical Approach

1. Inspect `RVVEmitCRouteProvider.h` memory-family validation contracts,
   `RVVEmitCRoutePlanning.cpp` metadata builders, and
   `RVVTargetArtifactRouteFamilyValidation.cpp` memory/segment2 target
   consumers around runtime-control, runtime-ABI, setvl/VL/loop, and candidate
   metadata labels.
2. Compare live code against the recent archived memory-family and conversion
   runtime mirror tasks so this round only closes residual memory-family
   metadata/validation gaps.
3. Make the smallest production diff that labels or validates retained runtime
   fields as route-local mirrors after selected-boundary validation.
4. Add focused target/export coverage for any residual stale candidate or
   route-local runtime mirror that is not already tested.
5. Run focused checks, scans, Trellis validation, archive, and commit if the
   bounded module is complete.

## Non-Goals

* No new memory operations, segment forms, dtype/LMUL cases, intrinsic coverage,
  or route expansion.
* No conversion, reduction, contraction, compare/select, elementwise, scalar
  splat-store, source-front-door, high-level frontend, IME, offload, TensorExt,
  or broad route-family sweep.
* No descriptor-driven computation, source-artifact route authority, or common
  EmitC/export RVV semantic inference.
* No broad smoke matrix or dashboard/report-only evidence.
* No `ssh rvv` unless emitted C/C++, runtime ABI order/counts, statement
  ordering, memory correctness, runtime behavior, or performance behavior
  changes.

## Evidence Plan

* Build focused tests:
  `rtk cmake --build build --target tianchenrv-target-artifact-export-test tianchenrv-rvv-extension-plugin-test -j 16`
* Run focused C++ tests:
  `rtk build/bin/tianchenrv-target-artifact-export-test`
  `rtk build/bin/tianchenrv-rvv-extension-plugin-test`
* Run focused memory/segment2 lit or generated-bundle dry-run filters after the
  touched slices are known.
* Run bounded greps before/after for memory and segment2
  `runtime_control_plan`, `runtime_abi_order`, `runtimeControlPlanID`, and
  `runtimeABIOrder` labels in touched planning/validation paths.
* Run added-line old-authority scan over touched files.
* Run `rtk git diff --check`.
* Run Trellis validation:
  `rtk python3 ./.trellis/scripts/task.py validate .trellis/tasks/06-04-stage2-rvv-memory-family-runtime-avl-vl-mirror-closure`

## SSH RVV Rationale

Do not run `ssh rvv` if the implementation remains a provider/target
validation label/order/test cleanup. The intended diff should not change
emitted C/C++, runtime ABI order/counts, statement ordering, memory behavior,
correctness behavior, or performance behavior.

## Definition Of Done

Memory-family target artifact validation and candidate metadata consume runtime
`n` / AVL / VL facts only through the embedded
`RVVRuntimeAVLVLSelectedBoundaryContract`; retained route-local runtime fields
and metadata are visibly mirror-only consistency checks. Focused checks and
scans pass, the task is finished/archived, and one coherent commit records the
production diff and evidence.
