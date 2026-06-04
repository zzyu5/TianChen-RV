# Stage2 RVV reduction runtime AVL/VL sole authority cleanup

## Goal

Make the RVV reduction-family target-artifact runtime-boundary consumers treat
`RVVRuntimeAVLVLSelectedBoundaryContract` as the sole acceptance authority for
runtime `n` / AVL / VL facts. Standalone reduction and vector reduction may keep
route-local runtime/control fields and candidate metadata only as generated
mirrors checked after the shared selected-boundary contract.

## What I already know

* The previous completed commit is `1e1ee6d1 rvv: demote contraction runtime mirrors`.
* The intended production chain is selected `tcrv.exec` runtime parameter `n`
  plus typed `tcrv_rvv` reduction body, then RVV plugin-owned runtime AVL/VL
  selected-boundary contract, then provider-built reduction
  `TCRVEmitCLowerableRoute`, then common EmitC materialization and target
  artifact validation.
* Stale or missing route-local runtime mirrors must fail closed with diagnostics
  that identify `route-local runtime AVL/VL mirror`; they must not override or
  replace the embedded selected-boundary contract.
* Existing contraction, compare/select, memory, conversion, and widening-dot
  cleanup tasks provide local migration patterns.
* This is a production cleanup task, not a coverage expansion task.

## Requirements

* Standalone reduction target-artifact validation must call the shared runtime
  AVL/VL selected-boundary validator before accepting route-local runtime
  mirror fields.
* Retained standalone reduction fields such as runtime control plan id, runtime
  ABI order, setvl/VL names, and loop/control names must be documented or
  validated as `route-local runtime AVL/VL mirror` facts only.
* Standalone and vector reduction candidate metadata runtime labels must be
  mirror-only. Labels such as selected typed runtime wording on
  `runtime_control_plan` or `runtime_abi_order` must be replaced when those
  fields are only mirrors.
* Existing vector reduction selected-boundary-first validation must remain
  intact, or be repaired if residual route-local runtime authority exists.
* Common EmitC/export code must remain neutral; RVV runtime/control semantics
  stay in the RVV plugin/provider and target validation contracts.

## Acceptance Criteria

* [x] Focused production diff aligns standalone/vector reduction route
      contracts, planning metadata, or target validation with the shared runtime
      AVL/VL selected-boundary contract.
* [x] Standalone reduction target validation consumes
      `RVVRuntimeAVLVLSelectedBoundaryContract` before route-local runtime
      mirror checks.
* [x] Stale or missing route-local runtime mirrors fail closed with diagnostics
      saying `route-local runtime AVL/VL mirror`.
* [x] Standalone and vector reduction candidate metadata runtime labels are
      explicit mirrors only.
* [x] Existing vector reduction selected-boundary-first validation remains
      green; the vector reduction contract now embeds the selected-boundary
      runtime contract.
* [x] Focused target/export tests cover stale route-local runtime mirrors not
      overriding the embedded selected-boundary contract.
* [x] Bounded grep / added-line scan shows no new old-authority route ids,
      dtype-prefixed helper growth, source-front-door authority, descriptor
      authority, or common/export RVV semantic branches in touched files.

## Non-Goals

* No new reduction operations, reduction coverage, conversion, contraction,
  compare/select, elementwise, scalar splat-store, memory, source-front-door,
  high-level frontend, or intrinsic cases.
* No dtype or LMUL expansion.
* No movement of RVV semantics into common EmitC/export.
* No runtime/correctness/performance claim unless `ssh rvv` evidence is added.
* No inference of runtime control from route ids, artifact names, test names,
  manifests, C strings, descriptors, or mirror metadata.

## Expected Evidence

* `git diff --check`
* Focused reduction target/export lit or C++ tests changed by this task.
* `tianchenrv-target-artifact-export-test` if target artifact validation changes.
* Focused reduction lit/export checks.
* `tianchenrv-rvv-extension-plugin-test` if provider/planning code changes.
* Before/after bounded grep for standalone/vector reduction runtime control
  labels in touched validation/planning paths.
* Added-line old-authority scan over touched files.
* Clean `git status --short`.
* `ssh rvv` only if emitted C, runtime ABI order/counts, statement ordering,
  reduction correctness, or runtime behavior changes.

## Technical Notes

* Read first:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  recent archived runtime sole-authority tasks under
  `.trellis/tasks/archive/2026-06/`,
  `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`,
  `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`,
  and focused reduction target/export tests.
* Preferred local pattern from contraction cleanup:
  target validation first calls
  `validateRVVRuntimeAVLVLSelectedBoundaryContract(...)`, then validates
  retained route-local runtime facts through route-local mirror helpers.

## Implementation Summary

* `RVVVectorReductionRouteValidationContract` now embeds
  `RVVRuntimeAVLVLSelectedBoundaryContract`; the provider builder constructs it
  from canonical selected typed config/runtime ABI facts.
* `RVVVectorReductionRouteValidationContract` and
  `RVVStandaloneReductionRouteValidationContract` document retained
  `runtimeControlPlanID` and `runtimeABIOrder` as route-local runtime/control
  mirrors whose acceptance authority is the embedded runtime contract.
* Standalone reduction target validation now calls
  `validateRVVRouteLocalRuntimeAVLVLMirrors(...)` after the shared
  selected-boundary validator and removes older direct route-local
  runtime-control / runtime-ABI-order provider checks.
* Vector reduction target validation now calls
  `validateRVVRuntimeAVLVLSelectedBoundaryContract(...)` before route id,
  payload, runtime ABI, statement-plan, and candidate mirror acceptance, then
  checks route-local runtime fields through
  `validateRVVRouteLocalRuntimeAVLVLMirrors(...)`.
* Standalone and vector reduction candidate metadata labels for
  `tcrv_rvv.runtime_control_plan` and `tcrv_rvv.runtime_abi_order` now use
  `route-local runtime AVL/VL control plan mirror` and
  `route-local runtime AVL/VL ABI order mirror`.
* `.trellis/spec/lowering-runtime/emitc-route.md` now records the standalone
  and vector reduction runtime selected-boundary contract and mirror-only
  metadata obligations.

## Evidence

* Built:
  `ninja -C build tianchenrv-target-artifact-export-test tianchenrv-rvv-extension-plugin-test tcrv-opt tcrv-translate`
* Ran:
  `build/bin/tianchenrv-target-artifact-export-test`
* Ran:
  `build/bin/tianchenrv-rvv-extension-plugin-test`
* Ran focused lit from `build/test`:
  `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'standalone-reduce|vector-reduction|vector-materialized-target-artifact-exporters'`
  with 43 passed and 434 excluded.
* Ran bounded old-label grep for
  `selected typed RVV (standalone|vector) reduction runtime...`: no matches.
* Ran added-line old-authority scan over touched files for `RVVI32M1`,
  `rvv-i32m1`, `tcrv_rvv.i32_`, `!tcrv_rvv.i32m`, source-front-door,
  source-artifact, descriptor, exact `__riscv_*_i32m1`, status/readiness, and
  `emission_plan`: no matches.
* Ran `git diff --check`: passed.
* Ran `python3 ./.trellis/scripts/task.py validate
  .trellis/tasks/06-04-06-04-stage2-rvv-reduction-runtime-avl-vl-sole-authority`:
  passed.
* Did not run `ssh rvv`: this round changes provider/target validation
  contracts, metadata mirror labels, tests, and specs only. It does not change
  emitted C, runtime ABI order/counts, statement ordering, reduction
  computation, runtime behavior, correctness behavior, or performance behavior.

## Spec Update Judgment

Spec update was required because vector reduction gained an executable
`runtimeAVLVLContract` field and target-consumer order. The update is limited
to `.trellis/spec/lowering-runtime/emitc-route.md`.
