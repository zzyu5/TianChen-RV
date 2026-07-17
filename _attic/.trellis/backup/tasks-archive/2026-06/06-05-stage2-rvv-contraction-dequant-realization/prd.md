# Stage2 RVV Selected-Body Contraction-Dequant Realization

## Goal

Implement one production RVV plugin-local selected-body realization submodule for the already-proven low-precision contraction-to-f32 chain. The path should accept a selected/pre-realized `tcrv_rvv` body whose structure explicitly carries the required typed facts, materialize the realized vector-level contraction-dequant body, and hand that realized body to the existing RVV route/artifact machinery.

This task moves the compiler one step earlier than the current hand-authored realized fixtures. It is not a new route-coverage batch and not an evidence-only task.

## What I Already Know

* Commit `3449dc31` closed executable `ssh rvv` evidence for the explicit selected-body contraction-dequant route and repaired multi-VL carry/epilogue semantics.
* The existing proven route expects a realized chain including signed i8 unit loads, widening product, i16 product, signed widening reduction, i32 accumulator/carry boundary, runtime-scale dequantize, and f32 output store.
* `tcrv.exec` binds ABI/runtime roles and selected variants; it must not own RVV compute semantics.
* `tcrv_rvv` owns low-level RVV body structure; the RVV plugin owns realization, legality, route construction, intrinsic mapping, and fail-closed diagnostics.
* Common EmitC/export must stay neutral and must not choose RVV dtype, compute, schedule, or policy.
* Route ids, artifact names, ABI strings, status/result fields, mirrors, and metadata are not authority.

## Assumptions

* The smallest coherent production owner is a plugin-local realization path that recognizes a bounded pre-realized contraction-dequant structure and materializes the already-supported realized op chain.
* Existing route planning should consume realized body facts after this transformation; it should not be taught a second metadata-driven route authority.
* Executable `ssh rvv` evidence is only required if this round claims new executable correctness after production realization. Route-supported validation and generated artifact/lit checks are the expected minimum.

## Requirements

* Add production code before closing the task.
* Define or use a selected/pre-realized `tcrv_rvv` input surface whose structure carries:
  * signedness,
  * source and result element types,
  * SEW, LMUL, and policy,
  * memory roles,
  * accumulator seed/carry boundary role,
  * runtime scale role,
  * AVL/VL facts,
  * f32 store boundary.
* Implement RVV plugin-local realization from that selected/pre-realized body into the realized contraction-dequant `tcrv_rvv` chain:
  * signed i8 unit loads,
  * widening product,
  * i16 product value,
  * signed widening reduction,
  * i32 accumulator/carry boundary,
  * runtime-scale dequantize,
  * f32 output store.
* Preserve computation semantics, dtype semantics, parameter roles, variant origin, required capabilities, dispatch/fallback behavior, and runtime `n`/AVL.
* Fail closed with targeted diagnostics for missing scale, missing accumulator boundary, dtype-chain mismatch, unsupported config, stale mirrors, or any route-string/artifact-name authority attempt.
* Keep common EmitC/export neutral.

## Acceptance Criteria

* [x] A focused positive test shows selected/pre-realized input materializes to the realized contraction-dequant `tcrv_rvv` body and reaches existing route/artifact validation.
* [x] Focused negative tests cover missing structural facts and stale metadata/authority.
* [x] `tianchenrv-rvv-extension-plugin-test` covers realization behavior.
* [x] Construction protocol tests cover the realization contract when relevant.
* [x] `tianchenrv-target-artifact-export-test` is run if target validation changes.
* [x] A generated artifact or lit check verifies the realized chain.
* [x] Bounded old-authority and q-name-authority scan is run over touched files.
* [x] `git diff --check` and `git diff --cached --check` pass.
* [x] Worktree is clean after the final commit.

## Implementation Summary

This round added a bounded `tcrv_rvv.typed_widening_product_reduce_dequantize_pre_realized_body` selected-body surface and a production RVV plugin-local realization path for the contraction-dequant chain. The realization materializes signed i8 unit loads, signed widening product to i16, standalone signed widening reduction to the i32 accumulator/carry boundary, runtime-scale i32-to-f32 dequantize, and f32 output store before existing route planning and artifact export consume the realized body facts.

Validation is fail-closed in the pre-realized op verifier and contraction route-family validator for missing runtime scale role, missing accumulator carry boundary, dtype/config-chain mismatch, unsupported policy/config, and stale authority metadata such as route ids. The common EmitC/export layer remains a consumer of realized route facts; it does not infer RVV compute, dtype, schedule, or policy.

Focused checks run:

* `cmake --build build --target tianchenrv-rvv-extension-plugin-test`
* `./build/bin/tianchenrv-rvv-extension-plugin-test`
* `cmake --build build --target tcrv-opt tcrv-translate`
* `/usr/lib/llvm-20/build/utils/lit/lit.py -v . --filter pre-realized-selected-body-artifact-widening-product-reduce-dequantize-f32` from `build/test`
* `cmake --build build --target tianchenrv-construction-protocol-common-test`
* `./build/bin/tianchenrv-construction-protocol-common-test`
* `./build/bin/tianchenrv-target-artifact-export-test`

`ssh rvv` evidence is not claimed in this task. The previous archived task already closed executable evidence for the fully realized body; this task claims route-supported selected-body realization and generated artifact validation only.

## Definition of Done

* PRD truthfully matches the implemented bounded owner.
* Focused tests pass, and any failures encountered during implementation are fixed or explicitly scoped as continuation.
* Trellis task context/status is updated.
* Task is finished/archived if the bounded owner is complete.
* One coherent commit is created unless the task remains unfinished.

## Out of Scope

* High-level Linalg/Vector/StableHLO frontend work.
* q8/q4/llama benchmark routes.
* Zero-point/clamp expansion.
* Multi-output tile expansion.
* New dtype/LMUL clone batches.
* Compatibility wrappers preserving old i32m1 authority.
* One-intrinsic wrapper dialects.
* Broad smoke matrices or dashboards.
* Report-only, guardrail-only, or repeated standalone evidence as the primary deliverable.

## Technical Notes

Read first:

* `.trellis/spec/index.md`
* `.trellis/spec/extension-plugins/rvv-plugin.md`
* `.trellis/spec/lowering-runtime/emitc-route.md`
* `.trellis/tasks/archive/2026-06/06-05-06-05-stage2-rvv-contraction-dequant-executable-abi-closure/`
* `test/Target/RVV/explicit-selected-body-artifact-widening-product-reduce-dequantize-f32.mlir`
* `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`
* `lib/Dialect/RVV/IR/RVVDialect.cpp`
* `lib/Plugin/RVV/Construction/RVVConstructionProtocol.cpp`
* `test/Plugin/ConstructionProtocolCommonTest.cpp`
* `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`
* `lib/Plugin/RVV/EmitC/RVVEmitCContractionRouteFamilyPlanOwners.cpp`
* `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
* `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`

Inspection findings for this round:

* `RVVSelectedBodyRealization.cpp` already dispatches selected/pre-realized bodies through a plugin-local owner registry before route construction.
* `RVVContractionSelectedBodyRealizationOwner.cpp` already realizes widening MAcc and widening dot-reduction pre-realized bodies, but does not yet cover the composed i8 product -> i16 product -> i32 reduction -> f32 dequant chain.
* `RVVOps.td` has realized `widening_product`, `standalone_reduce`, and `dequantize` ops but no bounded typed pre-realized body op for the composed product-reduction-dequant route.
* `RVVConfigContract.cpp` already defines the runtime ABI order and C types for `widening_product_reduce_dequantize_f32`: `lhs,rhs,acc,scale,out,n`.
* The existing explicit fixture proves the realized chain reaches route planning and target artifact validation; the new test should prove the pre-realized body materializes into that same realized chain.
