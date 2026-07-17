# Stage2 RVV runtime-scalar-cmp masked segment2 store executable artifact ABI boundary

## Goal

Make the existing RVV runtime-scalar-cmp masked segment2 store selected-body route family truthful at the executable artifact boundary. The selected or pre-realized typed `tcrv_rvv` body must carry runtime scalar comparison operands, computed predicate/mask facts, masked active/inactive lane policy, two payload source roles, output memory role/order, dtype/SEW/LMUL/config/policy, and runtime AVL/VL into an RVV plugin-owned route, common EmitC materialization, target artifact export, generated bundle ABI, and `ssh rvv` correctness evidence. If any required executable-boundary fact is missing or stale, the production path must fail closed with a focused diagnostic.

## What I already know

* The previous completed adjacent task hardened runtime-scalar-cmp masked segment2 load artifact ABI behavior in commit `5eb95000`.
* The current task is the store-side companion: two vector payload roles are stored to an output memory role/order under a runtime-scalar compare mask.
* The work must stay inside the compiler stack: C++ / MLIR / LLVM / TableGen / CMake / lit / FileCheck, with Python only for generated-bundle tooling and probes.
* Route authority must remain structural: typed `tcrv_rvv` body/config/runtime facts consumed by the RVV plugin, then common EmitC materialization. Route ids, metadata, helper names, test names, artifact names, and common EmitC logic must not decide RVV semantics.
* If the store production route is already complete, this task may close with focused fixture/evidence hardening that proves the store-specific ABI boundary. It must not drift into another route family.

## Requirements

* Inspect the current runtime-scalar-cmp masked segment2 store route path from selected and pre-realized fixtures through RVV selected-body realization, route planning, route provider, target artifact validation, generated bundle export, and runtime execution.
* Keep the owner bounded to the existing runtime-scalar-cmp masked segment2 store family.
* Ensure positive store-side executable evidence covers materialized selected boundary, emission plan, target artifact export, generated bundle compile, and `ssh rvv` runtime correctness when executable behavior is claimed.
* Ensure focused fail-closed coverage for at least one stale or missing store executable-boundary fact, prioritizing facts such as runtime scalar binding, compare operand role, computed mask binding, inactive-lane policy, payload source role, output memory role/order, header/prototype binding, route-family validation contract, generated C type, ABI value mapping, or runtime AVL/VL.
* Preserve common EmitC/export neutrality: common code may materialize plugin-provided route facts but must not infer RVV store semantics.
* Avoid broad segment2 matrix expansion, dtype/LMUL clone batches, computed-mask-only route expansion, high-level frontend work, performance tuning database work, source-front-door positive route work, and unrelated route rewrites.

## Acceptance Criteria

* [x] A selected-body runtime-scalar-cmp masked segment2 store fixture is accepted only when store-specific ABI/runtime facts line up.
* [x] A pre-realized selected-body runtime-scalar-cmp masked segment2 store fixture is accepted only when store-specific ABI/runtime facts line up.
* [x] At least one stale/missing store executable-boundary fact fails closed with a focused diagnostic in the relevant provider, target artifact validation, lit, generated-bundle, or C++ test boundary.
* [x] Generated bundle dry-run or equivalent focused `tcrv-opt` / `tcrv-translate` evidence proves the materialized artifact ABI for the explicit and pre-realized store paths.
* [x] Non-dry-run generated bundle execution on `ssh rvv` passes for the store path if runtime correctness is claimed.
* [x] `build/bin/tianchenrv-rvv-extension-plugin-test` and `build/bin/tianchenrv-target-artifact-export-test` pass after changes.
* [x] `git diff --check`, `git diff --cached --check`, and a bounded old-authority scan over touched files / added diff lines are clean.
* [x] Trellis task status and workspace journal accurately record the result.

## Completion Notes

* Inspection found the explicit and pre-realized runtime-scalar-cmp masked
  segment2 store production paths are already wired through RVV selected-body
  realization/provider facts, common EmitC materialization, target artifact
  validation, generated bundle export, and executable runtime evidence.
* This round hardened the store-specific artifact ABI regression surface in
  focused fixtures: both explicit and pre-realized target artifact export now
  prove stale `tcrv_rvv.route_operand_binding_operands` mirrors for the `src1`
  field-payload ABI role fail closed before target artifact acceptance.
* Positive fixture and generated-bundle evidence covers provider-derived
  `lhs,rhs_scalar,src0,src1,dst,n` ABI order and binding summary, runtime
  scalar splat compare producer, two payload loads, masked segment2 store,
  interleaved destination memory, and runtime AVL/VL.
* Non-dry-run `ssh rvv` generated-bundle execution passed for explicit and
  pre-realized selected-body inputs with counts `0,1,16,17,257`, runtime scalar
  values `-37,91`, and patterns `0,1`.
* No production runtime C++ code changed. The existing provider/target
  validators already reject the stale field-payload ABI summary; this round
  records that executable-boundary proof and refreshed runtime evidence.
* Spec update: `.trellis/spec/lowering-runtime/emitc-route.md` now states the
  runtime-scalar segment2 store binding summary as `lhs`, `rhs_scalar`, `src0`,
  `src1`, `dst`, and `n`, instead of inheriting the segment2-load
  `src/out0/out1` wording.

## Out of Scope

* No broad segment2 load/store/update matrix.
* No dtype or LMUL clone batch.
* No computed-mask-only route expansion beyond this runtime-scalar-cmp masked segment2 store boundary.
* No segment2 interleave/deinterleave work except as reference.
* No MAcc, reduction, dequant, compare/select, widening conversion, or unrelated mask route rework.
* No high-level Linalg / Vector / StableHLO frontend.
* No per-Linalg route authority.
* No dashboard, global autotuning database, or report-only endpoint.
* No source-front-door positive route.
* No common EmitC invention of RVV semantics.

## Technical Notes

Read first:

* `.trellis/spec/index.md`
* `.trellis/spec/extension-plugins/rvv-plugin.md`
* `.trellis/spec/lowering-runtime/emitc-route.md`
* `.trellis/tasks/archive/2026-06/06-08-06-08-stage2-rvv-runtime-scalar-cmp-masked-segment2-load-executable-abi/`
* `include/TianChenRV/Plugin/RVV/RVVEmitCSegment2RouteFamilyPlanOwners.h`
* `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`
* `lib/Plugin/RVV/RVVSegment2SelectedBodyRealizationOwner.cpp`
* `lib/Plugin/RVV/EmitC/RVVEmitCSegment2RouteFamilyPlanOwners.cpp`
* `lib/Plugin/RVV/EmitC/RVVEmitCCompareSelectStatementPlanOwners.cpp`
* `lib/Plugin/RVV/EmitC/RVVEmitCStatementPlanOwners.cpp`
* `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
* `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`
* `scripts/rvv_generated_bundle_abi_e2e.py`
* `test/Scripts/rvv-generated-bundle-abi-e2e-runtime-scalar-cmp-masked-segment2-store-dry-run.test`
* `test/Scripts/rvv-generated-bundle-abi-e2e-pre-realized-runtime-scalar-cmp-masked-segment2-store-dry-run.test`
* `test/Target/RVV/explicit-selected-body-artifact-runtime-scalar-cmp-masked-segment2-store.mlir`
* `test/Target/RVV/pre-realized-selected-body-artifact-runtime-scalar-cmp-masked-segment2-store.mlir`

## Definition of Done

* The bounded production/evidence seam is complete or the no-source-change justification is exact and evidence-backed.
* Focused checks pass or any environment limitation is documented with the exact command and failure.
* The task is finished/archived when complete.
* One coherent commit is created if the task is complete.
