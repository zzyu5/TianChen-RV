# Stage2 RVV computed-masked segment2 update executable artifact ABI boundary

## Goal

Make the existing RVV computed-masked segment2 update selected-body route family truthful at the executable artifact ABI boundary. The selected or pre-realized `tcrv_rvv` body, computed mask/predicate facts, active/inactive lane policy, lane0/lane1 source and destination roles, update/read-write binding facts, dtype/SEW/LMUL/config/policy, runtime AVL/VL, provider route facts, EmitC materialization, target artifact export, generated bundle ABI, and `ssh rvv` correctness evidence must agree without route-id, metadata, helper-name, test-name, descriptor, or common EmitC semantic authority.

## What I already know

* Hermes direction narrows this round to computed-masked segment2 update, after runtime-scalar-cmp masked segment2 store/load ABI closeouts.
* The owner is the executable artifact boundary for an existing selected-body route family, not broad segment2 coverage or a new frontend.
* Prior recent work often found the production path already wired and closed focused artifact ABI regression-proof gaps with stale binding negative tests plus refreshed generated-bundle and `ssh rvv` evidence.
* Runtime/correctness claims require real `ssh rvv` evidence.
* Common EmitC/export must remain neutral; RVV plugin/provider/target validation own RVV-specific facts and fail-closed diagnostics.

## Requirements

* Inspect the current computed-masked segment2 update explicit, pre-realized, and direct pre-realized fixtures and generated-bundle script routes.
* Confirm whether selected-body realization, RVV route provider facts, common EmitC materialization, target artifact export, and generated bundle execution are already production-wired.
* If the production seam is dry-run-only, stale, under-validated, or semantically conflated, fix the bounded production code path.
* If the production seam is already complete, close the exact executable evidence blocker with focused positive and fail-closed evidence instead of changing unrelated modules.
* Add or update focused fail-closed coverage for at least one missing or stale executable-boundary fact, preferably one specific to update/read-write or paired lane0/lane1 source/destination semantics.
* Refresh positive generated-bundle dry-run and `ssh rvv` correctness evidence for computed-masked segment2 update when executable behavior is claimed.
* Keep all new or changed behavior inside the computed-masked segment2 update artifact ABI seam and bounded segment2 reference files.

## Acceptance Criteria

* [x] Explicit selected-body computed-masked segment2 update materializes through emission plan, header/prototype, and target artifact export with lane0/lane1 source and destination/update ABI facts visible as provider-built mirrors.
* [x] Pre-realized selected-body computed-masked segment2 update materializes through the same boundary without stale pre-realized metadata replacing provider facts.
* [x] Generated bundle dry-run tests cover explicit, pre-realized, and direct pre-realized computed-masked segment2 update paths.
* [x] Real `ssh rvv` generated-bundle correctness evidence is refreshed for executable computed-masked segment2 update.
* [x] At least one focused negative test proves a stale or missing executable-boundary fact fails closed before target artifact acceptance.
* [x] `build/bin/tianchenrv-rvv-extension-plugin-test` and `build/bin/tianchenrv-target-artifact-export-test` pass.
* [x] Relevant lit/FileCheck or script tests pass. Local `FileCheck`/`llvm-lit` binaries are not installed, so the new RUN behavior was validated with equivalent `tcrv-opt` / `tcrv-translate` fail-closed commands and generated-bundle script checks.
* [x] `git diff --check`, `git diff --cached --check`, and a bounded old-authority/source-front-door scan over touched files or added diff lines are clean.

## Completion Notes

* Inspection found the explicit and pre-realized computed-masked segment2 update production paths are already wired through typed `tcrv_rvv` selected bodies, RVV plugin-local realization, provider-owned segment2/computed-mask/update route facts, common EmitC materialization, target artifact validation, generated bundle export, and executable runtime evidence.
* No production C++/Python compiler path changed. The existing target validator already rebuilds the provider route and compares `tcrv_rvv.route_operand_binding_operands` exactly against provider-built facts; this round closes the missing update-specific regression proof.
* Added explicit selected-body target artifact negative coverage proving stale `src1` route operand binding metadata cannot replace the provider summary for `src1=segment-field1-input-buffer` and its update-specific `add-rhs` use token.
* Added matching pre-realized selected-body target artifact negative coverage proving stale pre-realized metadata cannot replace the provider-built `src1`/`add-rhs` binding summary after selected-body materialization.
* Positive artifact/header export evidence covers the provider-derived ABI order `cmp_lhs,cmp_rhs,src0,src1,dst,n`, route operand binding summary, `rvv-route-operand-binding:cmseg2_update_unit_load.v1`, update arithmetic `add`, segment2 tuple creation/store intrinsics, computed mask facts, inactive-lane preservation policy, and generated C prototype.
* Generated-bundle dry-run evidence passed for explicit and pre-realized selected-body inputs with runtime counts `0,1,7,16,23,257`. The direct pre-realized route-entry path remains fail-closed with the retired shortcut diagnostic.
* Non-dry-run `ssh rvv` generated-bundle correctness passed for explicit and pre-realized selected-body inputs with counts `0,1,7,16,23,257` and patterns `0,1`, including active update lanes, inactive interleaved destination preservation, source preservation, and tail preservation.
* Spec update judgment: no `.trellis/spec/` change was needed. The provider operand binding summary contract and segment2 target export consumer contract already require exact provider-summary mirror validation and fail-closed stale field/update facts; this task added concrete regression evidence for that existing contract.

## Out of Scope

* Broad segment2 matrix expansion.
* Dtype or LMUL clone batches.
* Runtime-scalar-cmp segment2 update expansion unless it is the same already-complete seam.
* Segment2 load/store rework except as bounded ABI references.
* MAcc, reduction, dequant, clamp, compare/select, widening conversion, or unrelated mask route rework.
* High-level Linalg/Vector/StableHLO frontend work.
* Per-Linalg route authority, descriptor-driven computation, source-front-door positive routes, dashboards, tuning databases, or report-only work.
* Common EmitC invention of RVV semantics.

## Technical Notes

* Read first: `.trellis/spec/index.md`, `.trellis/spec/extension-plugins/rvv-plugin.md`, `.trellis/spec/lowering-runtime/emitc-route.md`.
* Bounded prior reference: `.trellis/tasks/archive/2026-06/06-08-stage2-rvv-runtime-scalar-cmp-masked-segment2-store-executable-abi/`.
* Likely production code: `lib/Plugin/RVV/`, `lib/Plugin/RVV/EmitC/`, `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`, `lib/Target/RVV/`.
* Likely tests/scripts: `scripts/rvv_generated_bundle_abi_e2e.py`, `test/Scripts/rvv-generated-bundle-abi-e2e-explicit-computed-masked-segment2-update-dry-run.test`, `test/Scripts/rvv-generated-bundle-abi-e2e-pre-realized-computed-masked-segment2-update-dry-run.test`, `test/Scripts/rvv-generated-bundle-abi-e2e-direct-pre-realized-computed-masked-segment2-update-dry-run.test`, `test/Target/RVV/explicit-selected-body-artifact-computed-masked-segment2-update.mlir`, `test/Target/RVV/pre-realized-selected-body-artifact-computed-masked-segment2-update.mlir`.
