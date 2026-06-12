# Stage2 RVV Gearbox multi-region selected-body realization

## Goal

Materialize the low-precision Gearbox product-reduction-dequantization resource decision into explicit provider-verifiable `tcrv_rvv` selected-body structure. The bounded owner is the RVV plugin-local selected-body realization and provider route planning path for widening product -> standalone reduction -> f32 dequantization.

## What I already know

* No active `.trellis/.current-task` existed at session start; this task was created from the Hermes Direction Brief.
* The repository is clean at `c38cc0ac rvv: require resource-realized low precision body facts`.
* The previous archived task added required pass-produced and realization-produced low-precision resource facts. It writes `realized_vsetvl_region_count = 2` on the realized `with_vl` and the provider checks those facts.
* Current production realization still materializes one `tcrv_rvv.setvl` plus one `tcrv_rvv.with_vl` for `typed_widening_product_reduce_dequantize_pre_realized_body`; the declared resource fact says two vsetvl regions, but actual realized structure does not yet reflect that.
* `collectRVVSelectedBodyRouteSlice` currently requires exactly one `tcrv_rvv.setvl` and one `tcrv_rvv.with_vl` across the selected variant. Because `with_vl` has no region results, this round should not split the dataflow across multiple independent `with_vl` bodies unless it also rewires the SSA/runtime boundary.
* The bounded first submodule is a plugin-owned structural marker path: realized body keeps the existing single selected `with_vl` dataflow but adds two ordered `tcrv_rvv.vsetvl_region_marker` ops for the resource-selected load/product/reduce and dequant/store placement phases, and the provider checks those markers against RVV-owned resource facts before route support.
* Relevant specs require RVV Stage 2 selected-body realization to consume performance/resource config into real `tcrv_rvv` structure, not route ids, artifact names, diagnostics, mirrors, or common EmitC inference.

## Requirements

* Keep implementation in C++/MLIR/TableGen/lit; Python may only be used for tooling.
* Implement the behavior inside RVV plugin-owned selected-body realization/provider code, not common EmitC semantics.
* For the low-precision product-reduction-dequantize route, realized body structure must reflect the selected resource facts for vsetvl region placement.
* Provider route planning must validate actual realized region structure against the pass/realization resource facts before route support.
* Stale or inconsistent resource facts versus actual realized region count must fail closed with targeted diagnostics.
* The route/provider must continue deriving dtype/config/operation from typed body/config/runtime facts and realized structure, not helper names, route ids, artifact names, ABI strings, or metadata mirrors.

## Acceptance Criteria

* [x] Production C++ changes in the RVV Gearbox selected-body realization and provider route planning path.
* [x] Positive lit coverage shows pre-realized product-reduction-dequantization materializes explicit two-phase `tcrv_rvv.vsetvl_region_marker` placement structure under the realized selected RVV `with_vl`.
* [x] Positive route/emission evidence shows the provider accepts the realized placement structure when facts and structure agree.
* [x] Negative lit coverage shows stale or inconsistent realized resource facts versus actual realized region structure fail closed before route support/export acceptance.
* [x] Focused module checks pass for changed tests and RVV plugin test where relevant.
* [x] Bounded old-authority scan over touched production/test diff shows no new positive legacy `i32m1` route authority, descriptor-driven compute, direct-C/source-export authority, or source-front-door positive route.
* [x] `git diff --check`, `git diff --cached --check`, and final clean `git status --short` are reported.

## Out of Scope

* No global autotuning database, dashboard, readiness state machine, or report-only work.
* No new high-level frontend, per-Linalg route authority, or source-front-door positive route.
* No dtype/LMUL clone batch and no unrelated MAcc/mask/reduction coverage expansion.
* No common EmitC invention of RVV semantics.
* No runtime/correctness/performance claim unless backed by `ssh rvv` evidence.

## Technical Notes

* Relevant specs read: `.trellis/spec/index.md`, `.trellis/spec/extension-plugins/rvv-plugin.md`, `.trellis/spec/lowering-runtime/emitc-route.md`, `.trellis/spec/variant-pipeline/index.md`, `.trellis/spec/testing/index.md`, `.trellis/spec/implementation-stack/index.md`, `.trellis/spec/guides/plugin-locality-review-guide.md`, and `.trellis/spec/guides/compute-boundary-review-guide.md`.
* Prior task read: `.trellis/tasks/archive/2026-06/06-08-stage2-rvv-gearbox-resource-aware-realization/prd.md`.
* Main implementation targets: `include/TianChenRV/Plugin/RVV/RVVGearboxSchedule.h`, `lib/Plugin/RVV/RVVContractionSelectedBodyRealizationOwner.cpp`, `lib/Plugin/RVV/EmitC/RVVEmitCContractionRouteFamilyPlanOwners.cpp`, and possibly `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`.
* Focused tests: `test/Target/RVV/pre-realized-selected-body-artifact-widening-product-reduce-dequantize-f32.mlir` and nearby Gearbox/resource tests.

## Completion Notes

* Added `tcrv_rvv.vsetvl_region_marker` as RVV plugin-local structural schedule evidence. It consumes the active `!tcrv_rvv.vl` token, is verified as nested under `tcrv_rvv.with_vl`, and is deliberately not an EmitC role op or route authority.
* The product-reduction-dequantization realization now emits two markers: `load-product-reduce` and `dequant-store`, both tied to the selected resource decision and two-region resource fact.
* Route slice collection records the markers, excludes them from construction role-order validation, and accepts them only for the bounded low-precision product-reduction-dequantization selected-body path.
* The contraction provider now checks marker count, order, phase, region index/count, resource decision, and bound `with_vl` token against RVV-owned resource facts before route support.
* The explicit selected-body product-reduction-dequantization artifact fixture was updated to represent an already-realized body with resource facts and placement markers.
* Continuation point: a future deeper realization may split actual dataflow across multiple `with_vl`/stripe regions only after defining the SSA/result and runtime boundary for cross-region values.
