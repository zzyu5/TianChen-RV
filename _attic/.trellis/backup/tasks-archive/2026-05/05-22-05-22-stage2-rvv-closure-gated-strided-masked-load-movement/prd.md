# Stage2 RVV Closure-Gated Strided Masked Load Movement

## Goal

Implement one bounded Stage2 RVV memory-movement owner: route-supported strided masked load movement on the corrected typed `tcrv_rvv` surface. The route must be owned by the RVV dialect/construction/selected-body realization/planning/provider path and gated by `RouteOperandBindingPlan` closure, not by route ids, helper strings, artifact names, metadata mirrors, descriptor residue, source-front-door exports, or common EmitC semantic inference.

## What I Already Know

* HEAD `4bc518fc` completed closure-gated masked strided store movement and left strided masked loads outside that boundary.
* The accepted RVV-first path is `tcrv.exec` selected RVV variant -> typed low-level `tcrv_rvv` body -> RVV plugin-owned legality/selected-body realization/route provider -> `TCRVEmitCLowerableRoute` -> common EmitC materializer -> target artifact -> `ssh rvv` evidence when runtime/correctness is claimed.
* Current Stage2 work must stay on low-level RVV capability classes and must not become high-level Linalg/Vector/StableHLO frontend lowering, indexed/segmented/gather/scatter expansion, or legacy `RVVI32M1` / `rvv-i32m1` / finite `tcrv_rvv.i32_*` route authority.
* The previous strided masked store task is the closest implementation pattern, but it is a regression anchor only; this task must not redo stores.

## Requirements

* Inventory current strided load, masked load, computed-mask load, masked store, and strided masked store support only across the relevant production owners: RVV dialect ODS/C++ verifier, construction protocol, selected-body realization, route planning, route provider, target support bundle, generated bundle runner, and directly relevant tests.
* Add or repair one coherent strided masked load route family where selected RVV bodies structurally carry:
  * compare-produced mask or explicit mask operand;
  * source memory base/window role;
  * byte stride role, either constant or runtime parameter as supported by the bounded family;
  * old-vector passthrough role for inactive lanes;
  * loaded vector result role;
  * runtime `n` / AVL;
  * dtype/config facts including SEW, LMUL, tail policy, and mask/inactive-lane policy.
* RVV plugin legality/realization/provider code must derive route support, materialized operands, target headers, route mirrors, and target intrinsic leaf from typed body/config/capability/runtime facts plus `RouteOperandBindingPlan` closure.
* Unsupported indexed, segmented, gather, and scatter variants are out of family and must fail closed if they reach this route.
* Missing mask, missing stride, missing source memory window, missing old-vector passthrough where required, missing runtime role, dtype/config mismatch, stale or wrong plan id, mirror/header mismatch, materialized-use mismatch, old masked_move fallback, route-id/helper-string fallback, descriptor/direct-C/source-front-door authority, and common/export semantic inference must fail closed with targeted diagnostics.

## Acceptance Criteria

* A positive explicit strided masked load route is supported through typed body facts and `RouteOperandBindingPlan`, with materialized source base, mask, old-vector passthrough, stride, AVL/runtime length, and output binding visible in route mirrors or tests as mirrors only.
* A positive pre-realized computed-mask strided masked load route is supported if it fits the bounded round; otherwise the task leaves a truthful continuation point after completing one coherent route family.
* Structural lit/FileCheck or focused tests prove that materialized operands are closure-gated and not reconstructed from mirrors, route ids, helper strings, artifact names, manifests, descriptors, direct C, source-front-door residue, or common/export RVV semantic inference.
* Negative tests cover missing mask, missing stride, missing source memory window, missing passthrough when required, missing runtime role, wrong/stale binding plan, mirror/header mismatch, materialized-use mismatch, old masked_move fallback, unsupported memory form, route-id/helper-string fallback, descriptor/source/direct-C authority, and common/export inference.
* Generated bundle dry-runs cover value-distinguishing source buffers, non-unit stride variation, true/false mask lanes, inactive-lane passthrough preservation, untouched strided gaps, tail/sentinel preservation, and runtime `n` / AVL variation for representative new routes.
* Real `ssh rvv` evidence passes for representative new route(s) before runtime/correctness is claimed.
* Focused checks, `check-tianchenrv`, `git diff --check`, active-authority scan, and clean `git status --short` pass before final commit.

## Non-Goals

* Do not redo strided masked store, contiguous masked load, masked unit load/store, computed masked unit load/store, or strided stores except as regression anchors.
* Do not add indexed, segmented, gather, scatter, reductions, compare/select expansion, contractions, conversions, dtype/LMUL clone batches, high-level frontend lowering, dashboards, report-only work, helper-only cleanup, source-front-door positive routes, or future plugin work.
* Do not route through legacy `RVVI32M1`, `rvv-i32m1`, finite `tcrv_rvv.i32_*` authority, exact intrinsic spelling authority, descriptor-driven direct C, or common EmitC/export semantic inference.

## Technical Context

Relevant specs:

* `.trellis/spec/index.md`
* `.trellis/spec/extension-plugins/rvv-plugin.md`
* `.trellis/spec/lowering-runtime/emitc-route.md`
* `.trellis/spec/testing/mlir-testing-contract.md`

Closest archived tasks:

* `.trellis/tasks/archive/2026-05/05-22-05-22-stage2-rvv-closure-gated-strided-masked-store-movement/`
* `.trellis/tasks/archive/2026-05/05-22-stage2-rvv-masked-memory-movement-coverage/`
* closure-gate and operand-binding tasks under `.trellis/tasks/archive/2026-05/05-21-*`

Expected production owners:

* `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`
* `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`
* `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`
* `lib/Dialect/RVV/IR/RVVDialect.cpp`
* `lib/Plugin/RVV/Construction/RVVConstructionProtocol.cpp`
* `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp`
* `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
* `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
* `lib/Target/RVV/RVVTargetSupportBundle.cpp`
* `scripts/rvv_generated_bundle_abi_e2e.py`

## Definition Of Done

* Task PRD and context files reflect this bounded owner.
* Implementation rewires production route support, not just helpers or metadata.
* Tests and evidence are tied to the new production route behavior.
* Task is finished/archived according to Trellis convention.
* One coherent commit records the completed round, or the task remains open with an exact continuation point if incomplete.
