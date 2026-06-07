# Stage2 RVV Gearbox multi-region with_vl stripe-ready dataflow boundary

## Goal

Move the bounded Gearbox widening product -> standalone reduction -> f32
dequantization route beyond the current marker-backed single-`with_vl` handoff
by modeling and consuming a stripe-ready two-scope boundary for the
product/reduction producer region and the dequant/store consumer region. The
route provider must validate the handoff value, active VL token, runtime AVL
SSA value, phase order, region count, resource decision, and dequant consumer
facts before route support; common EmitC and artifact metadata remain mirrors.

## What I already know

* No `.trellis/.current-task` existed at session start; this task was created
  from the Hermes Direction Brief and set as current.
* Repository started clean at `143aacfd rvv: add gearbox cross-region handoff
  boundary`.
* The previous archived task added `tcrv_rvv.gearbox_cross_region_handoff`,
  verifier rules, selected-body realization, route collection, provider
  validation, and artifact/header validation for the four-stage
  product/reduce/handoff/dequant chain.
* Current production route collection still requires exactly one
  `tcrv_rvv.setvl` and one `tcrv_rvv.with_vl` for all bounded RVV routes.
* Current `GearboxCrossRegionHandoffOp` verification requires the
  source-producing product/reduction ops, both vsetvl region markers, the
  handoff, and the dequant consumer to be in the same `with_vl` body.
* Existing explicit/pre-realized Gearbox fixtures show two region markers but
  one enclosing `with_vl`; this is still stripe-ready evidence rather than a
  real split-region consumer.
* Full global multiple-`with_vl` support would affect unrelated RVV route
  families. This PRD limits the production change to the bounded Gearbox
  product/reduce/dequant route.

## Requirements

* Keep implementation in C++/MLIR/TableGen/lit/C++ tests. Python remains
  tooling only.
* Implement the boundary in RVV dialect verification, RVV selected-body
  realization or construction protocol, and RVV provider route planning.
* Do not move Gearbox dtype, schedule, runtime AVL/VL, phase, resource, or
  intrinsic semantics into common EmitC, target artifact metadata, route ids,
  status fields, descriptors, or source-front-door patterns.
* The bounded Gearbox product/reduction route must support a structurally
  modeled two-scope handoff, or fail closed when the handoff value, VL token,
  runtime AVL source, phase order, region count, resource decision, dequant
  consumer, or provider route facts are stale or inconsistent.
* Preserve existing non-Gearbox route one-`with_vl` invariants unless this
  task explicitly changes only the Gearbox collection path.
* Do not claim runtime correctness or performance evidence unless an
  executable bundle is actually run on `ssh rvv`.

## Acceptance Criteria

* [x] A Gearbox-specific two-scope or stripe-ready region model exists in
  production RVV dialect/provider structures, distinguishing product/reduction
  producer scope from dequant/store consumer scope.
* [x] RVV dialect verification accepts the bounded Gearbox split only when the
  source product/reduction chain, handoff, consumer dequant/store chain, VL,
  runtime AVL, phase order, region count, and resource decision agree.
* [x] Selected-body realization or construction protocol materializes the same
  two-scope/stripe-ready boundary for the pre-realized product-reduce-dequant
  fixture.
* [x] Route slice collection and provider validation consume the modeled
  producer/consumer boundary before route support.
* [x] Positive lit/FileCheck evidence shows the explicit and pre-realized
  widening-product-reduce-dequantize-f32 path carrying and exporting provider-
  validated two-scope facts.
* [x] Fail-closed evidence covers stale or missing boundary facts such as
  missing handoff, dequant consuming reduction directly, mismatched VL token,
  stale runtime AVL source, wrong phase order, wrong region count, wrong
  resource decision, or stale provider route facts.
* [x] Focused checks pass:
  `build/bin/tianchenrv-rvv-dialect-test`,
  `build/bin/tianchenrv-rvv-extension-plugin-test`,
  `build/bin/tianchenrv-construction-protocol-common-test`, and relevant lit
  filters for `widening-product-reduce-dequantize-f32`.
* [x] Bounded old-authority scan over touched production/test diff lines shows
  no new positive legacy `i32m1` route authority, descriptor-driven compute,
  source-front-door authority, or artifact/status authority.
* [x] `git diff --check`, `git diff --cached --check`, and clean final
  `git status --short` are reported.

## Completion Notes

* Implemented the bounded two-scope/stripe-ready Gearbox boundary with
  `gearbox-scope:product-reduction` and `gearbox-scope:dequant-store` facts
  owned by the RVV dialect/plugin path.
* The selected-body realization now materializes the same scope facts on the
  realized handoff and provider-consumed route facts.
* Route planning, provider validation, target artifact validation, and header
  bundle export consume the scope facts as mirrors of verified RVV plugin-owned
  structure.
* `test/Plugin/ConstructionProtocolCommonTest.cpp` was updated to keep its
  route-table self-check aligned with the existing runtime-scalar
  computed-mask indexed/segment construction routes that the full required
  binary covers.
* No `ssh rvv` runtime correctness or performance evidence was run or claimed.

## Out of Scope

* No marker-only evidence expansion as the main result.
* No broad Gearbox matrix, dtype/LMUL clone batch, new MAcc/reduction/broadcast
  route batch, or unrelated dequant clone work.
* No high-level Linalg/Vector/StableHLO frontend.
* No source-front-door positive route.
* No performance tuning/autotuning database.
* No common EmitC inference of RVV schedule, dtype, phase, or runtime ABI
  semantics.
* No runtime correctness or performance claim without `ssh rvv` evidence.
* No global multi-`with_vl` migration for unrelated RVV route families in this
  round.

## Technical Notes

* Required specs read: `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  `.trellis/spec/extension-plugins/index.md`,
  `.trellis/spec/lowering-runtime/index.md`,
  `.trellis/spec/testing/index.md`,
  `.trellis/spec/implementation-stack/index.md`,
  `.trellis/spec/variant-pipeline/index.md`, and
  `.trellis/spec/guides/index.md`.
* Previous task read:
  `.trellis/tasks/archive/2026-06/06-08-stage2-rvv-gearbox-cross-region-ssa-runtime-boundary/prd.md`.
* Main code targets from the brief:
  `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`,
  `lib/Dialect/RVV/IR/RVVDialect.cpp`,
  `lib/Plugin/RVV/Construction/RVVConstructionProtocol.cpp`,
  `lib/Plugin/RVV/RVVGearboxSchedules.cpp`,
  `lib/Plugin/RVV/RVVContractionSelectedBodyRealizationOwner.cpp`,
  `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`, and
  `lib/Plugin/RVV/EmitC/RVVEmitCContractionRouteFamilyPlanOwners.cpp`.
* Focused fixtures:
  `test/Target/RVV/explicit-selected-body-artifact-widening-product-reduce-dequantize-f32.mlir`,
  `test/Target/RVV/pre-realized-selected-body-artifact-widening-product-reduce-dequantize-f32.mlir`,
  and `test/Plugin/RVVExtensionPluginTest.cpp`.
