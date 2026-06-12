# Stage2 RVV Gearbox cross-region SSA/runtime boundary

## Goal

Define and consume a bounded RVV plugin-owned structural boundary for the
Gearbox widening product -> standalone reduction -> f32 dequantization selected
body. The boundary must make the product/reduction/dequant handoff and runtime
AVL/VL facts visible in `tcrv_rvv` structure before route support, so the
provider no longer relies only on vsetvl placement markers as the bridge toward
multi-region selected-body realization.

## What I already know

* No active `.trellis/.current-task` existed at session start; this task was
  created from the Hermes Direction Brief.
* Repository started clean at `3e354570 rvv: realize gearbox vsetvl placement
  structure`.
* The previous archived task added `tcrv_rvv.vsetvl_region_marker` and provider
  validation for marker count/order/phase/index/count/resource decisions.
* The previous completion note explicitly left true multiple `with_vl`/stripe
  regions blocked on a cross-region SSA/result/runtime boundary.
* Current explicit/pre-realized Gearbox fixtures still contain one `setvl` and
  one `with_vl` with two ordered placement markers; product, reduction,
  dequantize, and store remain in one region.
* Relevant specs require RVV Stage 2 resource/schedule decisions to be consumed
  into plugin-owned `tcrv_rvv` structure before route construction, while common
  EmitC remains neutral.

## Requirements

* Keep implementation in C++/MLIR/TableGen/lit; Python remains tooling only.
* Implement the boundary in RVV dialect/provider/realization code, not in common
  EmitC and not as artifact/report metadata.
* The Gearbox product-reduction-dequantize realized body must carry a coherent
  structural boundary that names the value handoff from product/reduction into
  dequantization and binds it to runtime AVL/VL facts.
* Provider route planning must validate that boundary before route support:
  product/reduction/dequant phases, handoff value kinds, bound VL token,
  runtime AVL source/order, vsetvl region count, and resource decision must
  agree with RVV-owned facts.
* Stale, missing, or marker-only boundary facts must fail closed with targeted
  diagnostics.
* Preserve the existing marker-backed path as transitional evidence while the
  new structural handoff is added; do not claim runtime correctness or
  performance without `ssh rvv`.

## Acceptance Criteria

* [x] Production source changes define a structural Gearbox cross-region
  handoff/runtime boundary in `tcrv_rvv` and/or provider-owned RVV route
  planning structures.
* [x] The pre-realized Gearbox selected-body realizer emits the boundary for
  widening product -> standalone reduction -> f32 dequantization.
* [x] The explicit selected-body fixture represents the same already-realized
  boundary without running the realizer as the source of authority.
* [x] Provider route planning consumes and validates the boundary before route
  support and keeps common EmitC semantic-neutral.
* [x] Positive lit/FileCheck evidence covers the realized boundary and route
  metadata/export path for the bounded Gearbox fixture.
* [x] Negative evidence proves stale or missing handoff/runtime boundary facts
  fail closed before route support/export acceptance.
* [x] Focused builds/tests pass: `tcrv-opt`, `tcrv-translate`,
  `tianchenrv-rvv-extension-plugin-test`, `tianchenrv-rvv-dialect-test`, and
  focused lit tests for the explicit/pre-realized widening-product-reduce-
  dequantize-f32 fixtures.
* [x] Bounded old-authority scan over touched production/test diff lines shows
  no new positive legacy `i32m1` route authority, descriptor-driven compute,
  direct-C/source-front-door authority, or artifact/status authority.
* [x] `git diff --check`, `git diff --cached --check`, and final clean
  `git status --short` are reported.

## Out of Scope

* No broad Gearbox matrix, dtype/LMUL clone batch, or new math family.
* No high-level Linalg/Vector/StableHLO frontend or source-front-door positive
  route.
* No common EmitC invention of RVV dtype, schedule, route semantics, or runtime
  ABI semantics.
* No artifact dashboard/report-only completion.
* No runtime correctness or performance claim without `ssh rvv` evidence.
* Full multiple `with_vl`/stripe regions may remain a continuation point if
  this round lands the coherent structural handoff contract and one converted
  fixture.

## Technical Notes

* Required specs read: `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  `.trellis/spec/variant-pipeline/index.md`,
  `.trellis/spec/variant-pipeline/generation-selection-tuning.md`,
  `.trellis/spec/testing/index.md`,
  `.trellis/spec/implementation-stack/index.md`,
  `.trellis/spec/guides/index.md`,
  `.trellis/spec/guides/plugin-locality-review-guide.md`, and
  `.trellis/spec/guides/compute-boundary-review-guide.md`.
* Previous task read:
  `.trellis/tasks/archive/2026-06/06-08-stage2-rvv-gearbox-multi-region-selected-body-realization/prd.md`.
* Main code targets from the brief:
  `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`,
  `lib/Dialect/RVV/IR/RVVDialect.cpp`,
  `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`,
  `lib/Plugin/RVV/RVVContractionSelectedBodyRealizationOwner.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`, and
  `lib/Plugin/RVV/EmitC/RVVEmitCContractionRouteFamilyPlanOwners.cpp`.
* Focused tests from the brief:
  `test/Target/RVV/pre-realized-selected-body-artifact-widening-product-reduce-dequantize-f32.mlir`
  and
  `test/Target/RVV/explicit-selected-body-artifact-widening-product-reduce-dequantize-f32.mlir`.
