# Stage2 RVV contraction aggregate owner-consumed statement-plan fact boundary

## Goal

Move the remaining direct-contraction aggregate provider-plan consumption
boundary from central `RVVEmitCRouteProvider.cpp` into the RVV statement-plan
owner selection path. The central provider should gather neutral route facts,
ask the owner-selection module for one owner-built statement plan, and then
assemble `TCRVEmitCLowerableRoute`. It must not directly construct or consume
the direct-contraction provider plan for widening MAcc, product/reduction,
dot-reduction, computed-mask, strided-input, or dequant/dequant-clamp
contraction subfamilies.

## What I already know

* No `.trellis/.current-task` existed at session start; this task was created
  from the Hermes Direction Brief.
* The repository started clean on `main`; recent history includes
  `8c5aa3f9 rvv: consume contraction provider facts through owner`.
* The previous archived task
  `.trellis/tasks/archive/2026-06/06-08-stage2-rvv-contraction-family-statement-plan-provider-consumption-boundary/`
  moved direct-contraction provider-fact validation into
  `getRVVSelectedBodyRouteStatementPlanOwnerSelection(...)`.
* Current source inspection shows the central provider no longer calls
  `verifyRVVSelectedBodyDirectContractionRouteProviderFacts(...)` after owner
  selection. That verification is already owner-selection local.
* Current source inspection also shows the central provider still calls
  `getRVVSelectedBodyDirectContractionRouteProviderPlan(...)` before owner
  selection and passes the resulting aggregate into
  `getRVVSelectedBodyRouteStatementPlanOwnerSelection(...)`.
* `getRVVSelectedBodyDirectContractionRouteProviderPlan(...)` is the aggregate
  consumer for typed family facts, route-control facts, math operand-binding
  facts, ABI operands, leaf materialization facts, and low-precision resource
  facts across direct contraction subfamilies.
* The relevant specs require selected RVV bodies and RVV owner/provider code,
  not route ids, metadata, Common EmitC, or central glue, to own dtype/config,
  runtime/mask/scalar/accumulator roles, route-family validation, and
  fail-closed diagnostics.

## Requirements

* Keep implementation in C++ / MLIR / LLVM / TableGen / CMake / lit /
  FileCheck. Python is allowed only for support tooling and focused evidence
  scripts.
* Implement one production workflow submodule: direct-contraction aggregate
  provider-plan construction and validation must be owner-consumed before route
  construction.
* `RVVEmitCRouteProvider.cpp` must not call
  `getRVVSelectedBodyDirectContractionRouteProviderPlan(...)` or
  `verifyRVVSelectedBodyDirectContractionRouteProviderFacts(...)` directly.
* `getRVVSelectedBodyRouteStatementPlanOwnerSelection(...)` or its
  direct-contraction owner subpath must derive the direct-contraction provider
  plan from the same selected route analysis, materialization facts, and math
  operand-binding facts that the central provider already computed.
* The direct-contraction owner path must validate typed family facts, provider
  facts, route-control facts, math operand-binding facts, ABI roles,
  dtype/config/policy facts, materialization leaves, low-precision resource
  facts, and owner statement leaves before returning an owner selection.
* Central provider code may still gather generic materialization and operand
  binding facts, run unrelated family validators, attach headers/type/ABI
  mappings, and attach the owner-built statement plan. It must remain neutral
  after owner selection.
* Tests must cover a positive owner-consumed direct-contraction aggregate and a
  stale/fail-closed direct-contraction aggregate through owner selection.
* Do not add route coverage, broaden dtype/LMUL matrices, or move RVV semantics
  into Common EmitC.

## Acceptance Criteria

* [ ] `RVVEmitCRouteProvider.cpp` has no direct call to
  `getRVVSelectedBodyDirectContractionRouteProviderPlan(...)`.
* [ ] `RVVEmitCRouteProvider.cpp` has no direct call to
  `verifyRVVSelectedBodyDirectContractionRouteProviderFacts(...)`.
* [ ] The direct-contraction provider plan is built inside the statement-plan
  owner selection path from `analysis`, `materializationFacts`, and
  `mathOperandBindingFacts`.
* [ ] Owner selection remains fail-closed for stale or missing
  direct-contraction provider facts before `TCRVEmitCLowerableRoute`
  attachment.
* [ ] Focused regression proves a representative direct-contraction subfamily
  still selects an owner-built statement plan.
* [ ] Focused regression proves stale direct-contraction aggregate facts are
  rejected through owner selection, not central route glue.
* [ ] Representative generated-bundle dry-run positive and direct-pre-realized
  fail-closed script tests for touched contraction subfamilies are replayed.
* [ ] `build/bin/tianchenrv-rvv-extension-plugin-test` passes.
* [ ] `build/bin/tianchenrv-target-artifact-export-test` passes.
* [ ] Bounded residue scan over directly related owner/provider files confirms
  central `RVVEmitCRouteProvider.cpp` does not directly consume or validate
  direct-contraction provider facts after the migration.
* [ ] Bounded old-authority scan over touched files and added diff lines shows
  no new positive legacy `i32m1` route authority, source-front-door route
  authority, descriptor-driven compute, direct-C/source export authority, or
  mirror-only acceptance.
* [ ] `git diff --check`, `git diff --cached --check`, and final
  `git status --short` are reported.

## Out of Scope

* No new MAcc, dot-reduction, product-reduction, dequant, or contraction
  operation coverage.
* No dtype/SEW/LMUL/mask/runtime matrix expansion.
* No source-front-door positive route, high-level Linalg/Vector/StableHLO
  frontend, per-Linalg route authority, dashboard, or report-only work.
* No Common EmitC invention of RVV semantics.
* No compatibility wrapper that preserves central direct-contraction provider
  plan consumption.
* No unrelated memory, segment2, compare/select, conversion, scalar fallback,
  runtime-offload, or Gearbox work.
* No `ssh rvv` runtime/correctness/performance claim unless this round changes
  runtime behavior.

## Technical Notes

* Specs read:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`, and
  `.trellis/spec/testing/index.md`.
* Previous task reference:
  `.trellis/tasks/archive/2026-06/06-08-stage2-rvv-contraction-family-statement-plan-provider-consumption-boundary/`.
* Primary production files:
  `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCStatementPlanOwners.cpp`,
  `include/TianChenRV/Plugin/RVV/RVVEmitCStatementPlanOwners.h`,
  `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`, and
  `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`.
* Focused tests:
  `test/Plugin/RVVExtensionPluginTest.cpp` and directly relevant
  generated-bundle dry-run/fail-closed script tests for contraction families.
