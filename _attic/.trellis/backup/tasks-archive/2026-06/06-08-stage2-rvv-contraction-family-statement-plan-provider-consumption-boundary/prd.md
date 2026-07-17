# Stage2 RVV contraction-family statement-plan provider consumption boundary

## Goal

Move the remaining direct-contraction statement/provider-fact consumption
boundary out of central `RVVEmitCRouteProvider` validation and into the
RVV-owned statement-plan owner selection path. The central provider may still
obtain provider-owned route facts, instantiate `TCRVEmitCLowerableRoute`, attach
headers/type/ABI/source provenance, and attach the owner-built statements, but
it must not directly validate contraction-family statement/provider facts after
owner selection.

This round targets the bounded residue called out by the Direction Brief:
`RVVEmitCRouteProvider.cpp` still invokes
`verifyRVVSelectedBodyDirectContractionRouteProviderFacts(...)` after
`getRVVSelectedBodyRouteStatementPlanOwnerSelection(...)`. That validation must
be consumed through the contraction/direct statement owner boundary so direct
contraction, widening MAcc, and dot-reduction statement ownership remains
RVV-owner-local.

## What I already know

* No `.trellis/.current-task` existed at session start; this task was created
  from the Hermes Direction Brief.
* The repository started clean on `main`; recent history includes
  `0bfd7624 rvv: route migrated statement plans through owner`.
* The archived task
  `.trellis/tasks/archive/2026-06/06-08-06-08-stage2-rvv-migrated-statement-plan-provider-consumption-boundary/`
  moved the already-migrated families through an aggregate boundary, but left
  widening MAcc, dot-reduction routes, computed-mask standalone reduction
  variants, residual runtime scalar splat-store, and future families outside
  that aggregate.
* Current source inspection shows `RVVEmitCRouteProvider.cpp` builds
  `directContractionProviderPlan`, calls
  `getRVVSelectedBodyRouteStatementPlanOwnerSelection(...)`, then directly
  calls `verifyRVVSelectedBodyDirectContractionRouteProviderFacts(...)`.
* `RVVEmitCStatementPlanOwners.cpp` already owns the direct-contraction
  statement sequencing through
  `getRVVSelectedBodyDirectContractionRouteStatementPlan(...)`, and its
  provider plan covers direct contraction subfamilies including widening MAcc,
  widening product, product-reduction chains, dot-reduction, computed-mask, and
  strided-input cases.
* `.trellis/spec/extension-plugins/rvv-plugin.md` requires migrated and
  provider-facing statement owners to be the authority for statement
  construction; the provider must consume owner-built plans rather than rebuild
  or sequence family semantics centrally.
* `.trellis/spec/lowering-runtime/emitc-route.md` defines
  `TCRVEmitCLowerableRoute` as a provider-built common container, not a place
  to reconstruct RVV semantics from route ids, ABI strings, mirrors, artifacts,
  or Common EmitC code.

## Requirements

* Keep implementation in C++ / MLIR / LLVM / TableGen / CMake / lit /
  FileCheck. Python is allowed only for support tooling and evidence probes.
* Implement one production workflow submodule: the direct-contraction
  statement/provider-fact owner-consumption boundary.
* `RVVEmitCRouteProvider.cpp` must no longer call
  `verifyRVVSelectedBodyDirectContractionRouteProviderFacts(...)` directly.
* The owner selection path must validate direct-contraction provider facts using
  the same selected route analysis, materialization facts, math operand-binding
  facts, provider plan, and selected statement plan before route attachment.
* The validation must remain fail-closed for stale/missing contraction provider
  facts, stale typed config facts, stale math operand-binding facts, missing ABI
  roles, missing direct-contraction owner statements, and non-contraction routes
  carrying contraction provider facts.
* Central provider code may still derive neutral materialization facts and call
  owner selection once, but after owner selection it should only run unrelated
  family validators and attach the returned owner-built statement plan.
* Existing direct-contraction statement sequencing remains owner-local; this
  task must not add route coverage, broaden dtype/LMUL matrices, or move
  semantics into Common EmitC.
* Tests should cover both the positive owner-consumed validation path and at
  least one stale/missing provider-fact fail-closed path through owner
  selection.

## Acceptance Criteria

* [ ] `RVVEmitCRouteProvider.cpp` has no direct call to
  `verifyRVVSelectedBodyDirectContractionRouteProviderFacts(...)`.
* [ ] `getRVVSelectedBodyRouteStatementPlanOwnerSelection(...)` or the
  direct-contraction owner path validates provider facts before returning a
  direct-contraction owner selection.
* [ ] Existing standalone tests for
  `verifyRVVSelectedBodyDirectContractionRouteProviderFacts(...)` remain valid
  or are deliberately replaced with owner-selection tests that exercise the
  same fail-closed conditions.
* [ ] Focused positive regression covers a representative direct-contraction
  route statement plan being selected and attached without central provider
  validation.
* [ ] Focused fail-closed regression covers stale or missing direct-contraction
  owner/provider facts being rejected before `TCRVEmitCLowerableRoute`
  attachment.
* [ ] `build/bin/tianchenrv-rvv-extension-plugin-test` passes.
* [ ] `build/bin/tianchenrv-target-artifact-export-test` passes.
* [ ] Focused lit or generated-bundle dry-run regressions for the affected
  contraction/MAcc/dot routes are replayed when the changed code affects their
  outputs.
* [ ] Bounded provider residue scan shows central provider no longer performs
  the migrated direct-contraction provider-fact validation/sequencing targeted
  by this task.
* [ ] Bounded old-authority scan over touched files and added diff lines shows
  no new positive legacy `i32m1` route authority, source-front-door route
  authority, descriptor-driven compute, direct-C/source export authority, or
  mirror-only acceptance.
* [ ] `git diff --check`, `git diff --cached --check`, and final
  `git status --short` are reported.

## Out of Scope

* No new contraction operation coverage.
* No broad MAcc, dot, dtype, SEW, LMUL, mask, or runtime matrix.
* No source-front-door positive route, high-level Linalg/Vector/StableHLO
  frontend, per-Linalg route authority, artifact dashboard, or report-only
  closeout.
* No Common EmitC invention of RVV semantics.
* No compatibility wrapper that keeps central provider semantic validation
  alive.
* No unrelated memory, segment2, compare/select, reduction, Gearbox, or
  dequant-clamp rework unless required by the exact direct-contraction owner
  boundary.
* No `ssh rvv` runtime claim unless this round changes runtime correctness or
  performance behavior.

## Technical Notes

* Required specs:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`, and
  `.trellis/spec/testing/index.md`.
* Previous task reference:
  `.trellis/tasks/archive/2026-06/06-08-06-08-stage2-rvv-migrated-statement-plan-provider-consumption-boundary/`.
* Primary source seams:
  `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`,
  `include/TianChenRV/Plugin/RVV/RVVEmitCStatementPlanOwners.h`,
  `lib/Plugin/RVV/EmitC/RVVEmitCStatementPlanOwners.cpp`,
  `include/TianChenRV/Plugin/RVV/RVVEmitCContractionRouteFamilyPlanOwners.h`,
  `include/TianChenRV/Plugin/RVV/RVVEmitCMAccRouteFamilyPlanOwners.h`,
  `lib/Plugin/RVV/EmitC/RVVEmitCContractionRouteFamilyPlanOwners.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCMAccRouteFamilyPlanOwners.cpp`, and
  focused tests in `test/Plugin/RVVExtensionPluginTest.cpp` plus target export
  tests if route artifacts are affected.
