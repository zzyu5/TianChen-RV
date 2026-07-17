# Stage2 RVV migrated statement-plan provider consumption boundary

## Goal

Make the RVV selected-body route provider consume already-migrated RVV
statement-plan families through one aggregate owner-selected boundary. The
central `RVVEmitCRouteProvider` may still instantiate
`TCRVEmitCLowerableRoute`, attach provider-ready statements, and preserve
neutral route payloads, but it must not manually sequence migrated
family-specific statement-plan getters or rebuild migrated statements from
operation names, route ids, ABI strings, mirrors, artifacts, or Common EmitC.

This task closes the bounded production workflow seam named by the current
RVV plugin spec: route-family plans, route materialization facts, and operand
binding facts exist first; then the migrated statement-plan owner registry
selects exactly one owning family or returns an empty/default plan for
unrelated routes; then the provider attaches that owner-built plan to the
lowerable route.

## What I already know

* No `.trellis/.current-task` existed at session start; this task was created
  from the Direction Brief.
* The repository started clean on `main` at
  `d76f5cb0 rvv: make gearbox dequant clamp ABI executable`.
* The previous archived task made the Gearbox true multi-`with_vl`
  dequant-clamp path executable with provider/target validation and real
  `ssh rvv` generated-bundle evidence. It has no direct continuation point.
* `.trellis/spec/extension-plugins/rvv-plugin.md` requires a migrated
  statement-plan aggregate boundary selected through
  `getRVVSelectedBodyMigratedRouteStatementPlanOwners()`.
* The spec says the central provider must not manually call each migrated
  family statement-plan getter, infer ownership from whichever plan is
  non-empty, or rebuild migrated statement semantics locally.
* `.trellis/spec/lowering-runtime/emitc-route.md` keeps
  `TCRVEmitCLowerableRoute` as a provider-built common container, not an
  extension semantic interpreter.

## Requirements

* Keep implementation in C++ / MLIR / LLVM / TableGen / CMake / lit /
  FileCheck. Python is allowed only for support tooling and evidence probes.
* Implement one production submodule: the RVV plugin-local migrated
  statement-plan provider boundary involving
  `RVVSelectedBodyMigratedRouteStatementPlanOwner`,
  `getRVVSelectedBodyMigratedRouteStatementPlan`, provider-side
  selection/attachment, and fail-closed diagnostics.
* The central selected-body route provider must obtain route-family plans,
  route materialization facts, and all relevant operand-binding fact surfaces
  before calling the aggregate migrated statement-plan boundary.
* The aggregate boundary must classify through the owner registry and select
  exactly one migrated family for migrated routes. Each migrated owner must
  have a valid name, family tag, consumer hook, and builder hook.
* Positive migrated routes must receive provider-ready statements from the
  owning family plan and attach those statements to
  `TCRVEmitCLowerableRoute` without central provider semantic branching.
* Unrelated or non-migrated routes must receive an empty/default migrated plan
  and continue through their existing route surface unchanged.
* The boundary must fail closed before route statement construction on missing
  hooks, multiple matching owners, wrong family tags, a matched owner returning
  no migrated plan, or stale/missing family-specific dependencies surfaced by
  the owning builder.
* Provider diagnostics and tests must make it clear that route ids, operation
  names, ABI strings, mirrors, artifacts, source-front-door metadata,
  descriptors, and Common EmitC are not statement authority.

## Acceptance Criteria

* [ ] The owner registry has focused C++ coverage for membership, owner names,
  family tags, hook presence, and exact-once classification for already
  migrated families.
* [ ] `RVVEmitCRouteProvider.cpp` no longer manually sequences migrated
  family-specific statement-plan getters for migrated families before route
  construction.
* [ ] The provider calls the aggregate boundary once after route-family plans,
  materialization facts, and elementwise/select, memory, math, and residual
  operand-binding facts are available.
* [ ] Representative migrated families build positive provider-ready statement
  plans through the aggregate boundary and those statements are attached to the
  generated lowerable route.
* [ ] Unrelated/direct-provider routes retain empty/default behavior and are
  not forced through migrated family semantics.
* [ ] Focused fail-closed C++ coverage exercises at least one missing/stale
  migrated statement-plan dependency and at least one missing/multiple/wrong
  owner diagnostic path where practical with the existing test seams.
* [ ] Regression tests pass for
  `build/bin/tianchenrv-rvv-extension-plugin-test` and
  `build/bin/tianchenrv-target-artifact-export-test`.
* [ ] Any affected representative target/export or lit RUN-line fixtures are
  replayed if the changed code touches their behavior.
* [ ] A bounded scan over touched files and added diff lines shows no new
  positive legacy `i32m1` route authority, source-front-door route authority,
  descriptor-driven compute, direct-C/source export authority, or mirror-only
  acceptance.
* [ ] `git diff --check`, `git diff --cached --check`, and final
  `git status --short` are reported.

## Out of Scope

* No new RVV route-family coverage.
* No Gearbox/dequant-clamp rework except as regression reference.
* No migration of direct contraction, Widening MAcc, dot-reduction, residual
  routes, runtime-scalar splat-store, or future families unless current
  spec/code already declares them migrated into this aggregate.
* No source-front-door positive route, descriptor/direct-C/source export path,
  artifact metadata authority, mirror-only route authority, dashboard,
  performance tuning database, or report-only closeout.
* No Common EmitC invention of RVV statements.
* No broad smoke matrix beyond the focused provider/target checks needed for
  this boundary.

## Technical Notes

* Required specs:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`, and
  `.trellis/spec/lowering-runtime/emitc-route.md`.
* Relevant previous task:
  `.trellis/tasks/archive/2026-06/06-08-stage2-rvv-gearbox-true-multi-with-vl-dequant-clamp-epilogue-executable-artifact-abi-boundary/`.
* Initial production seam candidates from the brief:
  `include/TianChenRV/Plugin/RVV/RVVEmitCStatementPlanOwners.h`,
  `lib/Plugin/RVV/EmitC/RVVEmitCStatementPlanOwners.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`, migrated family owner
  files under `include/TianChenRV/Plugin/RVV/` and
  `lib/Plugin/RVV/EmitC/`, `test/Plugin/RVVExtensionPluginTest.cpp`, and
  representative target/export fixtures if affected.
