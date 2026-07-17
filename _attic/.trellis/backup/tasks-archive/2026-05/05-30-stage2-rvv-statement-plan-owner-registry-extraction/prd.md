# Expand: Stage2 RVV statement-plan owner registry extraction

## Goal

Move the production authority for migrated and direct-contraction selected-body
statement-plan owner registries, exact-one owner classification, missing-owner
diagnostics, and provider-facing statement attachment into the RVV EmitC
statement-plan owner boundary. `RVVEmitCRouteProvider.cpp` must remain a neutral
`TCRVEmitCLowerableRoute` assembler, and `RVVEmitCRoutePlanning.cpp` must stop
being the central home for provider-facing statement-plan owner registry
selection where this round moves that boundary.

## Direction Source

- Direction title: `Expand: Stage2 RVV statement-plan owner registry
  extraction`.
- Module owner: `RVVEmitCStatementPlanOwners.{h,cpp}` and any clearly named
  sibling under the same RVV EmitC selected-body statement-plan owner boundary.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean.
- Initial HEAD: `896077d8 rvv: extract statement-plan owner boundary`.
- `.trellis/.current-task` was absent, so this task was created from the Hermes
  Direction Brief before source edits.
- Serial worker constraint from the brief: one Codex worker; no subagents,
  spawned agents, parallel agents, or multi-agent workflows.

## What I Already Know

- The previous archived task
  `.trellis/tasks/archive/2026-05/05-30-stage2-rvv-statement-plan-owner-boundary`
  made `RVVEmitCRouteProvider.cpp` consume
  `getRVVSelectedBodyRouteStatementPlanOwnerSelection(...)` and
  `attachRVVSelectedBodyRouteStatementPlanOwnerSelection(...)` from
  `RVVEmitCStatementPlanOwners`.
- That prior boundary is real but shallow: `RVVEmitCStatementPlanOwners.cpp`
  still delegates to
  `getRVVSelectedBodyMigratedRouteStatementPlan(...)` and
  `getRVVSelectedBodyDirectContractionRouteStatementPlan(...)`.
- `RVVEmitCRoutePlanning.cpp` still contains
  `getRVVSelectedBodyMigratedRouteStatementPlanOwners()`,
  `isRVVSelectedBodyMigratedRouteStatementPlanConsumer(...)`,
  `getRVVSelectedBodyMigratedRouteStatementPlan(...)`,
  `getRVVSelectedBodyDirectContractionRouteProviderOwners()`,
  `isRVVSelectedBodyDirectContractionRouteProviderConsumer(...)`, and
  `getRVVSelectedBodyDirectContractionRouteStatementPlan(...)`.
- The same planning file also still owns the exact-one / missing / multiple
  owner diagnostics for those migrated and direct-contraction statement-plan
  owners.
- `RVVEmitCRoutePlanning.cpp` may continue to own neutral route analysis,
  materialization facts, operand-binding facts, provider-plan facts, and
  low-level family statement builders if moving all builders would make this
  round too large.
- The task is an owner-boundary extraction. It must preserve all existing
  migrated-family and direct-contraction route behavior exactly.

## Requirements

1. Move migrated statement-plan owner registry authority into
   `RVVEmitCStatementPlanOwners` or a clearly named sibling under the same owner
   boundary.
2. Move direct-contraction statement-plan owner registry authority into the same
   owner boundary. The direct-contraction provider-plan getter may remain in
   planning because it joins route-family, materialization, route-control, math
   binding, typed config, capability, and ABI facts before route construction.
3. Move exact-one owner classification, incomplete-entry checks,
   multiple-owner diagnostics, empty unrelated behavior, and missing-owner
   diagnostics for provider-facing statement-plan owners out of
   `RVVEmitCRoutePlanning.cpp`.
4. Keep low-level statement builder helpers and already validated family plan
   construction behavior unchanged unless a narrow signature adapter is required
   to let the owner module call them.
5. Keep `RVVEmitCRouteProvider.cpp` neutral: it may obtain route analysis,
   materialization facts, operand-binding facts, direct-contraction provider
   facts, instantiate `TCRVEmitCLowerableRoute`, add headers/type mappings/ABI
   mappings/provenance, then attach statements returned by the owner boundary.
6. Do not reintroduce provider-local migrated/direct sequencing, operation-name
   semantic switches, route-id authority, artifact-name authority,
   exact-intrinsic authority, descriptor/source-front-door authority, direct-C
   source export authority, or common EmitC RVV semantic branches.
7. Add focused C++ coverage proving the owner boundary, not the monolithic
   planning surface, exposes migrated/direct owner registries, owner
   classification, exact-one diagnostics, no-owner diagnostics, and provider
   consumption.
8. Do not add new RVV operation, dtype, LMUL, frontend, Linalg, Vector, IME,
   Offload, TensorExt, source-front-door, dashboard, or runtime/performance
   scope.

## Acceptance Criteria

- [x] `RVVEmitCStatementPlanOwners.h` or an owner-boundary sibling exposes the
      migrated and direct-contraction statement-plan owner registry/query APIs
      needed by production and tests.
- [x] `RVVEmitCRoutePlanning.h` no longer declares provider-facing migrated or
      direct-contraction statement-plan owner registry APIs as the authority.
- [x] `RVVEmitCRoutePlanning.cpp` no longer contains the provider-facing
      migrated/direct statement-plan owner registry tables or aggregate
      exact-one owner selection logic moved by this round.
- [x] Planning still exposes only the lower-level reusable builder hooks or
      neutral planning facts intentionally left there, with names that do not
      present planning as the provider-facing registry owner.
- [x] `RVVEmitCStatementPlanOwners.cpp` selects migrated and
      direct-contraction statement-plan ownership exactly once and reports
      incomplete, multiple-owner, wrong-family, missing-provider-plan, and
      missing-owner errors through the owner boundary.
- [x] `RVVEmitCRouteProvider.cpp` remains neutral and consumes only the
      provider-facing owner-selection / attach API; it does not regain migrated
      or direct-contraction sequencing.
- [x] Existing migrated-family route statements and direct-contraction route
      statements are preserved byte-for-byte in behavior unless tests require
      only diagnostic wording updates caused by the ownership move.
- [x] Focused C++ tests cover migrated owner registry membership/classification,
      direct-contraction owner registry membership/classification, exact-one or
      no-owner diagnostics, and provider consumption through the owner boundary.
- [x] Representative generated-bundle dry-runs for one migrated route and one
      direct-contraction route still report selected-boundary behavior with
      `route_entry_realization: false`.
- [x] `git diff --check` passes.
- [x] Focused C++ plugin build/test and `tcrv-opt` / `tcrv-translate` build
      pass.
- [x] `check-tianchenrv` passes, or an exact blocker is recorded.
- [x] Bounded authority scans over touched RVV files show no legacy-i32,
      source-front-door, descriptor, direct-C, source-export, artifact-name,
      route-id, exact-intrinsic, common-EmitC, metadata-only, mirror-only,
      direct-route-entry-only, pre-realized-fixture-only, or script-derived
      authority drift.
- [x] `ssh rvv` is not required because this round does not claim new runtime,
      correctness, or performance behavior.

## Out of Scope

- New RVV operation coverage, dtype/LMUL/SEW/policy coverage, frontend lowering,
  Linalg/Vector/StableHLO work, IME, Offload, TensorExt, Toy, Template, or
  future plugin work.
- Rewriting selected-body realization or route-family provider-plan
  verification.
- Moving unrelated materialization facts, operand-binding facts, route-control
  provider-plan facts, segment2 provider-plan ownership, target artifact
  mechanics, or common EmitC behavior.
- Reintroducing descriptor-driven computation, source-front-door positive RVV
  paths, direct route-entry realization, direct-C exporters, or compatibility
  wrappers preserving old authority.
- Treating reports, dashboards, broad smoke matrices, helper-only edits, or
  test-only edits as the milestone.

## Technical Approach

Keep `RVVEmitCRoutePlanning.cpp` as the owner of route analysis and reusable
planning/building mechanics, but split the provider-facing statement-plan owner
boundary out of it:

```text
selected RVV variant
  -> typed/realized tcrv_rvv body
  -> route analysis / materialization facts / operand-binding facts
  -> direct-contraction provider plan where applicable
  -> RVVEmitCStatementPlanOwners owner registry and exact-one selection
  -> owner-produced provider-ready statements
  -> RVVEmitCRouteProvider neutral TCRVEmitCLowerableRoute assembly
  -> common EmitC materialization
```

The likely code shape is:

- Move `RVVSelectedBodyMigratedRouteStatementPlanOwner` and
  `RVVSelectedBodyDirectContractionRouteProviderOwner` registry declarations to
  the owner-boundary header, or introduce equivalent owner-boundary aliases with
  production-visible names.
- Move registry tables, consumer aggregate predicates, aggregate statement-plan
  getters, and exact-one diagnostics into `RVVEmitCStatementPlanOwners.cpp`.
- Leave family-specific builder functions in `RVVEmitCRoutePlanning.cpp` if
  they are large and already validate family dependencies; expose them through
  narrow internal/public helper declarations only as builder hooks, not as
  registry authority.
- Keep `getRVVSelectedBodyRouteStatementPlanOwnerSelection(...)` as the
  provider-facing top-level API, but make it call owner-boundary aggregate logic
  rather than planning-owned aggregate getters.

## Validation Plan

1. Build and run focused C++ plugin coverage:
   `cmake --build build --target tianchenrv-rvv-extension-plugin-test -j2`
   and `./build/bin/tianchenrv-rvv-extension-plugin-test`.
2. Build touched production tools:
   `cmake --build build --target tcrv-opt tcrv-translate -j2`.
3. Run representative generated-bundle dry-runs for one migrated route such as
   `reduce_add` and one direct-contraction route such as
   `widening_dot_reduce_add`, checking `route_entry_realization: false`.
4. Run bounded provider/owner/planning scans proving registry authority moved
   and provider neutrality was preserved.
5. Run bounded authority drift scans over touched RVV production/test files.
6. Run `git diff --check`.
7. Run `cmake --build build --target check-tianchenrv -j2`, or record the exact
   blocker.

## Completion Evidence

- Production owner boundary moved:
  `include/TianChenRV/Plugin/RVV/RVVEmitCStatementPlanOwners.h` now exposes the
  migrated and direct-contraction statement-plan owner structs, owner registry
  getters, consumer predicates, aggregate statement-plan getters, and
  provider-facing selected owner-selection / attach APIs.
- Planning boundary narrowed:
  `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h` no longer declares the
  provider-facing migrated/direct statement-plan owner registries or aggregate
  getters. It intentionally retains only low-level migrated statement builder
  hooks and the direct-contraction statement builder hook so the owner module can
  reuse existing family validators without moving every large builder in this
  round.
- Registry and exact-one authority moved:
  `lib/Plugin/RVV/EmitC/RVVEmitCStatementPlanOwners.cpp` now owns migrated owner
  classification, direct-contraction owner classification, registry tables,
  aggregate exact-one selection, incomplete-entry diagnostics, multiple-owner
  diagnostics, missing-provider-plan diagnostics, missing-owner diagnostics, and
  movement of provider-ready pre-loop/loop statements into the selected owner
  payload.
- Planning implementation no longer contains provider-facing owner registry
  tables or aggregate migrated/direct statement-plan getter implementations.
  It still owns route analysis, materialization, operand-binding facts,
  route-control/provider-plan facts, and low-level builder mechanics.
- Provider neutrality preserved:
  `RVVEmitCRouteProvider.cpp` still obtains route analysis, materialization
  facts, operand-binding facts, and direct-contraction provider facts; builds
  `TCRVEmitCLowerableRoute`; records headers/type mappings/ABI mappings/source
  provenance; then consumes
  `getRVVSelectedBodyRouteStatementPlanOwnerSelection(...)` and
  `attachRVVSelectedBodyRouteStatementPlanOwnerSelection(...)`.
- Existing C++ coverage passed under the new boundary:
  `runMigratedRouteStatementPlanOwnerRegistryTest` covers the 11 migrated
  owner entries, owner names/family tags, exact-once classification, empty
  unrelated behavior, no-owner diagnostics, stale dependency diagnostics, and
  provider owner selection. The direct-contraction coverage checks the single
  direct owner, exact-once classification for all active direct-provider
  contraction routes, empty unrelated behavior, direct statement-plan
  construction, and provider selection.
- Representative generated-bundle dry-run passed:
  `python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run
  --pre-realized-selected-body --op-kind reduce_add --op-kind
  widening_dot_reduce_add --artifact-root
  artifacts/tmp/stage2_rvv_statement_plan_owner_registry_extraction --run-id
  representative-dry-run --overwrite`.
- Evidence files:
  `artifacts/tmp/stage2_rvv_statement_plan_owner_registry_extraction/representative-dry-run/reduce_add/evidence.json`
  and
  `artifacts/tmp/stage2_rvv_statement_plan_owner_registry_extraction/representative-dry-run/widening_dot_reduce_add/evidence.json`
  both report `route_entry_realization: false`.
- Bounded scans:
  no provider-facing migrated/direct owner registry or aggregate getter remains
  in `RVVEmitCRoutePlanning.{h,cpp}`; `RVVEmitCRouteProvider.cpp` has no direct
  migrated/direct statement-plan getter calls, local statement builders, route
  operation switches, `bound*ABI` reconstruction, descriptor/source-front-door,
  direct-C/source-export, route-entry, or exact-intrinsic authority drift.
- Diff-only authority scan over touched files produced no new legacy-i32,
  source-front-door, descriptor, direct-C, source-export, artifact-name,
  route-id, exact-intrinsic, common-EmitC, metadata-only, mirror-only,
  direct-route-entry-only, pre-realized-fixture-only, or script-derived
  authority hits.
- Checks passed:
  `cmake --build build --target tianchenrv-rvv-extension-plugin-test -j2`;
  `./build/bin/tianchenrv-rvv-extension-plugin-test`;
  `cmake --build build --target tcrv-opt tcrv-translate -j2`;
  representative generated-bundle dry-run; bounded scans; `git diff --check`;
  `cmake --build build --target check-tianchenrv -j2` passed 464/464.
- `ssh rvv` was not run because this round changes owner-boundary extraction
  only and makes no new runtime, correctness, or performance claim.
- Spec update judgment: no `.trellis/spec/` update is needed. This round
  implements the existing RVV plugin spec sections for migrated statement-plan
  provider consumption, direct-contraction route-provider ownership,
  provider-built `TCRVEmitCLowerableRoute`, and common EmitC neutrality without
  changing the durable contract.

## Technical Notes

- Specs read: `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`, and
  `.trellis/spec/lowering-runtime/emitc-route.md`.
- Previous task read:
  `.trellis/tasks/archive/2026-05/05-30-stage2-rvv-statement-plan-owner-boundary/prd.md`.
- Workspace journal read: `.trellis/workspace/codex/journal-18.md` entries for
  recent Stage2 selected-boundary/API closure context.
- Production files inspected:
  `include/TianChenRV/Plugin/RVV/RVVEmitCStatementPlanOwners.h`,
  `lib/Plugin/RVV/EmitC/RVVEmitCStatementPlanOwners.cpp`,
  `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`, and
  `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`.
- Focused test file inspected: `test/Plugin/RVVExtensionPluginTest.cpp`.
