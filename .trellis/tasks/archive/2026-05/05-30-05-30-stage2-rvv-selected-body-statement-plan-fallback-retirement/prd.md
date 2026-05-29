# Stage2 RVV selected-body statement-plan fallback retirement

## Goal

Retire the residual central selected-body RVV route statement fallback in
`RVVEmitCRouteProvider` after explicit migrated statement-plan owner dispatch
and direct-contraction statement-plan owner dispatch. Supported selected-body
routes must reach `TCRVEmitCLowerableRoute` through an explicit RVV-owned
statement-plan owner, or fail closed with a targeted diagnostic before the
provider reconstructs RVV operation semantics centrally.

## Direction Source

- Direction title: `Switch: Stage2 RVV selected-body statement-plan fallback
  retirement`.
- Module owner: RVV plugin-owned selected-body route statement-plan
  construction in `RVVEmitCRoutePlanning` and `RVVEmitCRouteProvider`.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean.
- Initial HEAD: `340e91b9 rvv: close selected-boundary route api cleanup`.
- `.trellis/.current-task` was absent, so this task was created from the
  supplied Hermes Direction Brief before source edits.
- Serial worker constraint from the brief: one Codex worker; no subagents,
  spawned agents, parallel agents, or multi-agent workflows.

## Current Repository Facts

- The predecessor task
  `.trellis/tasks/archive/2026-05/05-30-05-30-stage2-rvv-selected-boundary-api-closure`
  removed direct route-entry production APIs and archived with
  `check-tianchenrv` passing 464/464.
- `.trellis/spec/extension-plugins/rvv-plugin.md` now requires selected-body
  routes to flow through realized typed `tcrv_rvv` structure, route-family
  facts, provider plans, RVV-owned statement plans, and provider-built
  `TCRVEmitCLowerableRoute`.
- `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h` exposes
  `RVVSelectedBodyMigratedRouteStatementPlanOwner` and
  `RVVSelectedBodyDirectContractionRouteProviderOwner` surfaces.
- `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp` already has registries and
  aggregate getters for migrated statement-plan owners and direct-contraction
  provider/statement owners.
- `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp` already calls the migrated
  aggregate statement-plan getter and the direct-contraction statement-plan
  getter before a large residual fallback that binds ABI operands and emits
  setvl/load/compare/mask/compute/store statements by operation and memory-form
  switches inside the provider.
- That residual fallback is now the main architectural risk: after route-entry
  authority was retired, it can become the new central semantic owner if any
  selected-body route depends on it.

## Requirements

1. Keep `RVVEmitCRouteProvider` responsible for constructing the neutral
   `TCRVEmitCLowerableRoute`, adding provider-owned headers, type mappings,
   ABI mappings, and selected-boundary source provenance.
2. Ensure provider route statement construction dispatches through the
   migrated statement-plan owner aggregate first and direct-contraction
   statement-plan owner second, using the existing planning/provider facts for
   the same route analysis.
3. Remove active production dependency on the residual provider-local fallback
   that reconstructs operation semantics, memory form, dtype, mask/tail policy,
   runtime AVL/VL binding, stride binding, accumulator layout, or intrinsic
   choices from route descriptions or central switches.
4. If neither migrated nor direct-contraction owner produces a plan for a
   selected-body route, fail closed before central route statement
   reconstruction with a diagnostic that names the missing statement-plan owner
   boundary and the operation/memory-form context.
5. Preserve fail-closed diagnostics for incomplete, multiply matched, stale, or
   missing owner plans in the existing planning owner registries.
6. Add or update focused C++ tests proving migrated owner coverage,
   direct-contraction owner coverage, and fail-closed unowned selected-body
   behavior before provider fallback reconstruction.
7. Do not add new RVV operation coverage, dtype/LMUL clone batches, frontend
   lowering, direct route-entry support, descriptor/source-front-door route
   authority, common EmitC semantic selection, or evidence-only changes.

## Acceptance Criteria

- [x] PRD, implement context, and check context accurately describe the bounded
      production owner-boundary cleanup, specs, non-goals, and validation plan.
- [x] `RVVEmitCRouteProvider.cpp` no longer actively falls through from
      migrated/direct-contraction owner dispatch into central provider
      semantic statement reconstruction for selected-body routes.
- [x] A selected-body route not claimed by migrated or direct-contraction
      statement-plan owners fails closed before route statement construction,
      with a diagnostic naming the missing statement-plan owner boundary and
      the route operation/memory form.
- [x] Existing migrated-family routes still attach owner-produced pre-loop
      steps and loop statements into `TCRVEmitCLowerableRoute`.
- [x] Existing direct-contraction routes still attach owner-produced pre-loop
      steps and loop statements into `TCRVEmitCLowerableRoute`.
- [x] Focused C++ plugin tests cover migrated owner/provider consumption,
      direct-contraction owner/provider consumption, and the unowned fail-closed
      provider path.
- [x] Representative selected-boundary generated-bundle dry-runs include one
      migrated family and one direct-contraction family, and evidence still
      reports `route_entry_realization: false`.
- [x] Bounded scans over touched RVV production code show no active
      route-entry APIs and no active provider-local central semantic
      reconstruction after owner dispatch.
- [x] `git diff --check` passes.
- [x] Focused build/test targets for changed C++ code pass.
- [x] `check-tianchenrv` passes, or an exact blocker is recorded.
- [x] `ssh rvv` evidence is run only if emitted executable route statements
      change; otherwise the final report states no new runtime/correctness/
      performance claim was made.
- [x] Task status, journal, and archive state are truthful; a coherent commit
      is created only when the task is complete.

## Out of Scope

- No new RVV op family, dtype, LMUL, SEW, policy, mask, reduction, conversion,
  contraction, or memory coverage.
- No high-level Linalg/Vector/StableHLO frontend lowering.
- No source-front-door positive routes, descriptor-driven computation, direct-C
  exporter revival, one-intrinsic wrapper dialects, dashboards, broad smoke
  matrices, or report-only closure.
- No resurrection of direct route-entry realization or route-entry owner APIs.
- No common EmitC logic that selects RVV semantics, intrinsics, dtype, schedule,
  memory form, or route support.

## Technical Approach

The production path should stay:

```text
selected RVV variant
  -> public selected lowering-boundary materializer
  -> realized typed tcrv_rvv body
  -> route-family analysis and provider facts
  -> migrated statement-plan owner or direct-contraction statement-plan owner
  -> provider-built TCRVEmitCLowerableRoute
  -> neutral common EmitC
```

This task will keep the provider route object assembly, but close the residual
statement fallback after owner dispatch. The provider may attach owner-produced
steps, but it must not bind ABI operands and build route statements through a
central operation/memory-form switch after the owner boundary has returned no
plan.

## Validation Plan

1. Build and run the focused C++ plugin test:
   `cmake --build build --target tianchenrv-rvv-extension-plugin-test -j2`
   and `./build/bin/tianchenrv-rvv-extension-plugin-test`.
2. Build production tools touched by route construction:
   `cmake --build build --target tcrv-opt tcrv-translate -j2`.
3. Run representative selected-boundary generated-bundle dry-runs for one
   migrated family and one direct-contraction family, checking
   `route_entry_realization: false`.
4. Run a bounded authority scan over touched RVV production files for
   route-entry API revival and active provider-local fallback reconstruction.
5. Run `git diff --check`.
6. Run `cmake --build build --target check-tianchenrv -j2`, or record the exact
   blocker.
7. Run `ssh rvv` representative evidence only if emitted route statements
   changed.

## Completion Evidence

- Production movement: `RVVEmitCRouteProvider.cpp` now attaches migrated
  statement-plan owner output, then direct-contraction statement-plan owner
  output, and otherwise calls
  `diagnoseMissingRVVSelectedBodyRouteStatementPlanOwner(...)`; the residual
  central provider-local operation/memory-form statement fallback was removed.
- Owner boundary closed: ordinary vector `reduce_add` with `VectorRHSLoad` is
  now an explicit migrated statement-plan owner family, with provider-ready
  full-chunk `setvl`, loop `setvl`, lhs/rhs loads, reduction compute, and
  indexed output store statements produced in `RVVEmitCRoutePlanning`.
- Fail-closed behavior: unowned selected-body routes now report that selected
  RVV EmitC route construction requires an explicit migrated or
  direct-contraction statement-plan owner before provider-local statement
  construction, including the operation and memory form.
- Representative dry-run evidence:
  `artifacts/tmp/stage2_rvv_statement_plan_fallback_retirement/representative-dry-run`
  passed for migrated `reduce_add` and direct-contraction
  `widening_dot_reduce_add`; both evidence files record
  `local_bundle_generation.route_entry_realization: false`.
- `ssh rvv` was not run because this round makes no new runtime,
  correctness, or performance claim and the representative evidence is
  selected-boundary dry-run route construction evidence.
- Checks passed: focused RVV plugin C++ build and test, `tcrv-opt` and
  `tcrv-translate` build, representative generated-bundle dry-run, bounded
  authority scans, `git diff --check`, and `check-tianchenrv` 464/464.

## Technical Notes

- Specs read: `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  `.trellis/spec/testing/mlir-testing-contract.md`,
  `.trellis/spec/plugin-protocol/index.md`, and the shared guides.
- Predecessor read:
  `.trellis/tasks/archive/2026-05/05-30-05-30-stage2-rvv-selected-boundary-api-closure/prd.md`.
- Workspace journal read:
  `.trellis/workspace/codex/journal-18.md` entries for Sessions 322 and 323.
- Production files inspected:
  `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`, and
  `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`.
- Test file inspected:
  `test/Plugin/RVVExtensionPluginTest.cpp`.
