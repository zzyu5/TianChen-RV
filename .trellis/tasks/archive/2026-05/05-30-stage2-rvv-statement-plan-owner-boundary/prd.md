# Stage2 RVV selected-body statement-plan owner boundary

## Goal

Make the selected-body RVV EmitC statement-plan owner boundary a first-class
RVV plugin-local production module. The provider should remain a neutral
`TCRVEmitCLowerableRoute` assembler and consume one explicit owner-selection
surface for provider-ready statements, instead of knowing how to sequence the
migrated statement-plan aggregate and direct-contraction statement-plan
aggregate itself.

## Direction Source

- Direction title: `Switch: Stage2 RVV selected-body statement-plan owner
  module boundary`.
- Module owner: RVV EmitC selected-body statement-plan owner boundary:
  migrated statement-plan owner registry, direct-contraction statement owner
  registry, and provider-facing APIs.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean.
- Initial HEAD: `ec148291 rvv: retire selected-body statement fallback`.
- `.trellis/.current-task` was absent, so this task was created from the
  supplied Hermes Direction Brief before source edits.
- Serial worker constraint from the brief: one Codex worker; no subagents,
  spawned agents, parallel agents, or multi-agent workflows.

## Current Repository Facts

- The predecessor task
  `.trellis/tasks/archive/2026-05/05-30-05-30-stage2-rvv-selected-body-statement-plan-fallback-retirement`
  removed the residual provider-local selected-body statement fallback.
- `RVVEmitCRouteProvider.cpp` now obtains materialization facts, operand-binding
  facts, a direct-contraction provider plan, and segment2 provider plan before
  constructing a `TCRVEmitCLowerableRoute`.
- The provider then calls
  `getRVVSelectedBodyMigratedRouteStatementPlan(...)` and
  `getRVVSelectedBodyDirectContractionRouteStatementPlan(...)` directly, and
  attaches whichever provider-ready statement plan is produced.
- `RVVEmitCRoutePlanning.cpp` contains the migrated statement-plan registry,
  direct-contraction statement-plan registry, aggregate owner classification,
  missing/incomplete/multiple-owner checks, and many family-local statement
  builders inside one large planning surface.
- Existing C++ coverage already exercises migrated owner membership, direct
  contraction owner membership, representative migrated/direct plans, and
  unowned fail-closed provider behavior, but the provider-facing module boundary
  is not yet explicit.

## Requirements

1. Add or harden one RVV plugin-local statement-plan owner module surface that
   represents the production boundary between route planning facts and provider
   route assembly.
2. The module must consume only same-analysis typed body/config/runtime facts,
   materialization facts, operand-binding facts, route-control/direct provider
   facts, and validated family plans already exposed by RVV planning.
3. The module must classify migrated owners and direct-contraction owners as an
   exact-one provider-facing selected statement-plan owner. More than one owner
   class matching one selected route is a fail-closed error.
4. The module must return an empty/default plan for unrelated routes only where
   the underlying migrated/direct owner contracts define unrelated behavior; a
   route that reaches provider statement construction with no owner must fail
   closed with operation and memory-form diagnostics.
5. `RVVEmitCRouteProvider.cpp` must stay neutral: instantiate
   `TCRVEmitCLowerableRoute`, add neutral headers/type mappings/ABI mappings
   and provenance, then attach provider-ready statements returned by the new
   owner boundary.
6. The provider must not manually sequence every migrated family getter, branch
   on route operation names to rebuild statements, or choose RVV semantics from
   route ids, metadata, ABI strings, artifact names, exact intrinsics, source
   front-door markers, descriptors, or common EmitC.
7. Add focused C++ tests for the new provider-facing owner boundary: owner
   classification for migrated and direct-contraction representatives,
   empty/unrelated behavior, missing-owner diagnostics, and provider
   consumption for at least one migrated family and one direct-contraction
   family.
8. Do not add new RVV operation coverage, dtype/LMUL clone batches, frontend
   lowering, descriptor/source-front-door positive routes, direct route-entry
   support, one-intrinsic wrappers, common EmitC semantic branches, broad smoke
   matrices, dashboards, or standalone evidence packaging.

## Acceptance Criteria

- [x] PRD, implement context, and check context accurately describe the bounded
      production owner-boundary cleanup, specs, non-goals, and validation plan.
- [x] A first-class RVV statement-plan owner module/API exists in production
      code and is included in the RVV EmitC provider library build.
- [x] `RVVEmitCRouteProvider.cpp` consumes the provider-ready statement plan
      through the new owner boundary instead of locally sequencing migrated and
      direct-contraction statement-plan getters.
- [x] The new boundary preserves exact-one owner classification across migrated
      and direct-contraction owners and fails closed when no owner can satisfy a
      selected-body provider route.
- [x] Existing migrated-family routes still attach owner-produced pre-loop
      steps and loop statements into `TCRVEmitCLowerableRoute`.
- [x] Existing direct-contraction routes still attach owner-produced pre-loop
      steps and loop statements into `TCRVEmitCLowerableRoute`.
- [x] Focused C++ plugin tests cover owner boundary classification,
      empty/unrelated behavior, missing-owner diagnostics, migrated provider
      consumption, and direct-contraction provider consumption.
- [x] Representative selected-boundary generated-bundle dry-runs include one
      migrated route such as `reduce_add` and one direct-contraction route such
      as `widening_dot_reduce_add`, with `route_entry_realization: false`.
- [x] Bounded provider scan shows no residual central statement fallback,
      `bound*ABI`/loop-step reconstruction, or family statement switches inside
      `RVVEmitCRouteProvider.cpp`.
- [x] Bounded authority scan over touched RVV files shows no new legacy-i32,
      source-front-door, descriptor, direct-C, source-export, artifact-name,
      route-id, exact-intrinsic, common-EmitC, metadata-only, mirror-only, or
      direct-route-entry authority drift.
- [x] `git diff --check` passes.
- [x] Focused build/test targets for changed C++ code pass.
- [x] `check-tianchenrv` passes, or an exact blocker is recorded.
- [x] `ssh rvv` is not required unless emitted runtime behavior changes or a
      new runtime/correctness/performance claim is made.

## Out of Scope

- No new selected-body route coverage, dtype, LMUL, SEW, policy, mask,
  reduction, conversion, contraction, memory, Linalg/frontend, IME, Offload,
  TensorExt, Toy, Template, or source-front-door work.
- No restoration of provider-local fallback branches, direct route-entry
  realization, descriptor-driven computation, direct-C exporter paths, or
  source-artifact positive RVV routes.
- No common EmitC logic that chooses RVV intrinsics, dtype/config/schedule,
  memory form, route support, or statement semantics.
- No runtime/correctness/performance claim beyond preserving existing
  selected-boundary dry-run behavior.

## Technical Approach

Create a small RVV EmitC statement-plan owner boundary module. The module will
wrap the existing migrated statement-plan aggregate and direct-contraction
statement-plan aggregate behind one provider-facing API that returns a single
selected owner plan with provider-ready pre-loop steps and one loop. The
provider will build the route object and source provenance exactly as before,
then attach this selected owner plan or fail closed through the module.

The expected production flow is:

```text
selected RVV variant
  -> realized typed tcrv_rvv body
  -> route-family analysis and provider facts
  -> materialization and operand-binding facts
  -> direct-contraction provider facts where applicable
  -> RVV statement-plan owner module exact-one selection
  -> provider attaches returned statements to TCRVEmitCLowerableRoute
  -> neutral common EmitC
```

## Validation Plan

1. Build and run the focused C++ plugin test:
   `cmake --build build --target tianchenrv-rvv-extension-plugin-test -j2`
   and `./build/bin/tianchenrv-rvv-extension-plugin-test`.
2. Build production tools touched by route construction:
   `cmake --build build --target tcrv-opt tcrv-translate -j2`.
3. Run representative selected-boundary generated-bundle dry-runs for one
   migrated family and one direct-contraction family, checking
   `route_entry_realization: false`.
4. Run bounded provider and authority scans over touched RVV production/test
   files.
5. Run `git diff --check`.
6. Run `cmake --build build --target check-tianchenrv -j2`, or record the exact
   blocker.

## Completion Evidence

- Production module boundary: added
  `include/TianChenRV/Plugin/RVV/RVVEmitCStatementPlanOwners.h` and
  `lib/Plugin/RVV/EmitC/RVVEmitCStatementPlanOwners.cpp`, and included the new
  source in `lib/Plugin/RVV/EmitC/CMakeLists.txt`.
- Provider neutrality preserved: `RVVEmitCRouteProvider.cpp` still obtains
  verified route-family/materialization/operand-binding/direct-contraction
  facts, constructs `TCRVEmitCLowerableRoute`, adds headers/type mappings/ABI
  mappings/provenance, then calls the new
  `getRVVSelectedBodyRouteStatementPlanOwnerSelection(...)` and
  `attachRVVSelectedBodyRouteStatementPlanOwnerSelection(...)` boundary.
- Owner boundary changed: the new module wraps the migrated statement-plan
  aggregate and direct-contraction statement-plan aggregate, enforces exact-one
  provider-facing owner selection, moves provider-ready pre-loop steps and loop
  statements into a single selection object, and owns the missing-owner
  diagnostic implementation.
- C++ coverage updated: `test/Plugin/RVVExtensionPluginTest.cpp` now covers
  owner-module missing diagnostics for unrelated routes, migrated `reduce_add`
  selection, direct-contraction selection, and provider consumption through the
  new boundary.
- Representative selected-boundary dry-run:
  `scripts/rvv_generated_bundle_abi_e2e.py --dry-run
  --pre-realized-selected-body --op-kind reduce_add --op-kind
  widening_dot_reduce_add` passed under
  `artifacts/tmp/stage2_rvv_statement_plan_owner_boundary/representative-dry-run`;
  both per-op evidence files report
  `local_bundle_generation.route_entry_realization: false`.
- Bounded provider scan: `RVVEmitCRouteProvider.cpp` no longer directly calls
  migrated/direct statement-plan getters and has no provider-local statement
  fallback, `bound*ABI`/loop-step reconstruction, or operation/memory-form
  family switch for selected-body statements.
- Authority scan: new production files have no legacy-i32, source-front-door,
  descriptor, direct-C, source-export, artifact-name, route-id, exact-intrinsic,
  common-EmitC, metadata-only, mirror-only, or direct-route-entry authority
  drift. Hits in the touched C++ test file are pre-existing negative/stale-mirror
  checks and exact intrinsic expectations, not new authority.
- Checks passed:
  `cmake --build build --target tianchenrv-rvv-extension-plugin-test -j2`;
  `./build/bin/tianchenrv-rvv-extension-plugin-test`;
  `cmake --build build --target tcrv-opt tcrv-translate -j2`;
  representative generated-bundle dry-run;
  bounded scans; `git diff --check`;
  `cmake --build build --target check-tianchenrv -j2` passed 464/464.
- `ssh rvv` was not run because this round changes owner module boundaries and
  provider plumbing only; it makes no new runtime, correctness, or performance
  claim.

## Technical Notes

- Specs read: `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/core-dialect/tcrv-exec-contract.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`, and the relevant layer
  indexes for extension plugins, lowering runtime, core dialect, and testing.
- Predecessor read:
  `.trellis/tasks/archive/2026-05/05-30-05-30-stage2-rvv-selected-body-statement-plan-fallback-retirement/prd.md`.
- Workspace journal read:
  `.trellis/workspace/codex/journal-18.md` entries for Sessions 322, 323, and
  324.
- Production files inspected:
  `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`,
  `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`, and
  `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`.
- Test file inspected:
  `test/Plugin/RVVExtensionPluginTest.cpp`.
