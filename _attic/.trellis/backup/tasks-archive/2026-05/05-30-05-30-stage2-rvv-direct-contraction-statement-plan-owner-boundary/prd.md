# Stage2 RVV direct-contraction statement-plan builder boundary extraction

## Goal

Move direct-contraction statement construction for widening MAcc and widening
dot-reduce selected-body routes out of monolithic `RVVEmitCRoutePlanning` and
into the RVV EmitC statement-plan owner boundary. Planning remains responsible
for route analysis, materialization facts, math operand-binding facts,
route-control facts, and the prevalidated direct-contraction provider plan.
`RVVEmitCRouteProvider` remains a neutral `TCRVEmitCLowerableRoute` assembler.

## Direction Source

- Direction title: `Expand: Stage2 RVV direct-contraction statement-plan builder
  boundary extraction`.
- Module owner: RVV EmitC statement-plan owner boundary, centered on
  `RVVEmitCStatementPlanOwners.{h,cpp}`.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean.
- Initial HEAD: `14fc61f2 rvv: extract statement-plan owner registry`.
- `.trellis/.current-task` was absent, so this task was created from the Hermes
  Direction Brief before source edits.
- Serial worker constraint from the brief: one Codex worker; no subagents,
  spawned agents, parallel agents, or multi-agent workflows.

## What I Already Know

- The archived predecessor
  `.trellis/tasks/archive/2026-05/05-30-stage2-rvv-statement-plan-owner-registry-extraction`
  moved migrated/direct statement-plan owner registries, exact-one selection,
  diagnostics, and provider-facing attachment into
  `RVVEmitCStatementPlanOwners`.
- Current `RVVEmitCStatementPlanOwners.cpp` owns the direct-contraction owner
  registry and selects a single owner, but its direct-contraction owner still
  calls
  `buildRVVSelectedBodyDirectContractionRouteStatementPlanFromProviderPlan(...)`
  from `RVVEmitCRoutePlanning`.
- Current `RVVEmitCRoutePlanning.cpp` still owns
  `buildDirectContractionRouteStatementPlan(...)` plus the helper routines that
  assemble provider-ready pre-loop steps and the EmitC loop for the direct
  contraction route family.
- Current `RVVEmitCRoutePlanning.cpp` should continue to own
  `getRVVSelectedBodyDirectContractionRouteProviderPlan(...)`, because that is
  fact collection over the verified contraction family plan, materialization
  facts, route-control provider plan, math operand-binding facts, typed config,
  selected target capability, ABI bindings, and required materialization leaves.
- Existing C++ coverage in `test/Plugin/RVVExtensionPluginTest.cpp` already
  exercises direct owner registry membership, exact-once classification,
  provider-plan construction, direct statement-plan construction, owner module
  selection, provider attachment, and missing fact diagnostics.

## Requirements

1. Move direct-contraction provider-ready statement construction into
   `RVVEmitCStatementPlanOwners.cpp` or a clearly named sibling under the same
   owner boundary.
2. Remove the provider-facing direct-contraction statement-builder declaration
   from `RVVEmitCRoutePlanning.h`; planning must not expose
   `buildRVVSelectedBodyDirectContractionRouteStatementPlanFromProviderPlan(...)`
   or an equivalent statement-construction authority.
3. Keep direct-contraction provider-plan fact collection in
   `RVVEmitCRoutePlanning.cpp`, with names and diagnostics that make it clear it
   is validating provider-plan facts rather than owning statement construction.
4. Preserve statement semantics for `widening_macc_add`,
   `widening_dot_reduce_add`, `strided_input_widening_dot_reduce_add`,
   `computed_masked_widening_dot_reduce_add`, and
   `computed_masked_strided_input_widening_dot_reduce_add`.
5. Keep migrated statement-plan owner behavior unchanged.
6. Keep `RVVEmitCRouteProvider.cpp` neutral: it may obtain route analysis,
   materialization facts, operand-binding facts, and direct-contraction
   provider facts; instantiate `TCRVEmitCLowerableRoute`; record neutral
   headers/type mappings/ABI mappings/source provenance; and attach statements
   returned by the owner boundary.
7. Do not add new RVV operation coverage, dtype/LMUL clone batches,
   source-front-door routes, direct route-entry shortcuts, high-level frontend
   lowering, one-intrinsic wrapper dialects, or common EmitC RVV semantic
   branches.

## Acceptance Criteria

- [ ] `RVVEmitCStatementPlanOwners.cpp` owns direct-contraction statement
      construction for provider-ready pre-loop steps and loop statements.
- [ ] `RVVEmitCStatementPlanOwners.cpp` direct-contraction registry points to
      an owner-local builder, not a planning-owned builder hook.
- [ ] `RVVEmitCRoutePlanning.h` no longer declares provider-facing
      direct-contraction statement construction.
- [ ] `RVVEmitCRoutePlanning.cpp` no longer defines
      `buildDirectContractionRouteStatementPlan(...)` or
      `buildRVVSelectedBodyDirectContractionRouteStatementPlanFromProviderPlan(...)`.
- [ ] `RVVEmitCRoutePlanning.cpp` still owns
      `getRVVSelectedBodyDirectContractionRouteProviderPlan(...)` and only
      reusable provider-plan fact checks for direct contraction.
- [ ] Focused C++ coverage proves MAcc and widening dot-reduce
      direct-contraction routes obtain provider-ready statements through
      `RVVEmitCStatementPlanOwners`.
- [ ] Representative generated-bundle dry-runs pass for at least one
      plain/scalar MAcc route and one widening dot-reduce selected-boundary
      route with `route_entry_realization: false`.
- [ ] Bounded scans show `RVVEmitCRouteProvider.cpp` has no direct-contraction
      statement-building or family semantic authority.
- [ ] Bounded scans show `RVVEmitCRoutePlanning` no longer exposes
      provider-facing direct-contraction statement-builder authority.
- [ ] Bounded authority scans over touched RVV files show no legacy-i32,
      descriptor/source-front-door/artifact-name/route-id/exact-intrinsic/
      common-EmitC/direct-route-entry/pre-realized-fixture-only authority drift.
- [ ] `git diff --check` passes.
- [ ] Focused C++ plugin build/test and `tcrv-opt` / `tcrv-translate` build
      pass.
- [ ] `check-tianchenrv` passes, or an exact blocker is recorded.
- [ ] `ssh rvv` is not required because this round preserves statement
      construction semantics and does not claim new runtime/correctness/
      performance behavior.

## Out of Scope

- Migrating all migrated-route statement builders.
- Moving route analysis, materialization facts, operand-binding facts,
  route-control provider-plan facts, or direct-contraction provider-plan fact
  collection out of planning.
- Modifying common EmitC materialization, target artifact export semantics,
  selected-body realization, or route-family provider-plan verification.
- New op coverage, dtype/SEW/LMUL/policy expansion, Linalg/Vector/StableHLO
  frontend work, IME, Offload, TensorExt, Toy, Template, future plugin work,
  dashboards, broad smoke matrices, or runtime performance claims.

## Technical Approach

The intended production flow after this task is:

```text
selected RVV variant
  -> typed/realized tcrv_rvv body
  -> route analysis / materialization facts / math operand-binding facts
  -> direct-contraction provider plan in RVVEmitCRoutePlanning
  -> direct-contraction statement owner in RVVEmitCStatementPlanOwners
  -> owner-produced provider-ready pre-loop and loop statements
  -> RVVEmitCRouteProvider neutral TCRVEmitCLowerableRoute assembly
  -> common EmitC materialization
```

Implementation will move only the direct-contraction source provenance, call
opaque step assembly, pre-loop/loop step appenders, and route statement-plan
builder into `RVVEmitCStatementPlanOwners.cpp`. Planning will keep the provider
plan and rename its local direct-contraction fact validators away from
`StatementPlan` terminology so scans do not present planning as the statement
construction owner.

## Validation Plan

1. Build and run focused C++ plugin coverage:
   `cmake --build build --target tianchenrv-rvv-extension-plugin-test -j2`
   and `./build/bin/tianchenrv-rvv-extension-plugin-test`.
2. Build touched production tools:
   `cmake --build build --target tcrv-opt tcrv-translate -j2`.
3. Run representative generated-bundle dry-runs for one plain/scalar MAcc route
   and one widening dot-reduce route, checking `route_entry_realization: false`.
4. Run bounded provider/owner/planning scans proving authority moved and
   provider neutrality remained intact.
5. Run bounded authority drift scans over touched RVV production/test files.
6. Run `git diff --check`.
7. Run `cmake --build build --target check-tianchenrv -j2`, or record the
   exact blocker.

## Completion Evidence

- Direct-contraction statement construction now lives in
  `lib/Plugin/RVV/EmitC/RVVEmitCStatementPlanOwners.cpp`.
  The owner-local builder
  `buildDirectContractionRouteStatementPlanFromProviderPlan(...)` constructs
  provider-ready pre-loop `TCRVEmitCCallOpaqueStep` entries and the
  `TCRVEmitCForLoop` for widening MAcc, widening dot-reduce, strided-input
  widening dot-reduce, computed-mask widening dot-reduce, and computed-mask
  strided-input widening dot-reduce.
- `RVVEmitCStatementPlanOwners.cpp` direct-contraction registry now points to
  the owner-local builder. It no longer calls a planning-owned
  `buildRVVSelectedBodyDirectContractionRouteStatementPlanFromProviderPlan(...)`
  hook.
- `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h` no longer declares a
  provider-facing direct-contraction statement-builder API.
- `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp` no longer defines
  `buildDirectContractionRouteStatementPlan(...)` or
  `buildRVVSelectedBodyDirectContractionRouteStatementPlanFromProviderPlan(...)`.
  It still owns `getRVVSelectedBodyDirectContractionRouteProviderPlan(...)` and
  direct-contraction provider-plan fact validation, including ABI bindings,
  route-control plan, math operand-binding facts, materialized leaves, typed
  config, and selected capability facts.
- `RVVEmitCRouteProvider.cpp` remained unchanged and neutral. A bounded scan for
  direct-contraction statement-building terms over that file returned no hits.
- Focused C++ coverage passed:
  `cmake --build build --target tianchenrv-rvv-extension-plugin-test -j2` and
  `./build/bin/tianchenrv-rvv-extension-plugin-test`.
- Production tools built:
  `cmake --build build --target tcrv-opt tcrv-translate -j2`.
- Representative generated-bundle dry-run passed:
  `python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run
  --pre-realized-selected-body --op-kind macc_add --op-kind
  scalar_broadcast_macc_add --op-kind widening_dot_reduce_add --artifact-root
  artifacts/tmp/stage2_rvv_direct_contraction_statement_plan_owner_boundary
  --run-id owner-boundary-dry-run --overwrite`.
  Evidence files report `dry_run: true` and
  `local_bundle_generation.route_entry_realization: false` for all three
  routes.
- Bounded planning scan for direct-contraction statement-builder authority left
  only the shared `RVVSelectedBodyDirectContractionRouteStatementPlan` data
  structure in `RVVEmitCRoutePlanning.h`; direct construction helpers and
  provider-facing builder hooks moved out.
- Diff-only authority scan over touched RVV files found no newly added
  legacy-i32, descriptor/source-front-door/source-artifact, artifact-name,
  route-id, exact-intrinsic, common-EmitC, direct-route-entry, status/supported,
  or pre-realized-fixture-only authority drift.
- `git diff --check` passed.
- `cmake --build build --target check-tianchenrv -j2` passed: 464/464 tests.
- `ssh rvv` was not run because this round preserves statement construction
  semantics and makes no new runtime, correctness, or performance claim.
