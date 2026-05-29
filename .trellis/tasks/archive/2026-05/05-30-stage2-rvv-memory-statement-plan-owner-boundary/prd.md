# Stage2 RVV migrated memory statement-plan owner boundary extraction

## Goal

Move provider-ready statement construction for the migrated RVV memory families
out of monolithic `RVVEmitCRoutePlanning` and into the RVV EmitC
statement-plan owner boundary. The target families are base memory movement,
computed-mask memory, and segment2 memory. Planning remains responsible for
neutral route analysis, typed config/capability/runtime validation,
materialization facts, memory operand-binding facts, segment2 provider-plan
facts, and shared data structures. `RVVEmitCRouteProvider` remains a neutral
`TCRVEmitCLowerableRoute` assembler.

## Direction Source

- Direction title: `Expand: Stage2 RVV migrated memory statement-plan owner
  boundary extraction`.
- Module owner: RVV selected-body migrated memory statement-plan owner boundary.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean.
- Initial HEAD: `340809ce rvv: move direct contraction statements to owners`.
- `.trellis/.current-task` was absent, so this task was created from the
  Hermes Direction Brief before source edits.
- Serial worker constraint from the brief: one Codex worker; no subagents,
  spawned agents, parallel agents, or multi-agent workflows.

## What I Already Know

- The archived predecessor
  `.trellis/tasks/archive/2026-05/05-30-05-30-stage2-rvv-direct-contraction-statement-plan-owner-boundary`
  moved direct-contraction provider-ready statement construction into
  `RVVEmitCStatementPlanOwners.cpp` while leaving provider-plan fact collection
  in `RVVEmitCRoutePlanning.cpp`.
- Current `RVVEmitCStatementPlanOwners.cpp` owns the migrated statement-plan
  owner registry and has exact-one selection diagnostics for migrated families.
- Current migrated memory owner entries for `base memory movement`,
  `computed-mask memory`, and `segment2 memory` still call
  `buildRVVSelectedBody*MemoryMigratedRouteStatementPlan(...)` hooks declared
  in `RVVEmitCRoutePlanning.h` and implemented in
  `RVVEmitCRoutePlanning.cpp`.
- Current `RVVEmitCRoutePlanning.cpp` implements the provider-ready memory
  statement sequences for base memory movement, computed-mask memory, and
  segment2 memory. Those sequences include `setvl`, loads, compares, masks,
  strided/indexed addressing, segment tuple operations, segment loads/stores,
  and stores.
- Current planning should continue to own memory route analysis,
  materialization facts, memory operand-binding facts, mask/tail policy
  provider-plan facts, route-control provider facts, and segment2
  route-family provider-plan facts where they validate typed body/config,
  selected target capability, runtime ABI, and same-analysis dependencies.
- Existing C++ coverage in `test/Plugin/RVVExtensionPluginTest.cpp` exercises
  migrated statement-plan owner registry membership, memory statement plans,
  segment2 route-family planning owners, provider attachment, and missing/stale
  fact diagnostics.

## Requirements

1. Move provider-ready statement construction for base memory movement,
   computed-mask memory, and segment2 memory into
   `RVVEmitCStatementPlanOwners.cpp` or an owner-local package under the same
   boundary.
2. Remove memory migrated statement-plan builder declarations from
   `RVVEmitCRoutePlanning.h`; planning must not expose
   `buildRVVSelectedBodyBaseMemoryMovementMigratedRouteStatementPlan(...)`,
   `buildRVVSelectedBodyComputedMaskMemoryMigratedRouteStatementPlan(...)`,
   `buildRVVSelectedBodySegment2MemoryMigratedRouteStatementPlan(...)`, or an
   equivalent memory statement-construction authority.
3. Remove or demote memory statement-plan construction definitions from
   `RVVEmitCRoutePlanning.cpp`; planning may retain neutral/fact-oriented
   helpers only when they are genuinely shared validation or provider-plan
   facts rather than provider-ready statement assembly.
4. Keep base memory semantics unchanged for strided load/unit store, unit
   load/strided store, indexed gather/unit store, indexed scatter/unit load,
   static-mask unit load/store, and static-mask unit store.
5. Keep computed-mask memory semantics unchanged for runtime-scalar
   computed-mask store/load-store, computed-mask unit load/store, computed-mask
   strided store, computed-mask strided load/unit store, computed-mask indexed
   gather/unit store, and computed-mask indexed scatter/unit load.
6. Keep segment2 memory semantics unchanged for plain deinterleave/interleave,
   computed-mask segment2 load/store, and computed-mask segment2 update.
7. Preserve segment2 route-family planning owner behavior: segment2
   classification and provider-plan facts remain owner-selected before the
   segment2 statement plan consumes them.
8. Keep `RVVEmitCRouteProvider.cpp` neutral: it may obtain route analysis,
   materialization facts, operand-binding facts, route-control/provider-plan
   facts, instantiate `TCRVEmitCLowerableRoute`, record neutral headers/type
   mappings/ABI mappings/source provenance, and attach owner-returned
   statements.
9. Do not add new RVV operations, dtype/LMUL coverage, selected-body
   realization cases, direct route-entry support, source-front-door routes,
   high-level frontend lowering, performance tuning, dashboards, broad smoke
   matrices, or runtime/correctness/performance claims.

## Acceptance Criteria

- [ ] Memory-family migrated owner entries in
      `RVVEmitCStatementPlanOwners.cpp` construct or own the base memory
      movement, computed-mask memory, and segment2 memory statement plans.
- [ ] `RVVEmitCRoutePlanning.h` no longer declares memory migrated
      statement-plan builder hooks as provider-facing authority.
- [ ] `RVVEmitCRoutePlanning.cpp` no longer defines the three memory migrated
      builder hooks as statement-construction authority.
- [ ] Planning retains only justified neutral facts and shared data structures
      for memory route analysis, materialization facts, operand-binding facts,
      route-control facts, mask/tail policy facts, and segment2 provider-plan
      validation.
- [ ] Owner-registry diagnostics still prove exact-one owner selection,
      duplicate-owner fail-closed behavior, and missing/incomplete owner hook
      fail-closed behavior.
- [ ] Focused C++ coverage proves owner-local memory statement-plan selection
      and provider neutrality for representative base memory, computed-mask
      memory, and segment2 memory routes.
- [ ] Representative generated-bundle dry-runs pass for base memory,
      computed-mask memory, and segment2 memory paths covered by this task,
      with no `route_entry_realization` or metadata-authority drift.
- [ ] Bounded scans show `RVVEmitCRouteProvider.cpp` does not manually sequence
      memory family statement-plan getters or rebuild memory statement
      sequences.
- [ ] Bounded scans show touched RVV planning/provider/owner/test files do not
      introduce legacy-i32, descriptor, source-front-door/source-artifact,
      route-id, artifact-name, exact-intrinsic, common-EmitC,
      direct-route-entry-only, status/supported, or pre-realized-fixture-only
      authority.
- [ ] `git diff --check` passes.
- [ ] Focused C++ plugin build/test and `tcrv-opt` / `tcrv-translate` build
      pass.
- [ ] `check-tianchenrv` passes, or an exact blocker is recorded.
- [ ] `ssh rvv` is not required unless this round claims new executable,
      runtime, correctness, or performance behavior.

## Technical Approach

The intended production flow after this task is:

```text
selected RVV variant
  -> typed/realized tcrv_rvv memory body
  -> route analysis / materialization facts / memory operand-binding facts
  -> route-control and segment2 provider-plan facts where applicable
  -> memory-family statement-plan owner in RVVEmitCStatementPlanOwners
  -> owner-produced provider-ready pre-loop and loop statements
  -> RVVEmitCRouteProvider neutral TCRVEmitCLowerableRoute assembly
  -> common EmitC materialization
```

Implementation will move the memory statement-plan wrappers and provider-ready
statement construction for the three memory families into the owner boundary.
The owner registry will continue to select exactly one migrated family owner
for each migrated route. Planning will keep data structures and neutral
fact-validation helpers required by the owner builders, but exported memory
migrated builder hooks will no longer make planning the statement-construction
authority.

## Validation Plan

1. Build and run focused C++ plugin coverage:
   `cmake --build build --target tianchenrv-rvv-extension-plugin-test -j2`
   and `./build/bin/tianchenrv-rvv-extension-plugin-test`.
2. Build touched production tools:
   `cmake --build build --target tcrv-opt tcrv-translate -j2`.
3. Run representative generated-bundle dry-runs for base memory,
   computed-mask memory, and segment2 memory routes, checking
   `route_entry_realization: false` and typed selected-body provenance.
4. Run bounded provider/owner/planning scans proving memory statement
   construction moved and provider neutrality remained intact.
5. Run bounded authority drift scans over touched RVV production/test files.
6. Run `git diff --check`.
7. Run `cmake --build build --target check-tianchenrv -j2`, or record the
   exact blocker.

## Out of Scope

- Moving route analysis, materialization facts, memory operand-binding facts,
  route-control provider-plan facts, mask/tail provider-plan facts, segment2
  route-family planning owner facts, or shared data structures out of planning.
- Modifying common EmitC materialization, target artifact export semantics,
  selected-body realization, or route-family provider-plan verification.
- New op coverage, dtype/SEW/LMUL/policy expansion, Linalg/Vector/StableHLO
  frontend work, IME, Offload, TensorExt, Toy, Template, future plugin work,
  dashboards, broad smoke matrices, or runtime performance claims.

## Completion Evidence

- Memory-family migrated owner entries now own statement-plan construction for
  base memory movement, computed-mask memory, and segment2 memory through the
  RVV statement-plan owner boundary.
- `RVVEmitCRoutePlanning.h` no longer exposes memory migrated statement-plan
  builder hooks or memory statement-plan getters as planning-owned provider
  authority.
- `RVVEmitCRoutePlanning.cpp` no longer implements the memory migrated
  statement-plan builder hooks; it retains neutral provider-plan validation,
  typed facts, ABI/materialization facts, and shared statement-plan structs.
- `RVVEmitCStatementPlanOwners.h` declares the memory owner boundary APIs, and
  `RVVEmitCMemoryStatementPlanOwners.cpp` implements the owner-local base
  memory, computed-mask memory, and segment2 statement plans.
- `RVVEmitCRouteProvider.cpp` was not changed; provider behavior remains
  neutral route assembly and does not become the memory statement-construction
  authority.
- Spec update review: no `.trellis/spec/**` edit was needed because
  `.trellis/spec/extension-plugins/rvv-plugin.md` already contains the durable
  Stage2 selected-body realization, memory movement, statement-plan ownership,
  provider neutrality, fail-closed diagnostic, mirror metadata, and common
  EmitC neutrality rules implemented by this task.

Checks and evidence:

- `cmake --build build --target tianchenrv-rvv-extension-plugin-test -j2`
  passed.
- `./build/bin/tianchenrv-rvv-extension-plugin-test` passed.
- `cmake --build build --target tcrv-opt tcrv-translate -j2` passed.
- Representative generated-bundle dry-run passed for
  `strided_load_unit_store`,
  `computed_masked_strided_load_unit_store`, and
  `computed_masked_segment2_update_unit_load` under
  `artifacts/tmp/stage2_rvv_memory_statement_plan_owner_boundary/owner-boundary-dry-run`.
- Generated-bundle evidence kept `route_entry_realization: false` for the
  representative base memory, computed-mask memory, and segment2 paths.
- Bounded scans showed planning exposes only shared memory statement-plan
  structs, not memory migrated statement-plan builder/getter APIs.
- Bounded scans showed `RVVEmitCRouteProvider.cpp` does not call the memory
  statement-plan builders/getters or rebuild memory statement sequences.
- Bounded diff-only authority scans found no new legacy-i32,
  source-front-door/source-artifact, descriptor, route-id, artifact-name,
  exact-intrinsic, common-EmitC, direct-route-entry-only,
  status/supported/provider-supported, or pre-realized-fixture-only authority
  additions.
- `git diff --check` passed.
- `cmake --build build --target check-tianchenrv -j2` passed 464/464.
