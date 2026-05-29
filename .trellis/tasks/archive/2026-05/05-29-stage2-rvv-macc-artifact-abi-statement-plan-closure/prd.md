# Stage2 RVV MAcc artifact ABI statement-plan validation closure

## Goal

Close the target artifact ABI statement-plan validation boundary for the
production-active non-widening RVV MAcc selected-body routes:
`macc_add`, `scalar_broadcast_macc_add`, `computed_masked_macc_add`, and
`runtime_scalar_computed_masked_macc_add`.

The target artifact route-family validator must consume rebuilt provider route
facts and the provider-built `TCRVEmitCLowerableRoute` statement plan before
accepting artifact/export claims. Candidate metadata, route ids, artifact
names, ABI strings, exact intrinsic spellings, and callee presence are mirrors
or negative-test inputs only. They must not authorize MAcc route support or
artifact acceptance.

This task is complete only if production validation in
`lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp` moves for the MAcc
artifact consumer. Tests alone are not sufficient.

## Direction Source

- Direction title: `Switch: Stage2 RVV MAcc artifact ABI statement-plan validation closure`.
- Module owner: RVV target artifact route-family validator for non-widening
  MAcc selected-body routes.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean.
- Initial HEAD: `5f6eea25 rvv: close base memory ABI statement validation`.
- `.trellis/.current-task` was absent, so this task was created from the
  supplied Hermes Direction Brief before source edits.
- Serial worker constraint from the brief: one Codex worker, no spawned,
  parallel-agent, or multi-agent workflow.

## Current Repository Facts

- The recent base-memory artifact ABI closure changed
  `RVVTargetArtifactRouteFamilyValidation.cpp` so the target artifact consumer
  validates exact provider-built statement facts instead of accepting loop
  payloads by callee presence.
- The existing widening MAcc and widening dot-reduction validators already show
  the target-side pattern: validate exact pre-loop setvl, runtime AVL/VL loop,
  per-step operands, result names, result C types, ABI pointer bindings, store
  facts, and selected typed RVV provenance from the rebuilt provider route.
- The current MAcc target artifact consumer already checks route id, provider
  supported mirror, route operand binding facts, accumulator/result layout,
  per-subfamily family-plan mirrors, headers, type mappings, ABI mappings,
  candidate mirrors, and selected typed RVV provenance presence.
- The remaining production gap is
  `validateRVVMAccRouteStatementPlan`: after checking non-exact pre-loop setvl,
  loop setvl, loop bounds, runtime `n`, and provenance presence, it still
  accepts core payload facts through `routeLoopContainsCallee` for vector load,
  MAcc arithmetic, store, scalar broadcast, compare, masked merge, and runtime
  scalar splat. That proves a callee string appears somewhere, but not exact
  statement order, operands, result names/types, pointer expressions, VL use,
  mask/pass-through facts, runtime ABI order, or selected-body source facts.
- `TargetArtifactExportTest.cpp` already has positive fixtures for plain MAcc,
  scalar-broadcast MAcc, computed-mask MAcc, runtime-scalar computed-mask MAcc,
  and widening MAcc, plus route clone mutation style from adjacent validators.

## Requirements

1. Keep provider-built route facts as authority. Candidate metadata may only be
   checked as mirrors after provider route construction.
2. Replace MAcc callee-presence acceptance with exact rebuilt route statement
   validation for:
   - full-chunk pre-loop `setvl` from runtime `n`;
   - exactly one runtime AVL/VL loop with provider-derived induction, lower
     bound, upper bound, step, and runtime `n` relation;
   - loop `setvl` from remaining runtime AVL;
   - plain MAcc lhs/rhs/acc loads, MAcc operands/result, output store pointer,
     value, and VL;
   - scalar-broadcast MAcc lhs load, RHS scalar splat, accumulator load, MAcc
     operands/result, output store pointer, value, and VL;
   - computed-mask MAcc compare lhs/rhs loads, payload lhs/rhs loads,
     accumulator/pass-through load, compare mask creation, active MAcc,
     masked merge/pass-through, output store pointer, value, and VL;
   - runtime-scalar computed-mask MAcc compare lhs load, runtime scalar splat,
     payload lhs/rhs loads, accumulator/pass-through load, compare mask
     creation, active MAcc, masked merge/pass-through, output store pointer,
     value, and VL.
3. Validate exact operand expressions and C types for provider ABI pointers,
   runtime scalar values, loop induction, runtime AVL/VL, vector values,
   mask values, pass-through values, and output stores.
4. Validate exact result names and result C types for full-chunk VL, loop VL,
   vector loads, scalar splats, compare masks, active MAcc results, masked
   merge results, and final MAcc result channels.
5. Require selected typed RVV source provenance on every required pre-loop and
   loop statement.
6. Preserve existing checks for runtime ABI order/roles, route operand binding
   summary, memory form, accumulator/result layout, route-control plan, family
   plan mirrors, provider support mirror labels, headers, type mappings, ABI
   mappings, stale non-family facts, and stale candidate mirrors.
7. Add focused target/export C++ tests proving positive acceptance for all four
   supported non-widening MAcc forms and fail-closed behavior for stale route
   clones that mutate setvl, load, MAcc, store, scalar broadcast, compare,
   masked merge, runtime scalar splat, operands, result names/types, pointer
   expressions, policy/mask facts, runtime ABI order, provider mirrors, and
   selected-body provenance.
8. Keep route construction and EmitC semantics plugin-owned. Common EmitC/export
   must not infer RVV semantics, dtype/config, ABI roles, route support,
   statement order, mask policy, MAcc operands, or intrinsic choices.
9. Generated-bundle dry-run or existing executable non-regression should be run
   only if needed to prove the hardened target artifact validator still accepts
   real provider routes. No `ssh rvv` evidence is required unless this round
   changes runtime, correctness, or performance claims.

## Acceptance Criteria

- [x] PRD, implement context, and check context accurately describe the bounded
      module, specs, precedent, non-goals, and validation plan.
- [x] `RVVTargetArtifactRouteFamilyValidation.cpp` validates exact rebuilt
      provider route statement order, operands, result names, result C types,
      runtime AVL/VL facts, pointer expressions, mask/pass-through facts,
      runtime scalar splat facts, ABI mapping/order, and selected typed RVV
      provenance for the four non-widening MAcc forms.
- [x] The production MAcc artifact validator no longer uses
      `routeLoopContainsCallee` / callee-presence checks as acceptance for MAcc
      loop payload facts.
- [x] Positive C++ target/export coverage proves artifact acceptance for
      `macc_add`, `scalar_broadcast_macc_add`,
      `computed_masked_macc_add`, and
      `runtime_scalar_computed_masked_macc_add`.
- [x] Focused route-clone mutations fail closed for stale or missing pre-loop
      setvl AVL, loop setvl remaining AVL, lhs/rhs/acc loads, MAcc operands and
      result, output store pointer/value/VL, scalar broadcast operand/result,
      compare operands/result, masked merge operands/result, runtime scalar
      splat operand/result, runtime ABI order/roles, type mappings, stale
      candidate mirrors, stale provider mirrors, stale family mirrors, and
      selected typed RVV provenance.
- [x] Direct-route-entry-only, artifact-name/script-derived, route-id-derived,
      exact-intrinsic-as-authority, common-EmitC semantic invention,
      descriptor/source-front-door residue, pre-realized-fixture-only,
      callee-presence-only, and legacy-i32-derived authority remain fail-closed
      or absent.
- [x] Focused target artifact test(s) pass; generated-bundle dry-run or
      executable non-regression is recorded if needed.
- [x] Bounded touched-file authority scan finds no new central ad hoc,
      name-derived, metadata-derived, descriptor-derived, ABI-string-derived,
      script-derived, artifact-name-derived, common-EmitC-derived,
      source-front-door-derived, route-id-derived, exact-intrinsic-derived,
      direct-route-entry-only, pre-realized-fixture-only,
      callee-presence-only, or legacy-i32-derived executable authority.
- [x] `git diff --check` passes.
- [x] Focused checks pass; `check-tianchenrv` passes, or an exact blocker is
      recorded.
- [x] If complete, the task is finished, archived, journaled, and committed
      coherently.

## Technical Approach

1. Reuse `validateRVVProviderBuiltRouteStep` for exact callee, operand
   expression, operand C type, result name, and result C type checks.
2. Decode the selected MAcc ABI parameters from the rebuilt provider
   description by operation family:
   - plain: `lhs,rhs,acc,out,n`;
   - scalar broadcast: `lhs,rhs_scalar,acc,out,n`;
   - computed mask: `cmp_lhs,cmp_rhs,lhs,rhs,acc,out,n`;
   - runtime scalar computed mask: `cmp_lhs,rhs_scalar,lhs,rhs,acc,out,n`.
3. Encode expected loop statement sequences from provider facts:
   - plain MAcc: setvl, lhs load, rhs load, accumulator load, MAcc, store;
   - scalar broadcast MAcc: setvl, lhs load, scalar splat, accumulator load,
     MAcc, store;
   - computed-mask MAcc: setvl, compare lhs load, compare rhs load, payload lhs
     load, payload rhs load, accumulator load/pass-through, compare mask,
     active MAcc, masked merge, store;
   - runtime-scalar computed-mask MAcc: setvl, compare lhs load, runtime scalar
     splat, payload lhs load, payload rhs load, accumulator load/pass-through,
     compare mask, active MAcc, masked merge, store.
4. Extend `TargetArtifactExportTest.cpp` with positive and route-clone negative
   checks in the existing target artifact route-family validation style.
5. Run focused target artifact tests, optional generated-bundle dry-runs for the
   four non-widening MAcc forms, non-regression where cheap, authority scan,
   `git diff --check`, and `check-tianchenrv` if feasible.

## Out of Scope

- Do not add new MAcc variants, widening MAcc coverage, widening dot coverage,
  standalone reduction coverage, base-memory follow-up tests, source-front-door
  routes, high-level Linalg/frontend lowering, one-intrinsic wrapper dialects,
  runtime performance tuning, dashboards, reports, broad smoke matrices, or
  evidence-only tasks.
- Do not move RVV semantics into common EmitC/export code.
- Do not implement compiler core, dialects, passes, plugin registry,
  capability model, lowering, or emission as Python data structures.
- Do not make descriptors, ABI strings, artifact names, route ids, test names,
  metadata, exact intrinsic spellings, common EmitC code, source-front-door
  markers, direct route entries, pre-realized fixtures, callee presence, or
  legacy i32 helper names route authority.

## Validation Plan

1. `cmake --build build --target tianchenrv-target-artifact-export-test -j2`
2. `./build/bin/tianchenrv-target-artifact-export-test`
3. `cmake --build build --target tcrv-opt tcrv-translate -j2` if
   generated-bundle dry-runs need fresh tools.
4. Generated-bundle selected-boundary dry-run for the four non-widening MAcc
   forms if the target artifact test alone is insufficient to prove acceptance
   of real provider routes.
5. Focused non-regression for recent base-memory and widening MAcc/dot
   statement-plan validators if touched evidence or failures require it.
6. Bounded touched-file authority scan.
7. `git diff --check`
8. `cmake --build build --target check-tianchenrv -j2`, or exact blocker.

## Spec Update Judgment

Initial expectation: no `.trellis/spec/` update is required. The existing RVV
plugin, core dialect, EmitC route, and testing specs already require
provider-derived selected-body route facts, target artifact route-family
validators, mirror-only metadata, common EmitC neutrality, and evidence-bound
runtime claims. This task applies those existing contracts to the non-widening
MAcc target artifact consumer.

Final judgment: no `.trellis/spec/` update was needed. The implementation
applies the existing RVV plugin, EmitC route, core dialect, and testing
contracts without changing long-term architecture rules.

## Completion Evidence

- Production owner changed:
  `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`.
- Focused test owner changed:
  `test/Target/TargetArtifactExportTest.cpp`.
- The MAcc artifact validator now checks exact provider-derived runtime ABI
  order, route operand binding plan, target leaf profile,
  provider-supported mirror, required headers, C type mapping, accumulator and
  result layout, per-family route plan mirror, computed-mask/pass-through facts,
  selected typed RVV provenance, and statement-plan shape.
- Plain and scalar-broadcast MAcc require exact loop statement sequence:
  setvl, lhs load, rhs load or RHS scalar splat, accumulator load, MAcc compute,
  output store.
- Computed-mask and runtime-scalar computed-mask MAcc require exact loop
  statement sequence: setvl, compare lhs, compare rhs load or runtime scalar
  splat, payload lhs, payload rhs, accumulator/pass-through load, compare mask,
  active MAcc, masked merge/pass-through, output store.
- The production MAcc loop payload no longer accepts vector load, MAcc, store,
  scalar broadcast, compare, masked merge, or runtime scalar splat by
  `routeLoopContainsCallee`.
- Positive target artifact coverage was added for `macc_add`,
  `scalar_broadcast_macc_add`, `computed_masked_macc_add`, and
  `runtime_scalar_computed_masked_macc_add`.
- Negative route/provider/candidate mutations now prove fail-closed behavior for
  stale pre-loop setvl AVL, loop remaining AVL, load pointers, MAcc operands,
  store values/VL, scalar splat operand, compare RHS pointer, active MAcc
  payload, masked merge mask, selected-body provenance, provider mirror,
  binding plan, accumulation route-family plan, runtime scalar mask producer,
  runtime ABI mirror, and computed-mask producer mirror.
- Generated-bundle dry-run and `ssh rvv` evidence were not applicable: this
  round changed only target artifact validation and C++ target/export coverage,
  not runtime, correctness, or performance semantics.
- Checks:
  - `ninja -C build bin/tianchenrv-target-artifact-export-test`
  - `./build/bin/tianchenrv-target-artifact-export-test`
  - `git diff --check`
  - `cmake --build build --target check-tianchenrv -j2`, 459/459 passed
- Bounded authority scan:
  - MAcc validator diff only removes `routeLoopContainsCallee` payload
    acceptance and replaces it with exact provider-built route-step checks.
  - New `metadata-only` strings appear only in negative tests.
  - No new descriptor, source-front-door, artifact-name, direct-route-entry,
    pre-realized-fixture-only, legacy-i32, or exact-intrinsic-as-authority path
    was introduced in touched files.
