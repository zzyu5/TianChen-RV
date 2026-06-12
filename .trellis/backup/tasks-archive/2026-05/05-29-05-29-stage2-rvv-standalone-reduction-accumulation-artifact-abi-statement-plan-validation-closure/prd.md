# Stage2 RVV standalone reduction/accumulation artifact ABI statement-plan validation closure

## Goal

Close the target artifact ABI statement-plan validation boundary for active RVV
standalone reduction/accumulation artifact consumers. The target artifact
route-family validator must accept plain, computed-mask, and runtime-scalar
computed-mask standalone reduction routes only after validating the rebuilt
provider description and provider-built `TCRVEmitCLowerableRoute` statement
plan.

Callee or intrinsic spelling may remain one checked field inside an exact
provider-built statement. It must not authorize vector load, reduction,
store, seed splat, mask compare/merge, inactive neutralization, or runtime
scalar splat acceptance by itself.

This task is complete only if production validation in
`lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp` moves for the
standalone reduction/accumulation artifact consumer and focused C++ target
artifact tests prove the new fail-closed boundary.

## Direction Source

- Direction title: `Switch: Stage2 RVV standalone reduction/accumulation
  artifact ABI statement-plan validation closure`.
- Module owner: RVV target artifact route-family validator for standalone
  reduction/accumulation statement plans.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean.
- Initial HEAD: `264c167d rvv: close macc ABI statement validation`.
- `.trellis/.current-task` was absent, so this task was created from the
  supplied Hermes Direction Brief before source edits.
- Serial worker constraint from the brief: one Codex worker, no spawned,
  parallel-agent, or multi-agent workflow.

## Current Repository Facts

- Commit `264c167d` closed non-widening MAcc target artifact ABI
  statement-plan validation by replacing callee-presence acceptance with exact
  provider-derived ABI, operand, statement, result, loop, and selected typed
  RVV provenance validation.
- Archived standalone reduction/accumulation artifact validator migration
  already moved this family into
  `RVVTargetArtifactRouteFamilyValidation.cpp` and established provider-fact
  and candidate-mirror ownership.
- Archived vector reduction and standalone min/max closures show the same
  target-side pattern: rebuild the provider route, validate ABI order,
  headers, type mappings, operand bindings, source/result channel facts,
  loop/setvl facts, statement leaves, result names/types, and selected typed
  RVV provenance before accepting target artifact metadata.
- The next known gap is standalone reduction/accumulation loop payload
  validation that still authorizes required leaves by `routeLoopContainsCallee`
  for vector load, reduction intrinsic, result store, scalar seed splat,
  computed-mask compare/merge/inactive neutralization, and runtime-scalar RHS
  splat.
- Existing target/export tests already exercise standalone reduction artifact
  candidates and route-clone mutation style; this round should extend those
  tests instead of creating a broad smoke matrix.

## Requirements

1. Keep rebuilt provider route facts as authority. Candidate metadata and
   mirrors may only be checked after provider route construction.
2. Validate provider-built route-family plan IDs, provider-supported mirror,
   target leaf profile, required headers, C type mapping, ABI mappings, route
   operand binding plan/summary, runtime ABI order/count/roles, route-control
   facts, and selected typed RVV provenance.
3. Validate standalone reduction source/work and scalar accumulator/result
   channels, including vector type names, C types, SEW/LMUL relation, scalar
   seed, accumulator/result layout, lane-0 scalar result storage, and store VL.
4. Validate exact pre-loop full-chunk `setvl` from runtime `n` and exactly one
   runtime AVL/VL loop whose loop bounds, remaining AVL, induction, step, and
   loop VL match provider facts.
5. For plain standalone reductions, validate exact statement sequence and
   facts for loop `setvl`, source vector load, scalar seed or accumulator
   handling, reduction compute, scalar result extraction/store, output store
   pointer/value/VL, operand expressions, result names, result C types, and
   source provenance.
6. For computed-mask standalone reductions, additionally validate compare
   operand loads, mask result, inactive neutral source/work-channel splat,
   mask merge or pass-through binding, reduction source after neutralization,
   scalar accumulator/result handling, and output store.
7. For runtime-scalar computed-mask standalone reductions, additionally
   validate runtime scalar ABI binding, RHS scalar value expression, scalar
   splat statement, same-VL mask producer binding, mask/pass-through relation,
   reduction source, and scalar result store.
8. Fail closed for metadata-only provider support, stale mirror fields, wrong
   runtime ABI order/count/roles, missing selected typed RVV provenance, wrong
   setvl/AVL/VL relation, wrong scalar seed or accumulator layout, wrong
   mask/pass-through binding, wrong RHS scalar binding, wrong statement order,
   wrong callee paired with otherwise-valid metadata, missing required
   statement, stale source/result C type mapping, stale family-plan mirrors,
   route-id/artifact-name/source-front-door/script-derived claims, exact
   intrinsic-only claims, descriptor residue, direct-route-entry-only claims,
   pre-realized-fixture-only claims, legacy-i32 authority, or common EmitC
   semantic invention.
9. Do not remove `routeLoopContainsCallee` globally. Remove or demote it only
   for the standalone reduction/accumulation owner where exact statement-plan
   validation replaces it.
10. Preserve existing behavior for MAcc, widening conversion, base memory,
    segment2, vector reduction, standalone min/max, widening dot/MAcc, and
    other route families.

## Acceptance Criteria

- [x] PRD, implement context, and check context accurately describe the
      bounded module, specs, precedent, non-goals, and validation plan.
- [x] `RVVTargetArtifactRouteFamilyValidation.cpp` validates exact rebuilt
      provider route facts and statement-plan facts for active standalone
      reduction/accumulation artifact consumers.
- [x] Standalone reduction/accumulation validation no longer accepts loop
      payload facts through `routeLoopContainsCallee` / callee-presence checks.
- [x] Positive C++ target/export coverage proves valid standalone
      reduction/accumulation artifacts still pass for plain, computed-mask,
      and runtime-scalar computed-mask routes covered by the current provider.
- [x] Focused negative route/provider/candidate mutations fail closed for
      metadata-only provider support, stale provider mirrors, stale candidate
      mirrors, wrong ABI order/count/roles, wrong setvl/AVL/VL facts, missing
      selected typed RVV provenance, wrong source/result C type mapping,
      stale seed/accumulator/result layout, wrong source load, wrong reduction
      operands/result, wrong scalar result store, wrong mask compare/merge,
      wrong inactive neutral/pass-through binding, wrong runtime scalar splat,
      wrong output pointer/value/VL, missing required statement, and wrong
      intrinsic/callee paired with plausible metadata.
- [x] Direct-route-entry-only, artifact-name/script-derived, route-id-derived,
      exact-intrinsic-as-authority, common-EmitC semantic invention,
      descriptor/source-front-door residue, pre-realized-fixture-only,
      callee-presence-only, and legacy-i32-derived authority remain fail-closed
      or absent.
- [x] Focused target artifact test passes.
- [x] `git diff --check` passes.
- [x] `check-tianchenrv` passes, or an exact blocker is recorded.
- [x] If complete, the task is finished, archived, journaled, and committed
      coherently.

## Technical Approach

1. Inspect the current standalone reduction/accumulation validators and all
   remaining `routeLoopContainsCallee` uses in
   `RVVTargetArtifactRouteFamilyValidation.cpp`.
2. Reuse `validateRVVProviderBuiltRouteStep` or the adjacent exact
   statement-plan helpers to validate callee, operand expressions, operand C
   types, result names, and result C types for every required pre-loop and loop
   statement.
3. Derive expected standalone reduction statement facts from the rebuilt
   provider description and route facts, including runtime ABI parameters,
   source/work channel, scalar-result channel, accumulator/seed/result layout,
   mask producer facts, runtime-scalar facts, and store VL facts.
4. Replace standalone reduction/accumulation callee-presence checks with exact
   statement sequence and leaf validation while leaving unrelated families
   untouched.
5. Extend `test/Target/TargetArtifactExportTest.cpp` with focused positive and
   route-clone negative coverage in the existing target artifact validation
   style.
6. Run focused build/test, authority scan, `git diff --check`, and
   `check-tianchenrv` if feasible.

## Out of Scope

- Do not start widening conversion, segment2 memory, compare/select, base
  memory movement, MAcc rework, selected-body realization migration,
  direct-route-entry demotion, high-level frontend/Linalg work, broad smoke
  matrices, dashboard/report work, or evidence-only tasks.
- Do not add new standalone reduction operation coverage beyond validation
  hardening for the active provider route families.
- Do not move RVV semantics into common EmitC/export code.
- Do not use generated artifacts, route ids, metadata, ABI strings, artifact
  names, scripts, exact intrinsic spellings, descriptors, source-front-door
  markers, direct route entries, pre-realized fixtures, or legacy i32 helper
  names as route authority.
- Do not implement compiler core, dialects, passes, plugin registry,
  capability model, lowering, or emission as Python data structures.

## Validation Plan

1. `ninja -C build bin/tianchenrv-target-artifact-export-test`
2. `./build/bin/tianchenrv-target-artifact-export-test`
3. Bounded authority scan over touched production and test files for
   forbidden acceptance sources and remaining standalone reduction
   `routeLoopContainsCallee` acceptance.
4. `git diff --check`
5. `cmake --build build --target check-tianchenrv -j2`, or exact blocker.

Generated-bundle or `ssh rvv` evidence is not required unless this round
changes runtime, materialization, correctness, or performance semantics rather
than target artifact validation only.

## Spec Update Judgment

Initial expectation: no `.trellis/spec/` update is required. The existing RVV
plugin, EmitC route, and testing specs already require provider-derived typed
RVV route facts, target artifact route-family validators, mirror-only
metadata, common EmitC neutrality, standalone reduction source/result channel
validation, and evidence-bound runtime claims. This task applies those
existing contracts to a remaining target artifact validator acceptance hole.

Final judgment: no `.trellis/spec/` update was needed. The implementation
applies existing target artifact route-family validator, standalone reduction
source/result channel, route-control/provider-built route, and mirror-only
metadata contracts without changing long-term architecture rules.

## Completion Evidence

- Production owner changed:
  `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`.
- Focused test owner changed:
  `test/Target/TargetArtifactExportTest.cpp`.
- The standalone reduction/accumulation target artifact validator now validates
  exact provider-built runtime-scalar computed-mask loop statements:
  loop setvl, compare lhs load, RHS scalar splat, payload source load,
  compare predicate, inactive neutral splat, inactive-lane merge, loop scalar
  seed splat, signed min/max/add reduction, and scalar-result store.
- Existing exact validation for plain standalone and vector computed-mask
  standalone reduction remains active for pre-loop setvl, scalar seed splat,
  initial scalar-result store, loop setvl, source/compare loads, inactive
  neutralization, merge/pass-through, reduction, scalar-result store, C types,
  result names, runtime AVL/VL, store VL `1`, and selected typed RVV source
  provenance.
- The production standalone reduction/accumulation statement validator no
  longer uses `routeLoopContainsCallee` / callee-presence checks as acceptance
  for vector load, reduction, store, scalar seed splat, compare, merge, or RHS
  scalar splat loop payload facts.
- Positive target artifact coverage still accepts plain standalone
  reduce_add/min/max, computed-mask standalone reduce_add/min/max, and
  runtime-scalar computed-mask standalone reduce_add/min/max, including LMUL
  m2 and i64 runtime-scalar cases already present in the test fixture set.
- New negative route-clone coverage proves runtime-scalar computed-mask
  standalone reduction fails closed for stale RHS scalar splat operand/result,
  stale RHS scalar C type, stale payload source pointer, stale compare RHS
  operand, stale inactive-lane merge mask, stale reduction input, stale
  scalar-result store VL, and stale min inactive neutral literal. Existing
  provider/candidate negatives continue to cover stale typed op, runtime ABI
  order/roles, scalar-result channel, provider mirror, binding plan/summary,
  inactive-lane contract, RHS splat intrinsic, compare intrinsic, merge
  intrinsic, reduction intrinsic, scalar-result store intrinsic, runtime
  scalar producer source, scalar carry contract, and mirror mutations.
- Generated-bundle and `ssh rvv` evidence were not applicable: this round
  changed only target artifact validation and C++ target/export coverage, not
  runtime, materialization, correctness, or performance semantics.
- Checks:
  - `ninja -C build bin/tianchenrv-target-artifact-export-test`
  - `./build/bin/tianchenrv-target-artifact-export-test`
  - `git diff --check`
  - `cmake --build build --target check-tianchenrv -j2`, 459/459 passed
- Bounded authority scan:
  - The standalone reduction validator slice no longer contains
    `routeLoopContainsCallee`.
  - Added production lines contain no metadata-derived, route-id-derived,
    descriptor-derived, ABI-string-derived, script-derived,
    artifact-name-derived, common-EmitC-derived, source-front-door-derived,
    exact-intrinsic-only, direct-route-entry-only, pre-realized-fixture-only,
    callee-presence-only, or legacy-i32-derived executable authority.
  - Added `metadata_*` strings appear only in negative route-clone tests.

## Technical Notes

Specs and precedent read before implementation:

- `.trellis/spec/index.md`
- `.trellis/spec/extension-plugins/index.md`
- `.trellis/spec/extension-plugins/rvv-plugin.md`
- `.trellis/spec/lowering-runtime/index.md`
- `.trellis/spec/lowering-runtime/emitc-route.md`
- `.trellis/spec/testing/index.md`
- `.trellis/spec/testing/mlir-testing-contract.md`
- `.trellis/tasks/archive/2026-05/05-29-stage2-rvv-macc-artifact-abi-statement-plan-closure/prd.md`
- `.trellis/tasks/archive/2026-05/05-28-05-28-stage2-rvv-standalone-reduction-accumulation-artifact-validator-migration/prd.md`
- `.trellis/tasks/archive/2026-05/05-29-05-29-stage2-rvv-vector-reduction-artifact-abi-statement-plan-closure/prd.md`
