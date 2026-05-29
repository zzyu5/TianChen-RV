# Stage2 RVV segment2 memory artifact ABI statement-plan validation closure

## Goal

Close the target artifact ABI and statement-plan validation boundary for the
active RVV segment2-memory route family: computed-mask segment2 load/store/update
and plain segment2 deinterleave/interleave.

The RVV target artifact route-family validator must accept segment2-memory
artifacts only after validating rebuilt provider route facts, exact runtime ABI
roles/order, typed `tcrv_rvv` provenance, segment2/mask/layout facts, candidate
mirror metadata, and exact provider-built `TCRVEmitCLowerableRoute` statements.
Callee or intrinsic spelling may remain one field inside an exact statement
match, but callee presence must not authorize segment2 tuple construction,
field extraction, compare mask, vector load, segment load/store, update
arithmetic, ordinary stores, runtime AVL/VL, ABI order, selected typed
provenance, or artifact acceptance by itself.

This task is complete only if production validation in
`lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp` moves for the
segment2-memory artifact consumer and focused target/export C++ coverage proves
the new fail-closed boundary.

## Direction Source

- Direction title: `Switch: Stage2 RVV segment2 memory artifact ABI statement-plan validation closure`.
- Module owner: RVV target artifact route-family validation owner for the
  segment2-memory route family.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean.
- Initial HEAD: `76361035 rvv: close widening conversion ABI statement validation`.
- `.trellis/.current-task` was absent, so this task was created from the
  supplied Hermes Direction Brief before source edits.
- Serial worker constraint from the brief: one Codex worker; no subagents,
  spawned agents, parallel agents, or multi-agent workflows.

## Current Repository Facts

- Commit `76361035` closed widening conversion dtype-policy target artifact ABI
  statement-plan validation and left the worktree clean.
- The RVV spec already requires segment2 target export to rebuild the provider
  route, consume `TCRVEmitCLowerableRoute` plus provider description as
  authority, keep emission metadata as mirror-only, and validate runtime ABI
  order, segment layout, memory forms, mask facts, tuple/field facts, and
  provider-built statements.
- `validateRVVSegment2MemoryRoutePayloadFacts` already checks route id,
  provider-supported mirror, binding plan/summary, runtime control, route
  family plan mirrors, header declarations, type mappings, ABI mappings,
  segment layout, memory forms, segment count, field roles/names, mask facts,
  update arithmetic facts, and stale non-family provider/candidate mirrors.
- The remaining production gap is
  `validateRVVSegment2MemoryRouteStatementPlan`: after partial setvl/loop shape
  and provenance checks, it accepts compare, vector load, segment load/store,
  field extract, update arithmetic, and ordinary stores through
  `routeLoopContainsCallee`. That proves only that a callee string appears
  somewhere in the loop, not exact statement order, operands, result names,
  result C types, pointer expressions, VL operands, tuple field indices,
  runtime `n` relation, or selected statement provenance.
- `TargetArtifactExportTest.cpp` already has segment2 fixture generation,
  positive provider/candidate validation, and metadata/provider-fact negatives,
  but the segment2 section lacks route-clone negatives that mutate exact
  statement operands/results/source provenance.

## Requirements

1. Keep rebuilt provider route facts as authority. Candidate metadata may only
   be checked as mirrors after provider route construction.
2. Add exact runtime ABI order/role validation for all segment2-memory variants:
   - computed-mask load: `cmp_lhs,cmp_rhs,src,out0,out1,n`;
   - computed-mask store/update: `cmp_lhs,cmp_rhs,src0,src1,dst,n`;
   - plain deinterleave: `src,out0,out1,n`;
   - plain interleave: `src0,src1,dst,n`.
3. Replace segment2-memory callee-presence acceptance with exact rebuilt route
   statement validation for:
   - exactly one pre-loop `setvl` consuming runtime `n` / AVL and defining the
     provider full-chunk VL;
   - exactly one runtime AVL/VL loop with provider-derived induction, lower
     bound, upper bound, step, and runtime `n` relation;
   - loop `setvl` consuming remaining runtime AVL and defining per-iteration VL;
   - computed-mask compare loads, compare/mask statement, mask result name/type,
     and mask VL operands when required;
   - plain and computed-mask vector loads with exact source pointers, C types,
     result names/types, and loop VL;
   - segment2 load/store statements with exact tuple operands, pointer
     expressions, mask operands, VL operands, tuple C type, and result names;
   - tuple field extraction or tuple creation statements with exact field
     vector operands/results and field indices `0` / `1`;
   - computed-mask update arithmetic with exact payload operands, result, and
     VL;
   - ordinary field stores with exact destination pointers, vector values, and
     VL.
4. Require selected typed RVV source provenance on every required pre-loop and
   loop statement.
5. Preserve existing checks for provider-supported mirror, route-family plan
   mirrors, route operand binding facts, headers, type mappings, ABI mappings,
   source/destination memory forms, segment layout/count, field roles/names,
   mask/tail policy facts, update arithmetic facts, stale non-family facts, and
   stale candidate mirrors.
6. Fail closed for stale pre-loop AVL, loop remaining AVL, compare/mask operands
   or result, vector load pointer/result, segment load/store pointer/tuple/mask
   operand, tuple field index/result, update arithmetic operand/result, ordinary
   store pointer/value/VL, wrong ABI order/role, wrong memory form/layout,
   stale provider mirror, stale candidate mirror, stale route-family metadata,
   wrong selected typed provenance, route-id-only authority, artifact-name
   authority, exact-intrinsic-only authority, descriptor/source-front-door
   residue, common-EmitC semantic invention, and callee-presence-only
   authority.
7. Remove or demote `routeLoopContainsCallee` / `routeStepsContainCallee`
   acceptance only from the segment2-memory owner range. Helpers may remain for
   unrelated families unless replaced in this bounded module.

## Acceptance Criteria

- [x] PRD, implement context, and check context accurately describe the bounded
      module, specs, precedent, non-goals, and validation plan.
- [x] `RVVTargetArtifactRouteFamilyValidation.cpp` validates exact provider
      route facts and statement-plan facts for all active segment2-memory
      artifact consumers.
- [x] The segment2-memory owner range no longer accepts loop payload facts
      through `routeLoopContainsCallee` / `routeStepsContainCallee`.
- [x] Positive C++ target/export coverage proves valid computed-mask segment2
      load/store/update and plain segment2 deinterleave/interleave artifacts
      still pass through the production exporter and route-family validator.
- [x] Focused route-clone negatives fail closed for stale pre-loop AVL, loop
      remaining AVL, compare/mask, vector load, segment load/store, tuple
      field extract/create, update arithmetic, ordinary store pointer/value/VL,
      ABI order/role, provider mirror, candidate mirror, route-family metadata,
      and selected typed RVV provenance.
- [x] Direct-route-entry-only, artifact-name/script-derived, route-id-derived,
      exact-intrinsic-as-authority, common-EmitC semantic invention,
      descriptor/source-front-door residue, pre-realized-fixture-only,
      callee-presence-only, and legacy-i32-derived authority remain fail-closed
      or absent.
- [x] `ninja -C build tianchenrv-target-artifact-export-test` passes.
- [x] `./build/bin/tianchenrv-target-artifact-export-test` passes.
- [x] Bounded segment2 owner-slice authority scan confirms no remaining
      `routeLoopContainsCallee` or `routeStepsContainCallee` acceptance inside
      the segment2-memory validator range.
- [x] `git diff --check` passes.
- [x] `ninja -C build check-tianchenrv` passes, or an exact blocker is recorded.
- [x] If complete, the task is finished, archived, journaled, and committed
      coherently.

## Technical Approach

1. Reuse `validateRVVProviderBuiltRouteStep` for exact callee, operands, result
   name, and result C type checks.
2. Add segment2-specific runtime ABI role/order validation over provider
   `runtimeABIParameters`, using the existing runtime role enum rather than
   accepting only the last ABI parameter as `n`.
3. Strengthen `validateRVVSegment2MemoryRouteStatementPlan` to validate exact
   statement counts and ordered statement steps for each segment2 subfamily:
   computed-mask load, computed-mask store, computed-mask update, plain
   deinterleave, and plain interleave.
4. Extend `TargetArtifactExportTest.cpp` with focused route-clone negatives in
   the existing target artifact validation style.
5. Run focused build/test, bounded authority scan, `git diff --check`, and
   `check-tianchenrv` if feasible.

## Out of Scope

- Do not change RVV selected-body realization, route-provider semantics,
  generated-bundle scripts, generated RVV C emission, ssh rvv runtime behavior,
  performance, or high-level frontend lowering.
- Do not add new segment2 operation coverage, dtype/LMUL clone batches,
  one-intrinsic wrapper dialects, dashboards, reports, broad smoke matrices, or
  evidence-only bookkeeping.
- Do not change already-closed conversion, reduction, MAcc, base-memory,
  widening-dot, or contraction validators except for shared helper reuse needed
  by this bounded segment2 closure.
- Do not move RVV semantics into common EmitC/export code.
- Do not use generated artifacts, route ids, metadata, ABI strings, artifact
  names, scripts, exact intrinsic spellings, descriptors, source-front-door
  markers, direct route entries, pre-realized fixtures, or legacy i32 helper
  names as route authority.
- Do not implement compiler core, dialects, passes, plugin registry,
  capability model, lowering, or emission as Python data structures.
- Do not rerun generated-bundle or `ssh rvv` evidence by default because this
  task changes target artifact validation only. If runtime/materialization
  changes become necessary, add focused generated-bundle and `ssh rvv` evidence.

## Validation Plan

1. `ninja -C build tianchenrv-target-artifact-export-test`
2. `./build/bin/tianchenrv-target-artifact-export-test`
3. Bounded segment2 owner-slice authority scan for `routeLoopContainsCallee` /
   `routeStepsContainCallee`.
4. `git diff --check`
5. `ninja -C build check-tianchenrv`, or exact blocker.

Generated-bundle or `ssh rvv` evidence is not required unless this round changes
runtime, materialization, correctness, or performance semantics rather than
target artifact validation only.

## Spec Update Judgment

Initial expectation: no `.trellis/spec/` update is required. The current RVV
plugin, EmitC route, core dialect, and testing specs already require
provider-derived typed RVV route facts, segment2 target artifact
route-family validators, mirror-only metadata, common EmitC neutrality, and
evidence-bound runtime claims. This task applies those existing contracts to a
remaining target artifact validator acceptance hole.

Final judgment: no spec update was needed. The implementation applied the
existing segment2 target export consumer contract without discovering a new
durable rule.

## Completion Evidence

- Production owner file changed:
  `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`.
- Focused target/export coverage changed: `test/Target/TargetArtifactExportTest.cpp`.
- The segment2-memory statement-plan validator now requires exact provider-built
  pre-loop setvl, loop setvl, compare/mask, vector loads, segment load/store,
  tuple field extract/create, update arithmetic, ordinary stores, runtime
  n/AVL/VL relation, operand expressions, C types, result names, result C
  types, and selected typed RVV provenance.
- The owner slice now validates exact runtime ABI order, names, C types, roles,
  and target-export ownership for computed-mask segment2 load/store/update and
  plain segment2 deinterleave/interleave.
- Positive coverage covers computed-mask segment2 load/store/update and plain
  segment2 deinterleave/interleave through the production target artifact
  exporter and route-family validators.
- Negative route-clone coverage covers stale pre-loop AVL, loop remaining AVL,
  field vector load, compare/mask operand, update arithmetic operand/result,
  masked segment-store pointer, computed-mask segment-load pointer, field-store
  VL, tuple field index, field-store value, tuple field order, segment-store
  pointer, and selected typed RVV source provenance.
- Provider/candidate negatives cover ABI order/role, segment layout, memory
  forms, field roles, provider mirrors, candidate mirrors, route-family
  metadata, header/type mirrors, runtime plan mirrors, and route-id-derived
  support.
- Bounded owner-slice scan over
  `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp` lines 8970-9650
  found no `routeLoopContainsCallee` or `routeStepsContainCallee`; helpers
  remain only for unrelated route families.
- Checks run:
  - `ninja -C build tianchenrv-target-artifact-export-test`
  - `./build/bin/tianchenrv-target-artifact-export-test`
  - `sed -n '8970,9650p' lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp | rg -n "routeLoopContainsCallee|routeStepsContainCallee"` -> no matches
  - `rg -n "routeLoopContainsCallee|routeStepsContainCallee" lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp` -> helpers and unrelated families only
  - `git diff --check`
  - `ninja -C build check-tianchenrv` -> 459/459 passed
- Generated-bundle and `ssh rvv` evidence were not run because this task only
  changed target artifact validation and C++ target/export tests; it did not
  change runtime materialization, generated RVV C, correctness, or performance
  behavior.
