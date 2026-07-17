# Stage2 RVV widening conversion dtype-policy artifact ABI statement-plan validation closure

## Goal

Close the target artifact ABI statement-plan validation boundary for active RVV
widening conversion dtype-policy routes, centered on `widen_i16_to_i32` and the
selected typed `tcrv_rvv.widening_convert` body.

The RVV target artifact route-family validator must accept widening conversion
artifacts only after validating rebuilt provider route facts and exact
provider-built `TCRVEmitCLowerableRoute` statements. Callee or intrinsic
spelling may remain one checked field inside an exact provider-built statement,
but callee presence must not authorize source load, widening conversion, store,
runtime AVL/VL, dtype/SEW/LMUL policy, ABI order, selected typed provenance, or
artifact acceptance by itself.

This task is complete only if production validation in
`lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp` moves for the
widening conversion dtype-policy artifact consumer and focused target/export
C++ coverage proves the new fail-closed boundary.

## Direction Source

- Direction title: `Switch: Stage2 RVV widening conversion dtype-policy artifact ABI statement-plan validation closure`.
- Module owner: RVV target artifact route-family validator for active widening
  conversion dtype-policy route family.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean.
- Initial HEAD: `2347645c rvv: close standalone reduction ABI statement validation`.
- `.trellis/.current-task` was absent, so this task was created from the
  supplied Hermes Direction Brief before source edits.
- Serial worker constraint from the brief: one Codex worker; no subagents,
  spawned agents, parallel agents, or multi-agent workflows.

## Current Repository Facts

- Commit `2347645c` closed standalone reduction/accumulation target artifact ABI
  statement-plan validation and left the worktree clean.
- Archived vector reduction, MAcc, and standalone reduction/accumulation
  closures established the target-side pattern: rebuild the provider route,
  validate ABI order, headers, type mappings, operand bindings, loop/setvl
  facts, statement leaves, result names/types, and selected typed RVV
  provenance before accepting target artifact metadata.
- The current conversion dtype-policy target artifact consumer already checks
  route id, provider-supported mirror, operand-binding facts, runtime control
  facts, memory form `UnitStrideConversion`, widening plan presence,
  source/result SEW relation, source/result type mappings, ABI mappings, stale
  non-conversion route-family facts, and candidate metadata mirrors.
- The remaining production gap is
  `validateRVVConversionDtypePolicyRouteStatementPlan`: after checking only
  partial setvl/loop shape and provenance presence, it accepts the widening
  source load, conversion intrinsic, and store through
  `routeLoopContainsCallee`. That proves a callee string appears somewhere, but
  not exact statement order, operands, result names/types, source/result C
  types, pointer expressions, VL use, runtime `n` relation, or store target.
- `TargetArtifactExportTest.cpp` already has selected-body fixtures for
  `WidenI32ToI64` and `WidenI16ToI32`, route rebuild helpers, route-clone
  mutation helpers, and adjacent route-clone negative coverage style.

## Requirements

1. Keep rebuilt provider route facts as authority. Candidate metadata may only
   be checked as mirrors after provider route construction.
2. Replace widening conversion dtype-policy callee-presence acceptance with
   exact rebuilt route statement validation for:
   - exactly one full-chunk pre-loop `setvl` consuming runtime `n` / AVL and
     defining the provider full-chunk VL;
   - exactly one runtime AVL/VL loop with provider-derived induction, lower
     bound, upper bound, step, and runtime `n` relation;
   - loop `setvl` consuming remaining runtime AVL and defining per-iteration VL;
  - source vector load from `lhs + induction` using the provider source buffer
     ABI C type, provider source vector C type, and loop VL;
   - widening conversion statement consuming the source vector and loop VL,
     producing the provider result name with provider result vector C type;
   - output store consuming `out + induction`, the provider result vector, and
     loop VL using provider output ABI and VL C types.
3. Validate exact operand expressions and C types for runtime `n`, loop
   remaining AVL, source pointer, output pointer, source vector, result vector,
   and VL operands.
4. Validate exact result names and result C types for full-chunk VL, loop VL,
   source load result, and widening conversion result.
5. Require selected typed RVV source provenance on every required pre-loop and
   loop statement.
6. Validate conversion runtime ABI order and roles for the provider-owned
   `lhs,out,n` order, including `LHSInputBuffer`, `OutputBuffer`, and
   `RuntimeElementCount`; do not accept "last ABI parameter is n" as the only
   runtime AVL authority.
7. Validate source/result dtype policy for the active `widen_i16_to_i32` and
   existing widening conversion route facts through provider-derived
   source/result SEW, LMUL, vector type names, vector C types, C type mapping
   summary, conversion relation, route operand binding plan/summary, typed
   compute op, and memory form.
8. Preserve existing checks for provider-supported mirror, route-control plan,
   headers, type mappings, ABI mappings, stale non-conversion provider facts,
   and stale candidate mirrors.
9. Fail closed for stale source load operand/result/C type, stale conversion
   operand/result/C type, stale output pointer/VL, wrong runtime ABI role/order,
   wrong dtype relation, wrong selected typed provenance, stale mirror metadata,
   route-id/artifact-name-derived claims, exact-intrinsic-only authority,
   descriptor/source-front-door residue, and common-EmitC semantic invention.
10. Remove or demote `routeLoopContainsCallee` only from the widening conversion
    dtype-policy owner slice. Do not remove it globally unless unrelated
    callers are replaced in the same bounded module, which is not required here.

## Acceptance Criteria

- [x] PRD, implement context, and check context accurately describe the bounded
      module, specs, precedent, non-goals, and validation plan.
- [x] `RVVTargetArtifactRouteFamilyValidation.cpp` validates exact rebuilt
      provider route facts and statement-plan facts for active widening
      conversion dtype-policy artifact consumers.
- [x] Widening conversion dtype-policy validation no longer accepts loop payload
      facts through `routeLoopContainsCallee` / callee-presence checks.
- [x] Positive C++ target/export coverage proves valid `widen_i16_to_i32` /
      widening conversion artifacts still pass through the production exporter
      and route-family validator.
- [x] Focused negative provider/candidate/route mutations fail closed for stale
      source load pointer/result/C type, stale conversion operand/result/C type,
      stale store pointer/value/VL, wrong pre-loop setvl AVL, wrong loop setvl
      remaining AVL, wrong runtime ABI order/roles, wrong source/result
      SEW/LMUL or conversion relation, stale provider/candidate mirrors, wrong
      selected typed RVV provenance, and exact-intrinsic-only metadata
      substitution.
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

1. Reuse `validateRVVProviderBuiltRouteStep` to compare exact callee, operand
   expression, operand C type, result name, and result C type for conversion
   pre-loop and loop statements.
2. Add a conversion dtype-policy runtime ABI validator for the expected
   `lhs,out,n` ABI order and roles.
3. Strengthen the conversion dtype-policy payload validator with typed
   `tcrv_rvv.widening_convert` body facts, source/result SEW/LMUL relation,
   conversion relation, route operand binding facts, headers, type mappings,
   ABI mappings, stale non-family fact rejection, and exact statement-plan
   validation.
4. Extend `TargetArtifactExportTest.cpp` with focused positive and route-clone
   negative checks in the existing target artifact validation style.
5. Run focused build/test, bounded authority scan, `git diff --check`, and
   `check-tianchenrv` if feasible.

## Out of Scope

- Do not change RVV route planning/provider semantics, selected-body
  realization, generated-bundle fixtures, runtime materialization, or generated
  RVV C unless the validator exposes a real production inconsistency that must
  be repaired.
- Do not start segment2/interleave memory, elementwise/broadcast, base memory,
  MAcc, standalone reduction, widening dot, compare/select, new dtype/LMUL
  clone batches, high-level frontend lowering, one-intrinsic wrapper dialects,
  dashboard/report work, broad smoke matrices, or evidence-only tasks.
- Do not move RVV semantics into common EmitC/export code.
- Do not use generated artifacts, route ids, metadata, ABI strings, artifact
  names, scripts, exact intrinsic spellings, descriptors, source-front-door
  markers, direct route entries, pre-realized fixtures, or legacy i32 helper
  names as route authority.
- Do not implement compiler core, dialects, passes, plugin registry,
  capability model, lowering, or emission as Python data structures.
- Do not rerun `ssh rvv` by default because this task changes target artifact
  validation only. If runtime/materialization changes become necessary, add
  focused generated-bundle and `ssh rvv` evidence.

## Validation Plan

1. `ninja -C build tianchenrv-target-artifact-export-test`
2. `./build/bin/tianchenrv-target-artifact-export-test`
3. Bounded authority scan over touched production and test files for forbidden
   acceptance sources and remaining conversion dtype-policy
   `routeLoopContainsCallee` acceptance.
4. `git diff --check`
5. `ninja -C build check-tianchenrv`, or exact blocker.

Generated-bundle or `ssh rvv` evidence is not required unless this round
changes runtime, materialization, correctness, or performance semantics rather
than target artifact validation only.

## Completion Evidence

- Production owner file changed:
  `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`.
- Focused target/export coverage changed: `test/Target/TargetArtifactExportTest.cpp`.
- The widening conversion dtype-policy statement-plan validator now requires
  exact provider-built pre-loop setvl, loop setvl, source load, widening
  conversion, and output store statements with exact operands, C types, result
  names, result C types, loop/runtime `n` relation, and selected typed RVV
  provenance.
- The owner slice now validates exact provider facts for `lhs,out,n` ABI order
  and roles, `tcrv_rvv.widening_convert`, source/result SEW, LMUL, vector type
  names, vector C types, conversion relation, C type mapping summary, provider
  support mirror, and target leaf profile for `widen_i16_to_i32` and
  `widen_i32_to_i64`.
- Focused test self-repair: the stale dtype negative originally mutated
  `sourceSEW` and hit the older source-SEW/result-SEW guard first; it was
  corrected to mutate source vector C type so it reaches the new exact typed
  provider-facts guard.
- Checks run:
  - `ninja -C build tianchenrv-target-artifact-export-test`
  - `./build/bin/tianchenrv-target-artifact-export-test`
  - bounded conversion owner-slice authority scan for `routeLoopContainsCallee`
    / `routeStepsContainCallee`
  - `git diff --check`
  - `ninja -C build check-tianchenrv` -> 459/459 passed
- Generated-bundle and `ssh rvv` evidence were not run because this task only
  changed target artifact validation and C++ target/export tests; it did not
  change runtime materialization, generated RVV C, correctness, or performance
  behavior.

## Spec Update Judgment

Initial expectation: no `.trellis/spec/` update is required. The existing RVV
plugin, EmitC route, core dialect, and testing specs already require
provider-derived typed RVV route facts, target artifact route-family
validators, source/result dtype-policy validation, mirror-only metadata,
common EmitC neutrality, and evidence-bound runtime claims. This task applies
those existing contracts to a remaining target artifact validator acceptance
hole.
