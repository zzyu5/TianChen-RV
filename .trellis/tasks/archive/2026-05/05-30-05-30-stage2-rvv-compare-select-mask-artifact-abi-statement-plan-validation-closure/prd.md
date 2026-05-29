# Stage2 RVV compare/select mask artifact ABI statement-plan validation closure

## Goal

Close the target artifact ABI and statement-plan validation boundary for the
active RVV compare/select mask route-family target artifact consumer.

The RVV target artifact route-family validator must accept compare/select mask
artifacts only after validating rebuilt provider route facts, exact runtime ABI
roles/order/names/types/ownership, selected typed `tcrv_rvv` provenance,
route-family and mask/policy facts, candidate mirrors, and exact provider-built
`TCRVEmitCLowerableRoute` statement plans. Callee or intrinsic spelling may
remain one field inside an exact statement match, but callee presence must not
authorize compare, secondary compare, mask composition, select/merge, memory
movement, runtime AVL/VL, ABI order, selected typed provenance, or artifact
acceptance by itself.

This task is complete only if production validation in
`lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp` moves for the
compare/select mask artifact consumer and focused target/export C++ coverage
proves the new fail-closed boundary.

## Direction Source

- Direction title: `Switch: Stage2 RVV compare/select mask target artifact ABI
  statement-plan validation closure`.
- Module owner: RVV target artifact route-family validation owner for
  compare/select mask route-family consumers.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean.
- Initial HEAD: `5c428c1c rvv: close segment2 memory ABI statement validation`.
- `.trellis/.current-task` was absent, so this task was created from the
  supplied Hermes Direction Brief before source edits.
- Serial worker constraint from the brief: one Codex worker; no subagents,
  spawned agents, parallel agents, or multi-agent workflows.

## Current Repository Facts

- Commit `5c428c1c` closed segment2-memory target artifact validation and left
  the worktree clean.
- The RVV spec requires target artifact route-family validators to dispatch
  from rebuilt provider descriptions and rebuilt `TCRVEmitCLowerableRoute`,
  while artifact metadata remains mirror-only after route construction.
- The compare/select statement-plan spec requires the provider route to consume
  verified route-family plans, materialization facts, elementwise/select
  operand-binding facts, route-control provider plan, and an RVV-owned
  compare/select statement plan before common EmitC.
- `validateRVVCompareSelectMaskRoutePayloadFacts` already checks route id,
  provider-supported mirror presence, binding/runtime facts, family/mask facts,
  headers, type mappings, ABI mappings, and candidate mirror metadata for
  plain compare-select, computed-mask select, runtime-scalar select, dual
  runtime-scalar mask-and-select, and compare-produced computed-mask memory.
- The remaining production gap is
  `validateRVVCompareSelectMaskRouteStatementPlan`: after partial setvl/loop
  shape and selected-source provenance checks, it accepts compare, secondary
  compare, mask composition, select/primary compute, vector load, splat, index,
  masked load, strided/indexed load/store, and ordinary store through
  `routeLoopContainsCallee`. That proves only that a callee string appears
  somewhere in the loop, not exact statement order, operands, result names,
  result C types, pointer expressions, VL operands, mask operands, index/stride
  operands, runtime `n` relation, or selected statement provenance.
- `TargetArtifactExportTest.cpp` already has clone helpers for mutating route
  call operands, loop operands/results, and source provenance, plus precedent
  from the segment2-memory tests for focused route-clone negative validation.

## Requirements

1. Keep rebuilt provider route facts as authority. Candidate metadata may only
   be checked as mirrors after provider route construction.
2. Add exact runtime ABI role/order/name/type/ownership validation for every
   compare/select mask subfamily accepted by this owner:
   - plain compare-select: `lhs,rhs,out,n`;
   - computed-mask select: `cmp_lhs,cmp_rhs,true_value,false_value,out,n`;
   - runtime-scalar computed-mask select:
     `lhs,rhs_scalar,true_value,false_value,out,n`;
   - runtime-scalar dual compare-mask-and-select:
     `cmp_lhs_a,rhs_scalar_a,cmp_lhs_b,rhs_scalar_b,true_value,false_value,out,n`;
   - compare-produced computed-mask memory subfamilies according to their
     provider runtime ABI order and roles.
3. Replace compare/select mask callee-presence acceptance with exact rebuilt
   route statement validation for:
   - exactly one pre-loop `setvl` consuming runtime `n` / AVL and defining the
     provider full-chunk VL;
   - exactly one runtime AVL/VL loop with provider-derived induction, lower
     bound, upper bound, step, and runtime `n` relation;
   - loop `setvl` consuming remaining runtime AVL and defining per-iteration VL;
   - compare LHS/RHS loads or runtime-scalar splats with exact operands, result
     names/types, and loop VL;
   - optional secondary compare and `mask_and` composition with exact mask
     operands/results/types and loop VL;
   - true/false value loads, select/merge operands, result names/types, and
     output stores for compare/select subfamilies;
   - computed-mask memory source/destination loads/stores, masked loads,
     strided/indexed address steps, index scaling, masked merge/passthrough,
     store pointer/value/VL, and memory-form-specific operand order.
4. Require selected typed RVV source provenance on every required pre-loop and
   loop statement.
5. Preserve existing checks for provider-supported mirror,
   route-family/mask-tail policy mirrors, route operand binding facts, headers,
   type mappings, ABI mappings, source/destination memory forms, mask role,
   mask source, mask memory form, select layout, inactive-lane/passthrough facts,
   stale non-family provider facts, and stale candidate mirrors.
6. Fail closed for stale pre-loop AVL, loop remaining AVL, compare operands or
   mask result, secondary compare operands/result, mask composition operands,
   select operands/result, vector load pointer/result, rhs splat operands,
   masked load/store pointer/value/mask/VL, strided/indexed load/store
   pointer/index/scale/stride operands, wrong ABI order/role/name/type/
   ownership, wrong memory form/layout, stale provider mirror, stale candidate
   mirror, stale route-family metadata, wrong selected typed provenance,
   route-id-only authority, artifact-name authority, exact-intrinsic-only
   authority, descriptor/source-front-door residue, common-EmitC semantic
   invention, callee-presence-only authority, and legacy-i32-derived authority.
7. Remove or demote `routeLoopContainsCallee` / `routeStepsContainCallee`
   acceptance only from the compare/select mask owner range. Helpers may remain
   for unrelated families unless replaced in this bounded module.

## Acceptance Criteria

- [ ] PRD, implement context, and check context accurately describe the bounded
      module, specs, precedent, non-goals, and validation plan.
- [ ] `RVVTargetArtifactRouteFamilyValidation.cpp` validates exact provider
      route facts and statement-plan facts for every active compare/select mask
      artifact consumer it accepts.
- [ ] The compare/select mask owner range no longer accepts loop payload facts
      through `routeLoopContainsCallee` / `routeStepsContainCallee`.
- [ ] Positive C++ target/export coverage proves valid compare/select mask
      artifacts still pass through the production exporter and route-family
      validator for representative plain compare-select, computed-mask select,
      runtime-scalar compare-select, dual runtime-scalar compare-mask-and-select,
      and compare-produced computed-mask memory cases available in the current
      test surface.
- [ ] Focused route-clone negatives fail closed for stale pre-loop AVL, loop
      remaining AVL, compare operands/result, secondary compare, mask
      composition, select operands/result, load/store/index/stride operands,
      ABI order/role/name/type/ownership, provider mirror, candidate mirror,
      route-family metadata, and selected typed RVV provenance.
- [ ] Direct-route-entry-only, artifact-name/script-derived, route-id-derived,
      exact-intrinsic-as-authority, common-EmitC semantic invention,
      descriptor/source-front-door residue, pre-realized-fixture-only,
      callee-presence-only, and legacy-i32-derived authority remain fail-closed
      or absent.
- [ ] `ninja -C build tianchenrv-target-artifact-export-test` passes.
- [ ] `./build/bin/tianchenrv-target-artifact-export-test` passes.
- [ ] Bounded compare/select mask owner-slice authority scan confirms no
      remaining `routeLoopContainsCallee` or `routeStepsContainCallee`
      acceptance inside the compare/select mask validator range.
- [ ] `git diff --check` passes.
- [ ] `ninja -C build check-tianchenrv` passes, or an exact blocker is recorded.
- [ ] If complete, the task is finished, archived, journaled, and committed
      coherently.

## Technical Approach

1. Reuse `validateRVVProviderBuiltRouteStep` for exact callee, operands, result
   name, and result C type checks.
2. Add compare/select mask runtime ABI role/order validation over provider
   `runtimeABIParameters`, using the existing runtime role enum and
   provider-derived operation kind rather than accepting only a trailing `n`.
3. Strengthen `validateRVVCompareSelectMaskRouteStatementPlan` to validate
   exact statement counts and ordered statement steps for the currently active
   compare/select mask subfamilies.
4. Extend `TargetArtifactExportTest.cpp` with focused route-clone negatives in
   the existing target artifact validation style.
5. Run focused build/test, bounded authority scan, `git diff --check`, and
   `check-tianchenrv` if feasible.

## Out of Scope

- Do not change RVV selected-body realization, route-provider semantics,
  generated-bundle scripts, generated RVV C emission, ssh rvv runtime behavior,
  performance, or high-level frontend lowering.
- Do not change segment2-memory validation closed by the previous task,
  elementwise arithmetic validation, standalone reductions, MAcc, widening
  conversion, widening/contraction routes, or broad validation registry
  behavior except for shared helper reuse needed by this bounded closure.
- Do not add new compare/select, memory, dtype, LMUL, or one-intrinsic coverage
  beyond what is needed to prove this validator boundary.
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
3. Bounded compare/select mask owner-slice authority scan for
   `routeLoopContainsCallee` / `routeStepsContainCallee`.
4. `git diff --check`
5. `ninja -C build check-tianchenrv`, or exact blocker.

Generated-bundle or `ssh rvv` evidence is not required unless this round changes
runtime, materialization, correctness, or performance semantics rather than
target artifact validation only.

## Spec Update Judgment

Initial expectation: no `.trellis/spec/` update is required. The current RVV
plugin, EmitC route, core dialect, and target artifact validator specs already
require provider-derived typed RVV route facts, compare/select statement-plan
ownership, target artifact route-family validators, mirror-only metadata,
common EmitC neutrality, and evidence-bound runtime claims. This task applies
those existing contracts to a remaining target artifact validator acceptance
hole.
