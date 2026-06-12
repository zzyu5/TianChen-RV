# Stage2 RVV computed-mask accumulation route-control provider-plan integration

## Goal

Make the existing computed-mask accumulation MAcc route path consume the shared
`RVVSelectedBodyRouteControlProviderPlan` before computed-mask accumulation
statement/provider construction. Already-supported `computed_masked_macc_add`
and `runtime_scalar_cmp_masked_macc_add` must use the same RVV-owned structural
control boundary for runtime AVL/VL, SEW/LMUL, tail policy, mask policy,
runtime ABI order, selected capability facts, typed config facts,
mask-producer facts, accumulator/MAcc classification, materialization facts,
math operand binding, and same-analysis ownership now used by other migrated
RVV families.

This is a production-path migration for existing route support. It must not add
new accumulation operation kinds, new mask producer forms, new dtype/LMUL clone
batches, contraction/reduction expansion, high-level frontend lowering,
source-front-door positive routes, dashboards, broad smoke matrices, or
evidence-only fixture copying.

## Direction Source

- Direction title: `Stage2 RVV computed-mask accumulation route-control
  provider-plan integration`.
- Module owner: RVV plugin-local route-control provider-plan consumption by
  the existing computed-mask accumulation route-family/provider/statement path.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean.
- Initial HEAD: `63a774ee rvv: consume route control plan in runtime splat
  store`.
- `.trellis/.current-task` was absent, so this task was created from the
  supplied Hermes/User brief before source edits.
- Serial worker constraint: no subagents, spawned agents, parallel agents, or
  multi-agent workflow.

## Current Repository Facts

- Specs require the authority chain:
  selected `tcrv.exec` RVV variant -> typed or realized `tcrv_rvv` body -> RVV
  plugin-owned legality/realization/route provider -> `TCRVEmitCLowerableRoute`
  -> common EmitC -> target artifact mirrors.
- `RVVSelectedBodyRouteControlProviderPlan` currently validates typed config
  facts, selected target capability facts, runtime AVL/VL control, SEW/LMUL,
  tail policy, mask policy, config/runtime VL mirrors, runtime ABI order, and
  same-analysis materialization facts for several migrated families, including
  the previous runtime scalar splat-store task.
- `RVVSelectedBodyComputedMaskAccumulationRouteFamilyPlan`,
  `computedMaskAccumulationRouteFamilyPlan`, and
  `getRVVSelectedBodyComputedMaskAccumulationRouteStatementPlan(...)` already
  exist.
- The computed-mask accumulation family verifier already validates family
  mirrors, runtime-control mirrors, mask-producer facts, vector/runtime-scalar
  producer classification, MAcc vs standalone-reduction suffix classification,
  accumulator/result/inactive-lane contracts, runtime ABI parameters, and route
  operand binding closure.
- `RVVSelectedBodyRouteMaterializationFacts` already carries
  `computedMaskAccumulationPlan` and emits computed-mask accumulation leaves.
- The current gap is shared route-control ownership: computed-mask accumulation
  statement planning can be reached through the family/materialization/math
  binding path without first consuming a computed-mask accumulation consumer in
  `RVVSelectedBodyRouteControlProviderPlan`.

## Requirements

1. Production C++ planning/provider behavior must make only the existing
   computed-mask MAcc accumulation routes consume
   `RVVSelectedBodyRouteControlProviderPlan` before computed-mask accumulation
   statement-plan construction.
2. The route-control provider plan must recognize active computed-mask
   accumulation MAcc consumers:
   - `ComputedMaskedMAccAdd`
   - `RuntimeScalarComputedMaskedMAccAdd`
3. Computed-mask standalone reduction routes may remain outside this
   route-control consumer set unless code inspection proves they are
   inseparable from the MAcc statement-plan owner in this round.
4. Route-control construction for computed-mask accumulation must join and
   validate:
   - same-analysis typed config facts;
   - same-analysis selected target capability facts;
   - the owning runtime AVL/VL control plan;
   - runtime ABI order and runtime `n`/AVL binding;
   - SEW, LMUL, tail policy, and mask policy;
   - computed-mask accumulation family/materialization facts;
   - vector-compare or runtime-scalar mask-producer facts;
   - accumulation/MAcc classification and accumulator/result contracts;
   - math operand-binding facts through the statement-plan boundary;
   - route description and provider mirror consistency.
5. The computed-mask accumulation statement-plan boundary must fail closed
   before route construction when computed-mask accumulation requires
   route-control facts but any control plan, materialization facts, family
   facts, producer facts, MAcc classification facts, operand-binding facts,
   runtime ABI facts, policy facts, selected capability facts, or mirrors are
   stale, missing, or mismatched.
6. Common EmitC, target artifact export, generated headers, route descriptions,
   scripts, and metadata may mirror provider-built facts only after provider
   route construction. They must not infer AVL/VL, mask producer, accumulator,
   memory form, policy, dtype, or compute semantics.
7. No runtime/correctness/performance claim is made unless emitted executable
   behavior or ABI changes. If emitted code stays unchanged, historical runtime
   evidence remains sufficient and the final report must say so explicitly.

## Acceptance Criteria

- [x] `RVVSelectedBodyRouteControlProviderPlan` includes a computed-mask
      accumulation consumer flag or equivalent structural marker.
- [x] Route-control provider-plan consumer detection includes only existing
      computed-mask MAcc accumulation routes in this round.
- [x] `getRVVSelectedBodyRouteControlProviderPlan(...)` requires the verified
      computed-mask accumulation family plan and same-analysis materialization
      facts before exposing runtime AVL/VL, typed config, selected capability,
      policy, runtime ABI, mask-producer, accumulator/MAcc classification, and
      mirror facts for computed-mask accumulation consumers.
- [x] `getRVVSelectedBodyComputedMaskAccumulationRouteStatementPlan(...)`
      consumes the route-control provider plan before building setvl/compare
      load or splat/payload load/accumulator load/mask/MAcc/merge/store
      statement steps.
- [x] Focused C++ tests prove positive computed-mask accumulation
      route-control consumption and provider/statement-plan attachment for
      vector-compare and runtime-scalar computed-mask MAcc routes.
- [x] Focused negative C++ tests fail closed for representative stale or
      missing dependencies, including missing computed-mask accumulation
      family/materialization facts, stale same-analysis ownership, missing or
      wrong runtime AVL role, policy mismatch, unsupported selected
      capability/config, runtime ABI mirror mismatch, stale mask-producer or
      MAcc classification facts, and stale math operand binding.
- [x] Focused lit/FileCheck or generated-header checks cover explicit
      mirror-only labels if route metadata/header mirrors change.
- [x] Generated-bundle dry-run for affected computed-mask accumulation
      artifacts is rerun if emitted output changes; otherwise focused
      C++/provider evidence is sufficient and the final report records why
      `ssh rvv` was not rerun.
- [x] `git diff --check` passes.
- [x] Focused RVV plugin tests and `check-tianchenrv` pass, or an exact blocker
      is recorded.
- [x] A bounded authority scan over touched planning/provider/test/spec/target
      or script files finds no name-, route-id-, metadata-, descriptor-,
      ABI-string-, script-, artifact-, common-EmitC-, source-front-door-, or
      legacy-i32-derived AVL/VL, mask-producer, accumulator, memory-form,
      policy, dtype, or compute authority.

## Out Of Scope

- New computed-mask accumulation operation kinds or mask producer forms.
- Contraction, standalone reduction expansion, conversion expansion, memory or
  segment2 expansion, new dtype/LMUL clone batches, high-level frontend
  lowering, source-front-door positive routes, dashboards, broad smoke matrices,
  or evidence-only fixture copying.
- Treating runtime counts, route ids, metadata fields, manifests, artifact
  names, ABI strings, descriptors, scripts, tests, common EmitC, target artifact
  code, or legacy i32 spellings as AVL/VL, mask-producer, accumulator,
  memory-form, policy, dtype, or compute authority.
- Changing computation semantics, dispatch/fallback behavior, runtime ABI, or
  emitted statement order unless required by the control boundary and
  explicitly evidenced.

## Technical Approach

1. Validate and start this Trellis task.
2. Inspect current route-control, computed-mask accumulation family,
   materialization facts, math operand-binding facts, provider statement
   construction, selected-body realization, and focused C++ tests.
3. Add computed-mask accumulation MAcc to the route-control consumer set.
4. Extend `getRVVSelectedBodyRouteControlProviderPlan(...)` with a bounded
   computed-mask accumulation branch that validates same-analysis family and
   materialization facts, producer classification, MAcc classification,
   accumulator/result contracts, typed config, selected capability, runtime
   AVL/VL, policy, and mirrors.
5. Require the computed-mask accumulation statement-plan builder to obtain and
   check the route-control plan before binding statement operands.
6. Extend focused C++ tests rather than broad fixture copies.
7. Update `.trellis/spec/extension-plugins/rvv-plugin.md` only for durable
   route-control consumer boundary wording established by production code.
8. Run focused build/tests, bounded authority scans, `git diff --check`, and
   `check-tianchenrv`.

## Validation Plan

1. `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-25-stage2-rvv-computed-mask-accum-route-control`
2. `cmake --build build --target tianchenrv-rvv-extension-plugin-test -j2`
3. `./build/bin/tianchenrv-rvv-extension-plugin-test`
4. Focused lit/FileCheck or generated-bundle dry-run for computed-mask
   accumulation only if emitted target mirrors, ABI, artifact schema, or
   generated output changes.
5. Bounded authority scan over touched RVV planning/provider/test/spec files.
6. `git diff --check`
7. `cmake --build build --target check-tianchenrv -j2`

## Technical Notes

- Specs read: `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  `.trellis/spec/testing/index.md`, and
  `.trellis/spec/testing/mlir-testing-contract.md`.
- Relevant previous tasks read:
  `.trellis/tasks/archive/2026-05/05-25-stage2-rvv-runtime-vl-policy-provider-plan/prd.md`,
  `.trellis/tasks/archive/2026-05/05-25-stage2-rvv-computed-mask-select-route-control-provider-plan/prd.md`,
  `.trellis/tasks/archive/2026-05/05-25-stage2-rvv-computed-mask-memory-route-control/prd.md`,
  `.trellis/tasks/archive/2026-05/05-25-05-25-stage2-rvv-scalar-macc-route-control-provider-plan/prd.md`, and
  `.trellis/tasks/archive/2026-05/05-25-stage2-rvv-runtime-scalar-splat-store-route-control/prd.md`.
- Workspace journal read: `.trellis/workspace/codex/journal-15.md` sessions
  for computed-mask select, computed-mask memory, and runtime scalar
  splat-store route-control closure.
- Initial code surfaces inspected:
  `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`, and
  `test/Plugin/RVVExtensionPluginTest.cpp`.

## Implementation Result

- Added `controlsComputedMaskAccumulation` to
  `RVVSelectedBodyRouteControlProviderPlan`.
- Added route-control consumer detection for the existing
  `ComputedMaskedMAccAdd` and `RuntimeScalarComputedMaskedMAccAdd` routes with
  their current computed-mask accumulation memory forms only.
- Extended `getRVVSelectedBodyRouteControlProviderPlan(...)` so computed-mask
  accumulation consumers require same-analysis computed-mask accumulation
  family/materialization facts before route-control facts are exposed.
- The route-control owner now validates typed config facts, selected target
  capability facts, runtime AVL/VL control, tail/mask policy, runtime ABI
  order, vector/runtime-scalar mask-producer classification, vector-MAcc vs
  standalone-reduction classification, accumulator/result/inactive-lane
  contracts, materialization leaves, vector/mask type mirrors, and selected
  capability mirrors before statement construction.
- `getRVVSelectedBodyComputedMaskAccumulationRouteStatementPlan(...)` now
  requires same-analysis math operand-binding facts and the computed-mask
  accumulation route-control provider plan before building the provider-ready
  setvl/load/splat/compare/MAcc/merge/store sequence.
- Updated `test/Plugin/RVVExtensionPluginTest.cpp` with positive route-control
  coverage for vector-compare and runtime-scalar computed-mask MAcc, plus
  fail-closed coverage for missing family/materialization facts, stale
  same-analysis ownership, stale runtime AVL, policy mismatch, selected
  capability mismatch, stale MAcc classification, stale route-family mirrors,
  stale runtime ABI, and stale operand binding.
- Updated `.trellis/spec/extension-plugins/rvv-plugin.md` so computed-mask
  accumulation MAcc is recorded as a durable route-control provider-plan
  consumer and statement-plan dependency.
- No emitted statement order, runtime ABI, target mirror schema, generated
  bundle script behavior, runtime count handling, or executable semantics were
  changed.

## Validation Result

- [x] `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-25-stage2-rvv-computed-mask-accum-route-control`
- [x] `git diff --check`
- [x] `cmake --build build --target tianchenrv-rvv-extension-plugin-test -j2`
- [x] `./build/bin/tianchenrv-rvv-extension-plugin-test`
- [x] Bounded added-line authority scan over touched planning/provider/test/spec
      files. The broad scan only reported explicit spec mirror-only wording,
      local `result` test variables, and the intentional negative `metadata_n`
      runtime AVL test value; the narrow scan found no added route-id,
      descriptor, source-front-door, source-artifact, direct-C, source-export,
      legacy-i32, script-derived, common-EmitC-derived, metadata-derived, or
      artifact-derived authority.
- [x] `cmake --build build --target check-tianchenrv -j2`: 379/379 tests
      passed.

No generated-bundle dry-run or real `ssh rvv` rerun was required because this
round only adds provider-plan ownership and fail-closed validation before the
existing computed-mask accumulation statement construction. It does not change
emitted executable behavior, statement order, target ABI, target mirror schema,
or generated-bundle script behavior; historical runtime evidence remains the
evidence for the unchanged executable route.
