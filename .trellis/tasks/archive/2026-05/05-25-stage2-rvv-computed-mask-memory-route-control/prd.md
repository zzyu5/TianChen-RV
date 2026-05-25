# Stage2 RVV computed-mask memory route-control provider-plan integration

## Goal

Make the existing non-segment computed-mask memory RVV route family consume the
shared `RVVSelectedBodyRouteControlProviderPlan` before computed-mask memory
statement/provider construction. Already-supported runtime-scalar
computed-mask store/load-store, computed-mask unit load/store, strided store,
strided load/unit-store, indexed gather/unit-store, and indexed
scatter/unit-load routes must use the same RVV-owned structural boundary for
runtime AVL/VL, SEW/LMUL, tail policy, mask policy, runtime ABI order, typed
config, selected target capability, materialization facts, mask-producer facts,
memory-form facts, operand binding, and same-analysis ownership now used by
elementwise, scalar-broadcast, base memory, reduction, MAcc, plain
compare/select, and computed-mask select.

This is a production-path migration for an existing route family. It must not
add new computed-mask memory operation kinds, change computation semantics,
change dispatch/fallback behavior, or treat metadata/artifacts/scripts/common
EmitC as route authority.

## Source Direction

- Direction title: `Stage2 RVV computed-mask memory route-control provider-plan
  integration`.
- Module owner: RVV plugin-local route-control provider-plan consumption by the
  existing computed-mask memory route-family/provider path.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean.
- Initial HEAD: `f5721dac rvv: consume route control plan in computed-mask select`.
- `.trellis/.current-task` was absent, so this task was created from the
  Hermes/User brief before source edits.
- Serial worker constraint: no subagents, spawned agents, parallel agents, or
  multi-agent workflow.

## Current Repository Facts

- The stable RVV authority chain is selected `tcrv.exec` RVV variant -> typed
  or realized `tcrv_rvv` body -> RVV plugin-owned legality / selected-body
  realization / route provider -> `TCRVEmitCLowerableRoute` -> common EmitC ->
  target artifact mirrors.
- `RVVSelectedBodyRouteControlProviderPlan` already validates typed config
  facts, selected target capability facts, runtime AVL/VL control, SEW/LMUL,
  tail policy, mask policy, config/runtime VL mirrors, runtime ABI order, and
  same-analysis materialization facts.
- Recent archived tasks completed route-control consumption for base memory
  movement, standalone reduction, scalar MAcc, ordinary elementwise,
  scalar-broadcast elementwise, plain compare/select, and computed-mask select.
- `RVVSelectedBodyComputedMaskMemoryRouteFamilyPlan` already carries
  `runtimeControlPlan`, mask-producer source, runtime ABI order, memory-form
  facts, typed emission leaves, provider mirror fields, and route-family plan
  mirrors.
- `RVVSelectedBodyMemoryRouteOperandBindingFacts` already owns computed-mask
  memory operand bindings for compare inputs, runtime-scalar threshold,
  source/destination, index, stride, passthrough, and runtime `n`.
- `RVVSelectedBodyComputedMaskMemoryRouteStatementPlan` already owns the
  non-segment computed-mask memory statement sequence, but currently it can be
  constructed without first consuming the shared route-control provider plan.
- Computed-mask segment2 memory is deliberately routed through the segment2
  statement-plan owner and remains excluded from this computed-mask memory
  statement-plan boundary.

## Requirements

1. Production C++ planning/provider behavior must make only existing
   non-segment computed-mask memory routes consume
   `RVVSelectedBodyRouteControlProviderPlan` before computed-mask memory
   statement-plan construction.
2. The route-control provider plan must recognize active non-segment
   computed-mask memory consumers:
   - `RuntimeScalarComputedMaskStore`
   - `RuntimeScalarComputedMaskLoadStore`
   - `ComputedMaskUnitLoadStore`
   - `ComputedMaskStridedStore`
   - `ComputedMaskStridedLoadUnitStore`
   - `ComputedMaskIndexedGatherLoadUnitStore`
   - `ComputedMaskIndexedScatterStoreUnitLoad`
3. Computed-mask segment2 memory routes must continue to receive an empty
   computed-mask memory statement plan and remain owned by segment2 memory
   statement planning.
4. Route-control construction for computed-mask memory must join and validate:
   - same-analysis typed config facts;
   - same-analysis selected target capability facts;
   - the owning runtime AVL/VL control plan;
   - runtime ABI order and runtime `n`/AVL binding;
   - SEW, LMUL, tail policy, and mask policy;
   - computed-mask memory family/materialization facts;
   - mask-producer facts and memory-form classification;
   - memory operand-binding facts;
   - route description and provider mirror consistency.
5. The computed-mask memory statement-plan boundary must fail closed before
   route construction when computed-mask memory requires route-control facts
   but any control plan, materialization facts, family facts, mask-producer
   facts, memory-form facts, operand-binding facts, runtime ABI facts, policy
   facts, selected capability facts, or mirrors are stale, missing, or
   mismatched.
6. Common EmitC, target artifact export, generated headers, route descriptions,
   scripts, and metadata may mirror provider-built facts only after provider
   route construction. They must not infer AVL/VL, mask, memory-form, policy,
   dtype, or compute semantics.
7. No runtime/correctness/performance claim is made unless emitted executable
   behavior or ABI changes. If emitted code stays unchanged, historical runtime
   evidence remains sufficient and the final report must say so explicitly.

## Acceptance Criteria

- [x] `RVVSelectedBodyRouteControlProviderPlan` includes a computed-mask memory
      consumer flag or equivalent structural marker.
- [x] Route-control provider-plan consumer detection includes only the active
      non-segment computed-mask memory routes listed above; computed-mask
      segment2 routes remain excluded from this consumer set.
- [x] `getRVVSelectedBodyRouteControlProviderPlan(...)` requires the verified
      computed-mask memory family plan and same-analysis materialization facts
      before exposing runtime AVL/VL, typed config, selected capability,
      policy, runtime ABI, mask-producer, memory-form, and mirror facts for
      computed-mask memory consumers.
- [x] `getRVVSelectedBodyComputedMaskMemoryRouteStatementPlan(...)` consumes the
      route-control provider plan before building setvl/load/splat/compare/
      masked-load/masked-store/strided/indexed statement steps.
- [x] Focused C++ tests prove positive computed-mask memory route-control
      consumption and provider/statement-plan attachment for representative
      runtime-scalar, vector-compare, strided, and indexed paths.
- [x] Focused negative C++ tests fail closed for representative stale or
      missing dependencies, including missing computed-mask memory
      family/materialization facts, stale same-analysis ownership, missing or
      wrong runtime AVL role, policy mismatch, unsupported selected
      capability/config, runtime ABI mirror mismatch, stale mask-producer or
      memory-form marker, and stale operand binding.
- [x] Existing computed-mask segment2 tests prove the computed-mask memory
      statement-plan boundary remains empty for segment2 routes.
- [x] Focused lit/FileCheck or generated-header checks cover explicit
      mirror-only labels if route metadata/header mirrors change.
- [x] Generated-bundle dry-run for affected computed-mask memory artifacts is
      rerun if emitted output changes; otherwise focused C++/provider evidence
      is sufficient and the final report records why `ssh rvv` was not rerun.
- [x] `git diff --check` passes.
- [x] Focused RVV plugin tests and `check-tianchenrv` pass, or an exact blocker
      is recorded.
- [x] A bounded authority scan over touched planning/provider/test/spec/target
      or script files finds no name-, route-id-, metadata-, descriptor-,
      ABI-string-, script-, artifact-, common-EmitC-, source-front-door-, or
      legacy-i32-derived AVL/VL, mask, memory-form, policy, dtype, or compute
      authority.

## Out Of Scope

- New computed-mask memory operation kinds or predicates.
- Computed-mask segment2 migration onto this route-control consumer set.
- Conversion, segment2 expansion, contraction, accumulation, reduction, new
  dtype/LMUL clone batches, high-level frontend lowering, source-front-door
  positive routes, dashboards, broad smoke matrices, or evidence-only fixture
  copying.
- Treating runtime counts, route ids, metadata fields, manifests, artifact
  names, ABI strings, descriptors, scripts, tests, common EmitC, target
  artifact code, or legacy i32 spellings as AVL/VL, mask, memory-form, policy,
  dtype, or compute authority.
- Changing computation semantics, dispatch/fallback behavior, runtime ABI, or
  emitted statement order unless required by the control boundary and
  explicitly evidenced.

## Technical Approach

1. Start and validate the Trellis task.
2. Add a computed-mask memory control consumer marker to the route-control plan
   and consumer predicate.
3. Reuse the existing same-analysis/family/materialization validation pattern
   from computed-mask select and base memory movement.
4. Require the computed-mask memory statement-plan boundary to acquire and
   validate the route-control provider plan before statement construction.
5. Extend existing computed-mask memory C++ provider tests with positive
   route-control checks and targeted stale/missing dependency diagnostics.
6. Update `.trellis/spec/extension-plugins/rvv-plugin.md` after the production
   path and tests establish computed-mask memory as a route-control consumer.
7. Run focused build/tests, bounded authority scan, `git diff --check`, and
   `check-tianchenrv`.

## Validation Plan

1. `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-25-stage2-rvv-computed-mask-memory-route-control`
2. `cmake --build build --target tianchenrv-rvv-extension-plugin-test -j2`
3. `./build/bin/tianchenrv-rvv-extension-plugin-test`
4. `cmake --build build --target tcrv-opt tcrv-translate -j2` if focused lit or generated-bundle evidence is required.
5. Focused lit/FileCheck for computed-mask memory route artifacts if emitted
   mirrors or target artifacts change.
6. Generated-bundle dry-run for computed-mask memory only if emitted output,
   ABI evidence, or artifact mirrors change.
7. Bounded authority scan over touched RVV planning/provider/test/spec files.
8. `git diff --check`
9. `cmake --build build --target check-tianchenrv -j2`

## Implementation Result

- Added `controlsComputedMaskMemory` to
  `RVVSelectedBodyRouteControlProviderPlan`.
- Added route-control consumer detection for the existing non-segment
  computed-mask memory route kinds only. Computed-mask segment2 remains owned
  by segment2 statement planning and is not included in this consumer set.
- Extended computed-mask memory route-control planning so it requires the
  verified computed-mask memory family/materialization plan from the same
  selected route analysis, validates mask-producer and memory-form facts, and
  attaches the owning runtime control plan before route statement construction.
- Extended computed-mask memory statement planning so route-control ownership,
  control-plan pointer identity, materialization facts, memory operand binding
  same-analysis ownership, ABI mirrors, policy mirrors, typed config mirrors,
  and selected capability mirrors are checked before statement construction.
- Added focused positive and negative C++ coverage in
  `RVVExtensionPluginTest.cpp` for route-control attachment, runtime AVL role,
  policy mismatch, selected capability mismatch, missing/stale materialization
  facts, stale mask-producer or memory-form facts, stale operand binding, and
  runtime ABI mirror mismatch.
- Updated `rvv-plugin.md` to record non-segment computed-mask memory as a
  route-control consumer and to keep computed-mask memory authority inside the
  RVV plugin-local typed body/config/runtime/family/binding boundary.

## Validation Result

- `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-25-stage2-rvv-computed-mask-memory-route-control`
- `cmake --build build --target tianchenrv-rvv-extension-plugin-test -j2`
  initially exposed one missing `RVVSelectedBodyMemoryForm` using-declaration
  in the extended C++ test; the test include scope was fixed and the command
  was rerun successfully.
- `./build/bin/tianchenrv-rvv-extension-plugin-test`
- `cmake --build build --target tcrv-opt tcrv-translate -j2`
- `python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run --pre-realized-selected-body --artifact-root artifacts/tmp/stage2_rvv_computed_mask_memory_route_control --run-id pre-realized-computed-mask-memory-route-control-dry --overwrite --op-kind runtime_scalar_cmp_masked_store --op-kind runtime_scalar_cmp_masked_load_store --op-kind computed_masked_unit_load_store --op-kind computed_masked_strided_store --op-kind computed_masked_strided_load_unit_store --op-kind computed_masked_indexed_gather_load_unit_store --op-kind computed_masked_indexed_scatter_store_unit_load --runtime-count 7 --runtime-count 16 --runtime-count 23 --rhs-scalar 5 --rhs-scalar 11 --stride-bytes 4 --stride-bytes 12 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj`
  reported `dry_run_success` under
  `artifacts/tmp/stage2_rvv_computed_mask_memory_route_control/pre-realized-computed-mask-memory-route-control-dry`.
- `/usr/bin/python3.10 /usr/lib/llvm-20/build/utils/lit/lit.py -sv Target/RVV Scripts --filter='computed-mask|computed_mask|computed-masked|computed_masked|runtime-scalar-cmp-masked|runtime_scalar_cmp_masked'`
  from `build/test`: 53 selected tests passed.
- `git diff --check`
- Added-line authority scan found only explicit negative/spec wording that
  common EmitC/artifact metadata/status are not authority, C++ local `result`
  variables, and the `metadata_n` runtime ABI mirror mismatch negative test.
  A narrower scan found no added legacy-i32, descriptor, source-front-door,
  route-id, direct-C, or source-export authority.
- `cmake --build build --target check-tianchenrv -j2`: 379 tests passed.
- `ssh rvv` was not rerun because this round does not change emitted executable
  statement order, runtime ABI, or target artifact semantics; it inserts
  provider-plan ownership and fail-closed validation before the existing
  computed-mask memory statement/provider construction.

## Definition Of Done

The non-segment computed-mask memory route family has a reviewable
route-control authority chain from typed selected body/config and selected
target capability facts through runtime AVL/VL control, mask-producer and
memory-form family facts, operand-binding facts, RVV-owned statement planning,
and provider-built `TCRVEmitCLowerableRoute`. Unsupported or inconsistent
control dependencies fail closed before route/artifact authority. The task is
finished/archived and committed, or an exact blocker and continuation point is
recorded.
