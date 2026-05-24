# Stage2 RVV statement-plan provider-neutral closure

## Goal

Close the RVV provider-side statement-plan consumption boundary for the
production-active route families that have already moved statement ownership
into RVV planning: elementwise arithmetic, compare/select, base memory,
computed-mask memory, segment2 memory, and computed-mask accumulation.

`RVVEmitCRouteProvider` should request one RVV-owned migrated-family
statement-plan boundary after route-family verification, materialization facts,
and operand-binding facts are available. The provider remains responsible for
neutral `TCRVEmitCLowerableRoute` instantiation, headers, type mappings, ABI
mappings, selected-boundary source provenance, and attaching returned
statements. It must not manually sequence family-specific statement-plan calls
or recreate migrated-family setvl/load/splat/compare/mask/store/accumulator
statement sequences from operation names, ABI strings, route ids, mirrors, or
artifact metadata.

## Direction Source

- Direction title: `Stage2 RVV statement-plan provider-neutral closure`.
- Module owner: RVV plugin-owned route statement-plan consumption boundary
  across migrated production-active statement-plan families.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean before task creation.
- Initial HEAD: `10c0a74e rvv: own computed-mask accumulation statement plans`.
- `.trellis/.current-task` was absent, so this task was created from the
  Hermes Direction Brief before source edits.

## What I Already Know

- The stable route authority chain is selected `tcrv.exec` RVV variant ->
  typed low-level `tcrv_rvv` body -> RVV verifier/materialization/
  operand-binding facts -> RVV-owned family statement plan ->
  provider-built `TCRVEmitCLowerableRoute` -> common EmitC materialization.
- Recent commits already introduced family-owned statement plans for:
  elementwise arithmetic, compare/select, base memory, computed-mask memory,
  segment2 memory, and computed-mask accumulation.
- The current provider consumes those plans one family at a time with repeated
  provider-local sequencing and family-specific fail-closed guards before
  falling through to the older generic statement assembly surface.
- This task is not new Stage2 coverage. It is a boundary cleanup over
  migrated families that already have verified selected-body/materialization/
  operand-binding inputs.
- Common EmitC remains neutral. It must not choose RVV semantics, intrinsics,
  dtype/config, schedule, ABI, or route support.

## Requirements

1. Add a shared RVV planning API that returns one provider-ready migrated
   statement plan for these exact families:
   elementwise arithmetic, compare/select, base memory movement,
   computed-mask memory movement, segment2 memory movement, and computed-mask
   accumulation.
2. The shared API must be called only after
   `verifyRVVSelectedBodyRouteFamilyProviderPlans`, after
   `getRVVSelectedBodyRouteMaterializationFacts`, and after the relevant
   operand-binding fact boundaries for the same analysis:
   elementwise/select, memory, math, and residual.
3. The shared plan must preserve provider-ready pre-loop call steps, one
   provider-ready loop, source operation provenance, ABI/VL/mask/address/
   accumulator/source facts, sub-family identity, and route-family ownership
   without moving RVV semantics into common EmitC.
4. `RVVEmitCRouteProvider` must consume the shared migrated statement-plan
   boundary before the older generic provider-local statement assembly path.
   For migrated-family routes, a missing or stale valid plan must fail closed
   before statement construction and before common materialization.
5. Delete or factor the repeated provider-local calls to each migrated family
   statement-plan getter. If any older generic statement assembly branches for
   migrated operations remain in the file, they must be unreachable/fenced by
   the shared boundary with exact diagnostic coverage.
6. Preserve emitted target sequence, ABI order, operand order, VL/control
   behavior, source provenance, intrinsic spelling, route ids, and generated
   artifacts for valid selected-body routes.
7. Do not add route coverage, new memory/accumulation forms, reductions,
   contractions, dtype/LMUL clone batches, high-level frontend lowering,
   source-front-door positive routes, legacy i32 route authority,
   descriptor/direct-C/source-export paths, dashboards, broad smoke matrices,
   or common EmitC semantic logic.

## Acceptance Criteria

- [x] `prd.md`, `implement.jsonl`, and `check.jsonl` name the RVV plugin,
      EmitC route, testing specs, and previous computed-mask accumulation task
      context.
- [x] A shared RVV planning API exists for migrated statement-plan
      consumption and exposes an empty/default plan for unrelated route
      families.
- [x] `RVVEmitCRouteProvider.cpp` consumes the shared migrated statement-plan
      API once and attaches returned provider-ready statements without
      manually sequencing each migrated family getter in the provider body.
- [x] Focused C++ tests prove positive shared-boundary consumption for
      representative migrated families across the cluster, including at least
      one arithmetic/select-style route and one memory or accumulation route.
- [x] At least one focused C++ negative test proves a missing/stale migrated
      statement-plan dependency fails closed through the shared boundary before
      route statement construction.
- [x] Existing family-specific statement-plan tests remain passing and still
      prove the family plan payloads carry the route-specific ABI/VL/mask/
      address/accumulator/source facts.
- [x] Bounded provider scan shows migrated-family statement-plan consumption
      is reached through the shared boundary and that the provider no longer
      manually sequences the six family-specific statement-plan getter calls.
- [x] Active-authority scan over touched RVV planning/provider/test/spec/task
      files finds no newly introduced legacy `RVVI32M1`, `rvv-i32m1`, positive
      finite `tcrv_rvv.i32_*`, descriptor/source-front-door/direct-C/
      source-export, or mirror-only route authority.
- [x] `git diff --check` passes.
- [x] Focused build/tests and `check-tianchenrv` pass, or an exact blocker is
      documented.
- [x] Final git status is clean and one coherent commit records the completed
      task if all acceptance criteria are met.

## Non-Goals

- No new route coverage, memory forms, accumulation/reduction/contraction
  forms, dtype/LMUL clone batches, or high-level frontend work.
- No changes to emitted target sequence, ABI, runtime correctness, or
  performance claims. `ssh rvv` evidence is not required unless those claims
  change.
- No movement of RVV semantics into common EmitC/export.
- No source-front-door positive routes, descriptor-driven computation, or
  direct C/source-export route.
- No compatibility wrapper that preserves old i32m1 route authority.

## Validation Plan

1. Validate and start this Trellis task.
2. Build the focused C++ plugin test target.
3. Run `build/bin/tianchenrv-rvv-extension-plugin-test`.
4. Run representative focused lit/FileCheck coverage only if emitted fixtures
   change; otherwise rely on existing C++ route sequence checks plus
   `check-tianchenrv`.
5. Run bounded provider scan over
   `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`.
6. Run active-authority scan over touched RVV planning/provider/test/spec/task
   files.
7. Run `git diff --check`.
8. Run `cmake --build build --target check-tianchenrv -j2`.

## Technical Notes

Relevant specs read:

- `.trellis/spec/index.md`
- `.trellis/spec/implementation-stack/index.md`
- `.trellis/spec/implementation-stack/compiler-stack-contract.md`
- `.trellis/spec/extension-plugins/index.md`
- `.trellis/spec/extension-plugins/rvv-plugin.md`
- `.trellis/spec/lowering-runtime/index.md`
- `.trellis/spec/lowering-runtime/emitc-route.md`
- `.trellis/spec/testing/index.md`
- `.trellis/spec/testing/mlir-testing-contract.md`
- `.trellis/spec/variant-pipeline/index.md`
- `.trellis/spec/variant-pipeline/generation-selection-tuning.md`
- `.trellis/spec/guides/index.md`
- `.trellis/spec/guides/plugin-locality-review-guide.md`
- `.trellis/spec/guides/compute-boundary-review-guide.md`

Relevant previous task read:

- `.trellis/tasks/archive/2026-05/05-24-05-24-stage2-rvv-computed-mask-accumulation-statement-plan-ownership/prd.md`

Initial implementation surface:

- `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`
- `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
- `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
- `test/Plugin/RVVExtensionPluginTest.cpp`
- `.trellis/spec/extension-plugins/rvv-plugin.md`

## Definition Of Done

The six migrated RVV route families are consumed through one RVV-owned
provider-neutral statement-plan boundary, the provider only attaches the
returned plan before older fallback assembly, focused positive and fail-closed
C++ tests pass, authority scans show no drift, Trellis task metadata is
truthful, and the completed work is committed as one coherent change.

## Implementation Result

- Added `RVVSelectedBodyMigratedRouteStatementPlanFamily`,
  `RVVSelectedBodyMigratedRouteStatementPlan`, and
  `getRVVSelectedBodyMigratedRouteStatementPlan(...)` as the aggregate
  provider-neutral consumption boundary for the six migrated families:
  elementwise arithmetic, compare/select, base memory, computed-mask memory,
  segment2 memory, and computed-mask accumulation.
- The aggregate boundary consumes verified route-family/materialization/
  operand-binding facts by delegating to the existing RVV-owned family
  statement-plan builders. It preserves the family-built pre-loop steps, loop,
  source provenance, ABI/VL/mask/address/accumulator/source facts, and fails
  closed if more than one family claims a route or if a migrated route lacks a
  valid family plan.
- `RVVEmitCRouteProvider` now calls the aggregate boundary once, attaches the
  returned provider-ready statements, and returns before the older generic
  provider-local statement assembly path.
- The provider no longer manually sequences calls to the six migrated
  family-specific statement-plan getters. The older generic provider branches
  for some migrated operation names remain physically present as residual
  fallback code, but they are fenced by the aggregate boundary before generic
  assembly; the final provider scan found no direct family-specific getter
  calls or old family-specific statement-plan diagnostic guards in
  `RVVEmitCRouteProvider.cpp`.
- Focused C++ coverage now checks aggregate-boundary positive construction for
  elementwise arithmetic and computed-mask accumulation, empty/default
  aggregate plans for unrelated routes, and stale/missing plan dependencies
  failing through the aggregate boundary before statement construction.
- `.trellis/spec/extension-plugins/rvv-plugin.md` now records the durable
  aggregate migrated statement-plan consumption contract and review checks.

## Validation Result

- `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-24-stage2-rvv-statement-plan-provider-neutral-closure`
  passed.
- `cmake --build build --target tianchenrv-rvv-extension-plugin-test -j2`
  passed.
- `build/bin/tianchenrv-rvv-extension-plugin-test` passed with
  `RVV extension plugin smoke test passed`.
- Bounded provider scan:
  `rg -n "getRVVSelectedBody(ElementwiseArithmetic|CompareSelect|BaseMemoryMovement|ComputedMaskMemory|Segment2Memory|ComputedMaskAccumulation)RouteStatementPlan|provider requires the RVV-owned .*statement plan before generic provider-local" lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
  returned no matches.
- Aggregate provider scan:
  `rg -n "getRVVSelectedBodyMigratedRouteStatementPlan|RVVSelectedBodyMigratedRouteStatementPlan" lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
  found only the shared boundary call and local aggregate plan variable.
- Active-authority scan over newly added non-spec C++/task lines found no new
  `RVVI32M1`, `rvv-i32m1`, positive finite `tcrv_rvv.i32_*`,
  descriptor/source-front-door/direct-C/source-export, emission-plan status,
  artifact metadata, or mirror-only authority additions.
- The same scan over the spec diff only matched negative/prohibitive text for
  `artifact metadata`, `source-front-door`, `descriptor`, `direct-C`, and
  `source-export`.
- `git diff --check` passed.
- `cmake --build build --target check-tianchenrv -j2` passed: 363/363 lit
  tests passed.
- No `ssh rvv` evidence was collected because this task did not change or
  claim emitted target sequence, ABI, runtime correctness, or performance.

## Continuation Notes

- Non-migrated routes such as standalone reductions, plain MAcc,
  widening/conversion/dot routes, and residual runtime scalar splat-store
  remain outside the aggregate migrated statement-plan boundary.
- Some old generic provider statement assembly branches for migrated operation
  names remain in `RVVEmitCRouteProvider.cpp` as unreachable residue behind
  the aggregate boundary. A later cleanup can delete those residual branches
  once the remaining non-migrated generic route surface is split cleanly enough
  to avoid collateral changes.
