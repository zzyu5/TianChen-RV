# Stage2 RVV segment2 route-family owner completion

## Goal

Move segment2 route-family operand-binding authority and the segment2
provider-plan ownership boundary out of central `RVVEmitCRoutePlanning.cpp`
into `RVVEmitCSegment2RouteFamilyPlanOwners`, while keeping central route
planning neutral and dispatch-only for segment2 consumers.

The owner module must own the plan-ID and logical-operand-to-runtime-ABI-role
logic for plain segment2 deinterleave/interleave and computed-mask segment2
load/store/update. Central planning may keep shared route-analysis structs,
shared operand-binding containers, and generic closure checks, but it must no
longer encode segment2 semantics directly.

## Direction Source

- Direction title: `Stage2 RVV segment2 route-family owner completion`.
- Module owner: `RVVEmitCSegment2RouteFamilyPlanOwners` plus the segment2
  dispatch in `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: the worktree was already dirty when this task
  was created. Existing uncommitted changes in
  `lib/Plugin/RVV/EmitC/CMakeLists.txt`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`,
  `test/Plugin/RVVExtensionPluginTest.cpp`, and the untracked contraction
  owner files must be preserved and not reverted.
- Initial HEAD: `b1ac374f rvv: move macc operand bindings to owner`.
- `.trellis/.current-task` was absent, so this task was created from the
  supplied Hermes direction brief before source edits.
- Serial worker constraint: one Codex worker; no subagents, spawned agents,
  parallel agents, or multi-agent workflows.

## What I Already Know

- `.trellis/spec/index.md` and
  `.trellis/spec/extension-plugins/rvv-plugin.md` require RVV route support to
  derive from typed `tcrv_rvv` body/config/runtime facts through plugin-owned
  route-family plans and provider boundaries. Route ids, ABI strings,
  descriptors, artifact names, source-front-door markers, exact intrinsic
  spellings, and mirror metadata are not authority.
- `.trellis/spec/lowering-runtime/emitc-route.md` requires common EmitC to
  remain neutral and consume provider-built `TCRVEmitCLowerableRoute`
  payloads.
- `.trellis/spec/plugin-protocol/interfaces-and-registry.md` and
  `.trellis/spec/plugin-protocol/locality-contract.md` require extension
  semantics to live in plugin-local owner code, not in central orchestration.
- The existing segment2 owner module already owns the provider-plan boundary
  for the five active segment2 families:
  `ComputedMaskSegment2LoadUnitStore`,
  `ComputedMaskSegment2StoreUnitLoad`,
  `ComputedMaskSegment2UpdateUnitLoad`,
  `Segment2DeinterleaveUnitStore`, and
  `Segment2InterleaveUnitLoad`.
- Central `RVVEmitCRoutePlanning.cpp` still owns the segment2 operand-binding
  plan IDs, the segment2 logical-operand-to-runtime-ABI-role mapping, and the
  segment2 cases inside `deriveRVVRouteOperandBindingPlan(...)`.
- Central route planning also still exposes
  `verifyRVVSelectedBodySegment2MemoryRouteFamilyProviderPlans(...)`, which is
  currently a central segment2-specific validation surface rather than an
  owner-local one.
- `test/Plugin/RVVExtensionPluginTest.cpp` already hardcodes segment2 binding
  plan IDs and role checks in its central route-operand-binding tests, so the
  test surface will need to move to the owner API as well.

## Requirements

1. Add owner-owned segment2 route-operand binding APIs in
   `RVVEmitCSegment2RouteFamilyPlanOwners` for:
   - expected route operand binding plan ID lookup;
   - logical operand to `RuntimeABIParameterRole` mapping;
   - binding-plan derivation for the active plain and computed-mask segment2
     routes.
2. Move the segment2 operand-binding plan IDs and role-mapping bodies out of
   central `RVVEmitCRoutePlanning.cpp`.
3. Move the segment2 cases inside `deriveRVVRouteOperandBindingPlan(...)`
   into the segment2 owner module, with central code reduced to neutral owner
   dispatch.
4. Keep provider-plan ownership in the segment2 owner module and make any
   remaining central segment2 provider-plan helper a thin dispatch wrapper at
   most.
5. Preserve fail-closed behavior for:
   - unsupported memory forms;
   - wrong segment factor;
   - missing materialization facts;
   - missing or stale route-family plan mirrors;
   - wrong operand roles;
   - wrong runtime ABI order;
   - unsupported family/operation mismatches.
6. Update focused C++ tests so they assert the owner API directly rather than
   hardcoding segment2 authority in central helpers.
7. Keep the change bounded to segment2 memory movement and its provider-plan
   and operand-binding authority. Do not expand contraction, MAcc,
   compare/select, reduction, conversion, or other memory families in this
   round.

## Acceptance Criteria

- [ ] `RVVEmitCSegment2RouteFamilyPlanOwners.h/cpp` expose owner-owned
      segment2 plan-ID lookup, role lookup, and binding-plan derivation
      helpers.
- [ ] `RVVEmitCRoutePlanning.cpp` no longer defines segment2 operand-binding
      plan-ID constants or segment2 logical-operand role-mapping bodies.
- [ ] Central route planning no longer hardcodes segment2-specific binding
      cases inside `deriveRVVRouteOperandBindingPlan(...)`; it dispatches to
      the segment2 owner module instead.
- [ ] Any remaining segment2 provider-plan validation in central route
      planning is a neutral wrapper or is removed in favor of the owner module.
- [ ] Representative plain segment2 and computed-mask segment2 cases derive
      their binding-plan facts through the owner module and still fail closed
      on stale or mismatched plan/role/family inputs.
- [ ] Focused C++ tests cover owner plan-ID lookup, owner role lookup, owner
      binding-plan derivation, provider-plan validation, and wrong-role /
      wrong-plan fail-closed behavior for representative plain and
      computed-mask segment2 cases.
- [ ] Generated-bundle dry-runs for representative plain and computed-mask
      segment2 routes still pass.
- [ ] Bounded symbol scans show segment2 binding authority has moved out of
      central planner logic and into the owner module, leaving central code
      as neutral dispatch only.
- [ ] `git diff --check` passes.
- [ ] Focused RVV plugin build/test passes, and `check-tianchenrv` passes or
      the exact blocker is recorded.
- [ ] Trellis task status, journal, archive state, and commit state are
      truthful at the end of the round.

## Technical Approach

Use the owner-module pattern already established by the MAcc and segment2
provider-plan work:

```text
selected typed segment2 route analysis
  -> central neutral route analysis/fact collection
  -> segment2 owner derives expected plan ID / logical role mapping
  -> central generic closure comparison against route description mirrors
  -> segment2 owner derives the route operand binding plan
  -> segment2 owner derives or validates the provider plan
  -> provider-built route facts continue downstream
```

Central route planning should keep shared containers and generic closure
validation, but all segment2-specific binding semantics should live in the
segment2 owner module. The owner module can mirror the existing MAcc-owner
shape: static plan IDs, explicit logical operand role mapping, owner-local
binding-plan derivation, and owner-local provider-plan validation helpers.

## Out Of Scope

- Contracting MAcc or widening-contraction ownership again.
- Reworking already-completed MAcc, control-policy, or statement-plan owner
  boundaries.
- Adding new RVV operation coverage, segment widths, dtype/LMUL clone batches,
  source-front-door positives, broad runtime claims, dashboard/report work, or
  unrelated memory-family refactors.
- Changing generated C semantics, intrinsic spelling, runtime ABI semantics,
  target export mechanics, or artifact naming beyond what is needed to keep
  the segment2 owner boundary fail-closed.

## Validation Plan

1. Validate the task context:
   `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-30-05-30-stage2-rvv-segment2-route-family-owner-completion`.
2. Build the focused RVV plugin C++ target:
   `cmake --build build --target tianchenrv-rvv-extension-plugin-test -j2`.
3. Run the focused plugin binary:
   `./build/bin/tianchenrv-rvv-extension-plugin-test`.
4. Run focused generated-bundle dry-runs for representative plain and
   computed-mask segment2 routes.
5. Run fail-closed probes for stale plan ID, wrong logical role, missing
   materialization facts, and segment factor mismatch.
6. Run `git diff --check`.
7. Run `cmake --build build --target check-tianchenrv -j2`, or record the
   exact blocker if it does not pass in this round.

## Technical Notes

Required specs and prior context:

- `.trellis/spec/index.md`
- `.trellis/spec/extension-plugins/rvv-plugin.md`
- `.trellis/spec/lowering-runtime/emitc-route.md`
- `.trellis/spec/plugin-protocol/interfaces-and-registry.md`
- `.trellis/spec/plugin-protocol/locality-contract.md`
- `.trellis/spec/testing/mlir-testing-contract.md`
- `.trellis/workspace/codex/journal-18.md`
- archived segment2 provider-plan owner task:
  `.trellis/tasks/archive/2026-05/05-26-stage2-rvv-provider-route-family-planning-registry-owner/prd.md`
- archived MAcc owner completion task:
  `.trellis/tasks/archive/2026-05/05-30-05-30-stage2-rvv-macc-operand-binding-owner-completion/prd.md`

Primary production files:

- `include/TianChenRV/Plugin/RVV/RVVEmitCSegment2RouteFamilyPlanOwners.h`
- `lib/Plugin/RVV/EmitC/RVVEmitCSegment2RouteFamilyPlanOwners.cpp`
- `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
- `test/Plugin/RVVExtensionPluginTest.cpp`

## Definition Of Done

Segment2 route-operand binding and provider-plan authority are owner-local,
central planning is dispatch-only for segment2-specific semantics, focused
tests and dry-runs pass, and one coherent commit records the completed work.
