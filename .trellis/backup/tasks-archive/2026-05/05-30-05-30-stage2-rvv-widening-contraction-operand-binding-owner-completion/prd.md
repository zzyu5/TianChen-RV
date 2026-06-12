# Stage2 RVV widening-contraction operand-binding owner completion

## Goal

Complete the widening-contraction route-family operand-binding ownership split
by moving widening MAcc and widening dot/reduce route operand-binding plan
authority out of central `RVVEmitCRoutePlanning.cpp` and into an RVV
plugin-local owner boundary. The owner must cover `WideningMAccAdd`,
`WideningDotReduceAdd`, `StridedInputWideningDotReduceAdd`,
`ComputedMaskWideningDotReduceAdd`, and
`ComputedMaskStridedInputWideningDotReduceAdd` while preserving the existing
typed selected-body route path and generated-bundle behavior.

## Direction Source

- Direction title: `Switch: Stage2 RVV widening-contraction operand-binding owner completion`.
- Module owner: the RVV widening-contraction route-family plan owner for
  `WideningMAccAdd` and widening dot/reduce variants.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean.
- Initial HEAD: `b1ac374f rvv: move macc operand bindings to owner`.
- `.trellis/.current-task` was absent, so this task was created from the Hermes
  Direction Brief before source edits.
- Serial worker constraint: one Codex worker; no subagents, spawned agents,
  parallel agents, or multi-agent workflows.

## What I Already Know

- The predecessor archived task
  `.trellis/tasks/archive/2026-05/05-30-05-30-stage2-rvv-macc-operand-binding-owner-completion/`
  moved non-widening MAcc plan IDs, logical operand role mapping, binding-plan
  derivation, and provider verification plan-id checks into
  `RVVEmitCMAccRouteFamilyPlanOwners`.
- `.trellis/spec/index.md` and
  `.trellis/spec/extension-plugins/rvv-plugin.md` require RVV route support to
  derive from selected typed `tcrv_rvv` body/config/runtime facts through
  plugin-owned route-family plans and provider boundaries. Route ids, ABI
  strings, descriptors, artifact names, source-front-door markers, exact
  intrinsic spellings, and mirror metadata are not authority.
- `.trellis/spec/extension-plugins/rvv-plugin.md` requires direct contraction
  provider behavior to remain RVV owner-local for active routes:
  `widening_macc_add`, `widening_dot_reduce_add`,
  `strided_input_widening_dot_reduce_add`,
  `computed_masked_widening_dot_reduce_add`, and
  `computed_masked_strided_input_widening_dot_reduce_add`.
- `.trellis/spec/lowering-runtime/emitc-route.md` requires common EmitC to
  remain neutral and consume provider-built `TCRVEmitCLowerableRoute` payloads;
  common EmitC must not infer RVV dtype, ABI, schedule, or operation semantics.
- The task brief says bounded inspection after `b1ac374f` still found central
  plan-id, logical role, binding-plan assembly, and provider-plan verification
  dispatch authority for the widening-contraction routes.

## Requirements

1. Introduce or complete an RVV plugin-local widening-contraction
   route-family plan owner API for operand-binding plan identity, logical
   operand to `RuntimeABIParameterRole` mapping, binding-plan derivation, and
   provider-plan verification dispatch.
2. Cover all active direct widening-contraction routes:
   `WideningMAccAdd`, `WideningDotReduceAdd`,
   `StridedInputWideningDotReduceAdd`,
   `ComputedMaskWideningDotReduceAdd`, and
   `ComputedMaskStridedInputWideningDotReduceAdd`.
3. Move widening-contraction-specific plan IDs and logical operand role mapping
   out of central `RVVEmitCRoutePlanning.cpp`. Central code may call the owner
   through neutral dispatch but must not locally choose widening-contraction
   accumulator/source/result/stride/mask/runtime semantics.
4. Move or delegate widening-contraction binding-plan derivation so central
   `deriveRVVRouteOperandBindingPlan(...)` no longer assembles those logical
   operands from operation names.
5. Provider validation must consume owner-owned plan IDs and fail closed for
   wrong plan ID, wrong logical operand role, missing accumulator, wrong
   source/result dtype relation, missing or wrong mask binding, missing or
   wrong stride binding, wrong runtime ABI order, stale mirror metadata, or
   descriptor/name/route-id/artifact-derived binding claims.
6. Preserve existing route-family materialization, direct contraction provider
   plan/statement behavior, generated artifacts, and unrelated owner behavior.
7. Keep central route planning as shared containers, generic closure checks,
   typed fact collection, and neutral owner dispatch only.
8. Add focused C++ tests proving owner-owned plan ID lookup, logical operand
   role mapping, binding-plan derivation, provider validation, and wrong-role
   or wrong-plan fail-closed behavior for representative widening MAcc, plain
   widening dot, strided widening dot, computed-mask widening dot, and
   computed-mask strided widening dot cases.

## Acceptance Criteria

- [ ] A widening-contraction owner header/cpp owns operand-binding plan IDs,
      logical operand role mapping, binding-plan derivation, and provider-plan
      verification dispatch for the five active widening-contraction routes.
- [ ] `RVVEmitCRoutePlanning.cpp` no longer carries central
      widening-contraction operand-binding plan ID constants, role mapping
      switch bodies, binding-plan assembly, or family-specific provider-plan
      verification selection except neutral owner dispatch.
- [ ] Provider validation compares widening-contraction route descriptions and
      route operand binding closure against owner-owned plan IDs and roles.
- [ ] Focused C++ tests cover positive owner behavior and fail-closed wrong
      plan/wrong role behavior for representative widening MAcc, plain widening
      dot, strided widening dot, computed-mask widening dot, and computed-mask
      strided widening dot cases.
- [ ] Direct generated-bundle dry-runs for representative affected existing
      paths prove no artifact boundary regression.
- [ ] Bounded symbol scan proves migrated widening-contraction plan IDs and
      role/assembly cases no longer appear as central authority in
      `RVVEmitCRoutePlanning.cpp` or central headers except neutral dispatch or
      justified shared containers.
- [ ] Authority scan over touched production/test files finds no new
      central ad hoc, name-derived, metadata-derived, descriptor-derived,
      ABI-string-derived, script-derived, artifact-name-derived,
      common-EmitC-derived, source-front-door-derived, route-id-derived,
      exact-intrinsic-derived, direct-route-entry-only,
      pre-realized-fixture-only, or legacy-i32-derived authority.
- [ ] `git diff --check` passes.
- [ ] Focused RVV plugin C++ build/test passes.
- [ ] `tcrv-opt` and `tcrv-translate` build if affected.
- [ ] `check-tianchenrv` passes, or the exact blocker is recorded.
- [ ] `ssh rvv` is not required unless this round changes emitted runtime
      semantics or claims new runtime/correctness/performance behavior.

## Technical Approach

Reuse the owner-module pattern established by the MAcc and segment2 owners:

```text
selected typed widening-contraction route analysis
  -> central neutral route analysis/fact collection
  -> widening-contraction owner derives route-operand binding plan identity
  -> widening-contraction owner maps logical operands to runtime ABI roles
  -> widening-contraction owner derives the route operand binding plan
  -> central generic closure comparison against route description mirrors
  -> owner-owned provider verification dispatch
  -> materialization facts / direct contraction provider plan
  -> RVVEmitCRouteProvider builds TCRVEmitCLowerableRoute
```

The owner may live in a new widening-contraction owner module or in an existing
direct contraction family owner module if one already exists. The choice should
follow current code locality after inspection. Central route planning should
retain shared data structures and closure validation but delegate the
widening-contraction family semantics.

## Out of Scope

- New widening contraction operation coverage, dtype/LMUL clone batches,
  direct route-entry shortcuts, source-front-door routes, high-level
  Linalg/frontend lowering, one-intrinsic wrapper dialects, broad selected-body
  realization rewrites, performance tuning, dashboards, reports, or broad smoke
  matrices.
- Reworking completed non-widening MAcc owner behavior except for shared
  dispatch integration required by this module.
- Moving unrelated elementwise, memory, segment2, compare/select, standalone
  reduction, conversion, or residual owners.
- Changing emitted runtime semantics or making new runtime/correctness/
  performance claims.

## Validation Plan

1. Validate task context with
   `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-30-05-30-stage2-rvv-widening-contraction-operand-binding-owner-completion`.
2. Build focused plugin test:
   `cmake --build build --target tianchenrv-rvv-extension-plugin-test -j2`.
3. Run focused plugin C++ coverage:
   `./build/bin/tianchenrv-rvv-extension-plugin-test`.
4. Build touched tools if affected:
   `cmake --build build --target tcrv-opt tcrv-translate -j2`.
5. Run representative generated-bundle dry-runs for affected existing
   widening-contraction selected-body paths.
6. Run bounded central/owner symbol scans and authority scans.
7. Run `git diff --check`.
8. Run `cmake --build build --target check-tianchenrv -j2`, or record the
   exact blocker.
