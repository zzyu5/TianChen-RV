# Expand: Stage2 RVV segment2 route-family provider-boundary closure

## Goal

Close the RVV plugin-owned provider preflight boundary for the segment2
memory route family before `TCRVEmitCLowerableRoute` construction.

This round extends the provider-boundary invariant just closed for regular
computed-mask memory to the segment2 route family:

- `segment2_deinterleave_unit_store`;
- `segment2_interleave_unit_load`;
- `computed_masked_segment2_load_unit_store`;
- `computed_masked_segment2_store_unit_load`;
- `computed_masked_segment2_update_unit_load`.

## Direction Source

- Direction title: `Expand: Stage2 RVV segment2 route-family provider-boundary closure`.
- Module owner: RVV selected-body EmitC route-provider preflight for the
  segment2 memory route family.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean.
- Initial HEAD: `133bec75 rvv: close regular computed-mask memory provider boundary`.
- `.trellis/.current-task` was absent, so this task was created from the
  supplied Hermes Direction Brief before source edits.
- Serial worker constraint from the brief: one Codex worker; no subagents,
  spawned agents, parallel agents, or multi-agent workflows.

## What I Already Know

- The previous archived task added
  `verifyRVVSelectedBodyRegularComputedMaskMemoryRouteProviderFacts(...)` and
  wired it into `RVVEmitCRouteProvider` before route construction for five
  regular non-segment computed-mask memory routes.
- Segment2 route-family provider planning already has an explicit owner
  boundary for computed-mask segment2 load/store/update and plain segment2
  deinterleave/interleave.
- Segment2 statement-plan construction already consumes the owner-built
  provider plan and should not rediscover segment2 sub-family dispatch through
  operation names, ABI strings, route ids, artifact names, intrinsic mirrors,
  or common EmitC behavior.
- The remaining gap is one provider-boundary preflight immediately before
  accepting a segment2 `TCRVEmitCLowerableRoute`: the provider currently fetches
  segment2 plans and statement plans, then adds headers, type mappings, ABI
  mappings, and statement-plan owner selection, but the segment2 family lacks a
  dedicated same-analysis preflight analogous to runtime-scalar and regular
  computed-mask memory.
- `tcrv.exec.mem_window` and `tcrv.exec.runtime_param` bind ABI/runtime roles
  only. They do not define RVV compute, dtype, segment direction, memory form,
  mask semantics, inactive/pass-through behavior, field roles, or route support.
- Common EmitC must remain neutral and only materialize the provider-built
  route. It must not infer RVV dtype, SEW, LMUL, operation kind, memory form,
  ABI order, mask policy, segment tuple facts, or intrinsic spelling.

## Requirements

1. Add or repair one coherent RVV-owned provider-boundary preflight module for
   the five segment2 route forms:
   - `segment2_deinterleave_unit_store`;
   - `segment2_interleave_unit_load`;
   - `computed_masked_segment2_load_unit_store`;
   - `computed_masked_segment2_store_unit_load`;
   - `computed_masked_segment2_update_unit_load`.
2. The preflight must run after same-analysis route-family provider plans,
   materialization facts, memory operand-binding facts, route-control facts,
   and segment2 statement-plan construction are available, but before
   `TCRVEmitCLowerableRoute` is accepted.
3. The preflight must consume and validate:
   - typed config facts for dtype, SEW, LMUL, policy, vector/mask C types,
     tuple type, field type, setvl, segment load/store, masked segment load/store,
     tuple create/extract, field load/store, and arithmetic/update leaves where
     required;
   - segment2 route-family provider plan facts, including exactly one selected
     segment2 owner, segment count, operation kind, segment direction, memory
     form, tuple and field roles, provider-supported mirror fields, required
     headers, type mappings, ABI mappings, and intrinsic mirrors as mirrors only;
   - plain segment2 family facts for deinterleave/interleave routes;
   - computed-mask memory family facts for computed-mask segment2 routes,
     including computed-mask producer/source, compare inputs, mask type mapping,
     inactive-lane/pass-through policy, and update arithmetic kind where
     applicable;
   - memory operand-binding facts for runtime `src`, `dst`, `field0`, `field1`,
     `cmp_lhs`, `cmp_rhs`, pass-through/source field values, and runtime `n`
     as applicable to each form;
   - route-control facts for runtime `n`/AVL/VL, typed config, selected
     capability, tail/mask policy, runtime ABI order, and provider mirrors;
   - statement-plan owner readiness, including provider-ready setvl, loop,
     compare/mask, tuple, segment load/store, field load/store, update, and
     source-provenance leaves required by the selected form.
4. The provider must fail closed before route construction when selected
   segment2 input lacks fresh materialization facts, route-family provider plan
   facts, memory operand-binding facts, route-control facts, segment tuple or
   field-role facts, computed-mask producer facts, inactive/pass-through facts,
   mask type mapping, runtime `n`/AVL/VL facts, ABI order, provider mirror
   consistency, or statement-plan owner readiness.
5. The preflight diagnostics must name the relevant segment2 form, provider
   boundary context, and missing or stale logical fact where practical.
6. Preserve existing runtime-scalar and regular computed-mask memory
   provider-boundary behavior as non-regression paths. Do not reopen their
   semantics except for shared helper reuse that stays RVV-owned.
7. Keep common EmitC, target artifact export, scripts, route ids, ABI strings,
   artifact names, exact intrinsic strings, descriptors, and mirror/status
   fields out of route authority.
8. Do not add new segment2 route forms, dtype/LMUL clone batches, source
   front-door routes, high-level frontend lowering, one-intrinsic wrapper
   dialects, dashboards, report-only changes, or broad smoke matrices.

## Acceptance Criteria

- [ ] PRD and task context truthfully represent the Hermes Direction Brief,
      current repo state, previous archived task, and relevant Trellis specs.
- [ ] A focused production diff lands in RVV route planning/provider ownership
      code and directly necessary headers. Tests alone are not sufficient.
- [ ] `RVVEmitCRouteProvider` or the RVV planning layer calls the segment2
      provider preflight after same-analysis family/provider, materialization,
      memory-binding, route-control, and statement-plan facts exist, but before
      creating or accepting `TCRVEmitCLowerableRoute`.
- [ ] Positive C++ coverage proves provider-boundary acceptance for
      `segment2_deinterleave_unit_store`.
- [ ] Positive C++ coverage proves provider-boundary acceptance for
      `segment2_interleave_unit_load`.
- [ ] Positive C++ coverage proves provider-boundary acceptance for
      `computed_masked_segment2_load_unit_store`.
- [ ] Positive C++ coverage proves provider-boundary acceptance for
      `computed_masked_segment2_store_unit_load`.
- [ ] Positive C++ coverage proves provider-boundary acceptance for
      `computed_masked_segment2_update_unit_load`.
- [ ] C++ fail-closed coverage proves targeted diagnostics for missing or stale
      segment2 provider plan, wrong operation kind, missing field binding,
      wrong segment count or tuple type, missing mask mapping for computed-mask
      segment2, wrong inactive/pass-through policy, wrong runtime `n`/AVL/VL
      facts, ABI order mismatch, stale provider mirror, stale route id or
      metadata mirror, and missing statement-plan owner readiness.
- [ ] Focused lit non-regression passes for directly related segment2 selected
      tests if applicable.
- [ ] Runtime-scalar and regular computed-mask provider-boundary
      non-regression checks pass.
- [ ] Bounded production-diff authority scan finds no new central ad hoc,
      name-derived, metadata-derived, descriptor-derived, ABI-string-derived,
      script-derived, artifact-name-derived, common-EmitC-derived,
      source-front-door-derived, route-id-derived, exact-intrinsic-derived,
      direct-route-entry-only, pre-realized-fixture-only, or legacy-i32-derived
      authority.
- [ ] `git diff --check` passes.
- [ ] Focused tests pass, and `check-tianchenrv` passes or the exact blocker is
      recorded.
- [ ] Task status, context, and journal are truthful; if acceptance passes, the
      task is finished/archived and one coherent commit is created.
- [ ] No `ssh rvv` is required unless this round claims new runtime,
      correctness-on-hardware, or performance evidence.

## Technical Approach

The intended production flow after this task is:

```text
selected RVV segment2 variant
  -> typed/pre-realized tcrv_rvv segment2 body
  -> route-family provider plan verification
  -> route materialization facts
  -> memory operand-binding facts
  -> RVV-owned route-control provider plan
  -> RVV-owned segment2 route-family provider plan
  -> RVV-owned segment2 statement plan
  -> RVV-owned segment2 provider-boundary preflight
  -> TCRVEmitCLowerableRoute
  -> neutral common EmitC materialization
```

Implementation should prefer a planning/provider helper analogous to the
runtime-scalar and regular computed-mask memory preflights, while preserving
the existing segment2 owner boundary and statement-plan owner boundary. The
preflight should validate the already-built same-analysis facts rather than
rebuilding segment2 route support from route names, metadata mirrors, intrinsic
spellings, ABI strings, or target artifact facts.

## Validation Plan

1. Validate task context:
   `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-31-05-31-stage2-rvv-segment2-route-family-provider-boundary-closure`.
2. Build focused plugin test:
   `cmake --build build --target tianchenrv-rvv-extension-plugin-test -j2`.
3. Run focused C++ plugin coverage:
   `./build/bin/tianchenrv-rvv-extension-plugin-test`.
4. Run directly related segment2 lit filters if available.
5. Run runtime-scalar and regular computed-mask provider-boundary
   non-regression filters.
6. Run bounded authority scans over touched RVV planning/provider/test files.
7. Run `git diff --check`.
8. Run `cmake --build build --target check-tianchenrv -j2`, or record the
   exact blocker.

## Out of Scope

- Do not reopen regular computed-mask memory, runtime-scalar computed-mask
  memory, widening dot, MAcc, compare/select, standalone reduction, conversion,
  dtype/LMUL clone batches, high-level frontend lowering, one-intrinsic wrapper
  dialects, common EmitC semantics, target artifact semantics, dashboards,
  report work, broad smoke matrices, or evidence-only tasks.
- No edits to `tcrv.exec` that encode compute semantics.
- No RVV semantic choices in common EmitC or target artifact code.
- No Python implementation of compiler core, dialects, route provider,
  capability model, lowering, or emission. Python remains evidence tooling.
- No runtime, correctness, or performance claim without real `ssh rvv`
  evidence.

## Technical Notes

- Specs read before PRD:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/core-dialect/tcrv-exec-contract.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`, and
  `.trellis/spec/testing/index.md`.
- Previous archived task read:
  `.trellis/tasks/archive/2026-05/05-31-05-31-stage2-rvv-regular-computed-mask-memory-provider-boundary/`.
- Related segment2 archived tasks read:
  `.trellis/tasks/archive/2026-05/05-30-05-30-stage2-rvv-segment2-route-family-provider-plan-owner-boundary/` and
  `.trellis/tasks/archive/2026-05/05-31-stage2-rvv-segment2-memory-selected-body-owner-cleanup/`.
- Relevant workspace journal entry read:
  `.trellis/workspace/codex/journal-19.md`, Session 352 and related segment2 entries.
- Primary production files for inspection and possible edit:
  `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`, and
  `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`.
- Direct consumer evidence to keep focused:
  `test/Plugin/RVVExtensionPluginTest.cpp`, directly related segment2 selected
  lit tests if applicable, and runtime-scalar / regular computed-mask memory
  provider-boundary non-regression checks.
- No blocking user question remains; the supplied Direction Brief and specs are
  specific enough for one bounded implementation round.

## Completion Notes

Implemented the segment2 provider-boundary preflight as
`verifyRVVSelectedBodySegment2RouteProviderFacts(...)` in RVV route planning and
wired it from `RVVEmitCRouteProvider` after segment2 provider-plan and
statement-plan owner selection, before `TCRVEmitCLowerableRoute` construction.

Covered production forms:

- `segment2_deinterleave_unit_store`;
- `segment2_interleave_unit_load`;
- `computed_masked_segment2_load_unit_store`;
- `computed_masked_segment2_store_unit_load`;
- `computed_masked_segment2_update_unit_load`.

The preflight validates same-analysis typed config/materialization facts,
segment2 provider-plan facts, computed-mask producer and inactive/pass-through
facts, mask type mapping, memory operand binding, ABI order, field0/field1
source/destination bindings, runtime n/AVL/VL route-control facts,
provider-supported/header/type/intrinsic mirrors as mirrors only, route-id
mirror consistency, and migrated segment2 statement-plan owner readiness before
route acceptance. Unsupported or stale segment2 facts fail closed with targeted
diagnostics before route construction.

Evidence run:

- `cmake --build build --target tianchenrv-rvv-extension-plugin-test -j2`;
- `./build/bin/tianchenrv-rvv-extension-plugin-test`;
- focused lit filter for segment2 plus runtime-scalar and regular
  computed-mask memory non-regression: 34/34 selected tests passed;
- `cmake --build build --target check-tianchenrv -j2`: 464/464 passed;
- `git diff --check`;
- bounded diff authority scan over touched RVV planning/provider/test files.

Self-repair performed:

- Fixed an initial over-strict segment2 computed-mask preflight condition that
  incorrectly required non-segment computed-mask mask-tail route-family mirrors
  for segment2 routes.
- Tightened statement-plan readiness to reject empty callees and require the
  computed-mask segment2 load tuple-create leaf.

No `ssh rvv` evidence was required or run because this round makes compiler
provider-boundary and route-construction claims only; it does not claim new
runtime hardware correctness or performance.
