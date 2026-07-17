# Stage2 RVV Segment2 Provider Route-Construction Owner

## Goal

Make `RVVEmitCRouteProvider` consume the existing segment2 selected-body
route-family planning owner/provider-plan interface while constructing
`TCRVEmitCLowerableRoute` objects. The production route envelope for active
segment2 consumers must validate and derive route id, headers, type mappings,
ABI mappings, provider-supported mirrors, operand-binding facts, and statement
plan selection from typed body/config/runtime/mask/memory/arithmetic facts plus
the owner-built segment2 provider plan, not from route ids, metadata, ABI
strings, artifact names, or central ad hoc provider branches.

The primary active consumer is `computed_masked_segment2_update_unit_load`.
At least one adjacent segment2 family must go through the same route
construction interface; if the change is local, all five active segment2
families should be covered.

## Direction Source

- Direction title: `Stage2 RVV segment2 provider route-construction owner`.
- Module owner: RVV plugin-local `TCRVEmitCLowerableRoute` construction
  boundary for registered segment2 route families.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean.
- Initial HEAD: `e8d9110a rvv: add segment2 route-family planning owners`.
- No `.trellis/.current-task` existed, so this task was created from the
  supplied Hermes direction brief before source edits.

## What I Already Know

- The current authority chain is selected `tcrv.exec` RVV variant -> typed
  low-level `tcrv_rvv` body -> RVV plugin legality / selected-body
  realization / route provider -> `TCRVEmitCLowerableRoute` -> neutral common
  EmitC -> target artifact -> `ssh rvv` evidence for runtime/correctness
  claims.
- The archived predecessor task
  `.trellis/tasks/archive/2026-05/05-26-stage2-rvv-provider-route-family-planning-registry-owner`
  added `RVVSelectedBodySegment2RouteFamilyPlanningOwner` and
  `RVVSelectedBodySegment2RouteFamilyProviderPlan`, registered five segment2
  families, and rewired segment2 statement-plan construction through the
  provider-plan boundary.
- Current production code still constructs the `TCRVEmitCLowerableRoute`
  envelope in `RVVEmitCRouteProvider.cpp` before the segment2 provider plan is
  obtained. The provider plan is currently consumed only through
  `getRVVSelectedBodyMigratedRouteStatementPlan(...)` ->
  `getRVVSelectedBodySegment2MemoryRouteStatementPlan(...)`.
- `TCRVEmitCLowerableRoute` carries route id/kind, headers, type mappings,
  ABI value mappings, source provenance, call-opaque pre-loop steps, and
  structured loops. It has no metadata authority field, so route-construction
  proof must be visible through provider validation, route payload facts, and
  focused C++ tests that inspect the built route.
- Common EmitC must remain neutral. It must not infer RVV dtype, policy,
  memory form, route support, intrinsic spelling, ABI order, or segment2
  family from metadata or route ids.

## Requirements

1. Add a narrow RVV-local route-construction input derived from
   `RVVSelectedBodySegment2RouteFamilyProviderPlan`, or thread that provider
   plan directly into the route provider path, before the route envelope is
   finalized.
2. `RVVEmitCRouteProvider` must validate active segment2 route construction
   through the selected-body segment2 planning owner/provider-plan interface,
   not only through statement-plan construction.
3. The migrated provider path must include `computed-mask segment2 update` and
   at least one adjacent segment2 route family. Cover all five active segment2
   families if the implementation stays local.
4. The provider-built route for segment2 consumers must derive or validate:
   route id, required headers, C type mappings, ABI order/value mappings,
   operand binding closure, provider-supported mirror labels, statement-plan
   selection, source provenance, and intrinsic/type payloads from typed
   body/config/runtime/mask/memory/arithmetic facts plus the owner-built
   segment2 provider plan.
5. Unsupported or inconsistent selected-body family, missing provider plan,
   stale route id/metadata mirrors, wrong op kind, segment factor mismatch,
   memory form mismatch, stream role mismatch, mask producer mismatch,
   same-VL/runtime `n` mismatch, missing passthrough, arithmetic kind mismatch,
   dtype/config mismatch, unsupported policy, or wrong ABI order must fail
   closed with targeted diagnostics before common EmitC materialization.
6. If the provider already effectively consumes part of the plan, remove or
   demote duplicated provider-side ad hoc logic rather than adding wrappers.
7. Do not add new RVV operation coverage, segment widths, dtype/LMUL clone
   batches, source-front-door positives, high-level frontend/Linalg routes,
   common EmitC RVV semantics, dashboards, reports, or evidence-only changes.

## Acceptance Criteria

- [ ] Production C++ changes touch `RVVEmitCRouteProvider` and any narrow
      `RVVEmitCRoutePlanning` API needed to make segment2 provider-plan facts
      route-construction input.
- [ ] `computed_masked_segment2_update_unit_load` and at least one adjacent
      active segment2 route prove route construction consumes the selected-body
      family owner plus owner-built provider plan. Prefer all five active
      segment2 families exactly once.
- [ ] Focused C++ tests inspect built `TCRVEmitCLowerableRoute` payloads for
      segment2 consumers: route id, ABI mappings/order, headers/type mappings,
      pre-loop/loop statements, source roles, provider-supported mirror
      agreement, and owner-selected statement-plan facts.
- [ ] Focused fail-closed tests cover stale/missing provider plan or route
      construction facts, including selected-family mismatch, wrong op kind,
      segment factor mismatch, memory form mismatch, mask/runtime/ABI mismatch,
      unsupported arithmetic kind, dtype/config/policy mismatch, and stale
      route-id/metadata mirror attempts where the current API exposes them.
- [ ] Generated-bundle dry-run covers computed-mask segment2 update and one
      migrated adjacent segment2 family.
- [ ] Runtime evidence: rerun `ssh rvv` for computed-mask segment2 update, or
      prove exact artifact preservation and run one migrated representative.
- [ ] Focused non-regression covers computed-mask segment2 store/load,
      segment2 deinterleave/interleave, masked elementwise, reduction,
      scalar-broadcast, conversion, runtime-scalar MAcc, compare/select, MAcc,
      contraction, and base memory route-entry paths.
- [ ] Bounded touched-file authority scan finds no new positive legacy-i32,
      source-front-door, descriptor, ABI-string-derived, artifact-name-derived,
      script-derived, metadata-derived, route-id-derived,
      exact-intrinsic-derived, or common-EmitC-derived authority.
- [ ] `git diff --check` passes.
- [ ] Focused build/test targets pass, and `check-tianchenrv` passes or an
      exact blocker is recorded and the task remains open.
- [ ] Trellis task status, journal, archive, and one coherent commit are
      completed if all acceptance criteria are satisfied.

## Validation Plan

1. Start the Trellis task after PRD/context setup.
2. Implement the route-construction/provider-plan threading.
3. Build the focused RVV plugin test target.
4. Run `./build/bin/tianchenrv-rvv-extension-plugin-test`.
5. Run generated-bundle dry-runs for direct pre-realized
   `computed_masked_segment2_update_unit_load` and one adjacent segment2
   family.
6. Run direct route-entry non-regression dry-runs for the listed active owners.
7. Run representative `ssh rvv` evidence as required by the changed segment2
   route.
8. Run bounded authority scan, `git diff --check`, and `check-tianchenrv`.

## Out Of Scope

- New operation support or broad Stage2 coverage expansion.
- New segment factor beyond segment2.
- New dtype/LMUL clone batches.
- Linalg/Vector/StableHLO frontend lowering.
- Source-front-door positive routes.
- Descriptor/direct-C/source-export route resurrection.
- Common EmitC/export RVV semantic inference.
- Dashboard/report-only/helper-only changes.
- Weakening existing owners for computed-mask segment2 update/store/load,
  plain segment2 deinterleave/interleave, masked elementwise, reduction,
  scalar-broadcast, conversion, runtime-scalar MAcc, compare/select, MAcc,
  contraction, or base memory.

## Technical Notes

- Specs read:
  - `.trellis/spec/index.md`
  - `.trellis/spec/extension-plugins/rvv-plugin.md`
  - `.trellis/spec/lowering-runtime/emitc-route.md`
  - `.trellis/spec/plugin-protocol/interfaces-and-registry.md`
  - `.trellis/spec/testing/mlir-testing-contract.md`
  - `.trellis/spec/guides/index.md`
  - capability-first, plugin-locality, and compute-boundary guides
- Predecessor context read:
  - `.trellis/tasks/archive/2026-05/05-26-stage2-rvv-provider-route-family-planning-registry-owner/prd.md`
  - `.trellis/workspace/codex/journal-16.md`
- Initial code surface:
  - `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`
  - `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`
  - `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
  - `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
  - `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp`
  - `lib/Target/RVV/RVVTargetSupportBundle.cpp`
  - `scripts/rvv_generated_bundle_abi_e2e.py`
  - `test/Plugin/RVVExtensionPluginTest.cpp`

## Definition Of Done

Segment2 route construction in `RVVEmitCRouteProvider` consumes the same
selected-body family/provider-plan authority that statement planning consumes,
focused tests and route-entry evidence pass, Trellis state is truthful, and one
coherent commit records the completed work.
