# Stage2 RVV Compare/Select Mask Route-Family Owner

## Goal

Introduce or extract a plugin-local RVV compare/select mask route-family owner
across the active selected-body, planning/provider, and target artifact ABI
consumer path. The primary migrated consumer must be an existing compare/select
route. One adjacent existing masked elementwise or computed-mask consumer must
prove that compare-produced mask facts are carried as typed family-level route
facts, not as route names, metadata, ABI strings, scripts, artifact names, or
common EmitC decisions.

The completed route must stay on the current RVV-first chain:

```text
selected tcrv.exec RVV variant
  -> typed low-level tcrv_rvv compare/select or mask-consuming body
  -> RVV plugin-owned compare/select mask route-family ownership
  -> RVV provider route facts and statement plans
  -> provider-built TCRVEmitCLowerableRoute
  -> neutral common EmitC materialization
  -> target artifact ABI consumer
  -> generated RVV C artifact/harness
  -> ssh rvv evidence for executable claims
```

## Direction Source

- Direction title: `Stage2 RVV compare/select mask route-family owner`.
- Module owner: RVV plugin-local compare/select and mask-producing
  route-family boundary, carried from typed selected-body facts through route
  planning/provider construction and target artifact ABI validation.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean.
- Initial HEAD: `dd3569b7 rvv: add standalone reduction accumulation route owner`.
- `.trellis/.current-task` was absent, so this task was created from the
  supplied Direction Brief before source edits.
- Serial worker constraint from the brief: do not use subagents, spawned
  agents, parallel agents, or multi-agent workflows.

## What I Already Know

- The predecessor archived task
  `.trellis/tasks/archive/2026-05/05-27-05-27-stage2-rvv-standalone-reduction-accumulation-route-family-owner`
  completed standalone reduction/accumulation ownership and target artifact ABI
  validation on top of typed selected-body/provider-built route facts.
- Current specs define compare/select statement planning, elementwise/select
  operand-binding facts, route-control provider plans, migrated statement-plan
  consumption, and mirror-only emission diagnostics. This task should reuse or
  tighten those production surfaces rather than adding an independent wrapper.
- `tcrv.exec.mem_window` and `tcrv.exec.runtime_param` bind ABI/runtime roles
  only. Compare predicate, result mask type, selected value or passthrough
  layout, mask policy, dtype/config, VL/AVL use, intrinsic/header/type mapping,
  and ABI order must come from typed `tcrv_rvv` body/config/runtime facts and
  RVV-owned family plans.
- Plain compare/select and computed-mask select already have typed
  pre-realized dataflow fixtures. Computed-mask memory, computed-mask MAcc,
  computed-mask standalone reduction, contraction, and segment2 routes are
  adjacent consumers that rely on compare-produced mask facts; this round must
  choose one existing active adjacent consumer as evidence without broadening
  Stage2 coverage.
- Common EmitC/export must remain neutral. Target artifact export may validate
  provider mirrors after route construction, but it must not infer
  compare/select or mask support from scripts, artifact names, route ids, ABI
  strings, metadata, exact intrinsic spelling, descriptors, comments, or legacy
  i32 helper names.

## Requirements

1. Add or extract one explicit RVV plugin-local compare/select mask
   route-family owner surface. It must cover one existing compare/select route
   and one adjacent existing masked elementwise or computed-mask consumer.
2. The owner must derive support, compare predicate, produced mask role/source,
   result mask type, select true/false or passthrough layout, mask policy,
   dtype/config, VL/AVL use, operand bindings, ABI order, required headers,
   type/intrinsic facts, provider-supported mirror labels, statement-plan
   eligibility, and artifact ABI claims from typed `tcrv_rvv`
   body/config/runtime/predicate/select/passthrough/mask facts and selected
   capability facts.
3. The active provider route construction path must consume this owner before
   building or accepting `TCRVEmitCLowerableRoute` payload facts for the
   migrated compare/select and adjacent mask-consuming consumers. Existing
   one-off verifiers should be reused behind the family owner or demoted, not
   duplicated as parallel authority.
4. Target artifact/runtime ABI validation must consume the provider-built
   compare/select mask family facts for the migrated consumers before accepting
   generated artifact/header claims. Artifact metadata may mirror the family
   owner only after provider route construction.
5. Unsupported or inconsistent family, stale route id, stale mirror metadata,
   wrong predicate, missing mask/result binding, wrong select/passthrough
   layout, wrong mask/VL relation, dtype/config mismatch, wrong ABI order,
   missing operand binding, wrong type/header mapping, and artifact-name or
   script-derived authority attempts must fail closed with targeted diagnostics
   before common EmitC or artifact export.
6. Existing completed segment2, MAcc, standalone reduction/accumulation,
   scalar-broadcast, conversion, runtime-scalar, contraction, and base memory
   selected-body/provider/artifact paths must not be weakened or reclassified
   as compare/select mask evidence.
7. The task must not add broad route coverage batches, dtype/LMUL clone sets,
   high-level Linalg/frontend routes, source-front-door positive routes,
   descriptor/direct-C/source-export paths, dashboards, report-only work,
   one-intrinsic wrapper dialects, segment2 follow-on polish, MAcc follow-on
   polish, or reduction/accumulation follow-on polish.

## Acceptance Criteria

- [ ] A production compare/select mask route-family owner exists in RVV
      plugin-local C++ and is consumed by the active route/provider path.
- [ ] One existing compare/select route and one adjacent existing masked or
      computed-mask route are migrated behind that owner as active consumers.
- [ ] The provider-built route for the migrated consumers proves owner-derived
      typed body/config/runtime/predicate/select/passthrough/mask facts before
      materialization, including route-control, operand binding,
      materialization, statement-plan, header/type/intrinsic, ABI order, and
      provider-supported mirror agreement.
- [ ] Target artifact ABI validation consumes the provider-built
      compare/select mask family facts for the migrated consumers before
      accepting object/header artifact claims.
- [ ] Focused C++ or lit tests prove positive owner membership/dispatch and
      fail-closed diagnostics for stale route id, stale mirror metadata, wrong
      family, wrong predicate, missing mask/result binding, wrong
      select/passthrough layout, wrong mask/VL relation, dtype/config mismatch,
      wrong ABI order, missing operand binding, wrong type/header mapping, and
      artifact-name or script-derived authority attempts where the current test
      surface exposes them.
- [ ] Generated-bundle dry-run covers both migrated consumers.
- [ ] Real `ssh rvv` generated-bundle execution covers a compare/select or
      masked representative over counts including `0`, small, exact, tail, and
      stress cases.
- [ ] Focused non-regression covers completed segment2 update/store/load,
      segment2 deinterleave/interleave, computed-mask MAcc,
      scalar-broadcast MAcc, runtime-scalar MAcc, standalone
      reduction/accumulation, scalar-broadcast, conversion, runtime-scalar,
      contraction, and base memory paths.
- [ ] Bounded touched-file authority scan finds no new executable or route
      support depending on central ad hoc, name-derived, metadata-derived,
      descriptor-derived, ABI-string-derived, script-derived,
      artifact-name-derived, common-EmitC-derived, source-front-door-derived,
      route-id-derived, exact-intrinsic-derived, or legacy-i32-derived
      authority.
- [ ] `git diff --check` passes.
- [ ] Focused build/tests pass, and `check-tianchenrv` passes or an exact
      blocker is recorded with the task left open.
- [ ] Trellis task status/context/journal/archive are truthful, and one
      coherent commit is created if all acceptance criteria are satisfied.

## Validation Plan

1. Start and validate this Trellis task after PRD/context setup.
2. Inspect current compare/select, masked elementwise, computed-mask,
   route-family planning, provider construction, target artifact ABI
   validation, generated-bundle, and focused test surfaces.
3. Implement the compare/select mask family owner in the smallest production
   RVV plugin surface that currently owns duplicated or ad hoc compare/mask
   checks.
4. Wire provider construction and target artifact validation to consume the
   owner-derived facts for the migrated compare/select and adjacent mask
   consumer.
5. Add focused C++ and/or lit tests for owner membership, provider/target
   consumption, and fail-closed stale or missing family facts.
6. Run focused build/tests for touched RVV plugin and target surfaces.
7. Run generated-bundle dry-runs for both migrated consumers.
8. Run real `ssh rvv` evidence for one representative over counts `0`, small,
   exact, tail, and stress cases.
9. Run the required focused non-regression dry-runs/lit coverage.
10. Run bounded authority scan, `git diff --check`, task validation, and
    `check-tianchenrv`.

## Out Of Scope

- Segment2 follow-on polish or using segment2 as the active achievement.
- MAcc or standalone reduction/accumulation follow-on polish.
- New compare/select predicate variants or new computed-mask route variants
  beyond the two migrated active consumers.
- Dtype/LMUL clone batches, broad Stage2 coverage expansion, gather/scatter
  expansion, high-level frontend/Linalg/Vector/StableHLO routes, global
  tuning/profile databases, dashboards, reports-only changes, or smoke-only
  work.
- Any new `tcrv_rvv.i32_*` helper op, legacy `RVVI32M1`/`rvv-i32m1` route, or
  exact intrinsic spelling as route authority.
- Common EmitC or target export inferring compare/select or mask support,
  dtype, mask, selected value, passthrough, ABI order, policy, intrinsic, or
  executable authority from route ids, metadata, artifact names, scripts,
  descriptors, ABI strings, or source-front-door fixtures.

## Technical Notes

- Specs read:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`, and
  `.trellis/spec/core-dialect/tcrv-exec-contract.md`.
- Predecessor archived context read:
  `.trellis/tasks/archive/2026-05/05-27-05-27-stage2-rvv-standalone-reduction-accumulation-route-family-owner/prd.md`.
- Initial code surfaces to inspect:
  `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`,
  `lib/Target/RVV/RVVTargetSupportBundle.cpp`,
  `scripts/rvv_generated_bundle_abi_e2e.py`,
  `test/Plugin/RVVExtensionPluginTest.cpp`,
  and directly related compare/select, masked elementwise, computed-mask, and
  generated-bundle tests.

## Open Questions

None blocking. The Direction Brief and current specs point to a bounded
implementation: extract or add a compare/select mask owner over existing active
compare/select and adjacent computed-mask family validators, then make
provider/target consumers depend on that owner for one compare/select route plus
one adjacent mask-consuming route.

## Definition Of Done

The compare/select mask family owner is the active production boundary for the
migrated consumers, provider-built routes and target artifacts consume
owner-derived facts, focused positive and fail-closed tests pass,
generated-bundle and `ssh rvv` evidence are recorded, authority scans are
clean, Trellis state is truthful, and one coherent commit records the work.
