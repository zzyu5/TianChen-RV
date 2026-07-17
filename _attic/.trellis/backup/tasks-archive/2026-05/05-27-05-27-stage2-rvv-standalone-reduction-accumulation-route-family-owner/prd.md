# Stage2 RVV Standalone Reduction/Accumulation Route-Family Owner

## Goal

Introduce or extract a plugin-local RVV standalone reduction / standalone
accumulation route-family owner across the active selected-body,
planning/provider, and target artifact ABI consumer path. The primary migrated
consumer must be an existing standalone reduction route, and one adjacent
existing accumulation consumer must prove this is a family-level boundary
rather than another one-off route patch. Prefer the computed-mask standalone
accumulation path that the prior MAcc owner intentionally left outside the
MAcc route-family boundary.

The completed route must stay on the current RVV-first chain:

```text
selected tcrv.exec RVV variant
  -> typed low-level tcrv_rvv reduction/accumulation body
  -> RVV plugin-owned standalone reduction/accumulation family ownership
  -> RVV provider route facts and statement plans
  -> provider-built TCRVEmitCLowerableRoute
  -> neutral common EmitC materialization
  -> target artifact ABI consumer
  -> generated RVV C artifact/harness
  -> ssh rvv evidence for executable claims
```

## Direction Source

- Direction title: `Stage2 RVV standalone reduction/accumulation route-family owner`.
- Module owner: RVV plugin-local standalone reduction and standalone
  accumulation route-family boundary from selected-body facts through route
  planning/provider construction and target artifact ABI validation.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean.
- Initial HEAD: `60523413 rvv: add macc route-family owner`.
- `.trellis/.current-task` was absent, so this task was created from the
  supplied Direction Brief before source edits.
- Serial worker constraint from the brief: do not use subagents, spawned
  agents, parallel agents, or multi-agent workflows.

## What I Already Know

- The predecessor archived task
  `.trellis/tasks/archive/2026-05/05-27-05-27-stage2-rvv-macc-route-family-owner`
  completed a bounded MAcc owner and intentionally kept computed-mask
  standalone reductions/accumulations outside the MAcc statement-plan boundary.
- Current specs define the authority chain as selected `tcrv.exec` envelope,
  typed `tcrv_rvv` body, RVV plugin-owned legality/realization/provider,
  provider-built `TCRVEmitCLowerableRoute`, neutral common EmitC, and target
  artifact mirrors after provider route construction.
- `tcrv.exec.mem_window` and `tcrv.exec.runtime_param` bind ABI/runtime roles
  only. Standalone reduction/accumulation support, dtype/config, mask policy,
  accumulator/result layout, intrinsic/header/type mapping, and ABI order must
  come from typed `tcrv_rvv` body/config/runtime facts and RVV-owned family
  plans.
- Specs already describe broad route materialization facts, route-control
  provider-plan consumers, and math operand-binding facts for mature reduction
  and accumulation routes. This task should reuse or tighten those surfaces
  behind a specific standalone reduction/accumulation family owner rather than
  adding duplicate central provider branches.
- Specs also define a migrated statement-plan aggregate boundary, but list
  computed-mask standalone reduction variants as base/outside until their
  statement plans become migrated owners. This round may migrate the bounded
  active standalone reduction and adjacent accumulation consumer without
  broadening Stage2 route coverage.
- Common EmitC/export must remain neutral. Target artifact export may validate
  provider mirrors after route construction, but it must not infer standalone
  reduction or accumulation support from scripts, artifact names, route ids,
  ABI strings, metadata, exact intrinsic spelling, descriptors, comments, or
  legacy i32 helper names.

## Requirements

1. Add or extract one explicit RVV plugin-local standalone reduction /
   standalone accumulation route-family owner surface. It must cover one active
   standalone reduction route and one adjacent active accumulation route,
   preferably the computed-mask standalone accumulation path.
2. The owner must derive route support, reduction kind, accumulation kind,
   accumulator/result layout, initial value or passthrough semantics, mask
   policy when present, scalar result binding, dtype/config, VL/AVL use,
   operand bindings, ABI order, required headers, type/intrinsic facts,
   provider-supported mirror labels, and statement-plan eligibility from typed
   `tcrv_rvv` body/config/runtime/mask/reduction/accumulator/result facts and
   selected capability facts.
3. The active provider route construction path must consume this owner before
   building or accepting `TCRVEmitCLowerableRoute` payload facts for the
   migrated standalone reduction/accumulation consumers. Existing one-off
   verifiers should be reused behind the family owner or demoted, not
   duplicated as parallel authority.
4. Target artifact/runtime ABI validation must consume the provider-built
   standalone reduction/accumulation family facts for the migrated consumers
   before accepting generated artifact/header claims. Artifact metadata may
   mirror the family owner only after provider route construction.
5. Unsupported or inconsistent family, stale route id, stale mirror metadata,
   wrong reduction or accumulation kind, missing accumulator/result binding,
   wrong mask/VL relation, dtype/config mismatch, wrong ABI order, missing
   operand binding, wrong type/header mapping, and artifact-name or
   script-derived authority attempts must fail closed with targeted diagnostics
   before common EmitC or artifact export.
6. Existing completed segment2, MAcc, masked elementwise, scalar-broadcast,
   conversion, runtime-scalar, compare/select, contraction, and base memory
   selected-body/provider/artifact paths must not be weakened or reclassified
   as standalone reduction/accumulation evidence.
7. The task must not add broad route coverage batches, dtype/LMUL clone sets,
   high-level Linalg/frontend routes, source-front-door positive routes,
   descriptor/direct-C/source-export paths, dashboards, report-only work,
   one-intrinsic wrapper dialects, segment2 follow-on polish, or MAcc
   follow-on polish.

## Acceptance Criteria

- [ ] A production standalone reduction / standalone accumulation
      route-family owner exists in RVV plugin-local C++ and is consumed by the
      active route/provider path.
- [ ] One existing standalone reduction route and one adjacent existing
      accumulation route are migrated behind that owner as active consumers.
- [ ] The provider-built route for the migrated consumers proves owner-derived
      typed body/config/runtime/mask/reduction/accumulator/result facts before
      materialization, including route-control, math binding, materialization,
      statement-plan, header/type/intrinsic, ABI order, and
      provider-supported mirror agreement.
- [ ] Target artifact ABI validation consumes the provider-built standalone
      reduction/accumulation family facts for the migrated consumers before
      accepting object/header artifact claims.
- [ ] Focused C++ or lit tests prove positive owner membership/dispatch and
      fail-closed diagnostics for stale route id, stale mirror metadata, wrong
      family, wrong reduction or accumulation kind, missing accumulator/result
      binding, wrong mask/VL relation, dtype/config mismatch, wrong ABI order,
      missing operand binding, wrong type/header mapping, and artifact-name or
      script-derived authority attempts where the current test surface exposes
      them.
- [ ] Generated-bundle dry-run covers both migrated consumers.
- [ ] Real `ssh rvv` generated-bundle execution covers the standalone
      reduction or accumulation representative over counts including `0`,
      small, exact, tail, and stress cases.
- [ ] Focused non-regression covers completed segment2 update/store/load,
      segment2 deinterleave/interleave, computed-mask MAcc,
      scalar-broadcast MAcc, runtime-scalar MAcc, masked elementwise,
      scalar-broadcast, conversion, runtime-scalar, compare/select,
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
2. Inspect current standalone reduction, computed-mask standalone
   accumulation, route-family planning, provider construction, target artifact
   ABI validation, generated-bundle, and focused test surfaces.
3. Implement the standalone reduction/accumulation family owner in the smallest
   production RVV plugin surface that currently owns duplicated or ad hoc
   standalone reduction/accumulation checks.
4. Wire provider construction and target artifact validation to consume the
   owner-derived facts for the migrated reduction and accumulation consumers.
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
- MAcc follow-on polish or weakening the completed MAcc route-family owner.
- New reduction/accumulation math variants beyond the two migrated active
  consumers.
- Dtype/LMUL clone batches, broad Stage2 coverage expansion, gather/scatter
  expansion, high-level frontend/Linalg/Vector/StableHLO routes, global
  tuning/profile databases, dashboards, reports-only changes, or smoke-only
  work.
- Any new `tcrv_rvv.i32_*` helper op, legacy `RVVI32M1`/`rvv-i32m1` route, or
  exact intrinsic spelling as route authority.
- Common EmitC or target export inferring reduction/accumulation support,
  dtype, mask, accumulator, scalar result, ABI order, policy, intrinsic, or
  executable authority from route ids, metadata, artifact names, scripts,
  descriptors, ABI strings, or source-front-door fixtures.

## Technical Notes

- Specs read:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`, and
  `.trellis/spec/core-dialect/tcrv-exec-contract.md`.
- Predecessor archived context read:
  `.trellis/tasks/archive/2026-05/05-27-05-27-stage2-rvv-macc-route-family-owner/prd.md`.
- Initial code surfaces to inspect:
  `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`,
  `lib/Target/RVV/RVVTargetSupportBundle.cpp`,
  `scripts/rvv_generated_bundle_abi_e2e.py`,
  `test/Plugin/RVVExtensionPluginTest.cpp`,
  and directly related standalone reduction, computed-mask standalone
  accumulation, and generated-bundle tests.

## Open Questions

None blocking. The Direction Brief and current specs point to a bounded
implementation: extract or add a standalone reduction/accumulation owner over
existing active reduction/accumulation family validators and make
provider/target consumers depend on that owner for one standalone reduction
plus one adjacent accumulation route.

## Definition Of Done

The standalone reduction / standalone accumulation family owner is the active
production boundary for the migrated consumers, provider-built routes and
target artifacts consume owner-derived facts, focused positive and
fail-closed tests pass, generated-bundle and `ssh rvv` evidence are recorded,
authority scans are clean, Trellis state is truthful, and one coherent commit
records the work.
