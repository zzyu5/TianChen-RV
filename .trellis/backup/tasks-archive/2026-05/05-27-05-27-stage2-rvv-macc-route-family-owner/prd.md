# Stage2 RVV Multiply-Accumulate Route-Family Owner

## Goal

Introduce or extract a plugin-local RVV MAcc / accumulation route-family owner
across the active selected-body, planning/provider, and target artifact
consumer path. The owner must make `computed_masked_macc_add` the primary
migrated consumer and include one adjacent active MAcc variant, preferably
`scalar_broadcast_macc_add` or `runtime_scalar_cmp_masked_macc_add`, so this is
a family-level boundary rather than another one-off route patch.

The completed route must stay on the current RVV-first chain:

```text
selected tcrv.exec RVV variant
  -> typed low-level tcrv_rvv MAcc body
  -> RVV plugin-owned selected-body realization and family ownership
  -> RVV provider route facts and statement plans
  -> provider-built TCRVEmitCLowerableRoute
  -> neutral common EmitC materialization
  -> target artifact ABI consumer
  -> generated RVV C artifact/harness
  -> ssh rvv evidence for executable claims
```

## Direction Source

- Direction title: `Stage2 RVV multiply-accumulate route-family owner`.
- Module owner: RVV plugin-local multiply-accumulate / accumulation
  route-family boundary from selected-body realization through provider route
  construction and target artifact ABI consumption.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean.
- Initial HEAD: `10ea0b3c rvv: validate segment2 artifact ABI consumers`.
- `.trellis/.current-task` was absent, so this task was created from the
  supplied Direction Brief before source edits.
- Serial worker constraint from the brief: do not use subagents, spawned agents,
  parallel agents, or multi-agent workflows.

## What I Already Know

- The previous completed segment2 task
  `.trellis/tasks/archive/2026-05/05-27-stage2-rvv-route-family-artifact-abi-consumer-owner`
  made target artifact/runtime ABI export consume rebuilt provider route facts
  for segment2 before generated object/header claims.
- Existing MAcc work is not empty. Prior archived tasks added plain `macc_add`
  accumulator/runtime/artifact closure, vector-compare computed-mask MAcc
  artifact closure, and direct route-entry MAcc realization ownership.
- Current specs already define:
  - selected-body realization owners for `MAcc` and `computed-mask MAcc`;
  - plain/scalar-broadcast MAcc statement-plan boundary;
  - computed-mask accumulation statement-plan boundary;
  - route-control provider-plan consumers including scalar-broadcast MAcc and
    computed-mask accumulation MAcc;
  - math operand-binding facts for MAcc and accumulation routes;
  - mirror-only target artifact metadata.
- Current code inspection shows separate provider-plan verifiers for plain
  MAcc, scalar-broadcast MAcc, computed-mask accumulation, standalone
  reduction, and contraction under a broader reduction/accumulation/contraction
  registry. That registry is broader than the requested MAcc family boundary
  and does not name MAcc/accumulation as the family owner consumed by target
  artifact validation.
- Current target artifact metadata evidence already lists MAcc mirrors such as
  `plain_macc_route_family_plan`, `scalar_broadcast_macc_route_family_plan`,
  `accumulation_route_family_plan`, accumulator/result layout mirrors,
  route-control mirrors, runtime ABI bindings, required headers, and C type
  mapping. The production question is whether target acceptance for the MAcc
  family consumes a provider-built MAcc-family owner result rather than simply
  accepting optional generic mirrors.
- Common EmitC/export must remain neutral. It may consume provider-built route
  facts and mirrors after construction, but it must not infer MAcc semantics,
  mask semantics, accumulator layout, dtype/config, ABI order, intrinsic
  spelling, or route support from names, route ids, scripts, artifact names, or
  metadata.

## Requirements

1. Add or extract one explicit RVV plugin-local MAcc / accumulation
   route-family owner surface. It must cover `computed_masked_macc_add` and at
   least one adjacent active MAcc variant, preferably
   `scalar_broadcast_macc_add` or `runtime_scalar_cmp_masked_macc_add`.
2. The owner must derive route support, operation kind, accumulator/result
   layout, arithmetic kind, mask/passthrough behavior, scalar/runtime-scalar
   operand form, dtype/config, VL/AVL use, runtime ABI order, operand bindings,
   required headers, type/intrinsic facts, provider-supported mirror labels,
   and statement-plan eligibility from typed `tcrv_rvv` body/config/runtime
   facts and selected capability facts.
3. The active provider route construction path must consume this owner before
   building or accepting `TCRVEmitCLowerableRoute` payload facts for the
   migrated MAcc consumers. Existing one-off verifiers should be reused behind
   the family owner or demoted, not duplicated as parallel authority.
4. Target artifact/runtime ABI validation must consume the provider-built
   MAcc-family facts for the migrated consumers before accepting generated
   artifact/header claims. Artifact metadata may mirror the family owner only
   after provider route construction.
5. Unsupported or inconsistent family, stale route id, stale mirror metadata,
   wrong arithmetic kind, missing accumulator/pass-through, wrong scalar or
   runtime-scalar binding, wrong mask/VL relation, dtype/config mismatch,
   wrong ABI order, missing operand binding, wrong type/header mapping, and
   artifact-name or script-derived authority attempts must fail closed with
   targeted diagnostics before common EmitC or artifact export.
6. Existing completed segment2 selected-body, planning/provider, and artifact
   ABI consumer paths must not be weakened or reclassified as MAcc evidence.
7. The task must not add broad coverage batches, dtype/LMUL clone sets,
   high-level Linalg/frontend routes, source-front-door positive routes,
   descriptor/direct-C/source-export paths, dashboards, report-only work, or
   one-intrinsic wrapper dialects.

## Acceptance Criteria

- [x] A production MAcc / accumulation route-family owner exists in RVV
      plugin-local C++ and is consumed by the active route/provider path.
- [x] `computed_masked_macc_add` and one adjacent active MAcc variant are
      migrated behind that owner as active consumers.
- [x] The provider-built route for the migrated consumers proves owner-derived
      typed body/config/runtime/mask/arithmetic/accumulator facts before
      materialization, including route-control, math binding, materialization,
      statement-plan, header/type/intrinsic, ABI order, and
      provider-supported mirror agreement.
- [x] Target artifact ABI validation consumes the provider-built MAcc-family
      facts for the migrated consumers before accepting object/header artifact
      claims.
- [x] Focused C++ or lit tests prove positive owner membership/dispatch and
      fail-closed diagnostics for stale/missing MAcc family facts, wrong
      family, wrong arithmetic kind, missing accumulator or passthrough,
      wrong scalar/runtime-scalar binding, mask/VL mismatch, dtype/config
      mismatch, wrong ABI order, missing operand binding, wrong type/header
      mapping, and artifact-name or script-derived authority attempts where
      the current test surface exposes them.
- [x] Generated-bundle dry-run covers both migrated MAcc consumers.
- [x] Real `ssh rvv` generated-bundle execution covers
      `computed_masked_macc_add` or the migrated representative over counts
      including `0`, `7`, `16`, `23`, and `257`, with MAcc correctness,
      inactive-lane/pass-through behavior where applicable, and tail sentinel
      preservation.
- [x] Focused non-regression covers completed segment2 update/store/load,
      segment2 deinterleave/interleave, masked elementwise, reduction,
      scalar-broadcast, conversion, runtime-scalar, compare/select,
      contraction, and base memory paths.
- [x] Bounded touched-file authority scan finds no new executable or route
      support depending on central ad hoc, name-derived, metadata-derived,
      descriptor-derived, ABI-string-derived, script-derived,
      artifact-name-derived, common-EmitC-derived, source-front-door-derived,
      route-id-derived, exact-intrinsic-derived, or legacy-i32-derived
      authority.
- [x] `git diff --check` passes.
- [x] Focused build/tests pass, and `check-tianchenrv` passes or an exact
      blocker is recorded with the task left open.
- [x] Trellis task status/context/journal/archive are truthful, and one
      coherent commit is created if all acceptance criteria are satisfied.

## Validation Plan

1. Start and validate this Trellis task after PRD/context setup.
2. Implement the MAcc / accumulation family owner in the smallest production
   RVV plugin surface that currently owns the duplicated MAcc family checks.
3. Wire provider construction and target artifact validation to consume the
   owner-derived facts for `computed_masked_macc_add` and the selected adjacent
   MAcc variant.
4. Add focused C++ and/or lit tests for owner membership, provider/target
   consumption, and fail-closed stale or missing family facts.
5. Run focused build/tests for touched RVV plugin and target surfaces.
6. Run generated-bundle dry-runs for both migrated MAcc consumers.
7. Run real `ssh rvv` evidence for the representative MAcc route over counts
   `0`, `7`, `16`, `23`, and `257`.
8. Run the required focused non-regression dry-runs/lit coverage.
9. Run bounded authority scan, `git diff --check`, task validation, and
   `check-tianchenrv`.

## Out Of Scope

- Segment2 follow-on polish or using segment2 as the active achievement.
- New MAcc math variants beyond the two migrated active consumers.
- Dtype/LMUL clone batches, broad Stage2 coverage expansion, gather/scatter
  expansion, high-level frontend/Linalg/Vector/StableHLO routes, global
  tuning/profile databases, dashboards, reports-only changes, or smoke-only
  work.
- Any new `tcrv_rvv.i32_*` helper op, legacy `RVVI32M1`/`rvv-i32m1` route, or
  exact intrinsic spelling as route authority.
- Common EmitC or target export inferring MAcc support, dtype, mask,
  accumulator, ABI order, policy, intrinsic, or executable authority from
  route ids, metadata, artifact names, scripts, descriptors, ABI strings, or
  source-front-door fixtures.

## Technical Notes

- Specs read:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`, and
  `.trellis/spec/core-dialect/tcrv-exec-contract.md`.
- Predecessor/related archived context read:
  `.trellis/tasks/archive/2026-05/05-27-stage2-rvv-route-family-artifact-abi-consumer-owner/prd.md`,
  `.trellis/tasks/archive/2026-05/05-26-05-26-stage2-rvv-route-entry-macc-realization-owner/prd.md`,
  `.trellis/tasks/archive/2026-05/05-26-stage2-rvv-macc-accumulator-runtime-artifact-boundary/prd.md`,
  and
  `.trellis/tasks/archive/2026-05/05-26-05-26-stage2-rvv-computed-mask-macc-accumulator-mask-artifact-boundary/prd.md`.
- Initial code surfaces:
  `include/TianChenRV/Plugin/RVV/RVVSelectedBodyRealization.h`,
  `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp`,
  `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`,
  `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`,
  `lib/Target/RVV/RVVTargetSupportBundle.cpp`,
  `scripts/rvv_generated_bundle_abi_e2e.py`,
  `test/Plugin/RVVExtensionPluginTest.cpp`,
  and directly related MAcc generated-bundle/lit tests.

## Open Questions

None blocking. Current repository facts point to a bounded implementation:
extract or add a MAcc/accumulation family owner over existing MAcc family
validators and make provider/target consumers depend on that owner for
`computed_masked_macc_add` plus one adjacent MAcc route.

## Definition Of Done

The MAcc / accumulation family owner is the active production boundary for the
migrated computed-mask and adjacent MAcc consumers, provider-built routes and
target artifacts consume the owner-derived facts, focused positive and
fail-closed tests pass, generated-bundle and `ssh rvv` evidence are recorded,
authority scans are clean, Trellis state is truthful, and one coherent commit
records the work.
