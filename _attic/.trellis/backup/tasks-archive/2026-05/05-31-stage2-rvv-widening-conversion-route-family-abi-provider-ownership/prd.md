# PRD: Stage2 RVV widening conversion route-family ABI/provider ownership

## Direction Source

Hermes/user direction switches the next bounded owner to the RVV widening
conversion route-family ABI/provider boundary after commit `85765df4` closed
standalone reduction provider ABI ownership.

Requested production chain:

```text
selected tcrv.exec RVV widening-conversion variant
  -> typed tcrv_rvv widening conversion body
  -> RVV route analysis and widening conversion route-family plan
  -> provider-owned validation of source/result dtype, SEW/LMUL,
     conversion relation, materialization facts, operand binding,
     runtime ABI order, header/type/intrinsic mirrors, and target mirrors
  -> TCRVEmitCLowerableRoute
  -> neutral EmitC
  -> generated RVV artifact
  -> ssh rvv evidence when executable correctness is claimed
```

## Initial Repository State

- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean.
- Initial HEAD: `85765df4 rvv: close standalone reduction provider ABI boundary`.
- `.trellis/.current-task` was absent, so this task was created from the
  supplied Hermes Direction Brief before source edits.
- Serial worker constraint from the brief: one Codex worker; no subagents,
  spawned agents, parallel agents, or multi-agent workflow.

## Repository Findings

- Widening conversion already has a selected-body owner, route-family plan,
  conversion dtype-policy owner, route-control provider plan, math
  operand-binding facts, and an RVV-owned statement-plan path.
- The current supported widening conversion cases are bounded existing cases:
  `widen_i32_to_i64` and `widen_i16_to_i32`. This task must not add new dtype,
  LMUL, frontend, or conversion coverage.
- `RVVSelectedBodyWideningConversionRouteFamilyPlan` already records source
  and result SEW/LMUL/type facts, conversion relation, source load,
  conversion intrinsic, store intrinsic, runtime ABI order, provider support
  mirror, target leaf profile, required headers, and C type summary.
- `verifyRVVSelectedBodyWideningConversionRouteFamilyProviderPlans(...)`
  validates plan presence/stale-plan rejection and route description mirrors,
  but the production provider route-construction path lacks a dedicated
  widening conversion provider-fact closure analogous to the just-finished
  standalone reduction closure.
- `RVVEmitCRouteProvider.cpp` currently obtains math operand-binding facts and
  the aggregate migrated statement-plan owner selection, but it does not
  explicitly construct/check the widening conversion statement plan and
  materialization/operand-binding facts before creating
  `TCRVEmitCLowerableRoute`.
- Common EmitC remains a neutral consumer of provider-built route payloads and
  must stay that way.

## Goal

Close the widening conversion ABI/provider boundary for currently supported
selected-body widening conversion routes by requiring provider-side route
construction to consume validated widening conversion family-plan facts,
materialization facts, math operand-binding facts, route-control facts, and the
RVV-owned statement plan before building `TCRVEmitCLowerableRoute`.

This is a bounded production-plus-evidence task:

- production movement must land in RVV route planning/provider code, and in
  generated-bundle or target artifact validation only when needed as direct
  consumers;
- generated-bundle or script changes may validate mirrors of provider/target
  facts, but must not authorize route support.

## Requirements

1. Add a provider-local widening conversion route-construction preflight before
   `TCRVEmitCLowerableRoute` construction.
2. The preflight must require exactly the verified
   `RVVSelectedBodyWideningConversionRouteFamilyPlan` for widening conversion
   consumers and must reject stale widening conversion facts on non-conversion
   routes.
3. The preflight must verify selected typed body/config and family-plan facts
   agree for:
   - selected operation and unit-stride conversion memory form;
   - source element type/channel facts through source SEW, LMUL, vector type,
     vector C type, and source load leaf;
   - result element type/channel facts through result SEW, LMUL, vector type,
     vector C type, setvl leaf, and store leaf;
   - conversion relation and conversion intrinsic;
   - runtime AVL/VL control facts and runtime ABI order.
4. The preflight must verify materialization facts mirror the validated
   widening conversion plan for required headers, VL type, source/result vector
   types, setvl, source load, conversion, store, and no unrelated mask/scalar
   or standalone-reduction residue.
5. The preflight must verify math operand-binding facts come from the same
   selected route analysis and bind `lhs`, `out`, and runtime `n` with the
   expected runtime ABI roles.
6. The preflight must verify the RVV-owned widening conversion statement plan
   is present, selected for the exact conversion case, refers to the same
   family plan, and carries provider-ready setvl/source-load/convert/store
   statement leaves before route construction.
7. Route description mirrors must match provider plan fields for source/result
   dtype policy, source/result SEW/LMUL relation, conversion relation,
   provider support mirror, target leaf profile, header/type summaries,
   runtime ABI parameters, and route operand binding closure.
8. Non-consumer routes carrying stale widening conversion analysis,
   materialization, operand-binding, statement-plan, source/result type, or
   conversion-relation facts must fail closed before route construction.
9. Preserve selected-body realization semantics. Do not change computation
   semantics, dtype semantics, parameter roles, variant origin,
   dispatch/fallback behavior, runtime `n`/AVL/VL values, or required
   capabilities.
10. Keep common EmitC neutral. Common materialization must only consume the
    provider-built route payload and must not infer widening conversion
    semantics from route ids, ABI strings, metadata, artifact names, scripts,
    or exact intrinsic names.

## Acceptance Criteria

- [x] Production diff lands in RVV route planning/provider code, not only tests
      or scripts.
- [x] Provider materialization fails closed before `TCRVEmitCLowerableRoute`
      construction when widening conversion family facts are missing, stale,
      inconsistent with materialization facts, inconsistent with math
      operand-binding facts, inconsistent with statement-plan facts, or
      inconsistent with route description mirrors.
- [x] Provider materialization rejects stale widening conversion facts on
      non-conversion routes.
- [x] Focused C++ plugin tests cover positive provider-plan/route construction
      for representative existing widening conversion cases and fail-closed
      stale/missing facts before route construction.
- [x] Generated-bundle dry-run covers existing selected-boundary widening
      conversion artifacts and validates provider/artifact mirror facts when
      directly affected.
- [x] Direct pre-realized route-entry remains unsupported/fail-closed for
      production route construction.
- [x] Real `ssh rvv` generated-bundle correctness passes for representative
      widening conversion counts `0`, `1`, exact-VL, tail, and stress with
      signed inputs and expected widened outputs, or an exact external blocker
      is recorded.
- [x] Non-regression coverage for standalone reduction and adjacent
      selected-body route families passes through focused C++ tests and
      `check-tianchenrv` when available.
- [x] Bounded authority scan over touched production/test/script/task files
      finds no new central ad hoc, name-derived, metadata-derived,
      descriptor-derived, ABI-string-derived, script-derived,
      artifact-name-derived, common-EmitC-derived, source-front-door-derived,
      route-id-derived, exact-intrinsic-derived, direct-route-entry-only,
      pre-realized-fixture-only, or legacy-i32-derived route authority.
- [x] `git diff --check` passes.
- [x] Focused builds/tests pass, and `check-tianchenrv` passes or the exact
      blocker is recorded.
- [x] Task status, journal, archive state, and final commit state are truthful.

## Validation Plan

1. Build and run focused RVV plugin tests:
   `cmake --build artifacts/tmp/tianchenrv-build --target tianchenrv-rvv-extension-plugin-test -j2`
   and `artifacts/tmp/tianchenrv-build/bin/tianchenrv-rvv-extension-plugin-test`.
2. Compile-check generated-bundle tooling if changed:
   `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`.
3. Run generated-bundle dry-runs for existing selected-boundary widening
   conversion cases, including representative counts `0,1,16,17,257`.
4. Run direct pre-realized route-entry fail-closed probes for widening
   conversion if the generated-bundle script exposes the probe.
5. Run real `ssh rvv` generated-bundle correctness for representative
   widening conversion cases unless an exact external blocker occurs.
6. Run bounded authority scans over touched files.
7. Run `git diff --check`.
8. Run `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`,
   or record the exact blocker.

## Out of Scope

- New conversion operation coverage.
- New dtype or LMUL clone batches.
- High-level Linalg/Vector/StableHLO frontend lowering.
- One-intrinsic wrapper dialects.
- Selected-body realization framework rewrites.
- Direct route-entry resurrection.
- Source-front-door positive routes.
- Standalone reduction follow-up work.
- Segment2, compare/select, computed-mask memory, dashboard, report, or broad
  smoke-matrix work.

## Technical Notes

- Specs read:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/core-dialect/tcrv-exec-contract.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  `.trellis/spec/testing/index.md`, and
  `.trellis/spec/plugin-protocol/index.md`.
- Prior task context read:
  `.trellis/tasks/archive/2026-05/05-31-stage2-rvv-standalone-reduction-route-family-abi-provider-ownership/prd.md`
  and
  `.trellis/tasks/archive/2026-05/05-31-stage2-rvv-standalone-reduction-route-family-abi-provider-ownership/evidence-summary.md`.
- Relevant implementation files inspected:
  `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`,
  `include/TianChenRV/Plugin/RVV/RVVEmitCStatementPlanOwners.h`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCResidualStatementPlanOwners.cpp`,
  `scripts/rvv_generated_bundle_abi_e2e.py`, and
  `test/Plugin/RVVExtensionPluginTest.cpp`.

No blocking user question remains. The supplied direction brief, current
specs, archived task boundary, and code inspection are specific enough for one
bounded implementation round.
