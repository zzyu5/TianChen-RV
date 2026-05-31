# PRD: Stage2 RVV standalone reduction route-family ABI/provider ownership

## Direction Source

Hermes/user direction switches the next bounded owner to the RVV standalone
reduction route-family ABI/provider boundary after the elementwise arithmetic
ABI/provider boundary was closed in commit `a714d2a0`.

Requested production chain:

```text
selected tcrv.exec RVV standalone-reduction variant
  -> RVV plugin-local selected-body realization
  -> realized typed tcrv_rvv standalone reduction body
  -> standalone-reduction route-family provider facts
  -> runtime/control, operand binding, statement, header/type, inactive-lane,
     accumulator/result, and artifact ABI mirror validation
  -> TCRVEmitCLowerableRoute
  -> neutral EmitC
  -> generated RVV artifact
  -> ssh rvv evidence when executable correctness is claimed
```

## Initial Repository State

- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean.
- Initial HEAD: `a714d2a0 rvv: close elementwise arithmetic provider ABI boundary`.
- `.trellis/.current-task` was absent, so this task was created from the
  supplied Hermes Direction Brief before source edits.
- Serial worker constraint from the brief: one Codex worker; no subagents,
  spawned agents, parallel agents, or multi-agent workflow.

## Repository Findings

- Standalone reduction already has a route-family plan surface:
  `RVVSelectedBodyStandaloneReductionRouteFamilyPlan`, plan derivation,
  validation, route description mirror application, provider-plan verification,
  and an aggregate standalone reduction/accumulation owner.
- The existing standalone plan validates operation kind, memory form, runtime
  AVL/VL control, runtime ABI order, target leaf profile,
  `provider_supported_mirror`, header/type mapping summary, source vector
  type/C type, scalar-result vector type/C type, scalar seed/result layout,
  store VL, mask/inactive-lane facts for computed-mask routes, route operand
  binding, and runtime ABI parameter mirrors.
- `RVVEmitCRouteProvider.cpp` currently has an extra route-construction
  preflight for elementwise arithmetic typed config facts before
  `TCRVEmitCLowerableRoute` construction. Standalone reduction relies on the
  route-family verifier and generic materialization facts, but does not yet
  have an equivalent provider-local preflight proving that materialization
  facts, route description mirrors, and ABI/type mappings all come from the
  validated standalone family plan immediately before route construction.
- Target artifact validation already has a standalone reduction/accumulation
  validator that receives the rebuilt provider route and description. It
  checks route headers, type mappings, ABI mappings, pre-loop/loop statement
  plans, standalone route-family mirrors, source/scalar-result channel mirrors,
  runtime ABI order, inactive-lane policy, and stale non-standalone mirrors.
  It should be kept as a consumer of rebuilt provider route facts, not as route
  authority.
- Generated-bundle evidence already records many standalone reduction mirrors,
  but the reduction/accumulation metadata key set omits several mirrors that
  the target validator and provider description now treat as important for
  standalone reductions: source/scalar-result vector types, vector load,
  scalar seed splat, reduction intrinsic, scalar-result store, compare
  predicate/leaf, inactive merge leaf, and runtime-scalar RHS splat leaf.
- Current selected-boundary generated-bundle evidence includes plain
  `standalone_reduce_add`, computed-mask standalone reductions, and
  runtime-scalar computed-mask standalone reductions. Direct route-entry support
  is deliberately narrower and must not become the positive authority for this
  task.

## Goal

Close the standalone-reduction ABI/provider ownership gap for the currently
supported standalone reduction forms by requiring provider-side route
construction to consume validated standalone family-plan facts before building
`TCRVEmitCLowerableRoute`, and by tightening generated-bundle evidence so
standalone reduction artifact metadata mirrors the same provider/target facts
rather than authorizing support.

This is a bounded production-plus-evidence task:

- production movement must land in RVV standalone reduction route
  planning/provider and directly affected target-artifact validation if needed;
- generated-bundle script changes may only validate or expose mirrors of
  provider/target facts and must not authorize route support.

## Requirements

1. Add a provider-local standalone reduction route-construction preflight before
   `TCRVEmitCLowerableRoute` construction.
2. The preflight must require a validated
   `RVVSelectedBodyStandaloneReductionRouteFamilyPlan` for standalone reduction
   consumers and must reject stale standalone reduction materialization facts
   on non-consumer routes.
3. The preflight must verify materialization facts mirror the validated
   standalone family plan:
   - typed config facts exist and match source vector type/C type, VL type,
     setvl, vector load, scalar seed splat, reduction leaf, scalar-result store,
     source splat, RHS scalar splat where applicable, compare leaf, inactive
     merge leaf, source/scalar-result channel types, mask type, and required
     headers.
   - route description mirrors match provider plan fields for runtime ABI order,
     provider support, route-family plan id, header/type summary, source/scalar
     result channels, mask/inactive-lane policy, accumulator/result layout,
     store VL, scalar result runtime boundary, runtime ABI parameters, and
     route operand binding closure.
4. Preserve selected-body realization semantics. Do not change computation
   semantics, dtype semantics, accumulator/result layout, parameter roles,
   variant origin, dispatch/fallback behavior, runtime `n`/AVL/VL values, or
   required capabilities.
5. Preserve current supported standalone reduction forms only: plain
   standalone add/min/max, computed-mask standalone add/min/max, and
   runtime-scalar computed-mask standalone add/min/max where already supported.
   Do not add new reduction ops, dtype/LMUL clone batches, widening MAcc, dot,
   contraction, or frontend lowering.
6. Keep common EmitC neutral. Common materialization must only consume the
   provider-built route payload and must not infer standalone reduction
   semantics from route ids, ABI strings, metadata, artifact names, scripts, or
   exact intrinsic names.
7. Target artifact validation must remain based on the rebuilt provider route
   and provider description. If tightened, it must validate metadata only as
   mirrors after route reconstruction.
8. Generated-bundle evidence must check representative selected-boundary
   standalone reduction mirrors for:
   - `provider_supported_mirror`;
   - standalone route-family plan id;
   - route operand binding plan/operands;
   - runtime ABI order;
   - source vector type/C type;
   - scalar-result vector type/C type;
   - scalar result runtime boundary;
   - mask/inactive-lane policy for computed-mask forms;
   - header/type/intrinsic mirrors.
9. Direct pre-realized route-entry shortcuts must remain fail-closed for
   unsupported standalone reduction forms, and selected-boundary evidence with
   `route_entry_realization: false` must remain the positive path where
   applicable.

## Acceptance Criteria

- [x] Production diff lands in RVV standalone reduction route
      planning/provider code, not only tests or scripts.
- [x] Provider materialization fails closed before `TCRVEmitCLowerableRoute`
      construction when standalone reduction family facts are missing, stale,
      inconsistent with materialization facts, or inconsistent with route
      description mirrors.
- [x] Provider materialization rejects stale standalone reduction family facts
      on non-standalone routes.
- [x] Focused C++ plugin tests cover positive provider-plan/route construction
      for representative standalone reduction forms and fail-closed stale or
      missing facts before route construction.
- [x] Target artifact validation covers provider support, route-family plan id,
      operand binding, accumulator/result layout, runtime ABI order,
      mask/inactive-lane policy, header/type/intrinsic mirrors, stale
      non-reduction mirrors, and rebuilt route ABI/type mappings for
      standalone reduction consumers.
- [x] Generated-bundle dry-run covers representative selected-boundary
      standalone reductions including `standalone_reduce_add`,
      `computed_mask_standalone_reduce_add`, and
      `runtime_scalar_cmp_masked_standalone_reduce_add`, with
      `route_entry_realization: false` where applicable.
- [x] Direct route-entry fail-closed non-regression covers retired shortcuts
      for unsupported standalone reduction route-entry forms.
- [x] Real `ssh rvv` generated-bundle correctness passes for executable
      representative reductions over counts `0`, `1`, exact-VL, tail, and a
      stress count with signed data and runtime scalar thresholds where
      applicable, or an exact external blocker is recorded.
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
   `cmake --build build --target tianchenrv-rvv-extension-plugin-test -j2`
   and `./build/bin/tianchenrv-rvv-extension-plugin-test`.
2. Build and run target artifact tests if target validation changes:
   `cmake --build build --target tianchenrv-target-artifact-export-test -j2`
   and `./build/bin/tianchenrv-target-artifact-export-test`.
3. Compile-check the generated-bundle script:
   `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`.
4. Run generated-bundle dry-runs for representative selected-boundary
   standalone reductions:
   `standalone_reduce_add`,
   `computed_mask_standalone_reduce_add`, and
   `runtime_scalar_cmp_masked_standalone_reduce_add`, using counts
   `0,1,16,17,257` and at least two runtime scalar thresholds for runtime
   scalar forms.
5. Run direct pre-realized route-entry fail-closed probes for unsupported
   standalone reduction route-entry forms.
6. Run real `ssh rvv` generated-bundle correctness for the executable
   representative standalone reductions unless an exact external blocker
   occurs.
7. Run bounded authority scans over touched files.
8. Run `git diff --check`.
9. Run `cmake --build build --target check-tianchenrv -j2`, or record the
   exact blocker.

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
  `.trellis/tasks/archive/2026-05/05-31-05-31-stage2-rvv-elementwise-arithmetic-route-family-abi-provider-ownership/prd.md`,
  `.trellis/tasks/archive/2026-05/05-31-05-31-stage2-rvv-elementwise-arithmetic-route-family-abi-provider-ownership/evidence-summary.md`,
  `.trellis/tasks/archive/2026-05/05-28-stage2-rvv-standalone-reduction-route-family-authority/prd.md`,
  `.trellis/tasks/archive/2026-05/05-28-05-28-stage2-rvv-standalone-reduction-scalar-channel/prd.md`,
  `.trellis/tasks/archive/2026-05/05-25-05-25-stage2-rvv-standalone-reduction-route-family-boundary-reuse/prd.md`, and
  `.trellis/tasks/archive/2026-05/05-30-stage2-rvv-reduction-accumulation-owner-boundary/prd.md`.
- Relevant implementation files inspected:
  `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`,
  `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`,
  `include/TianChenRV/Plugin/RVV/RVVStandaloneReductionSelectedBodyRealizationOwner.h`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCReductionAccumulationStatementPlanOwners.cpp`,
  `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`,
  `scripts/rvv_generated_bundle_abi_e2e.py`,
  `test/Plugin/RVVExtensionPluginTest.cpp`,
  directly related standalone reduction target/lit tests, and generated-bundle
  standalone reduction script tests.

No blocking user question remains. The supplied direction brief, current
specs, archived task boundaries, and code inspection are specific enough for
one bounded implementation round.
