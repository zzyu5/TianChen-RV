# PRD: Stage2 RVV runtime-scalar splat-store provider ABI preflight

## Direction Source

Hermes/user direction switches the next bounded owner to the RVV route
planning/provider boundary for `runtime_scalar_splat_store` provider facts
before `TCRVEmitCLowerableRoute` construction.

Requested production chain:

```text
selected tcrv.exec RVV variant for runtime_scalar_splat_store
  -> realized typed tcrv_rvv runtime-scalar splat/store body
  -> runtime scalar splat-store route-family plan
  -> materialization facts
  -> residual operand-binding facts for scalar input, output, and runtime n
  -> route-control provider facts
  -> runtime scalar splat-store statement plan
  -> provider-facts preflight
  -> TCRVEmitCLowerableRoute
  -> common EmitC materializer
  -> generated bundle / target artifact
  -> ssh rvv evidence when executable correctness is claimed
```

## Initial Repository State

- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean.
- Initial HEAD: `a76aee17 rvv: close widening conversion provider ABI boundary`.
- `.trellis/.current-task` was absent, so this task was created from the
  supplied Direction Brief before source edits.
- Serial worker constraint from the brief: one Codex worker; no subagents,
  spawned agents, parallel agents, or multi-agent workflow.

## Repository Findings

- The RVV specs require selected typed `tcrv_rvv` body facts, RVV-owned
  provider plans, and a provider-built `TCRVEmitCLowerableRoute`; route ids,
  artifact names, ABI strings, metadata, scripts, common EmitC, descriptors,
  source-front-door markers, exact intrinsic spelling, and legacy i32 route
  surfaces are not route authority.
- Runtime scalar splat-store already has:
  - `RVVSelectedBodyRuntimeScalarSplatStoreRouteFamilyPlan`;
  - selected-analysis materialization facts for VL type, vector type, setvl,
    scalar splat, store, headers, and provider mirrors;
  - residual operand-binding facts for `rhs_scalar`, `out`, and `n`;
  - route-control provider-plan consumption;
  - `RVVSelectedBodyRuntimeScalarSplatStoreRouteStatementPlan`;
  - migrated statement-plan owner membership.
- The provider currently calls provider-facts preflights for compare/select,
  widening conversion, computed-mask memory, standalone reduction, and
  segment2 before route construction.
- Runtime scalar splat-store reaches route construction through family-plan
  verification, materialization, residual binding, route-control, and
  statement-plan surfaces, but lacks a named provider-facts preflight tying
  those facts together immediately before `TCRVEmitCLowerableRoute`
  construction.
- Common EmitC remains neutral and must stay a consumer of provider-built
  route payloads only.

## Goal

Close the runtime scalar splat-store provider ABI boundary by requiring the
RVV provider to prove selected-analysis runtime scalar splat-store family
facts, typed config/materialization facts, residual operand bindings,
route-control provider facts, runtime ABI order, and the RVV-owned statement
plan before constructing `TCRVEmitCLowerableRoute`.

This task is a bounded production-plus-evidence task. It must land in RVV
planning/provider code and focused tests. Generated-bundle or target-artifact
checks may validate mirrors of provider facts, but must not authorize route
support.

## Requirements

1. Add a provider-side runtime scalar splat-store route preflight before
   `TCRVEmitCLowerableRoute` construction.
2. The preflight must require exactly the same-analysis
   `RVVSelectedBodyRuntimeScalarSplatStoreRouteFamilyPlan` for
   `runtime_i32_splat_store` consumers and reject stale runtime scalar
   splat-store provider facts on non-consumer routes.
3. The preflight must verify selected typed body/config and family-plan facts
   agree for operation, `RuntimeScalarSplatStore` memory form, SEW, LMUL,
   policy, VL type, result vector type/C type, setvl leaf, scalar splat leaf,
   store leaf, result name, required headers, target leaf profile, provider
   support mirror, and C type mapping.
4. The preflight must verify materialization facts mirror the validated family
   plan for required headers, VL type, vector type, setvl, runtime scalar
   splat, and store, with no unrelated mask, source-load, conversion,
   reduction, accumulation, contraction, or segment facts.
5. The preflight must verify residual operand-binding facts come from the same
   selected route analysis and bind `rhs_scalar`, `out`, and runtime `n` with
   the expected ABI roles and materialized uses.
6. The preflight must verify the RVV-owned route-control provider plan for the
   same selected analysis, typed config, selected target capability,
   runtime AVL/VL control plan, policy, provider/legality mirrors, and runtime
   ABI order.
7. The preflight must verify the RVV-owned runtime scalar splat-store
   statement plan is present, targets `runtime_i32_splat_store`, points at the
   same family plan, and carries provider-ready setvl, scalar splat, and store
   leaves.
8. The provider must call the preflight after route-family verification,
   materialization facts, residual operand-binding facts, and runtime scalar
   splat-store statement-plan construction, but before constructing
   `TCRVEmitCLowerableRoute`.
9. The preflight must not build statements, choose intrinsics, infer
   dtype/config, read artifact metadata, consult route ids, call selected-body
   owner hooks, or move semantics into common EmitC/export.
10. Preserve selected-body realization semantics. Do not change computation
    semantics, dtype semantics, parameter roles, variant origin,
    dispatch/fallback behavior, runtime `n`/AVL/VL values, required
    capabilities, or emitted statement order except as directly required by
    fail-closed validation.

## Acceptance Criteria

- [ ] Production diff lands in RVV route planning/provider code, not only tests
      or scripts.
- [ ] Provider route construction calls the runtime scalar splat-store
      provider-facts preflight before `TCRVEmitCLowerableRoute` construction.
- [ ] Positive C++ coverage proves `runtime_i32_splat_store` reaches the
      provider preflight with selected typed body facts, family plan,
      materialization facts, residual operand bindings, route-control facts,
      runtime ABI order, and statement plan.
- [ ] Fail-closed C++ coverage covers missing/stale family plan, stale
      materialization facts, wrong scalar/result dtype or vector type, missing
      scalar splat/store leaves, missing or wrong residual operand binding for
      `rhs_scalar`/`out`/`n`, stale route-control provider facts, wrong runtime
      ABI order, stale statement plan, and non-consumer stale splat-store
      facts.
- [ ] Generated-bundle dry-run covers selected-boundary runtime scalar
      splat-store after the provider preflight is wired.
- [ ] Real `ssh rvv` correctness is collected for representative counts and
      scalar values if executable correctness is claimed.
- [ ] Non-regression for the latest widening conversion provider preflight
      passes through the focused C++ test.
- [ ] Bounded authority scan over touched production/test/script/spec files
      finds no new central ad hoc, name-derived, metadata-derived,
      descriptor-derived, ABI-string-derived, script-derived,
      artifact-name-derived, common-EmitC-derived, source-front-door-derived,
      route-id-derived, exact-intrinsic-derived, direct-route-entry-only,
      pre-realized-fixture-only, or legacy-i32-derived route authority.
- [ ] `git diff --check` passes.
- [ ] Focused builds/tests pass, and `check-tianchenrv` passes or the exact
      blocker is recorded.
- [ ] Task status, journal/archive, and one coherent commit complete if this
      task finishes.

## Validation Plan

1. Build focused RVV plugin test target:
   `cmake --build artifacts/tmp/tianchenrv-build --target tianchenrv-rvv-extension-plugin-test -j2`.
2. Run focused RVV plugin tests:
   `artifacts/tmp/tianchenrv-build/bin/tianchenrv-rvv-extension-plugin-test`.
3. Compile-check generated-bundle tooling if touched:
   `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`.
4. Run focused generated-bundle dry-run for selected-boundary
   `runtime_i32_splat_store` with representative counts and scalar values.
5. Run `ssh rvv` generated-bundle correctness for representative counts and
   scalar values only if this round claims executable correctness.
6. Run a focused widening conversion provider-preflight non-regression through
   the RVV plugin C++ test.
7. Run bounded authority scans over touched files.
8. Run `git diff --check`.
9. Run `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`,
   or record the exact blocker.

## Out of Scope

- New dtype, LMUL, runtime scalar, compare/select, memory, reduction,
  conversion, MAcc, contraction, segment2, TensorExt, IME, Offload, or future
  plugin coverage.
- High-level Linalg/Vector/StableHLO frontend lowering or generic frontend
  work.
- New descriptor-driven, source-front-door, source-artifact, direct-C/export,
  artifact-index dashboard, broad smoke-matrix, report-only, or evidence-only
  work.
- Refactoring the entire provider or target bundle beyond the smallest change
  needed for this runtime scalar splat-store preflight.
- Moving runtime scalar splat-store semantics into `tcrv.exec`, common EmitC,
  target export, scripts, metadata, manifests, descriptors, route ids, ABI
  strings, artifact names, test names, exact intrinsic strings, or legacy i32
  helper spellings.

## Technical Notes

- Specs read:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/core-dialect/tcrv-exec-contract.md`, and
  `.trellis/spec/lowering-runtime/emitc-route.md`.
- Prior task context read:
  `.trellis/tasks/archive/2026-05/05-31-stage2-rvv-widening-conversion-route-family-abi-provider-ownership/prd.md`,
  `.trellis/tasks/archive/2026-05/05-31-stage2-rvv-widening-conversion-route-family-abi-provider-ownership/evidence-summary.md`,
  `.trellis/tasks/archive/2026-05/05-25-stage2-rvv-runtime-scalar-splat-store-route-control/prd.md`, and
  `.trellis/tasks/archive/2026-05/05-23-05-23-stage2-rvv-runtime-scalar-splat-store-runtime-binding-closure/prd.md`.
- Relevant implementation files inspected:
  `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`,
  `include/TianChenRV/Plugin/RVV/RVVEmitCStatementPlanOwners.h`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCStatementPlanOwners.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCResidualStatementPlanOwners.cpp`, and
  `test/Plugin/RVVExtensionPluginTest.cpp`.

No blocking user question remains. The supplied Direction Brief, current
specs, archived provider-preflight pattern, and current code are specific
enough for one bounded implementation round.
