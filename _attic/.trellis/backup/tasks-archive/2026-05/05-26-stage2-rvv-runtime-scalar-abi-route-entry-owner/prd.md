# Stage2 RVV Runtime-Scalar ABI Route-Entry Owner

## Goal

Make selected `tcrv.exec.runtime_param` scalar values first-class in the RVV
direct route-entry selected-body realization and provider boundary for the
runtime-scalar computed-mask MAcc path. The concrete production path is the
current `runtime_scalar_cmp_masked_macc_add` route family, using the existing
typed runtime-scalar computed-mask MAcc body when available.

## What I Already Know

- The repository starts this task from clean `main` at
  `236e08c3 rvv: route computed-mask select entry owner`.
- There is no current Trellis task, so this task was created from the Hermes
  direction brief.
- The relevant long-term route is:
  selected `tcrv.exec` RVV variant -> typed/pre-realized `tcrv_rvv` body ->
  RVV selected-body realization owner -> RVV route-family/provider facts ->
  `TCRVEmitCLowerableRoute` -> common EmitC -> target artifact.
- Current code already has `TypedRuntimeScalarComputedMaskMAccPreRealizedBodyOp`
  and planning/provider references to
  `RuntimeScalarComputedMaskedMAccAdd`.
- Current dry-run tests exist for explicit and pre-realized
  `runtime_scalar_cmp_masked_macc_add`, but the task must still make a focused
  production owner change unless inspection proves the exact owner is already
  complete.
- Runtime/correctness claims require real `ssh rvv` evidence.

## Requirements

- Treat runtime scalar ABI values as structural selected-body inputs:
  `rhs_scalar` must be imported from `tcrv.exec.runtime_param`, typed as the
  scalar value consumed by `tcrv_rvv.splat`, and validated as the compare RHS
  for the same-VL mask producer.
- Keep runtime scalar semantics in RVV plugin-owned selected-body realization,
  planning, provider validation, operand-binding facts, route-control facts,
  and statement-plan facts.
- Route-entry realization must consume the pre-realized body before provider
  route facts are collected; provider/common EmitC must not synthesize missing
  runtime-scalar compare, splat, MAcc, mask, merge, accumulator, or store
  semantics.
- Provider validation must fail closed for missing or wrong scalar ABI role,
  stale scalar dtype/config relation, missing scalar splat use, wrong
  mask/VL/control facts, missing accumulator/result layout, missing runtime
  `n` ABI, stale route-control ownership, and unsupported route shape.
- Provider-supported, route ids, emission-plan fields, and artifact metadata
  may only appear as mirrors after route construction.
- Preserve non-regression for recent compare/select, MAcc, and contraction
  route-entry owners.

## Acceptance Criteria

- [x] A focused production diff touches the RVV selected-body realization and
      directly required planning/provider/test surfaces.
- [x] `runtime_scalar_cmp_masked_macc_add` direct route-entry realization is
      owner-scoped and route-entry eligible only through the computed-mask MAcc
      realization owner.
- [x] Provider facts for runtime-scalar computed-mask MAcc validate scalar ABI
      role, scalar splat, compare predicate, same-VL mask source, mask/tail
      policy, accumulator/result layout, runtime `n`/AVL, selected capability,
      operand bindings, route-control plan, materialization facts, and
      statement-plan facts before route construction.
- [x] Focused C++ or lit tests prove positive provider/route-entry consumption
      and fail-closed diagnostics for missing/wrong runtime scalar role or stale
      runtime-scalar MAcc facts.
- [x] Generated-bundle dry-run covers
      `runtime_scalar_cmp_masked_macc_add` with varied runtime counts and
      scalar values, including direct pre-realized route-entry mode if the
      production owner supports it.
- [x] At least one real `ssh rvv` run covers counts including `0`, a small
      count, exact-VL count, tail count, and a stress count with varied scalar
      values, or an exact blocker is recorded.
- [x] Correctness evidence distinguishes true runtime scalar use from constant
      folding/name-derived scalar, vector-vector fallback, unmasked path,
      inverted predicate, add-only, mul-only, accumulator clobbering,
      inactive-lane clobbering, and tail clobbering.
- [x] Focused non-regression checks cover compare/select, plain/scalar MAcc,
      computed-mask MAcc, and direct contraction route-entry owners.
- [x] Bounded authority scan finds no new dependency on descriptor, direct-C,
      source-export/source-front-door, ABI-string-derived, artifact-name-derived,
      script-derived, common-EmitC-derived, route-id-derived, metadata-derived,
      or legacy-i32-derived route authority.
- [x] `git diff --check` passes; `check-tianchenrv` passes or an exact blocker
      is reported.
- [x] Trellis task state is updated truthfully and the work is committed if
      complete.

## Out Of Scope

- Broad dtype/LMUL clone matrices.
- New high-level frontend/Linalg routes.
- New contraction variants or future plugin work.
- Descriptor-driven compute, source-front-door positive routes, direct-C/source
  export routes, dashboards, reports, or evidence-only commits.
- Moving RVV semantics into common EmitC or artifact export.

## Technical Notes

- Relevant specs:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  `.trellis/spec/variant-pipeline/generation-selection-tuning.md`,
  `.trellis/spec/testing/mlir-testing-contract.md`,
  `.trellis/spec/guides/index.md`.
- Relevant production files:
  `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp`,
  `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`,
  `scripts/rvv_generated_bundle_abi_e2e.py`.
- Relevant tests:
  `test/Plugin/RVVExtensionPluginTest.cpp`,
  `test/Target/RVV/*runtime-scalar-cmp-masked-macc-add*.mlir`,
  `test/Scripts/*runtime-scalar-cmp-masked-macc-add*.test`.
