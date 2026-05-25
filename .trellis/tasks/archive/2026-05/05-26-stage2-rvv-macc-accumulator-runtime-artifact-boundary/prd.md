# Stage2 RVV MAcc accumulator runtime artifact boundary

## Goal

Close the representative Stage2 plain `macc_add` accumulator/runtime/artifact
boundary from a selected `tcrv.exec` RVV variant with typed `tcrv_rvv.macc`
body through RVV-owned route-family validation, math operand binding,
plain-MAcc statement planning, provider-built `TCRVEmitCLowerableRoute`,
neutral common EmitC materialization, generated RVV bundle/harness, and real
`ssh rvv` correctness evidence.

The selected route is the existing plain vector-vector `macc_add` path:

```text
lhs[i], rhs[i], acc[i], n -> out[i] = acc[i] + lhs[i] * rhs[i]
```

This task is not a new MAcc family. It repairs the production boundary that
plain `macc_add` still lacks compared with neighboring mature families:
scalar-broadcast MAcc has a route-family plan and provider-supported mirror,
computed-mask MAcc has accumulation-family validation, but plain `macc_add`
currently relies on math operand-binding and statement-plan coverage without a
plain MAcc route-family validation surface that carries provider-supported
mirror facts into artifacts.

## What I already know

- Current HEAD is `affa0534 rvv: validate computed-mask standalone reduction boundary`.
- The repository started this task with a clean worktree and no current
  Trellis task.
- The previous archived standalone reduction tasks closed unmasked and
  computed-mask scalar-result reduction boundaries with provider-owned
  validation, generated bundle support, `ssh rvv` evidence, and mirror-only
  artifact metadata.
- The relevant long-term chain is `tcrv.exec` envelope -> selected RVV variant
  -> typed low-level `tcrv_rvv` body -> RVV plugin legality / selected-body
  realization / route provider -> `TCRVEmitCLowerableRoute` -> common EmitC ->
  target artifact -> `ssh rvv` evidence when correctness is claimed.
- Current code already has typed/pre-realized `macc_add` fixtures,
  selected-body realization, math operand-binding facts, migrated plain MAcc
  statement-plan ownership, generated-bundle script support, and historical
  `ssh rvv` evidence.
- Current live inspection shows scalar-broadcast MAcc has
  `RVVSelectedBodyScalarBroadcastMAccRouteFamilyPlan` and a validated
  `provider_supported_mirror`, while plain `macc_add` has no equivalent plain
  route-family plan/mirror field in `RVVSelectedBodyRouteAnalysis`,
  `RVVSelectedBodyRouteMaterializationFacts`, or artifact metadata.

## Scope

- Add the minimal plain `macc_add` route-family validation surface needed for
  provider-owned operation/runtime/accumulator/artifact facts.
- Validate before route construction:
  operation kind `add`, vector RHS-load memory form, lhs/rhs/acc/out/n ABI
  roles and order, accumulator input role/layout, output/result layout,
  runtime `n`/AVL and VL loop facts, typed SEW/LMUL/tail/mask policy, selected
  capability facts, required headers/C type mappings, provider-supported mirror,
  math operand-binding plan, and same-analysis materialization facts.
- Make target artifacts and generated evidence mirror plain MAcc route-family
  validation after provider route construction.
- Keep common EmitC/export neutral: it may consume the provider-built route and
  provider-ready statement plan, but it must not infer MAcc semantics, ABI,
  dtype/config, accumulator layout, intrinsic spelling, or support status.
- Extend focused C++/lit/script coverage for plain `macc_add` only where this
  boundary is observable.
- Run generated-bundle dry-run and real `ssh rvv` evidence for `macc_add` over
  multiple runtime counts including `n = 0`, small counts, and tail/sentinel
  cases.
- Run focused non-regression for the previous computed-mask standalone
  reduction scalar-result boundary and one existing compare/select or
  computed-mask memory path.

## Out of Scope

- Scalar-broadcast MAcc, computed-mask MAcc, runtime-scalar computed-mask MAcc,
  widening MAcc, dot reduction, contraction matrices, reductions beyond
  non-regression, dtype/LMUL clone batches, source-front-door positive routes,
  high-level frontend/Linalg routes, dashboards, reports-only work, or
  status-only work.
- Any new dtype-prefixed helper op such as `tcrv_rvv.i32_macc`.
- Any descriptor-derived, route-id-derived, artifact-name-derived,
  script-derived, ABI-string-derived, common-EmitC-derived, source-front-door,
  central ad hoc, metadata-only, or legacy-i32 route authority.
- Weakening the standalone reduction or computed-mask standalone reduction
  boundaries that just landed.

## Requirements

- Plain `macc_add` route-family support must be derived from typed
  `tcrv_rvv.macc` body/config/runtime facts and selected capability facts.
- The plain MAcc route-family plan must be RVV-local provider input. It may be
  mirrored into route metadata and target artifacts only after validation.
- Plain MAcc statement planning must consume the verified plain MAcc
  route-family plan, same-analysis materialization facts, and RVV-owned math
  operand-binding facts before emitting the setvl/load/load/acc-load/vmacc/store
  sequence.
- The generated artifact/harness must validate vector result semantics
  `out[i] = acc[i] + lhs[i] * rhs[i]`, include signed product coverage, and
  preserve output sentinels beyond runtime `n`.
- Runtime counts are execution cases, not authority for MAcc semantics,
  dtype/config, route support, or artifact metadata.

## Acceptance Criteria

- [ ] Plain `macc_add` has a provider-owned route-family validation surface or
  equivalent plan that carries operation, runtime, typed config, selected
  capability, accumulator/result layout, header/type/intrinsic, and
  provider-supported mirror facts before `TCRVEmitCLowerableRoute`
  construction.
- [ ] The route provider reaches plain `macc_add` through
  `RVVSelectedBodyRouteMaterializationFacts`,
  `RVVSelectedBodyMathRouteOperandBindingFacts`, and the RVV-owned plain MAcc
  statement plan; central provider code does not rebuild the MAcc statement
  sequence from names, ABI strings, route ids, or artifact metadata.
- [ ] Missing or stale plain MAcc route-family/mirror facts, math operand facts,
  accumulator/output/runtime ABI facts, materialization leaves, typed
  config/policy, or selected capability facts fail closed with targeted
  diagnostics before common EmitC.
- [ ] Explicit and pre-realized `macc_add` target artifact tests expose
  route-family mirror metadata, provider-supported mirror metadata, accumulator
  and result layout metadata, and route operand bindings as mirrors only.
- [ ] Generated-bundle dry-run evidence for explicit and pre-realized
  `macc_add` exposes typed body/config/runtime authority, selected ABI roles,
  accumulator/result layout, provider-supported mirror, RVV-owned statement
  plan, emitted `vmacc` operand order `acc,lhs,rhs,vl`, and mirror-only
  artifact metadata.
- [ ] Non-dry-run `ssh rvv` generated-bundle execution passes for `macc_add`
  over counts including `0`, a small count, one full-vector-ish count, and a
  tail case, with vector MAcc correctness and tail sentinel preservation.
- [ ] Previous computed-mask standalone reduction scalar-result behavior and
  at least one existing compare/select or computed-mask memory path still pass
  focused non-regression.
- [ ] A bounded authority scan over touched RVV planning/provider/test/script
  files finds no executable claim depending on central ad hoc, name-derived,
  metadata-derived, descriptor-derived, ABI-string-derived, script-derived,
  artifact-name-derived, common-EmitC-derived, source-front-door-derived,
  route-id-derived, or legacy-i32-derived authority.
- [ ] `git diff --check` passes. `check-tianchenrv` passes or the exact blocker
  is recorded. The task is finished/archived and a coherent commit is created
  if acceptance is met.

## Technical Notes

- Primary specs read before implementation:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  `.trellis/spec/testing/mlir-testing-contract.md`,
  `.trellis/spec/variant-pipeline/index.md`, and
  `.trellis/spec/guides/index.md`.
- Primary code surfaces:
  `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`,
  `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`,
  `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp`,
  `scripts/rvv_generated_bundle_abi_e2e.py`,
  `test/Plugin/RVVExtensionPluginTest.cpp`, and the `macc_add` fixtures under
  `test/Target/RVV` and `test/Scripts`.
- Archived context read:
  `05-26-stage2-rvv-standalone-reduction-scalar-result-runtime-boundary` and
  `05-26-05-26-stage2-rvv-computed-mask-standalone-reduction-scalar-boundary`.

## Open Questions

- None blocking. The task brief and current repository state identify one
  bounded production gap: plain `macc_add` needs the provider-owned
  route-family/accumulator/runtime/artifact boundary now present for adjacent
  Stage2 families.
