# Stage2 RVV computed-mask standalone reduction scalar-result boundary

## Goal

Extend the validated Stage2 standalone RVV scalar-result reduction boundary to
the computed-mask representative route `computed_mask_standalone_reduce_add`.
The production path must run from a selected `tcrv.exec` RVV variant with a
typed masked standalone `tcrv_rvv` body through RVV-owned planning, provider
validation, `TCRVEmitCLowerableRoute`, neutral common EmitC materialization,
generated RVV artifact/harness, and real `ssh rvv` correctness evidence.

## What I already know

- Current HEAD is `8eb4c24a rvv: validate standalone reduction scalar boundary`.
- The repository started this task with a clean worktree and no active Trellis
  task.
- The previous archived task
  `05-26-stage2-rvv-standalone-reduction-scalar-result-runtime-boundary` closed
  the unmasked `standalone_reduce_add` scalar-result runtime boundary with
  provider-owned validation, metadata mirrors only after validation, generated
  bundle support, and `ssh rvv` PASS evidence.
- The current code already has computed-mask standalone reduction operation
  kinds, typed/pre-realized realization hooks, route-family plan surfaces,
  operand-binding facts, and generated-bundle script hooks. This task must
  verify and repair the masked scalar-result/runtime boundary rather than add a
  new route family.
- Common EmitC/export must remain neutral. RVV semantics, mask policy,
  inactive-lane semantics, scalar-result ABI, intrinsic/header choices, and
  runtime AVL/VL facts must come from typed `tcrv_rvv` body/config/runtime facts
  consumed by the RVV plugin.

## Scope

- Representative route: `computed_mask_standalone_reduce_add`.
- Validate provider-owned route facts for reduction kind `add`, scalar output
  ABI, accumulator input role, runtime `n`/AVL, mask role/source, inactive-lane
  neutral semantics, active-lane contribution, `n = 0` behavior,
  all-inactive-mask behavior, tail preservation, typed config, operand
  bindings, selected capability, and provider-supported mirrors before route
  construction.
- Add or repair focused production C++/script/test behavior only where needed
  for the representative computed-mask standalone reduction boundary.
- Add focused fail-closed diagnostics for stale/missing/inconsistent
  scalar-result, mask, inactive-neutral, route-family, or metadata-mirror facts.
- Generate dry-run bundle evidence and non-dry-run `ssh rvv` evidence over
  multiple runtime counts, including `n = 0` and mixed active/inactive masks,
  with a scalar oracle that distinguishes active masked lanes from inactive
  neutral lanes and verifies `out[0]` separately from sentinel output slots.
- Run bounded non-regression for the previous unmasked `standalone_reduce_add`
  boundary and one existing computed-mask or compare/select path.

## Out of Scope

- Min/max matrices, dtype/LMUL clone batches, unrelated reductions, new
  compare/select/memory/contraction route families, source-front-door positive
  routes, high-level frontend/Linalg routes, dashboards, reports-only work, or
  status-only work.
- Any descriptor-driven, route-id-derived, artifact-name-derived,
  script-derived, ABI-string-derived, common-EmitC-derived, source-front-door,
  or legacy-i32 route authority.
- Weakening the just-landed `standalone_reduce_add` scalar-result boundary.

## Requirements

- The selected typed body must structurally carry the computed mask producer,
  source payload, accumulator/seed, scalar output, runtime count, typed config,
  and policy facts needed by the route provider.
- The RVV planning/provider layer must derive or validate the route-family plan,
  materialization facts, math operand-binding facts, route-control facts,
  selected capability facts, and scalar-result runtime boundary before creating
  the lowerable route.
- Mirror fields such as `provider_supported_mirror`, route metadata, artifact
  metadata, and generated bundle summaries may appear only after provider route
  validation and must not be used as authority.
- Runtime evidence must compile and run through the generated RVV artifact path
  on `ssh rvv`; local static checks are insufficient for executable claims.

## Acceptance Criteria

- [ ] `computed_mask_standalone_reduce_add` reaches provider-built
  `TCRVEmitCLowerableRoute` through RVV-owned planning and validation.
- [ ] Provider validation covers reduction kind `add`, runtime `n`/AVL,
  typed config, selected capability, mask role/source, inactive-neutral
  contract, active-lane contribution, accumulator input role, scalar output
  ABI, operand bindings, route-control facts, and mirror agreement.
- [ ] Missing or stale scalar-result, mask, inactive-neutral, route-family,
  runtime-control, operand-binding, selected-capability, or mirror facts fail
  closed with targeted diagnostics before common EmitC.
- [ ] Generated dry-run evidence exposes bounded mirror fields proving
  provider-derived typed body/config/runtime authority, mask-tail policy
  boundary, scalar result boundary, and mirror-only artifact metadata.
- [ ] Non-dry-run `ssh rvv` generated-bundle execution passes for multiple
  runtime counts including `n = 0` and mixed active/inactive masks, checks
  all-inactive masks, validates scalar `out[0]`, and preserves sentinel output
  slots.
- [ ] Previous `standalone_reduce_add` behavior and at least one existing
  computed-mask or compare/select path still pass focused non-regression.
- [ ] A bounded authority scan over touched RVV planning/provider/test/script
  files finds no executable claim depending on central ad hoc, name-derived,
  metadata-derived, descriptor-derived, ABI-string-derived, script-derived,
  artifact-name-derived, common-EmitC-derived, source-front-door-derived,
  route-id-derived, or legacy-i32-derived authority.
- [ ] `git diff --check` passes, relevant focused tests pass or exact blockers
  are recorded, task status is truthful, and a coherent commit is created if
  the task completes.

## Technical Notes

- Relevant specs: `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  `.trellis/spec/testing/mlir-testing-contract.md`,
  `.trellis/spec/variant-pipeline/generation-selection-tuning.md`.
- Relevant previous task:
  `.trellis/tasks/archive/2026-05/05-26-stage2-rvv-standalone-reduction-scalar-result-runtime-boundary`.
- Primary code surfaces:
  `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`,
  `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`,
  `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp`,
  `scripts/rvv_generated_bundle_abi_e2e.py`,
  `test/Plugin/RVVExtensionPluginTest.cpp`, relevant `test/Target/RVV` and
  `test/Scripts` fixtures.
