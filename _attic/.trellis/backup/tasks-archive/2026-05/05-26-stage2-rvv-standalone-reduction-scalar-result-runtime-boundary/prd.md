# Stage2 RVV standalone reduction scalar-result runtime boundary

## Goal

Close one representative Stage2 standalone RVV reduction scalar-result runtime
boundary from a selected `tcrv.exec` RVV variant with typed `tcrv_rvv` body
through RVV plugin-owned route planning/provider ownership, neutral common
EmitC materialization, generated RVV artifact/harness, and real `ssh rvv`
correctness evidence.

The representative route is a standalone scalar-result `reduce_add` unless live
code inspection proves another existing standalone reduction is the correct
minimal owner. The task must repair production validation where scalar-result
ABI/runtime ownership is incomplete; it must not finish as archive-only evidence
unless that boundary is already explicitly proven complete.

## What I already know

- Current HEAD is `97970ba4 chore(task): archive direct cmp-select executable closure`.
- The repository started this task with a clean worktree.
- Recent archived closures already cover direct contraction, computed-mask
  memory, and direct compare/select executable paths.
- TianChen-RV's RVV-first chain is `tcrv.exec` envelope -> selected RVV variant
  -> typed low-level `tcrv_rvv` body -> RVV plugin-owned legality/selected-body
  realization/route provider -> `TCRVEmitCLowerableRoute` -> common EmitC
  materializer -> target artifact -> `ssh rvv` evidence for runtime claims.
- `tcrv.exec` owns ABI/runtime roles and the execution envelope, not reduction
  compute semantics, dtype authority, RVV schedule, intrinsic spelling, or route
  selection.
- RVV plugin code, not route ids, metadata, artifact names, source front doors,
  common EmitC, scripts, ABI strings, descriptors, or legacy i32 helpers, must
  own standalone reduction route validation and construction.

## Scope

- Inspect and, if needed, repair the standalone reduction route-family boundary
  centered on one representative selected typed RVV standalone reduction body.
- Validate reduction kind, neutral value, accumulator/result layout, scalar
  output ABI role, runtime `n`/AVL, VL/tail/mask policy, operand bindings,
  `mem_window`/`runtime_param` roles, selected capability, and route-control
  facts before route construction.
- Preserve common EmitC/export neutrality: common materialization may consume a
  provider-built `TCRVEmitCLowerableRoute`, but must not choose RVV reduction
  semantics.
- Add focused tests or script checks only for repaired behavior and targeted
  fail-closed diagnostics.
- Produce non-dry-run `ssh rvv` correctness evidence for the chosen generated
  reduction artifact over multiple runtime counts with a scalar oracle.

## Out of Scope

- Broad reduction matrices, dtype/LMUL clone batches, Linalg/source-front-door
  positive routes, dashboards, readiness/status-only work, reports-only work,
  or unrelated scalar/IME/offload/TensorExt work.
- New dtype-prefixed helper ops or legacy positive i32 route authority.
- Changing compare/select, memory, contraction, dispatch/fallback, runtime ABI,
  or selected-body semantics except where directly required by the scalar-result
  reduction boundary and evidenced.

## Acceptance Criteria

- [ ] A selected typed standalone RVV reduction body reaches the RVV route
  provider through owner-driven planning/statement-plan ownership.
- [ ] Route construction validates all required scalar-result reduction facts:
  reduction kind, neutral value, accumulator/result layout, scalar output ABI
  role, runtime `n`/AVL, VL/tail/mask policy, operand bindings, mem-window and
  runtime-param roles, selected capability, and route-control facts.
- [ ] Missing or inconsistent boundary facts fail closed with targeted
  diagnostics in focused tests.
- [ ] The generated route lowers through `TCRVEmitCLowerableRoute` and common
  EmitC without common EmitC owning RVV semantics.
- [ ] A non-dry-run `ssh rvv` generated-bundle run validates scalar correctness
  for multiple runtime counts with an independent scalar oracle.
- [ ] At least one existing direct cmp-select or computed-mask memory
  non-regression check is run.
- [ ] A bounded authority scan finds no executable claim depending on central
  ad hoc, name-derived, metadata-derived, descriptor-derived,
  ABI-string-derived, script-derived, artifact-name-derived, common-EmitC-derived,
  source-front-door-derived, route-id-derived, or legacy-i32-derived authority.
- [ ] `git diff --check` passes and the final worktree is clean after commit or
  the exact blocker is recorded.

## Technical Notes

- Primary specs to read before coding:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  `.trellis/spec/lowering-runtime/emission-runtime-contract.md`,
  `.trellis/spec/plugin-protocol/interfaces-and-registry.md`,
  `.trellis/spec/testing/mlir-testing-contract.md`,
  `.trellis/spec/variant-pipeline/generation-selection-tuning.md`, and
  `.trellis/spec/guides/index.md`.
- Primary code surfaces from the brief:
  `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`,
  `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`,
  `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp`,
  `scripts/rvv_generated_bundle_abi_e2e.py`,
  `test/Plugin/RVVExtensionPluginTest.cpp`, and relevant `test/Target/RVV`
  standalone reduction fixtures.
- Archived context to compare against:
  route-provider statement-plan owner registry, direct contraction executable
  closure, computed-mask memory executable closure, direct cmp-select executable
  closure, standalone reduction ownership/provider-validation/route-family
  tasks.
