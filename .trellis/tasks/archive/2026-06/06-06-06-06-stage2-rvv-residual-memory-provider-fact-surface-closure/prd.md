# Stage2 RVV residual memory provider-fact surface closure owner

## Goal

Close the remaining RVV memory-family provider-fact ownership gap by moving the
runtime-scalar splat-store and computed-mask memory pre-route provider-fact
checks out of the generic `RVVEmitCRoutePlanning` public surface and onto
owner-local RVV plugin surfaces that already own the relevant statement plans
and family facts. The provider must still fail closed before
`TCRVEmitCLowerableRoute` construction whenever memory form, scalar/mask/data
roles, VL/control policy, SEW/LMUL, ABI order, route-family facts, or
statement-owner facts are stale or missing.

## Direction Source

Hermes Direction Brief:

`Stage2 RVV residual memory provider-fact surface closure owner`

## What I Already Know

- Initial `pwd` was `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short` was clean.
- Initial `git log --oneline -8` started at
  `df5c1926 rvv: own direct contraction provider facts`.
- No `.trellis/.current-task` existed at the start of this round, so this task
  was created from the Hermes brief before source edits.
- Local memory confirms the RVV authority chain and the rule that dtype/config
  facts must come from typed `tcrv_rvv` body/config and provider validation,
  not route ids, ABI strings, artifact names, descriptors, helper names, or
  Common EmitC/export code.
- Specs read:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/index.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`.
- Previous archived task
  `.trellis/tasks/archive/2026-06/06-06-stage2-rvv-direct-contraction-fact-contract/prd.md`
  moved direct contraction provider-fact verification to an owner-local
  statement/provider owner surface while preserving pre-route order.
- A bounded scan of `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`
  shows three remaining family-specific provider-fact verifier declarations on
  the generic planning surface:
  `verifyRVVSelectedBodyRuntimeScalarSplatStoreRouteProviderFacts(...)`,
  `verifyRVVSelectedBodyRuntimeScalarComputedMaskMemoryRouteProviderFacts(...)`,
  and `verifyRVVSelectedBodyRegularComputedMaskMemoryRouteProviderFacts(...)`.
- `RVVEmitCRouteProvider.cpp` already obtains computed-mask memory statement
  plans through `getRVVSelectedBodyComputedMaskMemoryRouteStatementPlan(...)`
  and calls the aggregate
  `verifyRVVSelectedBodyComputedMaskMemoryRouteProviderFacts(...)` before route
  construction for non-segment computed-mask memory consumers.
- The generic split computed-mask verifier names still appear in generic route
  planning, so this round must either move them under the computed-mask memory
  owner implementation or remove them if the aggregate owner-local verifier is
  the only production boundary.

## Requirements

- Move the runtime-scalar splat-store provider-fact verifier declaration and
  implementation to the runtime-scalar memory selected-body/statement owner
  surface that owns the splat-store statement plan.
- Move or remove the split runtime-scalar computed-mask memory and regular
  computed-mask memory provider-fact verifier declarations from the generic
  `RVVEmitCRoutePlanning` public surface. If they remain as helpers, they must
  be private owner-local implementation details under the computed-mask memory
  statement/provider owner path.
- Keep the public production computed-mask memory preflight as one owner-local
  boundary:
  `verifyRVVSelectedBodyComputedMaskMemoryRouteProviderFacts(...)`, called
  after route-family provider plans, materialization facts, memory
  operand-binding facts, and the computed-mask memory statement plan are
  available, but before constructing `TCRVEmitCLowerableRoute`.
- Preserve runtime-scalar splat-store fail-closed checks for operation kind,
  memory form, typed splat/store leaves, scalar and output ABI roles, runtime
  count role/order, SEW/LMUL/policy/config, runtime AVL/VL control,
  route-family plan, residual operand-binding facts, and statement-plan leaves.
- Preserve computed-mask memory fail-closed checks for runtime-scalar store and
  load-store, regular unit/strided/indexed computed-mask memory, mask
  provenance, scalar or compare RHS binding, source/destination/index/stride
  roles, memory form, route-control policy, SEW/LMUL, ABI order, family plan,
  memory operand-binding facts, and statement-plan leaves.
- Keep Common EmitC/export neutral. The implementation must not infer memory,
  scalar, mask, dtype, policy, ABI, intrinsic, or route support from route ids,
  artifact names, C strings, helper names, metadata mirrors, descriptors,
  source-front-door markers, or Common EmitC behavior.
- Keep this round bounded to runtime-scalar splat-store and non-segment
  computed-mask memory provider-fact ownership. Do not rewrite unrelated
  elementwise, segment2, compare/select, conversion/dequantization, standalone
  reduction, direct contraction, MAcc, or widening-dot owners.

## Acceptance Criteria

- [x] Production plugin code changes in the runtime-scalar splat-store and/or
      computed-mask memory realization/provider ownership path.
- [x] `RVVEmitCRouteProvider` still consumes the relevant owner-local
      provider-fact verifier(s) before constructing `TCRVEmitCLowerableRoute`.
- [x] The generic `RVVEmitCRoutePlanning.h` public surface no longer declares
      `verifyRVVSelectedBodyRuntimeScalarSplatStoreRouteProviderFacts(...)`,
      `verifyRVVSelectedBodyRuntimeScalarComputedMaskMemoryRouteProviderFacts(...)`,
      or `verifyRVVSelectedBodyRegularComputedMaskMemoryRouteProviderFacts(...)`.
- [x] Any remaining split runtime-scalar/regular computed-mask helper checks
      are private owner-local implementation details or are deleted because the
      aggregate computed-mask memory owner verifier supersedes them.
- [x] Focused positive evidence proves typed-body memory/scalar/mask facts feed
      route planning/header/export before route construction for the touched
      runtime-scalar or computed-mask memory family.
- [x] Focused fail-closed evidence rejects stale or missing facts such as
      route-id-selected memory form, metadata-selected ABI, missing runtime
      scalar operand, wrong scalar ABI role/order, missing computed-mask
      provenance, wrong masked input/output role, stale route-control policy,
      wrong SEW/LMUL, or stale statement/route-family plan.
- [x] Existing explicit and pre-realized memory regressions for the touched
      families remain passing.
- [x] `build/bin/tianchenrv-rvv-extension-plugin-test` and
      `build/bin/tianchenrv-target-artifact-export-test` pass.
- [x] If emitted route statements, ABI order, headers, generated bundle output,
      or runtime harness behavior changes, run a focused generated-bundle
      dry-run or `ssh rvv` smoke. If this remains pre-route owner/API tightening
      only, record why runtime evidence is not required.
- [x] Bounded scan over touched files shows no new positive legacy `i32m1`,
      helper-name, descriptor, source-front-door, source-export, route-id, or
      mirror-only authority drift, and shows the named generic provider-fact
      declarations were removed or justified.
- [x] `git diff --check`, `git diff --cached --check`, Trellis validation, and
      clean final git status pass.

## Definition Of Done

- Compiler behavior remains in C++/MLIR/LLVM/TableGen/CMake/lit/FileCheck.
- Python remains limited to Trellis/tooling support and is not used for
  compiler core behavior.
- Runtime-scalar and computed-mask memory facts stay in RVV plugin owner seams;
  Common EmitC only consumes provider-built `TCRVEmitCLowerableRoute` payloads.
- No new memory route family, dtype/LMUL clone batch, high-level frontend,
  per-Linalg authority, source-front-door positive route, direct pre-realized
  route shortcut, performance/autotuning, generated-bundle matrix, or mass
  rewrite of unrelated owner seams is introduced.
- Task status/context stay truthful.
- The task is finished/archived and one coherent commit is created when
  acceptance criteria are met.

## Technical Approach

Use the owner-local verifier pattern from standalone reduction and direct
contraction:

```text
typed/realized memory tcrv_rvv body
  -> RVV route-family provider plans
  -> route materialization facts
  -> RVV-owned operand-binding facts
  -> RVV-owned statement plan
  -> owner-local provider-fact verifier
  -> TCRVEmitCLowerableRoute
  -> neutral Common EmitC materialization
```

The expected narrow implementation is to relocate
`verifyRVVSelectedBodyRuntimeScalarSplatStoreRouteProviderFacts(...)` to the
runtime-scalar memory statement owner interface/implementation and to relocate
or privatize the two split computed-mask verifier helpers inside the
computed-mask memory statement/provider owner implementation, leaving the
aggregate computed-mask verifier as the production call surface.

## Out Of Scope

- New runtime-scalar or computed-mask memory route families.
- New dtype/LMUL clone batches.
- High-level Linalg/Vector/StableHLO frontend work.
- Per-Linalg route authority.
- Common EmitC inference of RVV memory, scalar, mask, dtype, ABI, policy,
  intrinsic, or route-support semantics.
- Direct pre-realized route-entry shortcuts.
- Source-front-door positive routes.
- Performance/autotuning or global profile systems.
- Broad generated-bundle matrices.
- Mass rewrite of direct contraction, standalone reduction,
  conversion/dequantization, compare/select, segment2, elementwise, MAcc,
  widening-dot, or non-memory owners outside the touched seam.

## Technical Notes

- Relevant production files:
  `include/TianChenRV/Plugin/RVV/RVVRuntimeScalarMemorySelectedBodyRealizationOwner.h`,
  `lib/Plugin/RVV/RVVRuntimeScalarMemorySelectedBodyRealizationOwner.cpp`,
  `include/TianChenRV/Plugin/RVV/RVVComputedMaskMemorySelectedBodyRealizationOwner.h`,
  `lib/Plugin/RVV/RVVComputedMaskMemorySelectedBodyRealizationOwner.cpp`,
  `include/TianChenRV/Plugin/RVV/RVVEmitCComputedMaskMemoryRouteFamilyPlanOwners.h`,
  `lib/Plugin/RVV/EmitC/RVVEmitCComputedMaskMemoryRouteFamilyPlanOwners.cpp`,
  `include/TianChenRV/Plugin/RVV/RVVEmitCStatementPlanOwners.h`,
  `lib/Plugin/RVV/EmitC/RVVEmitCResidualStatementPlanOwners.cpp`,
  `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`.
- Relevant tests:
  `test/Plugin/RVVExtensionPluginTest.cpp`,
  `test/Target/TargetArtifactExportTest.cpp`,
  focused RVV memory fixtures under `test/Target/RVV/`.
- Expected runtime boundary: no `ssh rvv` evidence if the round only moves
  pre-route verifier ownership and does not change emitted statements, ABI
  order, headers, generated bundle content, or harness behavior.

## Completion Evidence

- Production owner seam changed:
  `verifyRVVSelectedBodyRuntimeScalarSplatStoreRouteProviderFacts(...)` is now
  declared on `RVVEmitCStatementPlanOwners.h` and implemented in
  `RVVEmitCResidualStatementPlanOwners.cpp`, next to the runtime-scalar
  splat-store statement-plan owner.
- Computed-mask memory owner seam changed:
  `verifyRVVSelectedBodyRuntimeScalarComputedMaskMemoryRouteProviderFacts(...)`
  and `verifyRVVSelectedBodyRegularComputedMaskMemoryRouteProviderFacts(...)`
  are now private file-local helpers in
  `RVVEmitCComputedMaskMemoryRouteFamilyPlanOwners.cpp`, consumed only through
  the public aggregate owner-local
  `verifyRVVSelectedBodyComputedMaskMemoryRouteProviderFacts(...)`.
- Generic route-planning surface closed:
  `RVVEmitCRoutePlanning.h` and `RVVEmitCRoutePlanning.cpp` no longer declare
  or implement the three residual memory provider-fact verifiers named in this
  PRD. `RVVEmitCRoutePlanning.h` exposes only the existing runtime-scalar
  splat-store route-family plan validator needed by the relocated verifier.
- `RVVEmitCRouteProvider.cpp` call order is unchanged: route-family provider
  plans, materialization facts, operand-binding facts, and statement plans are
  produced before the owner-local provider-fact verifiers run and before
  `TCRVEmitCLowerableRoute` construction.
- The change does not alter emitted route statements, ABI order, required
  headers, generated bundle content, or runtime harness behavior. No `ssh rvv`
  smoke was required because this is a pre-route owner/API relocation with
  preserved provider logic.
- No `.trellis/spec/` update was needed. Existing specs already define the
  computed-mask memory statement-plan boundary, runtime-scalar splat-store
  route validation contract, provider-built `TCRVEmitCLowerableRoute`
  contract, and Common EmitC neutrality; this round aligns code with those
  contracts rather than introducing a new executable contract.
- Build target rebuild passed:
  `cmake --build build --target tianchenrv-rvv-extension-plugin-test tianchenrv-target-artifact-export-test`.
- Focused C++ tests passed:
  `build/bin/tianchenrv-rvv-extension-plugin-test` and
  `build/bin/tianchenrv-target-artifact-export-test`.
- Focused positive PLAN/HEADER FileCheck smokes passed for:
  `test/Target/RVV/pre-realized-selected-body-artifact-runtime-scalar-splat-store.mlir`,
  `test/Target/RVV/explicit-selected-body-artifact-runtime-scalar-cmp-masked-store.mlir`,
  `test/Target/RVV/pre-realized-selected-body-artifact-computed-masked-unit-load-store.mlir`,
  `test/Target/RVV/pre-realized-selected-body-artifact-computed-masked-strided-store.mlir`,
  and
  `test/Target/RVV/pre-realized-selected-body-artifact-computed-masked-indexed-gather-load.mlir`.
- Focused fail-closed checks passed:
  `test/Transforms/LoweringBoundary/rvv-pre-realized-runtime-scalar-splat-store-negative.mlir`
  with `--verify-diagnostics`, and
  `test/Target/RVV/explicit-selected-body-computed-masked-strided-load-negative.mlir`
  with explicit nonzero `tcrv-opt` status plus `FAIL` FileCheck.
- Trellis context validation passed:
  `python3 ./.trellis/scripts/task.py validate .trellis/tasks/06-06-06-06-stage2-rvv-residual-memory-provider-fact-surface-closure`.
- Diff hygiene passed:
  `git diff --check`.
- Bounded added-diff old-authority scan over touched RVV header/cpp files had
  no new positive legacy `i32m1`, descriptor, source-front-door,
  source-export, direct-C, route-id, helper-name, metadata-derived, or
  mirror-only authority matches.
