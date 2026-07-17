# Stage2 RVV standalone reduction realization-provider fact contract

## Goal

Strengthen the production RVV plugin seam for standalone reduction route
families. This round is bounded to moving the provider-fact contract for
plain, computed-mask, and runtime-scalar computed-mask standalone reductions
onto the RVV standalone reduction statement-plan owner surface, while
preserving the central provider call before `TCRVEmitCLowerableRoute`
construction.

## Direction Source

Hermes Direction Brief:

`Stage2 RVV standalone reduction realization-provider fact-contract owner`

## What I Already Know

- Initial `pwd` was `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short` was clean.
- Initial `git log --oneline -8` started at
  `3a293c7c rvv: own conversion dequant provider facts`.
- No `.trellis/.current-task` existed at the start of this round, so this task
  was created from the Hermes brief before source edits.
- Specs read:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/index.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  `.trellis/spec/testing/index.md`.
- Previous archived task
  `.trellis/tasks/archive/2026-06/06-06-stage2-rvv-conversion-dequant-fact-contract/prd.md`
  moved conversion/dequantization provider-fact verification to the statement
  plan owner surface while keeping route-provider preflight order.
- Current standalone reduction code already has owner-built statement plans:
  `getRVVSelectedBodyStandaloneReductionRouteStatementPlan(...)` and
  `buildRVVSelectedBodyStandaloneReductionMigratedRouteStatementPlan(...)`.
- The remaining ownership gap is that
  `verifyRVVSelectedBodyStandaloneReductionRouteProviderFacts(...)` is still
  declared on the generic `RVVEmitCRoutePlanning` surface and implemented in
  `RVVEmitCRoutePlanning.cpp`, even though it validates standalone reduction
  statement-plan owner facts before provider route construction.

## Requirements

- Expose the standalone reduction provider-fact verifier from
  `RVVEmitCStatementPlanOwners.h`, next to the standalone reduction
  statement-plan builder.
- Move or otherwise localize the verifier implementation to the RVV
  statement-plan owner implementation path, so the generic route-planning
  public surface no longer owns this standalone reduction provider-fact
  contract.
- Preserve `RVVEmitCRouteProvider` call order: after route-family provider
  plans, materialization facts, math operand-binding facts, and the standalone
  reduction statement plan are available, but before
  `TCRVEmitCLowerableRoute` construction.
- Preserve fail-closed validation for standalone reduction operation kind,
  source/work channel, scalar accumulator/result channel, scalar result
  runtime boundary, memory form, computed-mask and runtime-scalar mask
  provenance, inactive-lane zeroing/neutral policy, SEW/LMUL, VL/control
  policy, ABI order and roles, canonical route-family plan facts,
  materialization leaves, and statement-plan leaves.
- Preserve stale-fact rejection for non-standalone operations, plain routes
  carrying computed-mask residue, computed-mask routes missing shared
  accumulation facts, runtime-scalar computed-mask routes missing RHS scalar
  splat/ABI facts, and cross-family stale facts.
- Keep Common EmitC neutral. The implementation must not infer standalone
  reduction semantics, reduction kind, dtype, ABI, mask/tail behavior,
  inactive-lane policy, scalar-result layout, intrinsic, or route support from
  route ids, artifact names, metadata mirrors, helper names,
  source-front-door markers, descriptors, or Common EmitC behavior.

## Acceptance Criteria

- [x] Production plugin code changes in the standalone reduction
      statement-plan owner/provider ownership path.
- [x] `RVVEmitCRouteProvider` still consumes the standalone reduction
      provider-fact verifier before constructing `TCRVEmitCLowerableRoute`.
- [x] The generic `RVVEmitCRoutePlanning.h` public surface no longer declares
      the standalone reduction provider-fact verifier.
- [x] The verifier implementation is localized with the standalone reduction
      statement-plan owner implementation path and preserves the existing
      fail-closed checks for typed body/config facts, family plan facts,
      computed-mask accumulation facts, route-control facts, runtime ABI facts,
      materialization facts, statement-plan leaves, and stale non-family facts.
- [x] Focused positive C++ coverage still proves plain standalone reduction,
      computed-mask standalone reduction, and runtime-scalar computed-mask
      standalone reduction facts feed provider route construction.
- [x] Focused fail-closed C++ coverage still rejects stale or missing facts
      such as wrong/missing verified standalone reduction family plan, stale
      source/scalar-result type facts, stale computed-mask accumulation facts,
      missing inactive-lane neutral/zeroing policy, stale mask provenance,
      wrong scalar result channel, wrong ABI role/order, stale materialization
      leaves, and stale statement leaves.
- [x] Existing explicit and pre-realized standalone-reduce regressions for the
      touched family remain passing.
- [x] `build/bin/tianchenrv-rvv-extension-plugin-test` and
      `build/bin/tianchenrv-target-artifact-export-test` pass.
- [x] If emitted route statements, runtime ABI order, header export, generated
      bundle output, or runtime harness behavior changes, run one focused
      generated-bundle dry-run or `ssh rvv` smoke. If this remains a pre-route
      owner/provider API tightening only, record why runtime evidence is not
      required.
- [x] Bounded old-authority scan over touched RVV owner/provider/test files and
      added diff lines shows no positive legacy `i32m1`, descriptor,
      source-front-door, direct-C/source-export, route-id, helper-name, or
      mirror-only route authority drift.
- [x] `git diff --check`, `git diff --cached --check`, Trellis validation, and
      clean final git status pass.

## Definition Of Done

- Compiler behavior remains in C++/MLIR/LLVM/TableGen/CMake/lit/FileCheck.
- Python remains limited to tooling and is not used for compiler core behavior.
- RVV standalone reduction semantics stay in RVV plugin
  realization/route-family/statement-plan/provider owners; Common EmitC only
  consumes provider-built routes and payloads.
- No new reduction operation family, dtype/LMUL clone batch, high-level
  Linalg/Vector/StableHLO frontend, per-Linalg route authority, generated
  bundle matrix, source-front-door positive route, direct pre-realized route
  shortcut, performance/autotuning, or mass rewrite of unrelated owner seams is
  introduced.
- Task status/context stay truthful.
- The task is finished/archived and one coherent commit is created when
  acceptance criteria are met.

## Technical Approach

Align standalone reduction with the conversion/dequantization owner pattern
completed in the previous task:

```text
realized typed standalone reduction tcrv_rvv body
  -> standalone reduction route-family materialization facts
  -> RVV-owned math operand-binding facts
  -> standalone reduction statement-plan owner
  -> standalone reduction owner provider-fact contract
  -> TCRVEmitCLowerableRoute
```

The narrow implementation target is to move the public provider-facing
contract declaration for
`verifyRVVSelectedBodyStandaloneReductionRouteProviderFacts(...)` onto
`RVVEmitCStatementPlanOwners.h`, relocate the implementation to the
statement-plan owner implementation path, and keep the production provider/tests
calling the owner-level surface.

## Out Of Scope

- New standalone reduction kinds or route families.
- Dtype/LMUL clone batches.
- High-level Linalg/Vector/StableHLO frontend work.
- Per-Linalg route authority.
- Common EmitC inference of standalone reduction, dtype, ABI, intrinsic,
  inactive-lane, scalar-result, or mask semantics.
- Direct pre-realized route-entry shortcuts.
- Source-front-door positive routes.
- Performance/autotuning or global profile systems.
- Broad generated-bundle matrices.
- Mass rewrite of conversion/dequantization, compare/select, memory, segment2,
  elementwise, direct contraction, runtime-scalar splat-store, or MAcc owners
  outside the touched seam.

## Technical Notes

- Relevant production files:
  `include/TianChenRV/Plugin/RVV/RVVEmitCStatementPlanOwners.h`,
  `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`,
  `lib/Plugin/RVV/EmitC/RVVEmitCResidualStatementPlanOwners.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`.
- Relevant standalone reduction realization files:
  `include/TianChenRV/Plugin/RVV/RVVStandaloneReductionSelectedBodyRealizationOwner.h`,
  `lib/Plugin/RVV/RVVStandaloneReductionSelectedBodyRealizationOwner.cpp`.
- Relevant tests:
  `test/Plugin/RVVExtensionPluginTest.cpp`,
  `test/Target/TargetArtifactExportTest.cpp`,
  focused explicit and pre-realized standalone reduction MLIR fixtures under
  `test/Target/RVV/`.
- The expected runtime boundary is no `ssh rvv` evidence if this round only
  moves pre-route verifier ownership and does not change emitted statements,
  ABI order, headers, bundle content, or runtime harness behavior.

## Completion Evidence

- Production owner seam changed:
  `verifyRVVSelectedBodyStandaloneReductionRouteProviderFacts(...)` is now
  declared in `RVVEmitCStatementPlanOwners.h`, implemented in
  `RVVEmitCResidualStatementPlanOwners.cpp`, and removed from the generic
  `RVVEmitCRoutePlanning.h` / `RVVEmitCRoutePlanning.cpp` surface.
- Provider route construction still calls the verifier before
  `TCRVEmitCLowerableRoute` construction, after route-family verification,
  materialization facts, math operand-binding facts, and the standalone
  reduction statement plan are available.
- Focused runtime-scalar computed-mask provider preflight coverage was added to
  `runStandaloneReductionRouteFamilyProviderPlanTest`, including positive route
  construction through the registry and fail-closed checks for missing RHS
  scalar splat facts and wrong RHS scalar ABI role/order.
- Existing plain and computed-mask standalone reduction provider-plan coverage
  remains in the same focused C++ test.
- The change does not alter emitted statements, runtime ABI order, header
  export, generated bundle content, or runtime harness behavior; no `ssh rvv`
  runtime smoke is required for this pre-route owner/API tightening.
- Code-spec update added the standalone reduction route-provider facts
  preflight contract to `.trellis/spec/extension-plugins/rvv-plugin.md`.
- Focused standalone reduction MLIR fixture smoke passed with `tcrv-opt` for
  explicit and pre-realized plain standalone reduce-add, plus explicit and
  pre-realized runtime-scalar computed-mask standalone reduce-add.
- `FileCheck` / `lit` binaries were not available in this environment, so the
  standalone reduction fixture evidence was gathered through focused
  `tcrv-opt` runs instead of FileCheck invocation.
- Focused binaries passed:
  `build/bin/tianchenrv-rvv-extension-plugin-test` and
  `build/bin/tianchenrv-target-artifact-export-test`.
- Diff hygiene passed:
  `git diff --check`, `git diff --cached --check`, Trellis task validation, and
  a bounded added-diff old-authority scan over touched owner/provider/test
  files.
