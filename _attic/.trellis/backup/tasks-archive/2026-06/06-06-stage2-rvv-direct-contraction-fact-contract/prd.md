# Stage2 RVV direct contraction realization-provider fact contract

## Goal

Strengthen the production RVV plugin seam for direct contraction route
families. This round is bounded to making the pre-route provider-fact contract
for the active direct contraction routes owner-local to the RVV direct
contraction statement/provider owner surface, while preserving provider call
order before `TCRVEmitCLowerableRoute` construction.

## Direction Source

Hermes Direction Brief:

`Stage2 RVV direct contraction realization-provider fact-contract owner`

## What I Already Know

- Initial `pwd` was `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short` was clean.
- Initial `git log --oneline -8` started at
  `7cee34bf rvv: own standalone reduction provider facts`.
- No `.trellis/.current-task` existed at the start of this round, so this task
  was created from the Hermes brief before source edits.
- Local memory search did not return a direct hit for this direct contraction
  owner seam, so this PRD is derived from live repo state and Trellis specs.
- Specs read:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/index.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  `.trellis/spec/guides/index.md`,
  `.trellis/spec/guides/capability-first-design-guide.md`,
  `.trellis/spec/guides/plugin-locality-review-guide.md`,
  `.trellis/spec/guides/compute-boundary-review-guide.md`.
- Previous archived task
  `.trellis/tasks/archive/2026-06/06-06-stage2-rvv-standalone-reduction-fact-contract/prd.md`
  moved standalone reduction provider-fact verification to an owner-local
  statement-plan surface while preserving route-provider preflight order.
- Current direct contraction code already has an RVV-owned provider/statement
  owner surface:
  `RVVSelectedBodyDirectContractionRouteProviderOwner`,
  `getRVVSelectedBodyDirectContractionRouteProviderOwners(...)`,
  `getRVVSelectedBodyDirectContractionRouteProviderPlan(...)`, and
  `getRVVSelectedBodyDirectContractionRouteStatementPlan(...)`.
- The remaining bounded ownership gap is that
  `verifyRVVSelectedBodyDirectContractionRouteProviderFacts(...)` is still
  declared on the generic `RVVEmitCRoutePlanning` public surface and
  implemented in `RVVEmitCRoutePlanning.cpp`, even though it validates direct
  contraction owner/provider facts immediately before provider route
  construction.

## Requirements

- Expose `verifyRVVSelectedBodyDirectContractionRouteProviderFacts(...)` from
  `RVVEmitCStatementPlanOwners.h`, next to the direct contraction owner
  registry and statement-plan getter.
- Move or otherwise localize the verifier implementation into the direct
  contraction statement/provider owner implementation path, so the generic
  route-planning public surface no longer owns this direct contraction
  provider-fact contract.
- Preserve `RVVEmitCRouteProvider` pre-route call order: after route-family
  provider-plan verification, materialization facts, math operand-binding
  facts, direct contraction provider plan, and statement owner selection are
  available, but before constructing `TCRVEmitCLowerableRoute`.
- Preserve fail-closed validation for operation kind, direct contraction
  sub-family classification, accumulator/input/result roles, widening product
  flags, product-reduction/dequant/clamp flags, computed-mask and strided-input
  flags, memory and scalar-channel roles, mask/control policy provenance,
  SEW/LMUL and typed config mirrors, runtime AVL/VL control, ABI order and
  roles, route-family plan facts, provider plan leaves, and direct owner
  statement leaves.
- Preserve stale-fact rejection for non-contraction routes carrying direct
  contraction provider or owner-selection facts, mismatched family/materialized
  facts, stale route-control facts, missing same-analysis math operand-binding
  facts, wrong ABI roles, and missing provider/statement leaves.
- Keep Common EmitC/export neutral. The implementation must not infer direct
  contraction operation, dtype, ABI, control policy, mask behavior, product or
  reduction relation, intrinsic, or route support from route ids, artifact
  names, metadata mirrors, helper names, source-front-door markers,
  descriptors, or Common EmitC behavior.

## Acceptance Criteria

- [x] Production plugin code changes in the direct contraction
      realization/provider ownership path.
- [x] `RVVEmitCRouteProvider` still consumes the direct contraction
      provider-fact verifier before constructing `TCRVEmitCLowerableRoute`.
- [x] The generic `RVVEmitCRoutePlanning.h` public surface no longer declares
      the direct contraction provider-fact verifier.
- [x] The verifier implementation is localized with the direct contraction
      statement/provider owner implementation path and preserves existing
      fail-closed checks for typed body/config facts, family plan facts,
      control-policy facts, math operand-binding facts, ABI facts,
      materialization leaves, provider-plan leaves, and statement-plan leaves.
- [x] Focused positive C++ coverage still proves direct contraction provider
      plans and direct owner statements feed provider route construction for the
      touched direct-provider family.
- [x] Focused fail-closed C++ coverage still rejects stale or missing direct
      contraction facts such as wrong/missing provider plan, wrong/missing
      statement owner selection, wrong ABI role/order, stale control policy,
      stale family classification, or missing materialized/statement leaves.
- [x] Existing explicit and pre-realized direct contraction regressions for the
      touched family remain passing.
- [x] `build/bin/tianchenrv-rvv-extension-plugin-test` and
      `build/bin/tianchenrv-target-artifact-export-test` pass.
- [x] If emitted route statements, runtime ABI order, header export, generated
      bundle output, or runtime harness behavior changes, run one focused
      generated-bundle dry-run or `ssh rvv` smoke. If this remains a pre-route
      owner/API tightening only, record why runtime evidence is not required.
- [x] Bounded old-authority scan over touched RVV owner/provider/test files and
      added diff lines shows no positive legacy `i32m1`, descriptor,
      source-front-door, direct-C/source-export, route-id, helper-name, or
      mirror-only route authority drift.
- [x] `git diff --check`, `git diff --cached --check`, Trellis validation, and
      clean final git status pass.

## Definition Of Done

- Compiler behavior remains in C++/MLIR/LLVM/TableGen/CMake/lit/FileCheck.
- Python remains limited to tooling and is not used for compiler core behavior.
- Direct contraction semantics stay in RVV plugin realization, route-family,
  provider-plan, statement-plan, and route-provider owner seams; Common EmitC
  only consumes provider-built routes and payloads.
- No new contraction route family, dtype/LMUL clone batch, high-level
  Linalg/Vector/StableHLO frontend, per-Linalg route authority, generated
  bundle matrix, source-front-door positive route, direct pre-realized
  route-entry shortcut, performance/autotuning, or mass rewrite of unrelated
  RVV owner seams is introduced.
- Task status/context stay truthful.
- The task is finished/archived and one coherent commit is created when
  acceptance criteria are met.

## Technical Approach

Align direct contraction with the owner-local provider-fact pattern completed
for standalone reduction:

```text
realized typed direct-contraction tcrv_rvv body
  -> contraction route-family materialization facts
  -> RVV-owned math operand-binding facts
  -> direct contraction provider plan
  -> direct contraction statement/provider owner
  -> direct contraction owner-local provider-fact contract
  -> TCRVEmitCLowerableRoute
```

The narrow implementation target is to move the public provider-facing
contract declaration for
`verifyRVVSelectedBodyDirectContractionRouteProviderFacts(...)` onto
`RVVEmitCStatementPlanOwners.h`, relocate the implementation to the direct
contraction statement-plan owner implementation path, and keep the production
provider/tests calling the owner-level surface.

## Out Of Scope

- New contraction route families or intrinsic mappings.
- Dtype/LMUL clone batches.
- High-level Linalg/Vector/StableHLO frontend work.
- Per-Linalg route authority.
- Common EmitC inference of direct contraction, dtype, ABI, intrinsic,
  mask/tail behavior, product/reduction/dequant flags, or scalar/result
  semantics.
- Direct pre-realized route-entry shortcuts.
- Source-front-door positive routes.
- Performance/autotuning or global profile systems.
- Broad generated-bundle matrices.
- Mass rewrite of standalone reduction, conversion/dequantization,
  compare/select, memory, segment2, elementwise, runtime-scalar splat-store, or
  MAcc owners outside the touched seam.

## Technical Notes

- Relevant production files:
  `include/TianChenRV/Plugin/RVV/RVVEmitCStatementPlanOwners.h`,
  `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`,
  `lib/Plugin/RVV/EmitC/RVVEmitCStatementPlanOwners.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`.
- Relevant direct contraction realization/family files:
  `include/TianChenRV/Plugin/RVV/RVVContractionSelectedBodyRealizationOwner.h`,
  `lib/Plugin/RVV/RVVContractionSelectedBodyRealizationOwner.cpp`,
  `include/TianChenRV/Plugin/RVV/RVVEmitCContractionRouteFamilyPlanOwners.h`,
  `lib/Plugin/RVV/EmitC/RVVEmitCContractionRouteFamilyPlanOwners.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCControlPolicyPlanOwners.cpp`.
- Relevant tests:
  `test/Plugin/RVVExtensionPluginTest.cpp`,
  `test/Target/TargetArtifactExportTest.cpp`,
  focused explicit and pre-realized direct contraction fixtures under
  `test/Target/RVV/`.
- The expected runtime boundary is no `ssh rvv` evidence if this round only
  moves pre-route verifier ownership and does not change emitted statements,
  ABI order, headers, generated bundle content, or runtime harness behavior.

## Completion Evidence

- Production owner seam changed:
  `verifyRVVSelectedBodyDirectContractionRouteProviderFacts(...)` is now
  declared in `RVVEmitCStatementPlanOwners.h`, implemented in
  `RVVEmitCStatementPlanOwners.cpp`, and removed from the generic
  `RVVEmitCRoutePlanning.h` / `RVVEmitCRoutePlanning.cpp` surface.
- `RVVEmitCRouteProvider` still calls the verifier before
  `TCRVEmitCLowerableRoute` construction, after route-family provider plans,
  materialization facts, math operand-binding facts, direct contraction
  provider plan, and direct statement owner selection are available.
- Focused direct contraction C++ coverage still exercises the positive
  same-analysis provider-plan/statement-owner path and fail-closed stale
  route-control and missing statement-owner-selection cases.
- The change does not alter emitted statements, runtime ABI order, header
  export, generated bundle content, or runtime harness behavior; no `ssh rvv`
  runtime smoke is required for this pre-route owner/API tightening.
- No `.trellis/spec/` update was needed because the existing direct contraction
  owner boundary spec already covers the signature class, contract, validation
  matrix, good/base/bad cases, and tests. This round only relocated the
  implementation to match that existing spec.
- Focused direct contraction MLIR fixture smoke passed with `tcrv-opt` for
  pre-realized widening MAcc, widening dot-reduce, computed-mask widening
  dot-reduce, and computed-mask strided widening dot-reduce selected bodies.
- Focused binaries passed:
  `build/bin/tianchenrv-rvv-extension-plugin-test` and
  `build/bin/tianchenrv-target-artifact-export-test`.
- Diff hygiene passed:
  `git diff --check`, Trellis context validation, and a bounded added-diff
  old-authority scan over touched C++/header files.
