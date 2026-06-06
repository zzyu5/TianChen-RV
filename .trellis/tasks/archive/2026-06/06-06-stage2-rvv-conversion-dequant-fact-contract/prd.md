# Stage2 RVV conversion/dequantization realization-provider fact contract

## Goal

Strengthen the production RVV plugin seam for widening conversion and
standalone unit-stride dequantization routes. This round is bounded to moving
the provider-fact contract for the existing conversion/dequantization statement
plans onto the RVV statement-plan owner surface, while preserving the central
provider call before `TCRVEmitCLowerableRoute` construction.

## Direction Source

Hermes Direction Brief:

`Stage2 RVV conversion/dequantization realization-provider fact-contract owner`

## What I Already Know

- Initial `pwd` was `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short` was clean.
- Initial `git log --oneline -8` started at
  `d5998566 rvv: own compare-select provider facts`.
- No `.trellis/.current-task` existed at the start of this round, so this task
  was created from the Hermes brief before source edits.
- Specs read:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  `.trellis/spec/testing/index.md`.
- Previous archived task
  `.trellis/tasks/archive/2026-06/06-06-stage2-rvv-compare-select-fact-contract/prd.md`
  moved compare/select provider-fact verification to the compare/select
  statement-plan owner surface while keeping route-provider preflight order.
- Current conversion/dequant code already has owner-built statement plans:
  `getRVVSelectedBodyWideningConversionRouteStatementPlan(...)` and
  `getRVVSelectedBodyDequantizationRouteStatementPlan(...)`.
- The remaining ownership gap is that
  `verifyRVVSelectedBodyWideningConversionRouteProviderFacts(...)` and
  `verifyRVVSelectedBodyDequantizationRouteProviderFacts(...)` are still
  declared on the generic `RVVEmitCRoutePlanning` surface and implemented in
  `RVVEmitCRoutePlanning.cpp`, even though they validate statement-plan owner
  facts before provider route construction.

## Requirements

- Expose the widening conversion provider-fact verifier from
  `RVVEmitCStatementPlanOwners.h`, next to the widening conversion
  statement-plan builder.
- Expose the standalone dequantization provider-fact verifier from
  `RVVEmitCStatementPlanOwners.h`, next to the dequantization statement-plan
  builder.
- Move or otherwise localize the verifier implementation to the RVV
  statement-plan owner implementation path, so the generic route-planning
  public surface no longer owns these conversion/dequant provider-fact
  contracts.
- Preserve `RVVEmitCRouteProvider` call order: after route-family provider
  plans, materialization facts, math operand-binding facts, and the
  corresponding statement plan are available, but before
  `TCRVEmitCLowerableRoute` construction.
- Preserve fail-closed validation for source/result element facts, SEW/LMUL,
  conversion kind/relation, dequantization kind/relation, scale role/type/name,
  memory form, VL/policy, runtime ABI order, route-control facts, statement
  leaves, and stale non-family facts.
- Keep Common EmitC neutral. The implementation must not infer conversion,
  dequantization, dtype, ABI, intrinsic, or route support from route ids,
  artifact names, metadata mirrors, helper names, source-front-door markers,
  descriptors, or Common EmitC behavior.

## Acceptance Criteria

- [x] Production plugin code changes in the conversion/dequantization
      statement-plan owner/provider ownership path.
- [x] `RVVEmitCRouteProvider` still consumes widening conversion and
      dequantization provider-fact verifiers before constructing
      `TCRVEmitCLowerableRoute`.
- [x] The generic `RVVEmitCRoutePlanning.h` public surface no longer declares
      the widening conversion or standalone dequantization provider-fact
      verifiers.
- [x] Focused positive C++ coverage still proves `widen_i32_to_i64`,
      `widen_i16_to_i32`, and `dequantize_i32_to_f32` consume typed body facts,
      route-family plans, materialization facts, math operand-binding facts,
      route-control facts, and statement-plan leaves before route
      construction.
- [x] Focused fail-closed C++ coverage still rejects stale or missing facts
      such as wrong source/result type, wrong SEW/LMUL, route-id/metadata
      selected conversion kind, stale dequant scale role/type/name, wrong ABI
      role/order, wrong memory form, stale route-control policy, and stale
      statement leaves.
- [x] Existing explicit and pre-realized widening-conversion and
      dequantization regressions for the touched family remain passing.
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
- RVV conversion/dequantization semantics stay in RVV plugin
  realization/statement-plan/provider owners; Common EmitC only consumes
  provider-built routes and payloads.
- No new dtype/LMUL clone batch, high-level dequant frontend, gearbox or
  autotuning expansion, generated-bundle matrix, high-level
  Linalg/Vector/StableHLO frontend, per-Linalg route authority,
  source-front-door positive route, direct pre-realized route shortcut, or
  report-only closure is introduced.
- Task status/context stay truthful.
- The task is finished/archived and one coherent commit is created when
  acceptance criteria are met.

## Technical Approach

Align conversion/dequantization with the compare/select owner pattern completed
in the previous task:

```text
realized typed conversion/dequantization tcrv_rvv body
  -> conversion/dequantization route-family materialization facts
  -> RVV-owned math operand-binding facts
  -> conversion/dequantization statement-plan owner
  -> conversion/dequantization owner provider-fact contract
  -> TCRVEmitCLowerableRoute
```

The narrow implementation target is to move the public provider-facing
contract declarations for
`verifyRVVSelectedBodyWideningConversionRouteProviderFacts(...)` and
`verifyRVVSelectedBodyDequantizationRouteProviderFacts(...)` onto
`RVVEmitCStatementPlanOwners.h`, relocate the implementations to
`RVVEmitCResidualStatementPlanOwners.cpp` or the nearby statement-plan owner
implementation path, and keep the production provider/tests calling the
owner-level surface.

## Out Of Scope

- New conversion kinds or dequantization route families.
- Dtype/LMUL clone batches.
- High-level dequant frontend work.
- Gearbox/autotuning expansion beyond existing facts.
- Broad generated-bundle matrices.
- High-level Linalg/Vector/StableHLO frontend work.
- Per-Linalg route authority.
- Common EmitC inference of conversion, dequantization, dtype, ABI, or
  intrinsic semantics.
- Direct pre-realized route-entry shortcuts.
- Source-front-door positive routes.
- Performance/autotuning or global profile systems.
- Mass rewrite of compare/select, memory, segment2, elementwise, contraction,
  standalone reduction, or runtime-scalar owners outside the touched seam.

## Technical Notes

- Relevant production files:
  `include/TianChenRV/Plugin/RVV/RVVEmitCStatementPlanOwners.h`,
  `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`,
  `lib/Plugin/RVV/EmitC/RVVEmitCResidualStatementPlanOwners.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`.
- Relevant tests:
  `test/Plugin/RVVExtensionPluginTest.cpp`,
  `test/Target/TargetArtifactExportTest.cpp`,
  focused pre-realized widening-conversion and dequantization MLIR fixtures.
- The expected runtime boundary is no `ssh rvv` evidence if this round only
  moves pre-route verifier ownership and does not change emitted statements,
  ABI order, headers, bundle content, or runtime harness behavior.

## Completion Evidence

- Production seam changed:
  `verifyRVVSelectedBodyWideningConversionRouteProviderFacts(...)` and
  `verifyRVVSelectedBodyDequantizationRouteProviderFacts(...)` are now declared
  by `RVVEmitCStatementPlanOwners.h` and implemented in
  `RVVEmitCResidualStatementPlanOwners.cpp`, colocated with the widening
  conversion/dequantization statement-plan owner path.
- The generic `RVVEmitCRoutePlanning.h` declarations and
  `RVVEmitCRoutePlanning.cpp` implementations for those two provider-fact
  verifiers were removed.
- `RVVEmitCRouteProvider.cpp` continues to call both verifiers after obtaining
  materialization facts, math operand-binding facts, and the corresponding
  statement plan, and before constructing `TCRVEmitCLowerableRoute`.
- The moved verifier implementations preserve the same route-family,
  materialization, binding, route-control, description mirror, runtime ABI,
  statement-leaf, and stale-family fail-closed checks. Their family-plan
  validation is covered through the already-public
  `verifyRVVSelectedBodyRouteFamilyProviderPlans(...)` call rather than
  exposing private route-planning validate helpers across translation units.
- Spec updated:
  `.trellis/spec/extension-plugins/rvv-plugin.md` now records the widening
  conversion verifier as a statement-plan owner API and adds a standalone
  dequantization provider-fact preflight contract with scope, signature,
  contracts, error matrix, cases, required tests, and wrong/correct examples.
- Emitted route statements, runtime ABI order, header export, generated bundle
  output, and runtime harness behavior were not changed. This round only moves
  pre-route owner/provider API and implementation ownership, so no
  generated-bundle dry-run or `ssh rvv` runtime evidence was required or
  claimed.
- Checks passed:
  `cmake --build build --target tianchenrv-rvv-extension-plugin-test`;
  `build/bin/tianchenrv-rvv-extension-plugin-test`;
  `cmake --build build --target tianchenrv-target-artifact-export-test`;
  `build/bin/tianchenrv-target-artifact-export-test`;
  manual PLAN and HEADER FileCheck for
  `test/Target/RVV/explicit-selected-body-artifact-widen-i32-to-i64.mlir`;
  manual REALIZED, PLAN, and HEADER FileCheck for
  `test/Target/RVV/pre-realized-selected-body-artifact-widen-i16-to-i32.mlir`;
  manual PLAN and HEADER FileCheck plus STALE-RELATION negative FileCheck for
  `test/Target/RVV/explicit-selected-body-artifact-dequantize-i32-to-f32.mlir`;
  `python3 ./.trellis/scripts/task.py validate .trellis/tasks/06-06-stage2-rvv-conversion-dequant-fact-contract`;
  `git diff --check`.
- Self-repair performed:
  the first manual stale dequantization FileCheck used lit's `not` command,
  which is not on this host PATH. The negative check was rerun with an
  equivalent shell status check and passed. A first spec update also matched
  the compare/select "durable provider-side contract" phrase instead of the
  widening conversion phrase; that was corrected so compare/select remains
  documented as a compare/select owner implementation while conversion and
  dequantization point at the residual statement-plan owner path.
- Bounded old-authority scan over touched C++ files found no newly added
  positive legacy authority. Existing `RVVEmitCRoutePlanning.cpp` matches are
  pre-existing legacy rejection or route-id mirror diagnostics outside this
  moved verifier diff. Added-line scan only found spec text describing forbidden
  or bad cases for route ids, artifact names, descriptors, source-front-door,
  and exact-intrinsic authority.
