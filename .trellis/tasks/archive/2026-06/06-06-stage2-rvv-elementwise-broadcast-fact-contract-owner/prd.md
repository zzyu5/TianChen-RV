# Stage2 RVV elementwise/broadcast realization-provider fact-contract owner

## Goal

Strengthen one production RVV plugin seam between elementwise/broadcast
selected-body realization, route-family planning, migrated statement ownership,
and `TCRVEmitCLowerableRoute` construction. The bounded owner for this round is
elementwise arithmetic plus scalar-broadcast elementwise: add/sub/mul, masked
add/sub/mul, strided add, and scalar-broadcast add/sub/mul must carry realized
typed-body facts into a provider facts verifier before Common EmitC receives a
route.

## Direction Source

Hermes Direction Brief:

`Stage2 RVV elementwise/broadcast realization-provider fact-contract owner`

## What I Already Know

- Initial `pwd` was `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short` was clean.
- Initial `git log --oneline -8` started at
  `a60141de rvv: verify base memory provider facts`.
- No `.trellis/.current-task` existed at the start of this round, so this task
  was created from the Hermes brief before source edits.
- The previous archived base-memory fact-contract task added
  `verifyRVVSelectedBodyBaseMemoryMovementRouteProviderFacts` and wired it in
  `RVVEmitCRouteProvider` after statement owner selection and before route
  construction.
- Elementwise arithmetic already has route-family plans, scalar-broadcast
  plans, materialization facts, operand-binding facts, route-control provider
  plans, and migrated statement-plan ownership.
- The current provider still has a local
  `verifyElementwiseArithmeticMaterializationFactsBeforeRouteBuild` check in
  `RVVEmitCRouteProvider.cpp`. It validates materialization/description mirror
  facts but does not form the full owner-level fact contract over
  elementwise/select operand bindings, scalar broadcast source, mask
  provenance, route-control plan, and migrated statement owner selection.

## Requirements

- Add one RVV-plugin-local elementwise/broadcast provider-facts verifier that is
  consumed by `RVVEmitCRouteProvider` before constructing
  `TCRVEmitCLowerableRoute`.
- The verifier must consume existing realized typed-body route analysis,
  materialization facts, elementwise/select operand-binding facts, residual
  operand-binding facts for masked/strided elementwise forms, the route-control
  provider plan where applicable, and selected migrated statement owner
  selection.
- The verifier must cover plain add/sub/mul, masked add/sub/mul, strided add,
  and scalar-broadcast add/sub/mul through the same production seam.
- The verifier must return success for non-elementwise/broadcast routes only
  when they do not carry stale elementwise or scalar-broadcast provider facts.
- For elementwise/broadcast consumers, the verifier must fail closed if:
  elementwise or scalar-broadcast route-family plans are missing, stale, or from
  a different selected route analysis;
  typed config facts, element type/SEW/LMUL, VL/policy, header/type facts,
  runtime ABI order, or route description mirrors disagree with the validated
  plan;
  operand-binding facts do not bind the required vector/scalar/runtime roles;
  scalar-broadcast routes lack the `rhs_scalar` ABI role or provider-derived
  scalar splat leaf;
  masked routes lack provider-derived mask type, compare, merge,
  passthrough/inactive-lane, mask role/source, or mask provenance facts;
  strided elementwise routes lack stride ABI facts or strided load/store leaves;
  the route-control provider plan is stale or does not select the relevant
  elementwise/broadcast class;
  migrated statement owner selection does not select the elementwise arithmetic
  owner for an active elementwise/broadcast route;
  required statement leaves are absent before route construction.
- Keep Common EmitC neutral. The new verifier must not infer arithmetic or
  broadcast semantics from route ids, artifact names, helper names, ABI strings,
  test names, descriptors, source-front-door markers, or metadata mirrors.

## Acceptance Criteria

- [x] Production plugin code changes in the elementwise/broadcast
      realization/provider ownership path.
- [x] `RVVEmitCRouteProvider` calls the elementwise/broadcast provider-facts
      verifier after same-analysis facts and migrated statement owner selection
      are available, and before `TCRVEmitCLowerableRoute` construction.
- [x] Focused positive C++ coverage proves at least one plain elementwise, one
      masked elementwise, and one scalar-broadcast elementwise route reach route
      construction through typed body facts, operand-binding facts, statement
      owner selection, and the new verifier.
- [x] Focused fail-closed C++ coverage rejects stale or missing facts such as a
      stale route mirror, missing scalar broadcast source, stale mask
      provenance, wrong SEW/LMUL or policy, missing operand binding, or wrong
      migrated statement owner.
- [x] Existing explicit/pre-realized elementwise/broadcast regressions for the
      touched family remain passing.
- [x] `build/bin/tianchenrv-rvv-extension-plugin-test` and
      `build/bin/tianchenrv-target-artifact-export-test` pass.
- [x] If emitted route statements, ABI order, header export, or generated
      bundle behavior changes, run one focused generated-bundle dry-run or
      `ssh rvv` smoke for the affected route. If no emission behavior changes,
      record why runtime evidence was not required.
- [x] Bounded old-authority scan over touched RVV planning/provider/test files
      and added diff lines shows no positive legacy `i32m1`, descriptor,
      source-front-door, direct-C/source-export, route-id, helper-name, or
      mirror-only route authority drift.
- [x] `git diff --check`, `git diff --cached --check`, Trellis context
      validation, and clean final git status pass.

## Definition Of Done

- Compiler behavior remains in C++/MLIR/LLVM/TableGen/CMake/lit/FileCheck.
- Python remains limited to tooling and is not used for compiler core behavior.
- RVV semantics stay in the RVV plugin/provider; Common EmitC only consumes the
  provider-built route.
- No new arithmetic op family, dtype/LMUL clone batch, broad generated-bundle
  matrix, high-level Linalg/Vector/StableHLO frontend, direct pre-realized
  route-entry shortcut, source-front-door positive route, performance/autotuning,
  or report-only closure is introduced.
- Task status/context stay truthful.
- The task is finished/archived and one coherent commit is created when
  acceptance criteria are met.

## Technical Approach

Implement a narrow active owner improvement:

```text
realized typed tcrv_rvv elementwise/broadcast body
  -> elementwise or scalar-broadcast route-family/materialization facts
  -> elementwise/select operand-binding facts
  -> route-control provider plan
  -> migrated elementwise statement owner selection
  -> elementwise/broadcast provider-facts verifier
  -> TCRVEmitCLowerableRoute
```

The verifier will live in the RVV elementwise route-family owner layer and be
declared through the elementwise planning header so the central provider can
call it. It will reuse
`RVVSelectedBodyElementwiseArithmeticRouteFamilyPlan`,
`RVVSelectedBodyScalarBroadcastElementwiseRouteFamilyPlan`,
  `RVVSelectedBodyElementwiseSelectRouteOperandBindingFacts`,
  `RVVSelectedBodyResidualRouteOperandBindingFacts`, and
`RVVSelectedBodyRouteStatementPlanOwnerSelection`; it will not introduce a new
route family or reassemble statements from route ids or metadata mirrors.

## Decision (ADR-lite)

Context: The elementwise/broadcast route-family code validates route-family
plans and has statement-plan coverage, but the central provider still owns a
local materialization-only pre-route check. The Hermes brief asks for the same
kind of production fact contract that base-memory now has, but for the
elementwise/broadcast selected-body to provider seam.

Decision: Move/factor that pre-route check into an RVV elementwise owner
provider-facts verifier and expand it to validate operand binding, scalar
broadcast, mask, route-control, and migrated statement-owner facts. Wire the
verifier into production after statement owner selection and before route
construction.

Consequences: The elementwise/broadcast route seam becomes explicit and
testable against stale metadata-derived or fixture-derived facts. The change
should not alter emitted code unless it catches a previously accepted stale
route.

## Out Of Scope

- New arithmetic operation family or route coverage expansion.
- Dtype/LMUL clone batch.
- Broad generated-bundle matrix.
- High-level Linalg/Vector/StableHLO frontend work.
- Per-Linalg route authority.
- Common EmitC inference of arithmetic or broadcast semantics.
- Direct pre-realized route-entry shortcut.
- Source-front-door positive route.
- Performance/autotuning or global profile systems.
- Mass rewrite of memory, contraction, reduction, control-policy, target
  export, or unrelated route owners.

## Technical Notes

- Specs read:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  `.trellis/spec/testing/mlir-testing-contract.md`.
- Previous task read:
  `.trellis/tasks/archive/2026-06/06-06-06-06-stage2-rvv-base-memory-fact-contract-owner/prd.md`.
- Production files inspected:
  `include/TianChenRV/Plugin/RVV/RVVEmitCElementwiseRouteFamilyPlanOwners.h`,
  `include/TianChenRV/Plugin/RVV/RVVElementwiseSelectedBodyRealizationOwner.h`,
  `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`,
  `lib/Plugin/RVV/RVVElementwiseSelectedBodyRealizationOwner.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCElementwiseRouteFamilyPlanOwners.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCStatementPlanOwners.cpp`,
  and `test/Plugin/RVVExtensionPluginTest.cpp`.

## Completion Evidence

- Production seam changed:
  `verifyRVVSelectedBodyElementwiseBroadcastRouteProviderFacts` now lives in
  the RVV elementwise route-family owner layer and is called by
  `RVVEmitCRouteProvider` after route/materialization facts, operand-binding
  facts, residual facts, direct-contraction plan construction, and migrated
  statement owner selection are available.
- Spec updated:
  `.trellis/spec/extension-plugins/rvv-plugin.md` now records the executable
  elementwise/broadcast route provider fact contract, including inputs,
  validation matrix, good/base/bad cases, tests, and wrong-vs-correct route
  authority boundaries.
- The new verifier accepts plain elementwise, masked elementwise, and
  scalar-broadcast elementwise typed-body/provider facts before route
  construction, and rejects stale ordinary operand binding, missing migrated
  elementwise owner statements, wrong SEW/LMUL materialization config, missing
  masked compare provenance, and missing scalar-broadcast splat facts.
- Emitted route statements, ABI order, header export, generated bundles, and
  runtime harness behavior were not changed; the production change is a
  pre-route fail-closed verifier plus focused C++ coverage. Therefore no
  generated-bundle dry-run or `ssh rvv` runtime evidence was required or
  claimed in this round.
- Checks passed:
  `cmake --build build --target tianchenrv-rvv-extension-plugin-test`;
  `build/bin/tianchenrv-rvv-extension-plugin-test`;
  `cmake --build build --target tianchenrv-target-artifact-export-test`;
  `build/bin/tianchenrv-target-artifact-export-test`;
  `python3 ./.trellis/scripts/task.py validate .trellis/tasks/06-06-stage2-rvv-elementwise-broadcast-fact-contract-owner`;
  `git diff --check`.
- `clang-format`, `clang-format-18`, and `clang-format-17` were not available
  in `PATH`; formatting was kept consistent manually.
- Bounded added-diff old-authority scan over the touched files returned no
  positive legacy authority matches for `RVVI32M1`, `rvv-i32m1`,
  `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`, source-front-door, descriptor,
  direct-C/source-export, route-id, helper-name, or mirror-only route authority
  drift.
- The spec update mentions route ids, helper names, descriptors, metadata, and
  `add_i32m1` only in explicit forbidden/wrong-example contexts.
