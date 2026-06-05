# Stage2 RVV contraction realization/provider fact-contract owner

## Goal

Strengthen the active RVV plugin production contract between contraction
selected-body realization, provider planning, direct contraction statement-owner
selection, and `TCRVEmitCLowerableRoute` construction. The bounded owner for
this round is the direct contraction provider-facts seam: widening-MAcc,
widening-dot, computed-mask widening-dot, and strided-input widening-dot routes
must prove their realized typed-body, materialization, route-control,
math-operand-binding, mask, stride, accumulator/result, and runtime ABI facts
through one production verifier before common EmitC materialization receives a
route.

## Direction Source

Hermes Direction Brief:

`Stage2 RVV contraction realization/provider fact-contract owner`

## What I Already Know

- Initial `pwd` was `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short` was clean.
- Initial `git log --oneline -8` started at
  `7321ea5a rvv: close computed masked strided widening dot closure`.
- No `.trellis/.current-task` existed at the start of this round, so this task
  was created from the Hermes brief and activated before source edits.
- The previous archived computed-mask strided widening-dot task proved the
  executable selected-body path and `ssh rvv` evidence, but changed only a
  focused target fixture guard rather than the production owner seam.
- The repo already contains a direct contraction provider plan and statement
  owner:
  `RVVSelectedBodyDirectContractionRouteProviderPlan`,
  `getRVVSelectedBodyDirectContractionRouteProviderPlan`, and
  `getRVVSelectedBodyDirectContractionRouteStatementPlan`.
- Unlike standalone reduction, memory, segment2, and several migrated routes,
  the direct contraction path currently lacks a single production
  `verify...RouteProviderFacts` gate after statement-owner selection and before
  `TCRVEmitCLowerableRoute` construction.

## Requirements

- Add one production direct contraction provider-facts verifier that is consumed
  by `RVVEmitCRouteProvider` before constructing `TCRVEmitCLowerableRoute`.
- The verifier must be RVV-plugin-local and consume existing typed body/config,
  materialization, route-control provider plan, direct contraction provider
  plan, math operand-binding facts, and statement-owner selection.
- The verifier must return empty/success for non-direct-contraction routes
  without changing unrelated route families.
- For direct contraction consumers, the verifier must fail closed if:
  direct provider plan is absent or stale;
  contraction materialization facts do not point at the verified same-analysis
  contraction family plan;
  route-control provider plan does not bind the same typed config, selected
  capability, runtime-control, policy, runtime ABI, and provider mirrors;
  math operand-binding facts are missing or do not match the direct contraction
  sub-family;
  required ABI roles for lhs/rhs or compare/dot lhs/rhs, accumulator/output,
  runtime `n`, and optional strides are absent or wrong;
  required materialized leaves for setvl/load/strided-load/compare/mask/product/
  reduction/store are absent or missing from the selected owner statements.
- The production provider must not rebuild contraction semantics in common
  EmitC or use route ids, artifact names, descriptor/source-front-door markers,
  ABI strings, or fixture names as route authority.

## Acceptance Criteria

- [ ] Production plugin code changes in the direct contraction
      realization/provider ownership path.
- [ ] `RVVEmitCRouteProvider` calls the direct contraction provider-facts
      verifier before `TCRVEmitCLowerableRoute` construction.
- [ ] Focused C++ positive coverage proves the computed-mask strided
      widening-dot provider plan reaches route construction through the direct
      contraction owner and the verifier.
- [ ] Focused C++ fail-closed coverage rejects at least one stale/missing direct
      provider fact that could otherwise be represented by metadata/mirror data
      rather than realized typed-body/provider facts.
- [ ] Existing explicit and pre-realized direct contraction regressions for the
      touched family remain passing.
- [ ] `build/bin/tianchenrv-rvv-extension-plugin-test` and
      `build/bin/tianchenrv-target-artifact-export-test` pass.
- [ ] A bounded old-authority scan over touched RVV planning/provider/test files
      and added diff lines shows no positive legacy `i32m1`, descriptor,
      source-front-door, direct-C/source-export, or mirror-only route authority
      drift.
- [ ] `git diff --check`, `git diff --cached --check`, Trellis context
      validation, and clean final git status pass.

## Definition Of Done

- Compiler behavior remains in C++/MLIR/LLVM/TableGen/CMake/lit/FileCheck.
- Python remains limited to tooling and is not used for compiler core behavior.
- Common EmitC remains neutral; RVV semantics stay in the RVV plugin/provider.
- No new contraction op family, dtype/LMUL clone batch, broad smoke matrix,
  high-level frontend, direct pre-realized route-entry shortcut, source-front
  positive route, performance/autotuning, or report-only closure is introduced.
- Task status/context stay truthful.
- The task is finished/archived and one coherent commit is created when
  acceptance criteria are met.

## Technical Approach

Implement a small active owner improvement:

```text
realized typed tcrv_rvv contraction body
  -> contraction family/materialization facts
  -> math operand-binding facts
  -> route-control provider plan
  -> direct contraction provider plan
  -> direct contraction statement-owner selection
  -> direct contraction provider-facts verifier
  -> TCRVEmitCLowerableRoute
```

The verifier will live in the RVV plugin planning/provider layer, not common
EmitC. It will reuse the existing direct contraction provider plan and statement
owner output instead of introducing a new route family or reassembling
statements from operation names.

## Decision (ADR-lite)

Context: Existing code already has a direct contraction provider plan and
statement owner, but route construction lacks the same final provider-facts
gate that other mature migrated route families use. The Hermes brief asks for a
production fact contract, not another fixture-only evidence closure.

Decision: Add a direct contraction provider-facts verifier and call it in the
production provider before route construction. Keep the change narrow and reuse
existing plans/facts.

Consequences: The direct contraction seam becomes explicit and easier to test
against stale mask/stride/accumulator/runtime/provider facts, while leaving
route emission output unchanged unless the new guard catches stale facts.

## Out Of Scope

- New contraction op family or dtype/LMUL clone batch.
- Additional runtime or hardware evidence unless route emission changes.
- Broad contraction smoke matrix.
- High-level Linalg/Vector/StableHLO frontend work.
- Per-Linalg route authority or one-op-per-intrinsic wrapping.
- Common EmitC inference of contraction semantics.
- Direct pre-realized route-entry shortcut.
- Source-front-door positive route.
- Performance/autotuning or global profile systems.

## Technical Notes

- Specs read:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  and `.trellis/spec/testing/index.md`.
- Previous task read:
  `.trellis/tasks/archive/2026-06/06-06-stage2-rvv-computed-masked-strided-input-widening-dot-closure/prd.md`,
  `implement.jsonl`, and `check.jsonl`.
- Production files inspected:
  `lib/Plugin/RVV/RVVContractionSelectedBodyRealizationOwner.cpp`,
  `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCContractionRouteFamilyPlanOwners.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCMAccRouteFamilyPlanOwners.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCStatementPlanOwners.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`,
  `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`,
  `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`, and
  `include/TianChenRV/Plugin/RVV/RVVEmitCStatementPlanOwners.h`.

## Completion Evidence

- Added the production RVV plugin verifier:
  `verifyRVVSelectedBodyDirectContractionRouteProviderFacts`.
- `RVVEmitCRouteProvider` now calls this verifier after direct contraction
  provider-plan construction and statement-owner selection, but before creating
  `TCRVEmitCLowerableRoute`.
- The verifier succeeds only when direct contraction routes carry same-analysis
  contraction family/materialization facts, route-control provider facts,
  typed config and selected capability facts, exact math operand-binding facts,
  expected runtime ABI roles, and direct contraction owner statement leaves.
- The verifier returns success for non-direct-contraction routes only when they
  do not carry stale direct-contraction provider facts.
- Focused C++ positive coverage added in
  `runContractionTargetLeafProfileValidationTest`: the computed-mask strided
  widening-dot provider plan and direct contraction owner selection pass the new
  provider-facts verifier.
- Focused C++ fail-closed coverage added for stale route-control typed-config
  facts and missing direct contraction owner loop statements.
- Self-repair: the first verifier implementation was too strict for legal
  computed-mask strided routes. It incorrectly required unit `sourceLoadLeaf`
  and plain `wideningProductLeaf` statements where the realized route uses
  `stridedSourceLoadLeaf` and `maskedWideningProductLeaf`. The guard was
  narrowed to match the active typed-body facts.
- Rebuilt targets:
  `cmake --build build --target tianchenrv-rvv-extension-plugin-test` and
  `cmake --build build --target tianchenrv-target-artifact-export-test`.
- `build/bin/tianchenrv-rvv-extension-plugin-test` passed.
- `build/bin/tianchenrv-target-artifact-export-test` passed.
- Positive target fixture pipelines exited 0:
  `build/bin/tcrv-opt test/Target/RVV/pre-realized-selected-body-artifact-computed-masked-strided-input-widening-dot-reduce-add.mlir --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans`,
  the corresponding `tcrv-translate --tcrv-export-target-header-artifact`
  pipeline,
  `build/bin/tcrv-opt test/Target/RVV/explicit-selected-body-artifact-computed-masked-strided-input-widening-dot-reduce-add.mlir --tcrv-materialize-emission-plans`,
  and the corresponding header export pipeline.
- No new `ssh rvv` run was needed because this round did not change emitted
  route statements, ABI order, headers, or generated bundle code; it added a
  production fail-closed verifier before route construction.
- Bounded added-line authority scan over touched RVV planning/provider/test
  files found no new positive legacy `i32m1`, descriptor, source-front-door,
  direct-C, source-export, route-id, or mirror-only support authority strings.
- `git diff --check` and `git diff --cached --check` passed.
- Trellis context validation passed with 11 implement entries and 10 check
  entries.

## Spec Update Judgment

- No `.trellis/spec/` update was needed. The task implemented the existing
  Direct Contraction Route-Provider Owner Boundary from
  `.trellis/spec/extension-plugins/rvv-plugin.md`; it did not introduce a new
  route family, new selected-body owner API, new common EmitC contract, or new
  runtime evidence policy.
