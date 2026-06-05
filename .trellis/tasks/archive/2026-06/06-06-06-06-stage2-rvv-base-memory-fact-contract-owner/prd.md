# Stage2 RVV base-memory realization/provider fact-contract owner

## Goal

Strengthen one production RVV plugin seam between selected-body memory
realization, base-memory route-family planning, migrated statement ownership,
and `TCRVEmitCLowerableRoute` construction. The bounded owner for this round is
base memory movement: strided-load/unit-store, unit-load/strided-store, indexed
gather/unit-store, indexed scatter/unit-load, masked unit-load/store, and masked
unit-store routes must carry realized typed-body facts into a provider facts
verifier before common EmitC receives a route.

## Direction Source

Hermes Direction Brief:

`Stage2 RVV base-memory realization/provider fact-contract owner`

## What I Already Know

- Initial `pwd` was `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short` was clean.
- Initial `git log --oneline -8` started at
  `8e95a97a rvv: verify direct contraction provider facts`.
- No `.trellis/.current-task` existed at the start of this round, so this task
  was created from the Hermes brief before source edits.
- The previous archived contraction fact-contract task added
  `verifyRVVSelectedBodyDirectContractionRouteProviderFacts` and calls it in
  `RVVEmitCRouteProvider` before route construction.
- Base memory movement already has a route-family plan, canonical route facts,
  operand-binding derivation, a provider plan, migrated statement-plan owner
  participation, target metadata mirror contracts, and pre-realized selected
  body validators.
- The remaining gap is a single production verifier that consumes the
  base-memory provider plan plus statement owner selection after realized typed
  body analysis and before `TCRVEmitCLowerableRoute` construction.

## Requirements

- Add one RVV-plugin-local base-memory provider-facts verifier that is consumed
  by `RVVEmitCRouteProvider` before constructing `TCRVEmitCLowerableRoute`.
- The verifier must consume existing realized typed-body route analysis,
  materialization facts, memory operand-binding facts, the base-memory provider
  plan, and selected statement-plan owner selection.
- The verifier must return success for non-base-memory routes only when they do
  not carry stale base-memory provider-plan or statement-owner facts.
- For base-memory consumers, the verifier must fail closed if:
  base-memory route-family plan or materialization facts are absent or not from
  the same selected route analysis;
  provider plan mirrors are stale against the validated base-memory family plan;
  memory operand-binding facts are absent, from a different analysis, or miss
  required source/destination/runtime/stride/index/mask roles;
  runtime ABI order, runtime AVL/VL control, typed config, VL/policy, header,
  type, source/destination memory form, stride/index/mask, or element/config
  facts disagree with the realized typed body/provider plan;
  migrated statement owner selection did not select the base-memory movement
  owner for an active base-memory route;
  required pre-loop/loop statement leaves for the selected memory form are
  missing before route construction.
- Keep Common EmitC neutral. The new verifier must not infer memory semantics
  from route ids, artifact names, descriptor/source-front-door markers, ABI
  strings, test names, or metadata mirrors.

## Acceptance Criteria

- [x] Production plugin code changes in the base-memory
      realization/provider ownership path.
- [x] `RVVEmitCRouteProvider` calls the base-memory provider-facts verifier
      before `TCRVEmitCLowerableRoute` construction.
- [x] Focused positive C++ coverage proves at least one base-memory route
      reaches route construction through the provider plan, migrated statement
      owner, and new verifier.
- [x] Focused fail-closed C++ coverage rejects at least one stale or missing
      memory fact that could otherwise be represented by route metadata or
      mirrors.
- [x] Existing explicit/pre-realized memory regressions for the touched family
      remain passing.
- [x] `build/bin/tianchenrv-rvv-extension-plugin-test` and
      `build/bin/tianchenrv-target-artifact-export-test` pass.
- [x] If emitted route statements, ABI order, header export, or generated
      bundle behavior changes, run one focused generated-bundle dry-run or
      `ssh rvv` smoke for the affected memory route. If no emission behavior
      changes, record why runtime evidence was not required.
- [x] Bounded old-authority scan over touched RVV planning/provider/test files
      and added diff lines shows no positive legacy `i32m1`, descriptor,
      source-front-door, direct-C/source-export, route-id, or mirror-only route
      authority drift.
- [x] `git diff --check`, `git diff --cached --check`, Trellis context
      validation, and clean final git status pass.

## Definition Of Done

- Compiler behavior remains in C++/MLIR/LLVM/TableGen/CMake/lit/FileCheck.
- Python remains limited to tooling and is not used for compiler core behavior.
- RVV semantics stay in the RVV plugin/provider; Common EmitC only consumes the
  provider-built route.
- No new memory route family, dtype/LMUL clone batch, broad generated-bundle
  matrix, high-level Linalg/Vector/StableHLO frontend, direct pre-realized
  route-entry shortcut, source-front-door positive route, performance/autotuning,
  or report-only closure is introduced.
- Task status/context stay truthful.
- The task is finished/archived and one coherent commit is created when
  acceptance criteria are met.

## Technical Approach

Implement a narrow active owner improvement:

```text
realized typed tcrv_rvv base-memory body
  -> base-memory route-family/materialization facts
  -> memory operand-binding facts
  -> route-control provider plan
  -> base-memory provider plan
  -> migrated base-memory statement owner selection
  -> base-memory provider-facts verifier
  -> TCRVEmitCLowerableRoute
```

The verifier will live in the RVV base-memory planning/provider layer and be
declared through the RVV route planning header so the central provider can call
it. It will reuse `RVVSelectedBodyBaseMemoryMovementRouteProviderPlan` and
`RVVSelectedBodyRouteStatementPlanOwnerSelection`; it will not introduce a new
route family or reassemble statements from route ids or metadata mirrors.

## Decision (ADR-lite)

Context: The memory route-family code already validates typed family facts and
builds a provider plan, but the central provider only calls the analogous final
provider-facts gate for contraction and segment2. The Hermes brief asks for a
production fact contract between memory selected-body realization and route
planning, not fixture-only metadata checks.

Decision: Add a base-memory provider-facts verifier and call it in the
production provider after statement owner selection and before route
construction. Keep this round scoped to base memory movement; computed-mask and
segment2 memory remain existing adjacent paths unless the verifier requires a
small shared helper.

Consequences: The base-memory seam becomes explicit and testable against stale
stride/index/mask/runtime/header/type/provider facts. The change should not
alter generated code unless it catches a previously accepted stale route.

## Out Of Scope

- New memory route family or route coverage expansion.
- Dtype/LMUL clone batch.
- Broad generated-bundle matrix.
- High-level Linalg/Vector/StableHLO frontend work.
- Per-Linalg route authority.
- Common EmitC inference of memory semantics.
- Direct pre-realized route-entry shortcut.
- Source-front-door positive route.
- Performance/autotuning or global profile systems.
- Mass rewrite of contraction, elementwise, reduction, or unrelated route
  owners.

## Technical Notes

- Specs read:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`.
- Previous task read:
  `.trellis/tasks/archive/2026-06/06-06-stage2-rvv-contraction-fact-contract-owner/prd.md`.
- Production files inspected:
  `lib/Plugin/RVV/EmitC/RVVEmitCBaseMemoryRouteFamilyPlanOwners.cpp`,
  `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCMemoryStatementPlanOwners.cpp`,
  `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`,
  `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`,
  and `include/TianChenRV/Plugin/RVV/RVVEmitCBaseMemoryRouteFamilyPlanOwners.h`.

## Completion Evidence

- Added `verifyRVVSelectedBodyBaseMemoryMovementRouteProviderFacts` in the
  RVV base-memory provider owner layer. It validates same-analysis
  materialization/provider pointers, typed config and VL/policy facts, base
  memory operand binding roles, stride/index/mask ABI presence, provider-plan
  mirrors, mask provenance, and migrated base-memory statement owner leaves.
- `RVVEmitCRouteProvider` now obtains the base-memory provider plan and calls
  the verifier after statement owner selection and before constructing
  `TCRVEmitCLowerableRoute`.
- Extended `runBaseMemoryMovementRouteFamilyProviderPlanTest` so every focused
  base-memory fixture goes through the provider plan, migrated statement owner,
  new verifier, and route construction. Added fail-closed coverage for stale
  provider mirror payload, missing migrated owner statements, and missing
  `stride_bytes` ABI facts.
- Checks run:
  `cmake --build build --target tianchenrv-rvv-extension-plugin-test`;
  `build/bin/tianchenrv-rvv-extension-plugin-test`;
  `cmake --build build --target tianchenrv-target-artifact-export-test`;
  `build/bin/tianchenrv-target-artifact-export-test`;
  `git diff --check`;
  `git diff --cached --check`;
  `python3 ./.trellis/scripts/task.py validate .trellis/tasks/06-06-06-06-stage2-rvv-base-memory-fact-contract-owner`.
- Generated-bundle or `ssh rvv` evidence was not required for this round
  because no emitted route statements, ABI order, header export, generated
  bundle, or harness behavior changed; the change is a pre-route-construction
  fail-closed provider contract.
- Bounded old-authority scan over added diff lines found no legacy
  `RVVI32M1`, `rvv-i32m1`, `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`,
  source-front-door, descriptor-driven, direct-C, or source-export authority.
  The only metadata scan hit was the intentional negative test string
  `metadata-selected-abi`.
