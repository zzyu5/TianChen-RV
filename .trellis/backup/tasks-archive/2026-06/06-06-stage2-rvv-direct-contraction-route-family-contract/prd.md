# Stage2 RVV direct contraction route-family production contract consolidation

## Goal

Consolidate one production contract submodule shared by the already-proven
MAcc and direct contraction generated-artifact paths. The bounded owner for
this round is the provider-owned artifact validation contract core: runtime
AVL/VL identity, ABI order, provider support mirror, required headers,
C type mapping summary, operand-binding summary, typed compute op, and the
representative source/result/mask type facts that both MAcc and widening
dot-reduce target validators consume before accepting generated RVV bundles.

This round must make a meaningful production-source change. It must not add
another single fixture, report, generated-bundle proof, route id, metadata
mirror, or one-off operation case as the main achievement.

## Direction Source

Hermes Direction Brief:

`Stage2 RVV direct contraction route-family production contract consolidation`

## What I Already Know

- Initial `pwd` was `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short` was clean when read through `rtk proxy`.
- Initial `git log --oneline -8` started at
  `e553b8e5 rvv: prove computed masked strided dot reduce artifact abi`.
- No `.trellis/.current-task` existed, so this task was created from the
  Direction Brief before source edits.
- Specs read:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`, and
  `.trellis/spec/lowering-runtime/emitc-route.md`.
- Relevant live spec contract:
  `.trellis/spec/extension-plugins/rvv-plugin.md` defines the
  `Direct Contraction Route-Provider Owner Boundary`: provider construction
  must join same-analysis family plan, route-control, materialization facts,
  math operand-binding facts, typed config, target capability, and ABI bindings
  before `TCRVEmitCLowerableRoute`, then the owner statement plan consumes the
  prevalidated provider plan.
- Relevant archived tasks read:
  `.trellis/tasks/archive/2026-06/06-06-stage2-rvv-direct-contraction-fact-contract/`,
  `.trellis/tasks/archive/2026-06/06-06-stage2-rvv-macc-add-artifact-abi/`,
  `.trellis/tasks/archive/2026-06/06-06-stage2-rvv-runtime-scalar-cmp-masked-macc-lmul-m2-artifact-abi/`,
  and `.trellis/tasks/archive/2026-06/06-06-stage2-rvv-computed-masked-strided-widening-dot-reduce-add-artifact-abi/`.
- The direct contraction provider-fact verifier is already localized in
  `RVVEmitCStatementPlanOwners` and is not the missing active-source blocker.
- Recent MAcc and dot-reduce tasks repeatedly closed as executable evidence
  without production source changes. The next useful work must reduce
  duplicated/stale production contract surfaces, not prove another fixture.
- Current target validation already consumes two route-family contract structs:
  `RVVMAccRouteValidationContract` and
  `RVVWideningDotReduceRouteValidationContract`. Both carry parallel
  provider-owned runtime/control, ABI/header/type, operand-binding,
  typed-compute, and source/result/mask facts before target artifact
  acceptance.

## Requirements

- Extract a shared provider-owned contraction artifact contract core for facts
  common to MAcc and widening dot-reduce artifact validation.
- Populate the existing MAcc and widening dot-reduce validation contracts from
  that shared core, preserving all route-family-specific fields and behavior.
- Keep route authority structural: typed `tcrv_rvv` body/config/runtime facts,
  RVV family plans, route-control facts, operand-binding facts, and provider
  descriptions remain the source of truth.
- Keep target artifact metadata as mirror-only. Target validators may compare
  candidate metadata against provider facts, but must not infer support from
  route ids, artifact names, test names, scripts, ABI strings, helper names,
  exact intrinsic spellings, descriptors, source-front-door markers, or common
  EmitC behavior.
- Preserve current MAcc and widening dot-reduce generated artifact behavior,
  ABI/header mapping, operand binding summaries, runtime AVL/VL validation,
  and fail-closed stale mirror diagnostics.
- Avoid broad matrix expansion, new contraction/MAcc feature coverage, LMUL or
  dtype clone batches, source-front-door positive routes, Common EmitC RVV
  semantic branches, and runtime/performance claims.

## Acceptance Criteria

- [x] Production source changes extract and consume a shared provider-owned
      contraction artifact contract core for MAcc and widening dot-reduce
      validation-contract construction.
- [x] Existing MAcc and widening dot-reduce target validators still compare
      provider-derived headers, type mappings, ABI order, operand binding,
      runtime AVL/VL, statement shape, accumulator/result, mask/stride, and
      inactive-lane facts as before.
- [x] At least one representative MAcc path and one representative
      widening-dot/direct-contraction path still materialize/export through the
      existing focused generated-bundle or target artifact dry-run tests.
- [x] Focused fail-closed C++ evidence still rejects stale/missing provider
      facts such as wrong header/type mapping, wrong operand binding, stale
      accumulator/result role, stale mask/stride fact, or stale route-family
      mirror.
- [x] `build/bin/tianchenrv-rvv-extension-plugin-test` and
      `build/bin/tianchenrv-target-artifact-export-test` pass.
- [x] No `ssh rvv` evidence is required unless this round changes emitted code,
      ABI order, generated header content, runtime harness behavior, or makes a
      new executable correctness/runtime/performance claim.
- [x] Bounded old-authority scan over touched source/test files and added diff
      lines shows no positive legacy i32/source-front-door/descriptor/direct-C/
      source-export/route-id/artifact-name/helper-name/metadata authority drift.
- [x] `git diff --check`, `git diff --cached --check`, Trellis validation, and
      clean final git status pass.

## Technical Approach

Add a small RVV-local contract type such as
`RVVContractionArtifactContractCore` on the route-provider contract surface and
builder helpers for MAcc and widening dot-reduce descriptions. The helpers
copy only facts that are already provider-derived in the current route
description:

```text
provider route description
  -> shared contraction artifact contract core
  -> MAcc validation contract
  -> target artifact MAcc validator

provider route description
  -> shared contraction artifact contract core
  -> widening dot-reduce validation contract
  -> target artifact widening dot validator
```

Route-family-specific fields stay in their current MAcc and contraction owner
implementations. The target validator should continue to use the existing
MAcc and widening dot-reduce validation contract APIs, so this change tightens
the producer-side contract without broad target rewrites.

## Out Of Scope

- New MAcc, dot-reduce, reduction, memory, compare/select, dtype, LMUL, or
  source-front-door feature expansion.
- Broad generated-bundle or runtime smoke matrix.
- Common EmitC invention of RVV semantics.
- Per-Linalg route authority or high-level frontend work.
- Compatibility wrappers preserving old i32 route authority.
- One-op-per-intrinsic wrapper dialect work.
- Global autotuning, dashboards, readiness state machines, or performance
  claims.

## Technical Notes

- Relevant production files:
  `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`,
  `include/TianChenRV/Plugin/RVV/RVVEmitCMAccRouteFamilyPlanOwners.h`,
  `include/TianChenRV/Plugin/RVV/RVVEmitCContractionRouteFamilyPlanOwners.h`,
  `lib/Plugin/RVV/EmitC/RVVEmitCMAccRouteFamilyPlanOwners.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCContractionRouteFamilyPlanOwners.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCStatementPlanOwners.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`,
  and `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`.
- Relevant focused tests:
  `test/Plugin/RVVExtensionPluginTest.cpp`,
  `test/Target/TargetArtifactExportTest.cpp`,
  MAcc generated-bundle dry-run tests, and computed-mask strided widening
  dot-reduce generated-bundle dry-run tests.
- Runtime evidence is intentionally not part of the initial acceptance because
  the proposed source change should not alter generated statements, ABI order,
  headers, or runtime harness behavior.

## Completion Notes

Production source change:

- Added `RVVContractionArtifactContractCore` and
  `getRVVContractionArtifactContractCore(...)` on the RVV route-provider
  contract surface.
- `getRVVMAccRouteValidationContract(...)` now populates and retains the shared
  core for MAcc artifact validation facts, while keeping MAcc-specific
  accumulator/result, mask, arithmetic, and statement-plan fields in the MAcc
  owner contract.
- `getRVVWideningDotReduceRouteValidationContract(...)` now populates and
  retains the same shared core for widening dot-reduce/direct-contraction
  artifact validation facts, while keeping product/reduction/dequant/clamp,
  mask, stride, inactive-lane, and statement-plan fields in the contraction
  owner contract.
- `RVVTargetArtifactRouteFamilyValidation.cpp` now consumes `contract.core`
  for the common provider-owned target-acceptance facts in both MAcc and
  widening dot-reduce validators: route token, memory form, config contract,
  runtime control/ABI mirrors, required header declarations, C type mapping
  summary, route operand binding plan/summary, target leaf profile,
  provider-supported mirror, typed compute op, VL C type, and runtime ABI
  parameter order.

Behavioral boundary:

- No emitted statement, ABI order, generated header, generated bundle harness,
  or runtime behavior changed. The production change is a provider/target
  validation contract consolidation only, so no new `ssh rvv` evidence was
  required.
- Low-precision widening-product validation was left outside this round; an
  intermediate mistaken edit there was reverted after the focused build caught
  it.
- `.trellis/spec/extension-plugins/rvv-plugin.md` was updated with the new
  "MAcc And Direct-Contraction Artifact Contract Core" executable contract
  because this round introduced a durable public C++ contract surface.

Checks run:

- `cmake --build build --target tianchenrv-rvv-extension-plugin-test tianchenrv-target-artifact-export-test`
- `build/bin/tianchenrv-rvv-extension-plugin-test`
- `build/bin/tianchenrv-target-artifact-export-test`
- From `build/test`:
  `/usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'rvv-generated-bundle-abi-e2e-(pre-realized-)?macc-add-dry-run|rvv-generated-bundle-abi-e2e-(explicit-|pre-realized-)?computed-masked-strided-input-widening-dot-reduce-add-dry-run|rvv-generated-bundle-abi-e2e-direct-pre-realized-(macc-add|computed-masked-strided-input-widening-dot-reduce-add)-fail-closed'`
  passed 6 selected tests.
- `git diff --check`
- `git diff --cached --check`
- `python3 ./.trellis/scripts/task.py validate .trellis/tasks/06-06-stage2-rvv-direct-contraction-route-family-contract`
- Bounded added-diff old-authority scan over touched source files: no matches.

## Current Phase

Finish.
