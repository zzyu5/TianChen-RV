# Stage2 RVV computed-mask memory fact contract

## Goal

Strengthen the production RVV plugin seam between computed-mask memory
selected-body realization, computed-mask memory route-family planning, migrated
statement ownership, and `TCRVEmitCLowerableRoute` construction. The bounded
owner for this round is the non-segment computed-mask memory provider-fact
contract: runtime-scalar computed-mask store/load-store, computed-mask unit
load/store, strided store, strided load/unit-store, indexed gather/unit-store,
and indexed scatter/unit-load must enter route construction through the
computed-mask memory owner API with same-analysis typed body, mask, memory,
runtime ABI, route-control, and statement-plan facts.

## Direction Source

Hermes Direction Brief:

`Stage2 RVV computed-mask memory realization/provider fact-contract owner`

## What I Already Know

- Initial `pwd` was `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short` was clean under RTK output.
- Initial `git log --oneline -8` started at
  `91708373 rvv: verify elementwise broadcast provider facts`.
- No `.trellis/.current-task` existed at the start of this round, so this task
  was created from the Hermes brief before source edits.
- Specs read:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  `.trellis/spec/extension-plugins/index.md`,
  `.trellis/spec/lowering-runtime/index.md`,
  `.trellis/spec/guides/index.md`.
- The previous archived elementwise/broadcast task moved the elementwise
  provider-fact verifier into the elementwise owner layer and wired it before
  route construction.
- The current code already has split runtime-scalar and regular computed-mask
  memory provider-fact verifiers, and `RVVEmitCRouteProvider` already calls
  them before building the route.
- Unlike the current elementwise and base-memory owner pattern, the
  computed-mask memory provider-fact verifier surface still lives as split
  generic route-planning declarations and provider-local dispatch rather than a
  single computed-mask memory owner API.

## Requirements

- Add or expose one computed-mask memory owner-level provider-fact contract
  entry point for the active non-segment computed-mask memory families.
- Wire `RVVEmitCRouteProvider` to call that owner-level contract after
  materialization facts, memory operand-binding facts, and computed-mask memory
  statement-plan facts are available, and before `TCRVEmitCLowerableRoute`
  construction.
- The owner-level contract must cover the existing runtime-scalar and regular
  non-segment computed-mask memory subfamilies without adding new memory route
  coverage.
- Preserve the existing fail-closed checks for:
  verified computed-mask memory family plan identity;
  compare-produced mask or runtime-scalar mask provenance;
  unit/strided/indexed memory form;
  source/destination/passthrough/stride/index ABI roles;
  typed config, SEW, LMUL, VL, policy, target capability, header/type, and
  intrinsic facts;
  route-control provider plan;
  mask/tail provider plan;
  statement-plan owner leaves and step counts.
- Keep computed-mask segment2 memory outside this non-segment owner boundary.
- Keep Common EmitC neutral. The provider-fact contract must not infer RVV mask
  or memory semantics from route ids, artifact names, ABI strings, helper
  names, descriptors, source-front-door markers, or metadata mirrors.

## Acceptance Criteria

- [x] Production plugin code changes in the computed-mask memory
      realization/provider ownership path.
- [x] `RVVEmitCRouteProvider` consumes a computed-mask memory owner-level
      provider-fact contract before constructing `TCRVEmitCLowerableRoute` for
      active non-segment computed-mask memory routes.
- [x] Existing focused positive C++ coverage still proves runtime-scalar,
      unit, strided, indexed gather, and indexed scatter routes consume typed
      body facts, operand-binding facts, route-control facts, mask/tail facts,
      and computed-mask memory statement plans before route construction.
- [x] Existing focused fail-closed C++ coverage still rejects stale or missing
      facts such as stale mask provenance, metadata-selected runtime ABI,
      missing scalar/stride/index binding, wrong address role, wrong
      statement-plan subfamily, wrong SEW/LMUL or policy, and stale
      provider/mirror facts.
- [x] Existing explicit and pre-realized computed-mask memory regressions for
      the touched family remain passing.
- [x] `build/bin/tianchenrv-rvv-extension-plugin-test` and
      `build/bin/tianchenrv-target-artifact-export-test` pass.
- [x] If emitted route statements, ABI order, header export, generated bundle,
      or runtime harness behavior changes, run one focused generated-bundle
      dry-run or `ssh rvv` smoke. If this remains a pre-route contract/API
      tightening only, record why runtime evidence is not required.
- [x] Bounded old-authority scan over touched RVV owner/provider/test files and
      added diff lines shows no positive legacy `i32m1`, descriptor,
      source-front-door, direct-C/source-export, route-id, helper-name, or
      mirror-only route authority drift.
- [x] `git diff --check`, `git diff --cached --check`, Trellis validation, and
      clean final git status pass.

## Definition Of Done

- Compiler behavior remains in C++/MLIR/LLVM/TableGen/CMake/lit/FileCheck.
- Python remains limited to tooling and is not used for compiler core behavior.
- RVV semantics stay in RVV plugin owners/providers; Common EmitC only consumes
  provider-built routes and payloads.
- No new memory route family, segment2 expansion, dtype/LMUL clone batch,
  high-level Linalg/Vector/StableHLO frontend, source-front-door positive route,
  performance/autotuning, or report-only closure is introduced.
- Task status/context stay truthful.
- The task is finished/archived and one coherent commit is created when
  acceptance criteria are met.

## Technical Approach

Use the existing active verifier behavior but align the ownership surface with
the current elementwise/base-memory pattern:

```text
realized typed computed-mask memory tcrv_rvv body
  -> computed-mask memory family/materialization facts
  -> RVV-owned memory operand-binding facts
  -> RVV-owned computed-mask memory statement plan
  -> computed-mask memory owner provider-fact contract
  -> TCRVEmitCLowerableRoute
```

The narrow implementation target is a single computed-mask memory owner API
that dispatches to the existing runtime-scalar and regular non-segment
provider-fact checks. `RVVEmitCRouteProvider` should not keep a split local
subfamily verifier dispatch for this boundary. Existing positive and
fail-closed tests remain the behavioral evidence; add a focused assertion only
if the API shift creates an uncovered seam.

## Out Of Scope

- New memory route family or segment2 computed-mask memory expansion.
- Dtype/LMUL clone batch.
- Broad generated-bundle matrix.
- High-level Linalg/Vector/StableHLO frontend work.
- Per-Linalg route authority.
- Common EmitC inference of mask, memory, stride, index, dtype, or intrinsic
  semantics.
- Direct pre-realized route-entry shortcut.
- Source-front-door positive route.
- Performance/autotuning or global profile systems.
- Mass rewrite of elementwise, contraction, reduction, conversion,
  control-policy, segment2, or target export owners outside the touched seam.

## Completion Evidence

- Production seam changed:
  `verifyRVVSelectedBodyComputedMaskMemoryRouteProviderFacts` now lives on the
  computed-mask memory owner API in
  `RVVEmitCComputedMaskMemoryRouteFamilyPlanOwners.h/.cpp`. It is the single
  provider-facing entry point for non-segment computed-mask memory provider
  facts, and it dispatches to the existing runtime-scalar and regular
  subfamily checks only after the owner-level consumer boundary is selected.
- `RVVEmitCRouteProvider` now classifies active non-segment computed-mask
  memory routes through
  `isRVVSelectedBodyNonSegmentComputedMaskMemoryRouteFamilyConsumer`, obtains
  the RVV-owned computed-mask memory statement plan, and calls the owner-level
  provider-fact verifier before route construction.
- Focused C++ coverage in `test/Plugin/RVVExtensionPluginTest.cpp` now calls
  the owner-level provider-fact verifier for runtime-scalar, unit, strided,
  indexed gather, indexed scatter, and fail-closed stale/missing fact cases.
- Spec updated:
  `.trellis/spec/extension-plugins/rvv-plugin.md` now records the durable
  `verifyRVVSelectedBodyComputedMaskMemoryRouteProviderFacts` entry point and
  requires `RVVEmitCRouteProvider` to use it after computed-mask memory
  statement planning and before `TCRVEmitCLowerableRoute` construction.
- Emitted route statements, runtime ABI order, header export, generated bundle
  output, and runtime harness behavior were not changed. This round only
  tightens the pre-route owner/provider API and test entry point, so no
  generated-bundle dry-run or `ssh rvv` runtime evidence was required or
  claimed.
- Checks passed:
  `cmake --build build --target tianchenrv-rvv-extension-plugin-test`;
  `build/bin/tianchenrv-rvv-extension-plugin-test`;
  `cmake --build build --target tianchenrv-target-artifact-export-test`;
  `build/bin/tianchenrv-target-artifact-export-test`;
  `python3 ./.trellis/scripts/task.py validate .trellis/tasks/06-06-stage2-rvv-computed-mask-memory-fact-contract`;
  `git diff --check`;
  `git diff --cached --check`.
- Bounded added-diff scan over touched RVV owner/provider/test files found no
  positive legacy authority matches for `RVVI32M1`, `rvv-i32m1`,
  `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`, source-front-door, descriptor,
  direct-C/source-export, route-id, helper-name, or mirror-only route authority
  drift.
