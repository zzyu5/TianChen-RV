# Stage2 RVV segment2 memory realization provider fact contract

## Goal

Strengthen the production RVV plugin seam between segment2 selected-body
realization, segment2 route-family planning, migrated segment2 statement-plan
ownership, and `TCRVEmitCLowerableRoute` construction. This round is bounded to
the segment2 memory provider-fact contract: plain segment2
deinterleave/interleave and computed-mask segment2 load/store/update routes
must consume owner-built segment layout, field roles, computed-mask provenance,
memory-form, runtime ABI, VL/policy, route-control, and statement-owner facts
before the central provider constructs a route.

## Direction Source

Hermes Direction Brief:

`Stage2 RVV segment2 memory realization/provider fact-contract owner`

## What I Already Know

- Initial `pwd` was `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short` was clean.
- Initial `git log --oneline -8` started at
  `169e0cfd rvv: verify computed-mask memory provider facts`.
- No `.trellis/.current-task` existed at the start of this round, so this task
  was created from the Hermes brief before source edits.
- Specs read:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`.
- Previous archived task
  `.trellis/tasks/archive/2026-06/06-06-stage2-rvv-computed-mask-memory-fact-contract/prd.md`
  completed the non-segment computed-mask memory owner-level provider-fact
  contract and explicitly left computed-mask segment2 on its segment2 owner
  path.
- Current code already has:
  `RVVSelectedBodySegment2RouteFamilyPlanningOwner`,
  `getRVVSelectedBodySegment2RouteFamilyProviderPlan(...)`,
  `getRVVSelectedBodySegment2MemoryRouteStatementPlan(...)`, and a strong
  `verifyRVVSelectedBodySegment2RouteProviderFacts(...)` preflight.
- The remaining ownership gap is that the provider-fact verifier is declared
  on the generic `RVVEmitCRoutePlanning` surface and called as a generic route
  planning helper. The segment2 owner header does not expose the final
  provider-facing contract, even though the verifier semantically validates
  segment2 owner-built facts.

## Requirements

- Add or expose one segment2 owner-level provider-fact contract entry point for
  active segment2 memory families.
- Wire `RVVEmitCRouteProvider` to call the segment2 owner-level provider-fact
  contract after:
  `getRVVSelectedBodySegment2RouteFamilyProviderPlan(...)`,
  memory materialization/operand-binding facts, and migrated statement owner
  selection are available, and before `TCRVEmitCLowerableRoute` construction.
- The owner-level contract must cover:
  plain segment2 deinterleave/unit-store,
  plain segment2 interleave/unit-load,
  computed-mask segment2 load/unit-store,
  computed-mask segment2 store/unit-load, and
  computed-mask segment2 update/unit-load.
- Preserve fail-closed checks for same-analysis family plans,
  segment count/direction, field roles, mask producer/source/form,
  inactive-lane/passthrough policy, update arithmetic, memory form,
  runtime ABI order and roles, route-control facts, typed SEW/LMUL/policy,
  header/type/intrinsic facts, and migrated segment2 statement-owner leaves.
- Keep Common EmitC neutral. The contract must not infer segment2 or
  computed-mask semantics from route ids, artifact names, ABI strings, helper
  names, descriptors, source-front-door markers, exact intrinsic spellings, or
  metadata mirrors.

## Acceptance Criteria

- [x] Production plugin code changes in the segment2 memory
      realization/provider ownership path.
- [x] `RVVEmitCRouteProvider` consumes a segment2 owner-level provider-fact
      contract before constructing `TCRVEmitCLowerableRoute` for active
      segment2 memory routes.
- [x] Focused positive C++ coverage still proves computed-mask segment2 update
      and plain segment2 paths consume typed body facts, operand-binding facts,
      route-control facts, segment2 route-family provider plans, and migrated
      statement-owner leaves before route construction.
- [x] Focused fail-closed C++ coverage still rejects stale or missing facts
      such as metadata-selected route id, provider mirror, runtime ABI order,
      missing field binding, wrong segment count, stale inactive-lane policy,
      missing runtime-control plan, wrong operation subfamily, and missing
      migrated statement-owner leaves.
- [x] Existing explicit and pre-realized segment2 regressions for the touched
      family remain passing.
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
- RVV segment2 semantics stay in RVV plugin owners/providers; Common EmitC only
  consumes provider-built routes and payloads.
- No new segment width, segment3/segment4 expansion, dtype/LMUL clone batch,
  high-level Linalg/Vector/StableHLO frontend, source-front-door positive
  route, performance/autotuning, or report-only closure is introduced.
- Task status/context stay truthful.
- The task is finished/archived and one coherent commit is created when
  acceptance criteria are met.

## Technical Approach

Align segment2 with the computed-mask memory owner pattern completed in the
previous task:

```text
realized typed segment2 tcrv_rvv body
  -> segment2/computed-mask family materialization facts
  -> RVV-owned memory operand-binding facts
  -> segment2 route-family planning owner
  -> migrated segment2 statement-plan owner selection
  -> segment2 owner provider-fact contract
  -> TCRVEmitCLowerableRoute
```

The narrow implementation target is to move the public provider-facing
contract declaration for `verifyRVVSelectedBodySegment2RouteProviderFacts(...)`
onto `RVVEmitCSegment2RouteFamilyPlanOwners.h`, keep the implementation in the
segment2 owner translation unit or delegate there from existing code, and make
provider/tests call the owner-level surface. Existing positive and fail-closed
tests should remain the behavioral evidence; add a focused assertion only if
the API shift exposes an uncovered seam.

## Out Of Scope

- Segment3/segment4 or any new segment width.
- New dtype/LMUL route clone batch.
- Broad generated-bundle matrix.
- High-level Linalg/Vector/StableHLO frontend work.
- Per-Linalg route authority.
- Common EmitC inference of segment, mask, memory, dtype, or intrinsic
  semantics.
- Direct pre-realized route-entry shortcut.
- Source-front-door positive route.
- Performance/autotuning or global profile systems.
- Mass rewrite of non-segment memory, elementwise, contraction, reduction,
  conversion, or control-policy owners outside the touched seam.

## Completion Evidence

- Production seam changed:
  `verifyRVVSelectedBodySegment2MemoryRouteProviderFacts(...)` is now the
  segment2 memory owner-level provider-fact contract declared by
  `RVVEmitCSegment2RouteFamilyPlanOwners.h`. The old generic
  `verifyRVVSelectedBodySegment2RouteProviderFacts(...)` declaration was
  removed from `RVVEmitCRoutePlanning.h`.
- `RVVEmitCRouteProvider` now calls the segment2 memory owner-level contract
  after obtaining the segment2 route-family provider plan and migrated
  statement-plan owner selection, and before constructing
  `TCRVEmitCLowerableRoute`.
- Focused C++ coverage in `test/Plugin/RVVExtensionPluginTest.cpp` now uses
  the segment2 owner-level contract for the existing positive preflight helper
  and fail-closed stale/missing fact checks, including stale route id, provider
  mirror, runtime ABI order, missing field binding, wrong segment count, stale
  inactive-lane policy, missing runtime-control plan, stale subfamily flags,
  and missing migrated statement-owner leaves.
- Spec updated:
  `.trellis/spec/extension-plugins/rvv-plugin.md` now records the durable
  `verifyRVVSelectedBodySegment2MemoryRouteProviderFacts(...)` API and requires
  provider consumption before `TCRVEmitCLowerableRoute` construction.
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
  `python3 ./.trellis/scripts/task.py validate .trellis/tasks/archive/2026-06/06-06-stage2-rvv-segment2-memory-fact-contract`;
  `git diff --check`;
  `git diff --cached --check`.
- Self-repair performed:
  after archiving, Trellis validation reported that `check.jsonl` still pointed
  at the pre-archive PRD path. The path was corrected to the archived PRD path
  and validation passed.
- Bounded added-diff scan over touched RVV owner/provider/test files found no
  positive legacy authority matches for `RVVI32M1`, `rvv-i32m1`,
  `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`, source-front-door, descriptor,
  direct-C/source-export, route-id, helper-name, or mirror-only route
  authority drift.
