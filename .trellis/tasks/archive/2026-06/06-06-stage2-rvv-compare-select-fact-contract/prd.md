# Stage2 RVV compare-select realization provider fact contract

## Goal

Strengthen the production RVV plugin seam between compare/select selected-body
realization, compare/select statement-plan ownership, and
`TCRVEmitCLowerableRoute` construction. This round is bounded to the
compare/select provider-fact contract: plain compare-select, computed-mask
select, runtime-scalar computed-mask select, and runtime-scalar dual
compare-mask-and-select routes must consume typed predicate, compare operands,
scalar operand, true/false value roles, selected result role, mask/result
facts, VL/policy, ABI order, and statement-owner facts before the central
provider constructs a route.

## Direction Source

Hermes Direction Brief:

`Stage2 RVV compare/select realization/provider fact-contract owner`

## What I Already Know

- Initial `pwd` was `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short` was clean.
- Initial `git log --oneline -8` started at
  `e9a11400 rvv: own segment2 memory provider facts`.
- No `.trellis/.current-task` existed at the start of this round, so this task
  was created from the Hermes brief before source edits.
- Specs read:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  `.trellis/spec/testing/mlir-testing-contract.md`.
- Previous archived task
  `.trellis/tasks/archive/2026-06/06-06-stage2-rvv-segment2-memory-fact-contract/prd.md`
  moved segment2 provider-fact verification to the segment2 owner-level API and
  made the central provider consume that API before route construction.
- Current compare/select code already has:
  `getRVVSelectedBodyCompareSelectRouteStatementPlan(...)`,
  `buildRVVSelectedBodyCompareSelectMigratedRouteStatementPlan(...)`, and a
  production provider preflight named
  `verifyRVVSelectedBodyCompareSelectRouteProviderFacts(...)`.
- The remaining ownership gap is that the compare/select provider-fact
  verifier is still declared on the generic `RVVEmitCRoutePlanning` surface
  instead of the compare/select statement-plan owner surface, even though it
  semantically validates compare/select owner-built statement-plan facts.

## Requirements

- Add or expose one compare/select owner-level provider-fact contract entry
  point for active compare/select families.
- Wire `RVVEmitCRouteProvider` to call the compare/select owner-level
  provider-fact contract after route materialization facts,
  elementwise/select operand-binding facts, and the compare/select
  statement-plan owner output are available, and before
  `TCRVEmitCLowerableRoute` construction.
- The owner-level contract must cover:
  plain compare-select,
  computed-mask select,
  runtime-scalar computed-mask select, and
  runtime-scalar dual compare-mask-and-select.
- Preserve fail-closed checks for same-analysis family plans, predicate facts,
  mask producer/source/form, scalar operand binding, compare operand roles,
  true/false value roles, selected result role, runtime ABI order and roles,
  route-control facts, mask/tail facts, typed SEW/LMUL/policy, header/type/
  intrinsic facts, and compare/select statement-owner leaves.
- Keep Common EmitC neutral. The contract must not infer compare/select
  semantics from route ids, artifact names, ABI strings, helper names,
  descriptors, source-front-door markers, exact intrinsic spellings, or
  metadata mirrors.

## Acceptance Criteria

- [x] Production plugin code changes in the compare/select
      realization/provider ownership path.
- [x] `RVVEmitCRouteProvider` consumes a compare/select owner-level
      provider-fact contract before constructing `TCRVEmitCLowerableRoute` for
      active compare/select routes.
- [x] Focused positive C++ coverage still proves plain compare-select,
      computed-mask select, runtime-scalar computed-mask select, and
      runtime-scalar dual compare-mask-and-select consume typed body facts,
      operand-binding facts, route-control or mask/tail facts, compare/select
      route-family provider plans, and statement-owner leaves before route
      construction.
- [x] Focused fail-closed C++ coverage still rejects stale or missing facts
      such as route-id selected predicate/intrinsic, stale typed config,
      wrong scalar operand role, stale true/false value role, wrong selected
      result role, missing mask provenance, stale mask/tail facts, or wrong
      SEW/LMUL.
- [x] Existing explicit and pre-realized compare/select regressions for the
      touched family remain passing.
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
- RVV compare/select semantics stay in RVV plugin owners/providers; Common
  EmitC only consumes provider-built routes and payloads.
- No new compare predicate family, dtype/LMUL clone batch, high-level
  Linalg/Vector/StableHLO frontend, source-front-door positive route,
  performance/autotuning, direct pre-realized route shortcut, or report-only
  closure is introduced.
- Task status/context stay truthful.
- The task is finished/archived and one coherent commit is created when
  acceptance criteria are met.

## Technical Approach

Align compare/select with the segment2 owner pattern completed in the previous
task:

```text
realized typed compare/select tcrv_rvv body
  -> compare/select route-family materialization facts
  -> RVV-owned elementwise/select operand-binding facts
  -> compare/select statement-plan owner
  -> compare/select owner provider-fact contract
  -> TCRVEmitCLowerableRoute
```

The narrow implementation target is to move the public provider-facing
contract declaration for `verifyRVVSelectedBodyCompareSelectRouteProviderFacts`
onto `RVVEmitCStatementPlanOwners.h`, keep its implementation in the
compare/select owner translation unit or delegate there from existing code, and
make provider/tests call the owner-level surface. Existing positive and
fail-closed tests are already broad; add one focused assertion only if the API
move exposes an uncovered stale-fact seam.

## Out Of Scope

- New compare predicate families or mask-policy expansion beyond facts needed
  by compare/select.
- Dtype/LMUL clone batches.
- Broad generated-bundle matrices.
- High-level Linalg/Vector/StableHLO frontend work.
- Per-Linalg route authority.
- Common EmitC inference of compare/select, predicate, mask, dtype, ABI, or
  intrinsic semantics.
- Direct pre-realized route-entry shortcuts.
- Source-front-door positive routes.
- Performance/autotuning or global profile systems.
- Mass rewrite of memory, segment2, elementwise arithmetic, contraction,
  reduction, conversion, or control-policy owners outside the touched seam.

## Completion Evidence

- Production seam changed:
  `verifyRVVSelectedBodyCompareSelectRouteProviderFacts(...)` is now declared
  by `RVVEmitCStatementPlanOwners.h` and implemented in
  `RVVEmitCCompareSelectStatementPlanOwners.cpp`, colocated with the
  compare/select statement-plan owner. The generic
  `RVVEmitCRoutePlanning.h` declaration and `RVVEmitCRoutePlanning.cpp`
  implementation were removed.
- `RVVEmitCRouteProvider` continues to call the verifier after obtaining
  compare/select materialization facts, elementwise/select operand-binding
  facts, and `RVVSelectedBodyCompareSelectRouteStatementPlan`, and before
  constructing `TCRVEmitCLowerableRoute`.
- Focused C++ coverage in `test/Plugin/RVVExtensionPluginTest.cpp` now uses
  the owner-level declaration for the existing positive provider-consumption
  and fail-closed stale/missing fact checks, including stale statement-plan
  flags, stale typed config, stale materialization leaves, stale operand
  bindings, stale mask/tail facts, missing family plans, and copied/stale
  analysis facts.
- Spec updated:
  `.trellis/spec/extension-plugins/rvv-plugin.md` now records
  `verifyRVVSelectedBodyCompareSelectRouteProviderFacts(...)` as the
  compare/select statement-plan owner API and requires provider consumption
  before `TCRVEmitCLowerableRoute` construction.
- Emitted route statements, runtime ABI order, header export, generated bundle
  output, and runtime harness behavior were not changed. This round only moves
  the pre-route owner/provider API and implementation ownership, so no
  generated-bundle dry-run or `ssh rvv` runtime evidence was required or
  claimed.
- Checks passed:
  `cmake --build build --target tianchenrv-rvv-extension-plugin-test`;
  `build/bin/tianchenrv-rvv-extension-plugin-test`;
  `cmake --build build --target tianchenrv-target-artifact-export-test`;
  `build/bin/tianchenrv-target-artifact-export-test`;
  manual lit RUN reproduction for
  `test/Dialect/RVV/computed-mask-select-dataflow.mlir`;
  manual `REALIZED`, `PLAN`, and `HEADER` FileCheck runs for
  `test/Target/RVV/pre-realized-selected-body-artifact-computed-mask-select-lmul-m2.mlir`;
  manual `REALIZED`, `PLAN`, and `HEADER` FileCheck runs for
  `test/Target/RVV/pre-realized-selected-body-artifact-runtime-scalar-dual-cmp-mask-and-select.mlir`;
  `python3 ./.trellis/scripts/task.py validate .trellis/tasks/06-06-stage2-rvv-compare-select-fact-contract`;
  `python3 ./.trellis/scripts/task.py validate .trellis/tasks/archive/2026-06/06-06-stage2-rvv-compare-select-fact-contract`;
  `git diff --check`.
- Self-repair performed:
  an initial manual lit reproduction omitted the dialect test's
  `--split-input-file` flag and tried to use `FileCheck` through PATH. The
  command was corrected using the lit-configured
  `/usr/lib/llvm-20/bin/FileCheck` path and the exact RUN-line flags, then the
  focused compare/select lit checks passed. After archiving, Trellis validation
  reported that `check.jsonl` still pointed at the pre-archive PRD path. The
  path was corrected to the archived PRD path and validation passed.
- Bounded added-diff scan over touched RVV owner/provider/test files found no
  positive legacy authority matches for `RVVI32M1`, `rvv-i32m1`,
  `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`, source-front-door, descriptor,
  direct-C/source-export, route-id, helper-name, or mirror-only route
  authority drift. A full touched-file scan only found pre-existing
  fail-closed/negative fixture references and generic legacy rejection logic.
