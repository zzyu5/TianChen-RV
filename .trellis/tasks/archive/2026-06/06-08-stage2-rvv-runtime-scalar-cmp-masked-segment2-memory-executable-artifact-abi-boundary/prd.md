# Stage2 RVV runtime-scalar-cmp masked segment2 memory executable artifact ABI boundary

## Goal

Prove or repair the current HEAD production workflow for runtime-scalar compare
masked segment2 load/store selected-body routes from typed `tcrv_rvv` body
facts through RVV provider route validation, common EmitC materialization,
target artifact export, generated bundle ABI, and real `ssh rvv` correctness
evidence. This round is specifically after commit `9807ed7e`, which added the
provider-owned runtime-scalar computed-mask splat contract but did not claim new
runtime behavior.

## What I Already Know

* No `.trellis/.current-task` existed at session start; this task was created
  from the Hermes Direction Brief.
* Repository inspection started from clean `main` at commit `9807ed7e`.
* The archived provider-contract task added
  `RVVRuntimeScalarComputedMaskMemorySplatProviderContract` and wired it into
  runtime-scalar computed-mask segment2 load/store provider preflight before
  `TCRVEmitCLowerableRoute` construction.
* Prior journal evidence recorded runtime-scalar-cmp masked segment2 explicit
  and pre-realized load/store bundles passing on `ssh rvv`, but that evidence
  predates the current task's required post-`9807ed7e` proof boundary and must
  not be reused as current executable evidence without rerunning.
* `.trellis/spec/extension-plugins/rvv-plugin.md` requires the selected typed
  or realized RVV body, not route id or metadata, to carry runtime scalar
  compare, splat, mask, segment2 field layout, inactive-lane, dtype/config, and
  runtime AVL/VL facts.
* `.trellis/spec/lowering-runtime/emitc-route.md` requires provider-built
  `TCRVEmitCLowerableRoute` facts and exact target mirror validation before
  accepting generated artifacts.
* `.trellis/spec/testing/mlir-testing-contract.md` requires real `ssh rvv`
  evidence for runtime/correctness claims and harness checks for active lanes,
  inactive/passthrough preservation, source preservation, field distinction,
  runtime counts, and tail sentinels.
* Source inspection found the current provider path calls
  `verifyRVVSelectedBodySegment2MemoryRouteProviderFacts(...)` before route
  construction; runtime-scalar segment2 consumers invoke the shared splat
  provider contract against ABI order, provider mirror, `rhs_scalar` ABI role,
  materialized splat leaf, and statement-plan splat leaf.
* Target validation rebuilds the provider route and checks runtime-scalar
  segment2 load/store statement sequence, including `rhs_scalar` splat before
  compare, segment2 tuple/load/store layout, pointer expressions, runtime
  `n`/AVL loop bounds, route operand binding, header/prototype, and mirror
  agreement.
* The generated-bundle script already has explicit and pre-realized
  expectations for `runtime_scalar_cmp_masked_segment2_load_unit_store` and
  `runtime_scalar_cmp_masked_segment2_store_unit_load`, including route operand
  binding, runtime ABI order, provider-supported mirror, harness active/inactive
  checks, two data patterns, and `rhs_scalar` compare use.

## Requirements

* Keep the task bounded to runtime-scalar-cmp masked segment2 load/store
  executable artifact ABI boundary.
* Start with the load/store pair. If a production failure is discovered and both
  routes are too large to repair in one round, finish one coherent route
  submodule and leave the exact continuation point for the other route.
* Do not add new dtype/LMUL clones, unrelated memory route families, segment2
  update expansion, frontend work, source-front-door routes, descriptor-driven
  compute, Common EmitC RVV semantics, dashboard/report-only work, or broad
  smoke matrices.
* Runtime-scalar `rhs_scalar` binding, splat leaf, computed mask facts,
  inactive-lane policy, segment2 field mapping, pointer behavior, dtype/config,
  ABI/header/prototype binding, and runtime AVL/VL must remain provider-owned
  and target-validated mirrors after provider route construction.
* If current dry-run, lit, target, or generated-bundle evidence exposes a stale
  or under-validated production seam, repair that seam in production C++/MLIR
  code and add focused fail-closed coverage before claiming executable support.
* If the current production path is already valid, this round may be a
  no-source-change evidence closeout, but it must include current HEAD focused
  checks and non-dry-run `ssh rvv` evidence for explicit and pre-realized
  runtime-scalar-cmp masked segment2 load/store bundles.

## Acceptance Criteria

* [x] Current task context is valid, started, and records this bounded PRD.
* [x] Focused production inspection confirms whether a source change is needed;
      if not, the no-source-change justification identifies the existing
      provider/target/script seams that satisfy the PRD.
* [x] Generated-bundle dry-runs pass for explicit and pre-realized
      `runtime_scalar_cmp_masked_segment2_load_unit_store` and
      `runtime_scalar_cmp_masked_segment2_store_unit_load`.
* [x] Non-dry-run generated bundle execution on `ssh rvv` passes for explicit
      and pre-realized runtime-scalar-cmp masked segment2 load/store, with
      active/inactive lanes, runtime scalar compare, field distinction, source
      preservation, inactive/passthrough preservation, runtime counts, and tail
      preservation covered by the harness evidence.
* [x] Focused target/RVV lit coverage for runtime-scalar-cmp masked segment2
      remains passing.
* [x] `build/bin/tianchenrv-rvv-extension-plugin-test` passes.
* [x] `build/bin/tianchenrv-target-artifact-export-test` passes.
* [x] If source changes are made, focused fail-closed evidence covers at least
      one stale executable-boundary fact such as rhs_scalar ABI role/order,
      stale splat materialization leaf, computed mask source, inactive-lane
      contract, segment2 field mapping, header/prototype binding, or route
      operand binding.
* [x] Bounded old-authority scan over touched files and added diff lines finds
      no new positive legacy `RVVI32M1`, `rvv-i32m1`, `tcrv_rvv.i32_*`,
      `!tcrv_rvv.i32m*`, descriptor/direct-C, source-front-door, or mirror-only
      route authority.
* [x] `git diff --check`, `git diff --cached --check`, and final
      `git status --short` are clean or explicitly reported.

## Definition Of Done

* The runtime-scalar-cmp masked segment2 load/store artifact ABI boundary is
  either repaired and verified, or explicitly closed as current-production
  valid with current HEAD dry-run, lit, C++ test, target artifact, and `ssh rvv`
  runtime evidence.
* Trellis task status, workspace journal, archive state, and one coherent
  commit are completed if all checks pass.

## Out Of Scope

* No broad memory matrix.
* No segment2 update expansion unless it is a direct blocker for load/store.
* No dtype/LMUL clone batch.
* No unrelated scalar-broadcast, MAcc, reduction, product-dequant, conversion,
  compare/select, mask route, or performance tuning work.
* No high-level Linalg/Vector/StableHLO frontend.
* No per-Linalg route authority.
* No source-front-door positive route.
* No Common EmitC invention of RVV semantics.
* No report-only closeout unless current executable evidence is refreshed.

## Technical Notes

* Specs read:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`, and
  `.trellis/spec/testing/mlir-testing-contract.md`.
* Archived task read:
  `.trellis/tasks/archive/2026-06/06-08-06-08-stage2-rvv-runtime-scalar-computed-mask-memory-provider-contract-owner/`.
* Workspace journal entries read:
  `.trellis/workspace/codex/journal-26.md` sessions 556, 560, 561, and 562.
* Source/test files inspected before implementation:
  `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCComputedMaskMemoryRouteFamilyPlanOwners.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCStatementPlanOwners.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`,
  `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`,
  `scripts/rvv_generated_bundle_abi_e2e.py`,
  the four runtime-scalar-cmp masked segment2 generated-bundle dry-run tests,
  the four target RVV explicit/pre-realized load/store fixtures, and
  `test/Plugin/RVVExtensionPluginTest.cpp`.

## Results

* No production source change was required. Current HEAD already routes the
  runtime-scalar-cmp masked segment2 load/store seam through selected typed or
  realized `tcrv_rvv` facts, RVV-owned computed-mask memory and segment2
  provider plans, the shared runtime-scalar splat provider contract, migrated
  segment2 statement-plan owner selection, `TCRVEmitCLowerableRoute`, target
  rebuilt-route validation, and generated bundle ABI checks.
* The no-source-change justification is concrete:
  `verifyRVVSelectedBodySegment2MemoryRouteProviderFacts(...)` checks
  operation-specific route operand binding, ABI roles/order, provider mirrors,
  route-control/runtime AVL/VL, statement-plan owner readiness, and the shared
  `RVVRuntimeScalarComputedMaskMemorySplatProviderContract` for runtime-scalar
  segment2 load/store before route construction.
* Target validation rebuilds and validates the segment2 route and checks
  runtime-scalar statement sequences: lhs vector load, `rhs_scalar` splat,
  compare/mask construction, field payload or passthrough loads, tuple
  create/extract, masked segment2 load/store, interleaved pointer expressions,
  runtime `n` loop bound, header/prototype ABI order, route operand binding,
  provider-supported mirror, and C type mapping.
* Current HEAD generated-bundle dry-runs passed for explicit and pre-realized
  runtime-scalar-cmp masked segment2 load/store.
* Current HEAD non-dry-run generated bundles compiled and ran on `ssh rvv`:
  explicit load, pre-realized load, explicit store, and pre-realized store all
  reported `rvv_generated_bundle_abi_e2e: success`, `remote_compile_succeeded:
  true`, `remote_run_succeeded: true`, `dry_run: false`, `ssh_evidence: true`,
  and PASS summaries.
* Load evidence covered counts `0,1,16,17,257`, patterns `0,1`, active and
  inactive mask lanes, runtime scalar compare, field distinction, source
  preservation, old-field passthrough preservation, and tail preservation.
* Store evidence covered explicit counts `0,1,16,17,257` and pre-realized
  counts `0,1,7,16,23,257`, patterns `0,1`, active and inactive mask lanes,
  runtime scalar compare, field distinction, source preservation, inactive
  interleaved destination preservation, and tail preservation.
* Checks run:
  `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`,
  `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`,
  `python3 ./.trellis/scripts/task.py validate <task>`,
  four direct generated-bundle dry-runs using explicit build tool paths,
  `/usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter
  'runtime-scalar-cmp-masked-segment2'` from `build/test`,
  `cmake --build build --target tianchenrv-rvv-extension-plugin-test`,
  `build/bin/tianchenrv-rvv-extension-plugin-test`,
  `cmake --build build --target tianchenrv-target-artifact-export-test`,
  `build/bin/tianchenrv-target-artifact-export-test`, four non-dry-run
  `ssh rvv` generated-bundle runs, evidence JSON spot checks, bounded
  old-authority scans, diff checks, task validation, and final git status.
* Self-repair performed:
  the first dry-run attempt failed because `tcrv-opt` and `llvm-readobj` were
  not on PATH. The rerun used `build/bin/tcrv-opt`,
  `build/bin/tcrv-translate`, and `/usr/bin/llvm-readobj-20`, after which all
  dry-run and non-dry-run evidence passed.
* No new fail-closed source test was needed because no production seam failed;
  the prior provider-contract fail-closed coverage remains the targeted stale
  splat-boundary regression, while this round refreshed executable artifact ABI
  behavior after that contract landed.
