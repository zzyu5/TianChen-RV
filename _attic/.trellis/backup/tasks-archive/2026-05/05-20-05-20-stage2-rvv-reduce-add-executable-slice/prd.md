# Stage2 RVV reduce-add executable slice

## Goal

Close the Hermes `Stage2 RVV reduction/accumulation executable slice` brief
against current HEAD truth. The bounded module is one signed i32 / SEW32 /
LMUL m1 / unit-stride `reduce_add` path whose selected RVV boundary carries
explicit input, accumulator seed, result/output, policy, runtime `n`/AVL, and
ABI roles into typed `tcrv_rvv` reduction structure; the RVV plugin then owns
selected-body realization, route planning, target artifact metadata, and
generated-bundle correctness evidence.

This is not a broad reduction framework. If current HEAD already satisfies the
bounded slice, this task completes by validating and documenting that fact. If
focused checks reveal a real gap, the implementation scope is limited to that
gap.

## Direction Source

- Hermes Direction Brief: `Stage2 RVV reduction/accumulation executable slice`.
- Module owner: RVV plugin-owned reduction and accumulator-layout route
  derivation for one bounded i32 unit-stride reduce-add path.
- Required evidence: focused compiler/script checks, fail-closed negative
  coverage, generated-bundle dry-run, real `ssh rvv` correctness for
  representative tail-sensitive counts such as `7,16,23`, active-authority
  scan, `git diff --check`, and clean final worktree.

## Current Repository Facts

- Repo root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean.
- Initial HEAD: `fbf3a162 rvv: add widening conversion executable slice`.
- No `.trellis/.current-task` existed, so this task was created from the
  Hermes brief before source edits.
- Recent commits already completed Stage2 slices for scalar broadcast,
  compare/select executable ABI, i64 SEW64 route policy, LMUL m2 route policy,
  tail/mask policy, and widening conversion.
- Archived reduction tasks show three earlier reduce-related milestones:
  generic `tcrv_rvv.reduce` route skeleton, explicit reduce-add executable
  closure, and pre-realized reduce-add selected-body realization.
- Current HEAD code inspection confirms those reduce surfaces are still active:
  `tcrv_rvv.typed_reduce_pre_realized_body`,
  `tcrv_rvv.reduce`, RVV selected-body realization, RVVEmitCRoutePlanning
  reduction layout mirrors, target artifact tests, generated-bundle dry-run
  tests, and `scripts/rvv_generated_bundle_abi_e2e.py --op-kind reduce_add`.
- Stage1/Stage2 guardrails remain active. This task must not reintroduce
  positive `RVVI32M1`, `rvv-i32m1`, finite positive `tcrv_rvv.i32_*`,
  `!tcrv_rvv.i32m*`, source-front-door/source-seed, descriptor/direct-C/
  source-export, status/artifact-name route authority, or common/export RVV
  semantic authority.

## Module Boundary

In scope:

- One signed i32 / SEW32 / LMUL m1 / unit-stride `reduce_add` selected-body
  path.
- Explicit runtime ABI roles for `lhs` vector input, `rhs` accumulator seed,
  `out` result/output, and `n` runtime count / AVL source.
- Explicit accumulator role/layout:
  `rhs-input-buffer` and `rhs-vector-seed-lane0-per-vl-chunk`.
- Explicit result layout:
  `store-reduction-lane0-to-output-chunk-base`.
- Explicit typed `tcrv_rvv` structure: `setvl`, `with_vl`, input load,
  accumulator load, `reduce {kind = "add"}`, and store.
- RVV plugin-owned route planning/provider derivation for runtime ABI order,
  scalar/vector C types, header metadata, intrinsic leaves, reduction layout
  mirrors, and target artifact handoff.
- Focused validation of existing or repaired behavior, including fresh
  `ssh rvv` evidence if executable correctness is claimed in this round.

Out of scope:

- Broad reduction matrix, floating point reductions, min/max/and/or batches,
  segmented reductions, contractions/matmul, high-level frontend lowering,
  source-front-door positive routes, one-intrinsic wrapper dialects, dtype/LMUL
  clone batches, dashboards, performance claims, or future plugin work.

## Requirements

1. `tcrv.exec` remains an execution envelope and ABI/runtime boundary only; it
   must not own reduction compute, dtype/config, accumulator layout, result
   layout, or intrinsic spelling.
2. The selected/pre-realized body or typed body must structurally carry
   reduction kind, input dtype/config, accumulator seed role/layout,
   result/output role/layout, memory form, policy, runtime `n`/AVL, and ABI
   roles.
3. RVV selected-body realization may only materialize legal generic typed
   reduction structure from explicit selected facts. It must not infer compute
   semantics from helper names, route ids, ABI strings, artifact names,
   descriptors, source-front-door metadata, exact intrinsic spelling, or C
   strings.
4. RVVEmitCRoutePlanning must derive route metadata and intrinsic leaves from
   typed body/config/runtime facts, and fail closed for unsupported reduction
   kind, missing accumulator/result layout, mismatched dtype/config, missing
   runtime AVL role, unsupported policy, wrong memory/result form, or
   incomplete reduction body.
5. Provider/materializer/target export must consume provider-built route
   payloads. Common EmitC/export may carry mirrors only and must not choose
   reduction semantics or RVV intrinsics.
6. Generated-bundle tooling may drive, inspect, and run artifacts only. It is
   not compiler route authority.
7. Runtime/correctness claims require fresh `ssh rvv` evidence in this task or
   must be reported as older archived evidence only.

## Acceptance Criteria

- [x] Task PRD, implement/check context, and metadata describe the bounded
      reduce-add executable slice and do not drift into a reduction framework.
- [x] Positive selected/pre-realized reduce-add body carries input,
      accumulator seed, result/output, SEW32, LMUL m1, agnostic policy,
      unit-stride memory form, runtime `n`/AVL, and ABI roles explicitly.
- [x] RVV selected-body realization produces `setvl`, `with_vl`, input load,
      accumulator load, `tcrv_rvv.reduce {kind = "add"}`, and store before
      provider route construction.
- [x] RVVEmitCRoutePlanning/provider derives runtime ABI, C types, reduction
      layout mirrors, header/artifact metadata, and intrinsic leaves from
      typed facts.
- [x] Negative fail-closed coverage exists for unsupported/missing operation,
      wrong memory form, missing/wrong accumulator role or layout, missing/
      wrong result role or layout, mismatched config/dtype, unsupported policy,
      missing runtime `n`/AVL, and mixed/incomplete typed body structure.
- [x] Generated-bundle dry-run for the pre-realized reduce-add path covers
      counts `7,16,23` and guards against descriptor/direct-C/source-export/
      source-front-door/finite `tcrv_rvv.i32_*` authority.
- [x] Real `ssh rvv` generated-bundle correctness evidence passes for
      `reduce_add` counts `7,16,23` if this round claims executable
      correctness.
- [x] Focused build/lit/C++/script checks pass for touched or validated RVV
      dialect/config, selected-body realization, route planning/provider,
      target artifact, and generated-bundle paths.
- [x] Active-authority scan confirms no positive legacy i32 route authority,
      source-front-door/source-seed authority, descriptor/direct-C/
      source-export authority, or common/export RVV semantic authority is
      reintroduced.
- [x] `git diff --check`, task validation, final task status update, archive,
      commit, and clean worktree checks pass if the task completes.

## Validation Plan

1. Start and validate the Trellis task.
2. Build focused targets:
   `tcrv-opt`, `tcrv-translate`, `tianchenrv-rvv-dialect-test`,
   `tianchenrv-rvv-extension-plugin-test`,
   `tianchenrv-construction-protocol-common-test`, and
   `tianchenrv-target-artifact-export-test`.
3. Run focused C++ tests for RVV dialect/plugin/construction/export.
4. Run focused lit from `build/test` for reduce-add selected-body
   realization, explicit/pre-realized artifact materialization, reduction
   negative coverage, generated-bundle dry-run, and relevant regression tests.
5. Run `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py` and
   `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`.
6. Run generated-bundle dry-run:
   `python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run --pre-realized-selected-body --artifact-root artifacts/tmp/rvv_generated_bundle_abi_e2e --run-id 20260520T-pre-realized-reduce-add-closure-dry --overwrite --op-kind reduce_add --runtime-count 7 --runtime-count 16 --runtime-count 23 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj`.
7. Run real `ssh rvv` evidence with the same selected path and counts if
   executable correctness is claimed.
8. Run active-authority scans over active RVV include/lib/script/test paths.
9. Run `git diff --check`.
10. Run `check-tianchenrv` if focused validation or source changes touch shared
    compiler behavior broadly enough to justify the full gate.

## Technical Notes

Relevant specs read:

- `.trellis/spec/index.md`
- `.trellis/spec/extension-plugins/index.md`
- `.trellis/spec/extension-plugins/rvv-plugin.md`
- `.trellis/spec/lowering-runtime/index.md`
- `.trellis/spec/lowering-runtime/emitc-route.md`
- `.trellis/spec/plugin-protocol/index.md`
- `.trellis/spec/core-dialect/index.md`
- `.trellis/spec/variant-pipeline/index.md`
- `.trellis/spec/testing/index.md`

Prior task context read:

- `.trellis/tasks/archive/2026-05/05-19-05-19-stage2-generic-rvv-reduction-accumulation-route/prd.md`
- `.trellis/tasks/archive/2026-05/05-19-stage2-generic-rvv-reduction-executable-closure/prd.md`
- `.trellis/tasks/archive/2026-05/05-19-stage2-rvv-selected-reduce-add-realization/prd.md`
- `.trellis/tasks/archive/2026-05/05-20-stage2-rvv-route-provider-planning-extraction/prd.md`
- `.trellis/tasks/archive/2026-05/05-20-stage2-rvv-vector-scalar-broadcast-executable-path/prd.md`
- `.trellis/tasks/archive/2026-05/05-20-stage2-rvv-tail-mask-policy-executable-slice/prd.md`
- `.trellis/tasks/archive/2026-05/05-20-stage2-rvv-widening-conversion/prd.md`

Initial code/test surfaces inspected:

- `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`
- `include/TianChenRV/Dialect/RVV/IR/RVVConfigContract.h`
- `lib/Dialect/RVV/IR/RVVConfigContract.cpp`
- `lib/Dialect/RVV/IR/RVVDialect.cpp`
- `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp`
- `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`
- `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
- `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
- `lib/Plugin/RVV/Construction/RVVConstructionProtocol.cpp`
- `scripts/rvv_generated_bundle_abi_e2e.py`
- `test/Target/RVV/pre-realized-selected-body-artifact-reduce-add.mlir`
- `test/Target/RVV/explicit-selected-body-artifact-reduce-add.mlir`
- `test/Transforms/LoweringBoundary/rvv-pre-realized-reduce-selected-body-negative.mlir`
- `test/Conversion/EmitC/rvv-generic-stage2-reduction-negative.mlir`
- `test/Scripts/rvv-generated-bundle-abi-e2e-pre-realized-reduce-add-dry-run.test`

## Definition Of Done

- The current HEAD or repaired code demonstrably satisfies the bounded
  reduce-add executable slice.
- Evidence clearly separates current fresh evidence from older archived
  evidence.
- The task remains open if fresh validation exposes an unresolved gap; otherwise
  the task is finished/archived with one coherent commit.

## Implementation Results

- No compiler source change was needed. Current HEAD already contains the
  bounded Stage2 reduce-add production path requested by the Hermes brief.
- Verified current active surfaces:
  `tcrv_rvv.typed_reduce_pre_realized_body`,
  `tcrv_rvv.reduce`, `RVVSelectedBodyRealization`,
  `RVVEmitCRoutePlanning`, `RVVEmitCRouteProvider`,
  target artifact emission, generated-bundle dry-run, and real RVV harness
  support for `--pre-realized-selected-body --op-kind reduce_add`.
- Verified the positive pre-realized selected body carries explicit input,
  accumulator seed, result/output, SEW32, LMUL m1, agnostic policy,
  unit-stride memory form, runtime `n`/AVL, accumulator role/layout, result
  layout, and ABI roles before realization.
- Verified selected-body realization consumes that pre-realized body into
  `setvl`, `with_vl`, input load, accumulator load,
  `tcrv_rvv.reduce {kind = "add"}`, and store before route planning.
- Verified route planning/provider derive the reduction route and metadata
  mirrors from typed facts, including:
  `tcrv_rvv.reduction_accumulator_layout =
  rhs-vector-seed-lane0-per-vl-chunk`,
  `tcrv_rvv.reduction_result_layout =
  store-reduction-lane0-to-output-chunk-base`, and
  `tcrv_rvv.reduction_store_vl = 1`.
- Verified the generated harness checks non-vacuous expected sums for counts
  `7,16,23`, including lane-0-per-dynamic-VL-chunk result placement and
  untouched non-result lanes.
- The active-authority scan introduced no new source/code high-risk matches in
  this task. Full-path matches are pre-existing deprecated parse-only
  inventory, negative fail-closed tests, source-front-door rejection code,
  script guardrails, or provider-derived intrinsic leaves after typed-body
  validation.

## Validation Results

- [OK] `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-20-05-20-stage2-rvv-reduce-add-executable-slice`
- [OK] `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-rvv-dialect-test tianchenrv-rvv-extension-plugin-test tianchenrv-construction-protocol-common-test tianchenrv-target-artifact-export-test -j2`
- [OK] `build/bin/tianchenrv-rvv-dialect-test`
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `build/bin/tianchenrv-construction-protocol-common-test`
- [OK] `build/bin/tianchenrv-target-artifact-export-test`
- [OK] `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
- [OK] `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- [OK] generated-bundle dry-run:
  `python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run --pre-realized-selected-body --artifact-root artifacts/tmp/rvv_generated_bundle_abi_e2e --run-id 20260520T-pre-realized-reduce-add-closure-dry --overwrite --op-kind reduce_add --runtime-count 7 --runtime-count 16 --runtime-count 23 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj`
- [OK] focused lit from `build/test`: 7/7 passed for pre-realized reduce-add
  artifact, explicit reduce-add artifact, reduce-add negative diagnostics,
  reduction materialization, generic RVV dataflow, and generated-bundle dry-run.
- [OK] real `ssh rvv` generated-bundle correctness:
  `python3 scripts/rvv_generated_bundle_abi_e2e.py --pre-realized-selected-body --artifact-root artifacts/tmp/rvv_generated_bundle_abi_e2e --run-id 20260520T-pre-realized-reduce-add-closure-ssh --overwrite --op-kind reduce_add --runtime-count 7 --runtime-count 16 --runtime-count 23 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj --ssh-target rvv --timeout 180 --connect-timeout 10`
- [OK] `ssh rvv` output:
  `PASS op=reduce_add counts=7,16,23`
- [OK] Evidence root:
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/20260520T-pre-realized-reduce-add-closure-ssh`
- [OK] `git diff --check`
- [OK] diff active-authority scan over active include/lib/script/test paths:
  no new high-risk RVV route-authority matches in source/code. The only
  diff-added high-risk strings are task PRD guardrail text.
- [OK] full active-authority scan classified remaining matches as pre-existing
  deprecated parse-only inventory, negative fail-closed tests, source-front-door
  rejection code, script guardrails, or provider-derived intrinsic leaves after
  typed-body validation.
- [OK] `cmake --build build --target check-tianchenrv -j2`: 193/193 passed.

## Spec Update Decision

No `.trellis/spec/` update was needed. The existing specs already require RVV
plugin-owned typed-body reduction authority, explicit accumulator/result layout,
common EmitC neutrality, fail-closed legacy/source/descriptor paths, and real
`ssh rvv` evidence for correctness claims. This task validated the existing
bounded implementation against those rules rather than adding a new durable
architecture rule.
