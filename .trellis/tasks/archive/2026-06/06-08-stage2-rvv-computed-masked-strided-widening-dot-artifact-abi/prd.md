# Stage2 RVV computed-masked strided widening dot artifact ABI

## Goal

Complete one bounded production workflow submodule: the existing RVV
computed-masked strided-input widening dot-reduce add selected-body route must
line up from typed `tcrv_rvv` body facts through RVV plugin-owned stride,
computed-mask, inactive-lane, widening product, reduction, accumulator, and ABI
route validation, common EmitC materialization, target artifact export,
generated bundle ABI, and real `ssh rvv` correctness evidence. If current
production code already has the executable path, this round closes the
non-dry-run evidence gap and records the exact no-source-change justification.
If inspection finds a stale or under-validated executable boundary, this round
repairs only that computed-masked strided-input widening dot-reduce artifact/ABI
seam.

## What I Already Know

* No `.trellis/.current-task` existed at session start; this task was created
  from the Hermes Direction Brief.
* Repository inspection started from clean `main` at commit `330cd245 rvv:
  archive strided widening dot evidence`.
* The previous archived task closed the base strided-input widening dot-reduce
  executable evidence gap on `ssh rvv`, including ABI order
  `lhs,rhs,acc,out,n,lhs_stride,rhs_stride`, stride pairs `2:3` and `3:2`,
  counts `0,1,16,17,257`, signed widened dot, seed contribution,
  skipped-source ignoring, scalar-output-only behavior, and tail preservation.
* The next bottleneck is the computed-masked strided-input seam, because it
  composes stride/runtime-parameter binding with computed predicate and inactive
  lane semantics inside the same widening dot/reduce route family.
* Existing files and tests already name explicit/pre-realized computed-masked
  strided-input widening dot-reduce add paths, including dry-run and
  direct-pre-realized fail-closed tests. Inspection found the production
  owner/provider/target-validation seam already consumes the combined route
  facts, but the generated runtime harness did not snapshot compare/source/acc
  buffers before execution. This round repairs that evidence gap and proves the
  pre-realized generated bundle on `ssh rvv`.

## Requirements

* Keep compiler implementation in C++ / MLIR / LLVM / TableGen / CMake / lit /
  FileCheck. Python remains support tooling only.
* Stay on the computed-masked strided-input widening dot-reduce add
  selected-body route family. Do not broaden to an unrelated dot-reduce matrix,
  dtype/LMUL clone batch, MAcc/product-reduce/dequant/clamp route, high-level
  frontend, or performance tuning database.
* Preserve authority placement: typed `tcrv_rvv` body/config/runtime facts and
  RVV plugin-owned selected-body realization, statement owners, route provider,
  and target validation define route support; common EmitC/export only
  materializes and packages provider-built facts.
* Treat route ids, artifact names, emission-plan metadata, status/result fields,
  helper names, test names, and generated evidence files as mirrors only.
* Validate or prove fail-closed handling for stale/missing executable-boundary
  facts, especially stride ABI binding, computed mask binding, inactive-lane
  policy, widened product/reduction type, accumulator seed/result roles,
  scalar result type, dtype/SEW/LMUL/config/policy, header/prototype binding,
  route-family validation contract, generated C type mapping, ABI order,
  runtime AVL/VL, and statement facts.
* Positive executable evidence must run generated artifacts on real `ssh rvv`
  before claiming runtime correctness.
* The generated harness/evidence must distinguish combined semantics from base
  strided and computed-mask-only behavior: at minimum it must cover meaningful
  counts, at least two stride-pair patterns, computed mask patterns with active
  and inactive lanes, signed widened product/reduction, accumulator seed
  preservation on active lanes, inactive accumulator preservation,
  skipped-source ignoring, scalar-output-only behavior, and tail preservation.
* If the executable path is currently dry-run-only or under-validated, repair
  the production seam rather than closing as report-only evidence.

## Acceptance Criteria

* [x] Current code inspection proves the computed-masked strided-input widening
  dot-reduce route is owner/provider-consumed before route construction, or a
  focused source diff repairs only that executable artifact/ABI seam.
* [x] Positive generated-bundle evidence covers materialized selected boundary,
  emission plan, target artifact export, generated bundle compile, and `ssh rvv`
  correctness for computed-masked strided-input widening dot-reduce add.
* [x] Positive evidence records and checks stride ABI order, stride-source
  facts, computed mask binding, active/inactive lane policy, widened
  dot/reduction facts, accumulator seed/result preservation, scalar output
  behavior, dtype/config/policy, runtime AVL/VL, header/prototype binding, and
  source/accumulator/tail preservation.
* [x] A focused fail-closed check rejects at least one stale or missing
  executable-boundary fact, such as stale stride binding, stale computed mask
  binding, stale inactive-lane policy, stale widened product/reduction type,
  stale accumulator/result role, stale header/prototype binding, stale
  route-family validation contract, wrong generated C type, or wrong ABI value
  mapping.
* [x] `build/bin/tianchenrv-rvv-extension-plugin-test` passes.
* [x] `build/bin/tianchenrv-target-artifact-export-test` passes.
* [x] Relevant generated-bundle dry-run lit tests for explicit/pre-realized
  computed-masked strided-input widening dot-reduce and direct pre-realized
  fail-closed behavior pass.
* [x] Bounded old-authority scan over touched files and added diff lines shows
  no new positive legacy `i32m1` route authority, source-front-door route
  authority, descriptor-driven compute, direct-C/source export authority, or
  mirror-only acceptance.
* [x] `git diff --check`, `git diff --cached --check`, and final
  `git status --short` are clean/reported.
* [x] Task status, workspace journal, archive, and one coherent commit are
  completed if the task finishes.

## Implementation Notes

* Production C++ inspection found the combined seam already routes through
  `ComputedMaskStridedInputWideningDotReduceAdd`, `masked_strided_wdot.v1`,
  `ComputedMaskStridedInputWideningDotReduce`, computed-mask predicate facts,
  strided lhs/rhs source loads, accumulator/result layouts, exact runtime ABI
  order `cmp_lhs,cmp_rhs,lhs,rhs,acc,out,n,lhs_stride,rhs_stride`, and target
  artifact mirror validation. Existing C++ tests reject stale combined operand
  binding, stale dot-lhs role, stale accumulator/output roles, stale stride
  roles, stale mask source, stale predicate, stale target leaf profile, and
  stale header facts.
* The generated bundle harness for
  `computed_masked_strided_input_widening_dot_reduce_add` now snapshots
  `cmp_lhs`, `cmp_rhs`, `lhs`, `rhs`, and `acc`, computes expected results from
  snapshots, checks the buffers after the generated call, and prints
  `source_preserved accumulator_preserved tail_preserved` only after the
  preservation checks pass.
* The explicit and pre-realized dry-run script tests now check the snapshot
  buffers, mutation diagnostics, and preservation success markers in the
  generated harness.
* `ssh rvv` evidence passed for the pre-realized generated bundle with counts
  `0,1,16,17,257`, stride pairs `2:3` and `3:2`, mask/input patterns `0` and
  `1`, signed widened dot, seed contribution, inactive-lane skipping,
  skipped-source ignoring, scalar-output-only behavior, and source/accumulator
  and tail preservation.

## Out Of Scope

* No broad dot-reduce or contraction matrix.
* No dtype/SEW/LMUL clone batch.
* No MAcc/product-reduce/dequant/clamp rework except bounded reference reading.
* No high-level Linalg/Vector/StableHLO frontend work.
* No source-front-door positive route.
* No per-Linalg route authority.
* No performance tuning database or dashboard.
* No artifact/report-only completion when the production seam is stale.
* No common EmitC RVV semantic selection.
* No route-id, metadata, helper-name, status, result, or mirror-field
  acceptance authority.

## Technical Notes

* Specs queued for this task:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  `.trellis/spec/testing/index.md`, and
  `.trellis/spec/testing/mlir-testing-contract.md`.
* Previous completed task:
  `.trellis/tasks/archive/2026-06/06-08-stage2-rvv-strided-input-widening-dot-reduce-executable-artifact-abi-boundary/`.
* Primary code paths to inspect:
  `include/TianChenRV/Plugin/RVV/RVVEmitCContractionRouteFamilyPlanOwners.h`,
  `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`,
  `lib/Plugin/RVV/RVVContractionSelectedBodyRealizationOwner.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCContractionRouteFamilyPlanOwners.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCCompareSelectStatementPlanOwners.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCReductionAccumulationStatementPlanOwners.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCStatementPlanOwners.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`, and
  `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`.
* Primary evidence paths to inspect:
  `scripts/rvv_generated_bundle_abi_e2e.py`,
  `test/Scripts/rvv-generated-bundle-abi-e2e-explicit-computed-masked-strided-input-widening-dot-reduce-add-dry-run.test`,
  `test/Scripts/rvv-generated-bundle-abi-e2e-pre-realized-computed-masked-strided-input-widening-dot-reduce-add-dry-run.test`,
  `test/Scripts/rvv-generated-bundle-abi-e2e-direct-pre-realized-computed-masked-strided-input-widening-dot-reduce-add-fail-closed.test`,
  `test/Target/RVV/explicit-selected-body-artifact-computed-masked-strided-input-widening-dot-reduce-add.mlir`, and
  `test/Target/RVV/pre-realized-selected-body-artifact-computed-masked-strided-input-widening-dot-reduce-add.mlir`.
