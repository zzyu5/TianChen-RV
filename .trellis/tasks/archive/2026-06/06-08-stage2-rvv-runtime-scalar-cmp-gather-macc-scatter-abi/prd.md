# Stage2 RVV runtime scalar cmp masked indexed gather MAcc scatter artifact ABI

## Goal

Complete one bounded production workflow submodule: the existing RVV
runtime-scalar compare masked indexed gather-MAcc-scatter selected-body route
must line up from typed `tcrv_rvv` body facts through RVV plugin-owned composite
realization, computed-mask/indexed-memory/MAcc/scatter route validation, common
EmitC materialization, target artifact export, generated bundle ABI, and real
`ssh rvv` correctness evidence. If current production code already has the
executable path, this round closes the non-dry-run evidence gap and records the
exact no-source-change justification. If inspection finds a stale or
under-validated executable boundary, this round repairs only that composite
artifact/ABI seam.

## What I Already Know

* No `.trellis/.current-task` existed at session start; this task was created
  from the Hermes Direction Brief.
* Repository inspection started from clean `main` at commit `0428621d rvv:
  prove computed masked strided dot evidence`.
* The previous archived task closed the computed-masked strided-input widening
  dot-reduce generated-bundle evidence gap by proving `ssh rvv` execution with
  compare/source/accumulator snapshots and `source_preserved`,
  `accumulator_preserved`, and `tail_preserved` checks.
* The next bounded bottleneck is the runtime-scalar compare masked indexed
  gather-MAcc-scatter seam because it composes runtime scalar predicate
  construction, indexed gather, masked MAcc, indexed scatter, inactive-lane
  preservation, ABI order, and generated bundle execution in one selected-body
  workflow.
* The relevant RVV plugin spec already defines a dedicated
  `Runtime-Scalar Indexed Gather-MAcc-Scatter Route Contract` with ABI order
  `cmp_lhs,rhs_scalar,gather_src,payload,acc,index,dst,n`, compare predicate
  `sle`, shared compare-produced mask, shared index vector, gather passthrough,
  MAcc accumulator roles, masked scatter result role, provider facts, and
  fail-closed validation matrix.
* Existing code and tests already name explicit and pre-realized
  `runtime_scalar_cmp_masked_indexed_gather_macc_scatter` paths, including
  materialized selected-body fixtures, target-header stale-provider checks,
  generated-bundle dry-run tests, and Python harness branches. This round must
  inspect whether those branches already provide truthful executable evidence
  or still need source/harness hardening.

## Requirements

* Keep compiler implementation in C++ / MLIR / LLVM / TableGen / CMake / lit /
  FileCheck. Python remains support tooling only.
* Stay on the runtime-scalar compare masked indexed gather-MAcc-scatter
  selected-body route family. Do not broaden to a gather/scatter matrix,
  dtype/LMUL clone batch, standalone gather/scatter detour, unrelated dot
  rework, high-level frontend, performance tuning database, or dashboard.
* Preserve authority placement: typed `tcrv_rvv` body/config/runtime facts and
  RVV plugin-owned composite realization, route family plans, statement plans,
  route provider, and target validation define route support; common EmitC and
  target export only materialize and package provider-built facts.
* Treat route ids, artifact names, emission-plan metadata, status/result
  fields, helper names, test names, and generated evidence files as mirrors
  only.
* Validate or prove fail-closed handling for stale/missing executable-boundary
  facts, especially runtime scalar binding, mask construction, index mapping,
  gather/scatter roles, MAcc vector/addend/accumulator roles, inactive-lane
  policy, source/accumulator/tail preservation, dtype/SEW/LMUL/config/policy,
  header/prototype binding, ABI order, runtime AVL/VL, and statement facts.
* Positive executable evidence must run generated artifacts on real `ssh rvv`
  before claiming runtime correctness.
* The generated harness/evidence must distinguish the composite semantics from
  standalone gather, standalone scatter, and MAcc-only behavior: at minimum it
  must cover meaningful counts, two runtime scalar thresholds, active and
  inactive lanes, noncontiguous unique indices, signed products, gather source
  preservation, payload and accumulator preservation, masked inactive-lane
  destination preservation, scatter destination correctness, untouched lanes,
  tail preservation, and runtime `n`/AVL behavior.
* If the executable path is currently dry-run-only or under-validated, repair
  the production seam rather than closing as report-only evidence.

## Acceptance Criteria

* [x] Current code inspection proves the runtime-scalar compare masked indexed
  gather-MAcc-scatter route is realized and provider-consumed before route
  construction, or a focused source diff repairs only that executable
  artifact/ABI seam.
* [x] Positive generated-bundle evidence covers materialized selected boundary,
  emission plan, target artifact export, generated bundle compile, and `ssh rvv`
  correctness for runtime-scalar compare masked indexed gather-MAcc-scatter.
* [x] Positive evidence records and checks ABI order, runtime scalar predicate
  effects, shared compare-produced mask, index mapping, gather source roles,
  MAcc operand/accumulator roles, scatter destination role, inactive-lane
  policy, dtype/config/policy, runtime AVL/VL, header/prototype binding,
  source/payload/accumulator preservation, untouched destination lanes, and
  tail preservation.
* [x] A focused fail-closed check rejects at least one stale or missing
  executable-boundary fact, such as stale provider mirror, stale runtime scalar
  binding, stale index mapping, stale gather/scatter role, stale accumulator
  role, stale mask policy, stale header/prototype binding, stale route-family
  validation contract, wrong generated C type, or wrong ABI value mapping.
* [x] `build/bin/tianchenrv-rvv-extension-plugin-test` passes.
* [x] `build/bin/tianchenrv-target-artifact-export-test` passes.
* [x] Relevant generated-bundle dry-run tests for explicit/pre-realized
  runtime-scalar compare masked indexed gather-MAcc-scatter pass.
* [x] `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test` passes if
  the harness changes.
* [x] Bounded old-authority scan over touched files and added diff lines shows
  no new positive legacy `i32m1` route authority, source-front-door route
  authority, descriptor-driven compute, direct-C/source export authority, or
  mirror-only acceptance.
* [x] `git diff --check`, `git diff --cached --check`, and final
  `git status --short` are clean/reported.
* [ ] Task status, workspace journal, archive, and one coherent commit are
  completed if the task finishes.

## Implementation Notes

* Production C++ inspection found the composite seam already routes through the
  named RVV plugin-local composite realization owner, provider-owned
  `RuntimeScalarComputedMaskIndexedGatherMAccScatter` route facts, target
  artifact validation, and stale mirror checks. The owner requires exactly one
  runtime-scalar indexed gather body, one runtime-scalar MAcc body, and one
  runtime-scalar indexed scatter body; it validates shared runtime scalar, mask,
  runtime n/AVL, index, destination, dtype/config/policy, inactive-lane policy,
  MAcc accumulator/result layout, and ABI roles before realization. Target
  artifact validation requires provider-owned composite route-family plan,
  typed compute chain, legal resource selection, runtime ABI order, headers,
  type mappings, ABI mappings, and statement plan shape before export.
* The generated bundle harness for
  `runtime_scalar_cmp_masked_indexed_gather_macc_scatter` now computes product,
  expected result, and mismatch diagnostics from pre-call snapshots:
  `gather_src_before`, `payload_before`, and `acc_before`. It still separately
  compares live buffers against those snapshots and prints
  `source_preserved payload_acc_preserved tail_preserved` only after the
  preservation checks pass.
* The explicit and pre-realized generated-bundle dry-run tests now assert the
  snapshot-backed expected expression in the generated C harness.
* The `--rhs-scalar` help text now names this composite op, matching the
  existing implementation that runs multiple runtime scalar thresholds for the
  same generated artifact.
* `ssh rvv` evidence passed for the pre-realized generated bundle at
  `build/trellis-validation/stage2-runtime-scalar-cmp-indexed-gather-macc-scatter/pre-composite-gms/runtime_scalar_cmp_masked_indexed_gather_macc_scatter/evidence.json`.
  Remote compile used `clang -O2 -march=rv64gcv -mabi=lp64d` on `riscv64`.
  Remote runtime passed counts `0,1,16,17,257`, RHS scalars `-37,91`, patterns
  `0,1`, active/inactive lane mixes, inactive destination preservation,
  noncontiguous unique indices, signed products, source preservation,
  payload/accumulator preservation, and tail preservation.
* The reusable evidence rule learned in this round was recorded in
  `.trellis/spec/testing/mlir-testing-contract.md`: when preservation snapshots
  are part of generated-bundle evidence, expected-value computation and
  mismatch diagnostics must use the pre-call snapshots, not post-call live
  buffers.

## Out Of Scope

* No broad gather/scatter route matrix.
* No dtype/SEW/LMUL clone batch.
* No standalone indexed gather or scatter detour unless it is the named blocker
  for this composite seam.
* No additional dot-reduce, strided-dot, or MAcc-only rework except bounded
  reference reading.
* No high-level Linalg/Vector/StableHLO frontend work.
* No source-front-door positive route.
* No per-Linalg route authority.
* No performance tuning database or dashboard.
* No artifact/report-only completion when the production seam is stale.
* No common EmitC RVV semantic selection.
* No route-id, metadata, helper-name, status, result, or mirror-field
  acceptance authority.

## Technical Notes

* Specs for this task:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  `.trellis/spec/testing/index.md`, and
  `.trellis/spec/testing/mlir-testing-contract.md`.
* Previous completed task:
  `.trellis/tasks/archive/2026-06/06-08-stage2-rvv-computed-masked-strided-widening-dot-artifact-abi/`.
* Primary code paths to inspect:
  `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`,
  `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`,
  `lib/Plugin/RVV/RVVCompositeGatherMAccScatterSelectedBodyRealizationOwner.cpp`,
  `lib/Plugin/RVV/RVVComputedMaskMemorySelectedBodyRealizationOwner.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCMAccRouteFamilyPlanOwners.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCBaseMemoryRouteFamilyPlanOwners.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCComputedMaskMemoryRouteFamilyPlanOwners.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCMemoryStatementPlanOwners.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`, and
  `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`.
* Primary evidence paths to inspect:
  `scripts/rvv_generated_bundle_abi_e2e.py`,
  `test/Scripts/rvv-generated-bundle-abi-e2e-runtime-scalar-cmp-masked-indexed-gather-macc-scatter-dry-run.test`,
  `test/Target/RVV/explicit-selected-body-artifact-runtime-scalar-cmp-masked-indexed-gather-macc-scatter.mlir`, and
  `test/Target/RVV/pre-realized-selected-body-artifact-runtime-scalar-cmp-masked-indexed-gather-macc-scatter.mlir`.
