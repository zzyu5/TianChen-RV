# Stage2 RVV computed-masked strided dot source/accumulator acceptance boundary

## Goal

Close one bounded production workflow submodule: the existing
`computed_masked_strided_input_widening_dot_reduce_add` route family must make
source-before, skipped-source, accumulator seed/carry, inactive-lane,
scalar-output-only, tail-preservation, stride ABI, dtype/config/runtime, and
header/prototype binding an RVV provider-owned and target-consumed acceptance
contract before target artifact acceptance.

This task is not another runtime evidence-only closeout. If the production
route already has an equivalent named contract, this round must prove it from
provider/target code and add or repair the precise missing stale-boundary
fail-closed check. If the named contract is missing or under-validated, this
round must add the bounded provider fact, route validation contract, target
consumer check, positive mirror evidence, and focused fail-closed coverage for
this route family.

## What I Already Know

* No `.trellis/.current-task` existed at session start; this task was created
  from the Hermes Direction Brief.
* Repository inspection started from a clean `main` at commit `b4336c56 rvv:
  enforce runtime scalar indexed write boundary`.
* The previous completed task promoted runtime-scalar indexed write-side
  source-before/destination-before inactive-tail behavior into provider route
  facts, route-family planning, target validation, generated bundle evidence,
  focused fail-closed tests, and refreshed `ssh rvv` evidence.
* The archived computed-masked strided widening dot artifact/ABI task proved
  executable behavior on `ssh rvv` and fixed harness snapshot evidence, but
  this round's scope is stricter: source/accumulator/scalar-result semantics
  must be accepted or rejected by the provider/target production seam, not
  remain only in harness/oracle evidence.
* Relevant specs require typed `tcrv_rvv` body/config/runtime facts as route
  authority, RVV plugin-owned route validation, common EmitC neutrality,
  target artifact consume-only validation, and real `ssh rvv` evidence for any
  refreshed runtime correctness claim.

## Requirements

* Stay on the existing computed-masked strided-input widening dot-reduce add
  family. Do not choose another route family.
* Keep compiler implementation in C++ / MLIR / LLVM / TableGen / CMake / lit /
  FileCheck. Python may only support generated-bundle evidence or self-tests.
* Acceptance must be provider-owned and target-consumed before object/header or
  bundle artifact acceptance.
* The contract must cover source-before snapshots, skipped-source ignoring,
  explicit `lhs_stride` / `rhs_stride` ABI roles, accumulator seed/carry
  semantics, inactive-lane zeroing/preservation behavior, scalar-output-only
  result behavior, tail preservation, dtype/SEW/LMUL/config/policy, runtime
  AVL/VL, route operand binding, and header/prototype participation.
* Common EmitC/export must remain neutral and consume provider-built route
  payload only; it must not infer RVV dot, mask, stride, accumulator, or result
  semantics.
* Route ids, artifact names, emission-plan metadata, helper names, test names,
  status/result fields, exact intrinsic spelling, and generated evidence files
  are mirrors only and must not become acceptance authority.
* If runtime correctness evidence is newly refreshed, it must use real
  `ssh rvv`; otherwise the final report must state that prior runtime evidence
  was not re-claimed.

## Acceptance Criteria

* [ ] Current production code inspection proves a named equivalent provider
  route fact and target validation contract already carries the strided
  source/accumulator/scalar-result contract, or source changes add the missing
  bounded contract.
* [ ] Positive provider route facts and target artifact validation consume or
  expose the contract for explicit and/or pre-realized
  `computed_masked_strided_input_widening_dot_reduce_add`.
* [ ] Target validation fails closed for at least one stale/missing boundary
  fact such as stale stride-source contract, stale source-before contract,
  stale accumulator seed/carry contract, stale skipped-source contract, stale
  inactive-lane policy, stale scalar-output result contract, stale ABI order,
  stale route operand binding, or stale header/prototype binding.
* [ ] Positive generated-bundle dry-run or target fixture evidence mirrors the
  provider/target contract without treating the mirror as route authority.
* [ ] `build/bin/tianchenrv-rvv-extension-plugin-test` passes.
* [ ] `build/bin/tianchenrv-target-artifact-export-test` passes.
* [ ] Relevant computed-masked strided widening dot lit tests pass.
* [ ] If Python evidence tooling changes, the focused generated-bundle script
  tests and self-tests pass.
* [ ] Bounded old-authority scan over touched files and added diff lines shows
  no new positive legacy `i32m1`, source-front-door, descriptor/direct-C, or
  mirror-only acceptance authority.
* [ ] `git diff --check`, `git diff --cached --check`, and final
  `git status --short` are clean/reported.
* [ ] Trellis task status, workspace journal, archive, and one coherent commit
  are completed if the task finishes.

## Out Of Scope

* No broad dot-reduce or contraction matrix.
* No dtype/LMUL clone batch.
* No indexed-memory rework except bounded reference reading.
* No segment2, Gearbox, MAcc, product-reduce, dequant, or clamp expansion.
* No high-level Linalg/Vector/StableHLO frontend.
* No source-front-door positive route.
* No per-Linalg route authority.
* No performance database or dashboard.
* No report-only evidence closeout when the production seam is stale.
* No common EmitC invention of RVV semantics.

## Technical Notes

* Specs read or queued:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/index.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  `.trellis/spec/testing/index.md`,
  `.trellis/spec/testing/mlir-testing-contract.md`,
  `.trellis/spec/variant-pipeline/index.md`, and shared guides.
* Previous archived tasks to use as context only:
  `.trellis/tasks/archive/2026-06/06-08-stage2-rvv-runtime-scalar-indexed-write-boundary/`
  and
  `.trellis/tasks/archive/2026-06/06-08-stage2-rvv-computed-masked-strided-widening-dot-artifact-abi/`.
* Primary code/evidence paths named by the brief:
  `include/TianChenRV/Plugin/RVV/RVVEmitCContractionRouteFamilyPlanOwners.h`,
  `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`,
  `lib/Plugin/RVV/RVVContractionSelectedBodyRealizationOwner.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCContractionRouteFamilyPlanOwners.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCCompareSelectStatementPlanOwners.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCReductionAccumulationStatementPlanOwners.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`,
  `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`,
  `test/Target/TargetArtifactExportTest.cpp`,
  `scripts/rvv_generated_bundle_abi_e2e.py`,
  the explicit/pre-realized computed-masked strided widening dot dry-run tests,
  and the explicit/pre-realized target artifact MLIR fixtures.
