# Stage2 RVV runtime-scalar-cmp masked indexed scatter-store executable artifact ABI boundary

## Goal

Complete one bounded production workflow submodule: the existing RVV
runtime-scalar compare masked indexed scatter-store selected-body route must
line up from typed `tcrv_rvv` indexed scatter body facts through RVV
plugin-owned runtime-scalar comparison, compare-produced mask, index mapping,
masked indexed store facts, common EmitC materialization, target artifact
export, generated bundle ABI, and real `ssh rvv` correctness evidence. If the
production path is already complete, this round closes the exact executable
evidence blocker with focused evidence. If inspection finds dry-run-only,
stale, or under-validated executable-boundary behavior, this round repairs only
that scatter-store artifact/ABI seam.

## What I Already Know

* No `.trellis/.current-task` existed at session start; this task was created
  from the Hermes Direction Brief.
* Repository inspection started from clean `main` at commit `f661456c rvv:
  prove runtime scalar indexed gms evidence`.
* The previous archived task closed the runtime-scalar compare masked indexed
  gather-MAcc-scatter composite evidence gap by hardening snapshot-backed
  expected values and proving the pre-realized composite generated bundle on
  `ssh rvv`.
* This round intentionally switches away from the composite route and isolates
  the standalone runtime-scalar compare masked indexed scatter-store memory
  movement boundary: runtime scalar binding, compare-produced mask, source
  payload, unique indexed destination mapping, inactive-lane preservation,
  header/prototype ABI order, runtime AVL/VL, generated bundle ABI, and target
  artifact validation.
* The relevant standalone route already exists in the repository under
  operation kind `runtime_scalar_cmp_masked_indexed_scatter_store_unit_load`.
  Current fixtures expose explicit and pre-realized selected-body artifact
  paths, pre-realized dry-run generated-bundle evidence, provider-owned operand
  binding summaries, stale mask-producer fail-closed checks, and target
  artifact validation constants.
* Existing specs require computed-mask indexed routes to consume provider-owned
  facts for memory form, typed compute op, runtime-scalar RHS splat, mask
  producer source, index source/EEW/offset/uniqueness, inactive-lane policy,
  header/type summaries, ABI order, binding summary, intrinsic leaves, and
  candidate mirror validation before artifact acceptance.
* The pre-realized dry-run harness currently checks a representative generated
  call, runtime scalar thresholds, patterns, indexed scatter contract, inactive
  destination preservation diagnostic, source preservation, tail preservation,
  and `PASS` output. This round must inspect whether that harness computes all
  expected values and mismatch diagnostics from pre-call snapshots where
  preservation is asserted, and then prove the non-dry-run generated bundle on
  real RVV hardware when executable correctness is claimed.

## Requirements

* Keep compiler implementation in C++ / MLIR / LLVM / TableGen / CMake / lit /
  FileCheck. Python remains support tooling and harness generation only.
* Stay on the standalone runtime-scalar compare masked indexed scatter-store
  selected-body route family. Do not broaden to a gather/scatter matrix,
  standalone gather detour, gather-MAcc-scatter composite rework, dtype/LMUL
  clone batch, unrelated dot/reduction/MAcc/dequant work, high-level frontend,
  dashboard, or report-only task.
* Preserve authority placement: typed `tcrv_rvv` body/config/runtime facts and
  RVV plugin-owned realization, route family plans, statement plans, provider
  route descriptions, and target validation define route support; common EmitC
  and target export only materialize and package provider-built facts.
* Treat route ids, artifact names, emission-plan metadata, status/result
  fields, helper names, test names, and generated evidence files as mirrors
  only.
* Validate or prove fail-closed handling for stale/missing executable-boundary
  facts, especially runtime scalar binding, compare operand role, mask fact,
  inactive-lane policy, index mapping, scatter destination role, source payload
  role, dtype/SEW/LMUL/config/policy, header/prototype binding, ABI order,
  runtime AVL/VL, C type mapping, and statement facts.
* Positive executable evidence must run generated artifacts on real `ssh rvv`
  before claiming runtime correctness.
* Generated evidence must use pre-call snapshots for expected values and
  mismatch diagnostics whenever source, destination, inactive-lane, or tail
  preservation is asserted.
* If the executable path is dry-run-only or under-validated, repair the
  production seam rather than closing as Trellis-only evidence.

## Acceptance Criteria

* [x] Current code inspection proves the runtime-scalar compare masked indexed
  scatter-store route is realized and provider-consumed before route
  construction, or a focused source diff repairs only that executable
  artifact/ABI seam.
* [x] Positive generated-bundle evidence covers materialized selected
  boundary, emission plan, target artifact export, generated bundle compile,
  and `ssh rvv` correctness for
  `runtime_scalar_cmp_masked_indexed_scatter_store_unit_load`.
* [x] Positive evidence records and checks ABI order, runtime scalar predicate
  effects, compare-produced mask facts, unique noncontiguous index mapping,
  source payload role, scatter destination role, inactive-lane policy,
  dtype/config/policy, runtime AVL/VL, header/prototype binding, source
  preservation, inactive destination lane preservation, untouched destination
  slots, and tail preservation.
* [x] A focused fail-closed check rejects at least one stale or missing
  executable-boundary fact, such as stale runtime scalar binding, stale compare
  producer source, stale mask binding, stale inactive-lane policy, stale index
  mapping, stale scatter destination role, stale header/prototype binding,
  stale route-family validation contract, wrong generated C type, wrong ABI
  value mapping, or unsupported direct pre-realized route claim.
* [x] `build/bin/tianchenrv-rvv-extension-plugin-test` passes.
* [x] `build/bin/tianchenrv-target-artifact-export-test` passes.
* [x] Relevant generated-bundle dry-run tests or direct RUN-line FileCheck
  equivalents for explicit/pre-realized runtime-scalar compare masked indexed
  scatter-store pass.
* [x] `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test` passes if
  the harness changes.
* [x] Bounded old-authority scan over touched files and added diff lines shows
  no new positive legacy `i32m1` route authority, source-front-door route
  authority, descriptor-driven compute, direct-C/source export authority, or
  mirror-only acceptance.
* [x] `git diff --check`, `git diff --cached --check`, and final
  `git status --short` are clean/reported.
* [x] Task status, workspace journal, archive, and one coherent commit are
  completed if the task finishes.

## Implementation Notes

* Production inspection found the standalone runtime-scalar computed-mask
  indexed scatter-store seam already routes through RVV-owned validation before
  `TCRVEmitCLowerableRoute`: the pre-realized body validator requires
  `runtime_scalar_cmp_masked_indexed_scatter_store_unit_load`, predicate `sle`,
  memory form `computed-mask-unit-load-indexed-scatter-store`, unique i32
  element indices, compare-produced mask facts, inactive-lane policy
  `preserve-output-on-false-lanes`, SEW32/LMUL m1 agnostic policy, and ABI
  roles `lhs,rhs_scalar,src,index,dst,n`. The computed-mask memory route owner
  then checks canonical runtime-scalar indexed facts, binding summary, ABI
  parameters, source/index/destination forms, statement leaves, and typed
  config before route construction. Target artifact validation rebuilds and
  validates provider route payload before candidate metadata mirrors.
* The executable evidence gap was in generated harness quality, not in the C++
  provider path. The runtime-scalar indexed scatter-store harness now computes
  expected active-lane values and mismatch diagnostics from `src_before[index]`
  rather than the post-call live `src[index]`. It still checks the live source
  buffer against `src_before`, inactive destination lanes against `old_dst`, and
  tail destination slots against `old_dst` before printing
  `source_preserved tail_preserved`.
* The explicit and pre-realized dry-run FileCheck tests now assert the
  snapshot-backed expected expression:
  `lhs[index] <= rhs_scalar ? src_before[index] : old_dst[indices[index]]`.
* The `--rhs-scalar` help text now names
  `runtime_scalar_cmp_masked_indexed_scatter_store_unit_load`, matching the
  existing multi-threshold evidence path.
* `ssh rvv` evidence passed for the pre-realized generated bundle at
  `build/trellis-validation/stage2-runtime-scalar-cmp-indexed-scatter-store/ssh/pre-rtscatter/runtime_scalar_cmp_masked_indexed_scatter_store_unit_load/evidence.json`.
  Remote compile used `clang -O2 -march=rv64gcv -mabi=lp64d` on `riscv64`
  with Ubuntu clang 18.1.3. Remote runtime passed counts `0,1,16,17,257`, RHS
  scalars `-37,91`, patterns `0,1`, active/inactive lane mixes,
  noncontiguous unique indexed destination writes, inactive destination
  preservation, source preservation, and tail preservation:
  `PASS op=runtime_scalar_cmp_masked_indexed_scatter_store_unit_load counts=0,1,16,17,257 rhs_scalars=-37,91 patterns=0,1`.
* Spec update judgment: no `.trellis/spec/` change is needed in this round. The
  reusable rule already exists in
  `.trellis/spec/testing/mlir-testing-contract.md`: preservation evidence must
  use pre-call snapshots in expected-value computation and mismatch diagnostics.
  This round applies that existing rule to the standalone scatter-store
  harness.

## Checks Run

* `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
* Explicit dry-run generated bundle for
  `runtime_scalar_cmp_masked_indexed_scatter_store_unit_load` with
  `build/bin/tcrv-opt`, `build/bin/tcrv-translate`, and
  `/usr/lib/llvm-20/bin/llvm-readobj`.
* Pre-realized dry-run generated bundle for
  `runtime_scalar_cmp_masked_indexed_scatter_store_unit_load` with the same
  local toolchain.
* Direct FileCheck on explicit dry-run `ROOT`, `RTSCATTER`, and `HARNESS`
  artifacts.
* Direct FileCheck on pre-realized dry-run `ROOT`, `RTSCATTER`, and `HARNESS`
  artifacts.
* Explicit target artifact fixture `PLAN` and `HEADER` RUN-line equivalents.
* Explicit stale producer fail-closed RUN-line equivalent:
  `runtime-scalar-splat-compare-rhs` changed to `vector-compare-rhs-load` is
  rejected before target artifact acceptance.
* Pre-realized target artifact fixture `REALIZED`, `PLAN`, and `HEADER`
  RUN-line equivalents.
* `build/bin/tianchenrv-rvv-extension-plugin-test`
* `build/bin/tianchenrv-target-artifact-export-test`
* Non-dry-run pre-realized generated bundle on `ssh rvv`.
* Bounded added-line old-authority scan.
* `git diff --check`
* `git diff --cached --check`

## Out Of Scope

* No broad gather/scatter route matrix.
* No indexed gather-load detour unless inspection proves it is the single
  blocker for this scatter-store seam.
* No gather-MAcc-scatter composite rework except bounded reference reading.
* No dtype/SEW/LMUL clone batch.
* No standalone MAcc, dot, reduction, dequant, or conversion work.
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
  `.trellis/tasks/archive/2026-06/06-08-stage2-rvv-runtime-scalar-cmp-gather-macc-scatter-abi/`.
* Primary code paths to inspect:
  `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`,
  `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`,
  `lib/Plugin/RVV/RVVComputedMaskMemorySelectedBodyRealizationOwner.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCBaseMemoryRouteFamilyPlanOwners.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCComputedMaskMemoryRouteFamilyPlanOwners.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCMemoryStatementPlanOwners.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`, and
  `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`.
* Primary evidence paths to inspect:
  `scripts/rvv_generated_bundle_abi_e2e.py`,
  `test/Scripts/rvv-generated-bundle-abi-e2e-runtime-scalar-cmp-masked-indexed-scatter-store-dry-run.test`,
  `test/Scripts/rvv-generated-bundle-abi-e2e-pre-realized-runtime-scalar-cmp-masked-indexed-scatter-store-dry-run.test`,
  `test/Target/RVV/explicit-selected-body-artifact-runtime-scalar-cmp-masked-indexed-scatter-store.mlir`, and
  `test/Target/RVV/pre-realized-selected-body-artifact-runtime-scalar-cmp-masked-indexed-scatter-store.mlir`.
