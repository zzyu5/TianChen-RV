# RVV selected config VL dataflow materialization

## Goal

Complete one bounded RVV plugin/dialect submodule that makes the existing
`RVVBinarySelectedConfigContract` drive or validate concrete finite-binary
`tcrv_rvv` VL/control/dataflow materialization. The selected compile-time
config must be the source for SEW, LMUL, tail/mask policy, vector type,
setvl suffix, and RVV dataflow op family, while runtime element count/AVL/VL
remains runtime control and descriptor-local element_count remains finite
descriptor metadata.

## Background

The previous archived task
`.trellis/tasks/archive/2026-05/05-11-rvv-selected-config-boundary-family-contract/`
added the compiler-owned selected config contract and routed planning,
selected emission metadata, and dispatch/export consumers through it. That task
did not claim new RVV runtime evidence.

Current code already has finite `tcrv_rvv.setvl`, `tcrv_rvv.with_vl`,
i32m1/i32m2/i64m1 dataflow types, and generated finite binary microkernel
bodies. This task should not broaden that into a generic RVV backend. The
module gap is to make the selected-config contract the explicit boundary for
materializing and validating those bodies, rather than allowing materialization
or verification to drift through locally reconstructed shape/descriptor
strings.

## Requirements

- Consume `RVVBinarySelectedConfigContract` in the RVV plugin/dialect/target
  dataflow path for finite binary selected plans.
- Materialize or validate the same selected-config-driven dataflow logic for
  at least one i32m2 path and one i64m1 path.
- Preserve the layering:
  - hardware/capability facts remain target capability/profile metadata;
  - compile-time selected config carries SEW, LMUL, tail/mask policy, vector
    type, selected suffixes, and selected capability ids;
  - runtime element count/AVL/VL is represented by runtime ABI/control surfaces
    and the `tcrv_rvv.setvl` / `tcrv_rvv.with_vl` token flow;
  - descriptor-local metadata carries dtype/operator/family, lowering
    descriptor, finite element_count, and runtime ABI identity.
- Ensure materialized `tcrv_rvv.setvl`, `tcrv_rvv.with_vl`, load, arithmetic,
  and store ops derive their bounded shape and family facts from the selected
  config contract or a helper that first validates that contract.
- Fail closed when selected config and descriptor dtype/operator disagree, when
  selected i32m2/i64m1 shape metadata is stale, when tail/mask policy is not
  the selected finite policy, or when descriptor-local element_count is used as
  runtime AVL/VL control.
- Keep generic transforms extension-neutral. Generic passes may call plugin
  interfaces; they must not branch on RVV family, dtype, shape, vendor, or
  target-specific strings.
- Keep `tcrv.exec` compute-free. Any executable RVV dataflow must remain under
  the RVV extension dialect/plugin/target ownership.

## Acceptance Criteria

- Positive lit/FileCheck and/or C++ coverage proves an i32m2 path and an i64m1
  path both show contract-driven `tcrv_rvv.setvl`, `tcrv_rvv.with_vl`, and the
  correct finite load/op/store dataflow.
- The implementation contains an explicit C++ contract-facing materialization
  or verification helper, so dataflow code does not independently re-derive
  SEW/LMUL/vector suffix from stale strings when the selected-config contract
  already provides them.
- Negative coverage rejects:
  - selected config versus descriptor dtype/operator mismatch;
  - stale i32m2 versus i64m1 vector-shape metadata;
  - invalid tail/mask policy;
  - runtime element-count/AVL confusion with compile-time vector config or
    descriptor-local element_count;
  - missing selected config where a contract-driven dataflow path is required.
- Existing i64 vadd/vsub/vmul front-door dispatch/export assumptions remain
  valid.
- No new RVV runtime correctness, throughput, latency, or performance claim is
  made unless real `ssh rvv` evidence is collected, which is out of scope for
  this task.
- Focused tests pass, `git diff --check` passes, and `check-tianchenrv` is run
  if practical after focused checks.

## Non-Goals

- No generic RVV backend.
- No MLIR vector/scalable-vector lowering route.
- No new benchmark, throughput, latency, performance, or broad evidence
  matrix.
- No arbitrary dtype expansion, reductions, masked arithmetic semantics,
  tensor/tile IR, or new high-level compute ops.
- No compute semantics in `tcrv.exec`.
- No compiler internals implemented in Python.
- No docs-only, helper-only, smoke-only, report-only, or metadata-only
  closeout.

## Validation Plan

- Build focused targets touched by RVV planning/materialization/export tests:
  `tcrv-opt`, `tcrv-translate`,
  `tianchenrv-rvv-binary-planning-test`,
  `tianchenrv-rvv-extension-plugin-test`,
  `tianchenrv-rvv-selected-lowering-boundary-test`, and target/export tests as
  needed.
- Run focused C++ tests covering selected-config planning/materialization
  helpers and selected lowering-boundary behavior.
- Run focused lit/FileCheck tests for RVV dialect verification, selected
  lowering-boundary, emission readiness/materialized emission plans,
  LinalgToExec i32m2/i64m1 cases, and RVV scalar dispatch/export regressions
  affected by the change.
- Run `git diff --check`.
- Run `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv
  -j2` after focused checks if practical.
- Validate the Trellis task before finish/archive.

## Technical Notes

- Specs read for this task:
  `.trellis/spec/index.md`,
  `.trellis/spec/core-dialect/tcrv-exec-contract.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/variant-pipeline/generation-selection-tuning.md`,
  `.trellis/spec/lowering-runtime/emission-runtime-contract.md`, and
  `.trellis/spec/testing/mlir-testing-contract.md`.
- Previous context read:
  `.trellis/tasks/archive/2026-05/05-11-rvv-selected-config-boundary-family-contract/prd.md`.
- Relevant current code surfaces:
  `include/TianChenRV/Target/RVV/RVVSelectedConfigContract.h`,
  `include/TianChenRV/Plugin/RVV/RVVBinaryPlanning.h`,
  `include/TianChenRV/Plugin/RVV/RVVBinarySelectedEmissionPlanning.h`,
  `include/TianChenRV/Plugin/RVV/RVVBinaryMicrokernelMaterialization.h`,
  `lib/Plugin/RVV/RVVBinaryPlanning.cpp`,
  `lib/Plugin/RVV/RVVBinarySelectedEmissionPlanning.cpp`,
  `lib/Plugin/RVV/RVVBinaryMicrokernelMaterialization.cpp`,
  `lib/Target/RVV/RVVBinaryMicrokernelBodyVerifier.cpp`,
  `lib/Target/RVV/RVVMicrokernel.cpp`,
  and the RVV dialect ODS under
  `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`.

## Definition Of Done

The task is complete when the selected-config contract is an active
contract-facing input to finite RVV VL dataflow materialization or validation
across i32m2 and i64m1, focused positive and negative tests pass, generic core
logic remains extension-neutral, the Trellis task validates and archives, and
one coherent commit records the completed module. If unfinished, leave the task
open and record the exact continuation point: contract producer, RVV dialect
builder/verifier, lowering-boundary consumer, emission-readiness consumer,
dataflow positive case, dataflow negative diagnostic, or exporter
compatibility.

## Completion Notes

- Added `RVVBinaryVLDataflowMaterialization` as the explicit C++ bridge from
  `RVVBinarySelectedConfigContract` to bounded RVV VL/control/dataflow
  materialization facts.
- `materializeRVVBinaryMicrokernelOp` now first builds that contract-facing
  dataflow object, validating that the selected intrinsic descriptor, selected
  family, selected shape, SEW, LMUL, policy, vector type/suffix, setvl suffix,
  arithmetic op, and descriptor-local element_count all match the selected
  config contract before it creates `tcrv_rvv.setvl`, `tcrv_rvv.with_vl`, and
  load/op/store body ops.
- Positive C++ coverage now proves i32m2 `i32-vsub` and i64m1 `i64-vmul`
  derive vector type, operation names, SEW/LMUL, vector suffix, setvl suffix,
  and descriptor-local element_count from the same selected-config-to-VL
  dataflow helper.
- Negative C++ coverage rejects stale i32m1 descriptor data crossing an i32m2
  selected-config boundary and rejects a materialization plan with no selected
  config contract.
- Existing lit coverage continues to cover invalid runtime/config boundary
  confusion, stale selected-shape metadata, invalid setvl/with_vl policy, and
  i32m2/i64m1 frontend/export dataflow behavior.
- Generic transforms were not given RVV family branches. The changed behavior
  remains RVV plugin/dialect-owned, and no compute semantics were added to
  `tcrv.exec`.
- No `ssh rvv` evidence was collected, and no new RVV runtime correctness or
  performance claim was made.

## Checks Run

- `cmake --build artifacts/tmp/tianchenrv-build --target
  tianchenrv-rvv-binary-planning-test
  tianchenrv-rvv-selected-lowering-boundary-test tcrv-opt tcrv-translate -j2`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-rvv-binary-planning-test`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-rvv-selected-lowering-boundary-test`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-rvv-extension-plugin-test`
- `git diff --check`
- `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`:
  200/200 lit tests passed.
- `clang-format -i ...` was attempted but `clang-format` is not installed in
  this environment; formatting was kept manually and checked with
  `git diff --check`.
