# Stage2 RVV widening multiply-accumulate executable slice

## Goal

Implement one bounded Stage2 RVV signed widening multiply-accumulate executable
slice:

```text
out_i32[i] = acc_i32[i] + sign_extend(lhs_i16[i]) * sign_extend(rhs_i16[i])
```

The selected RVV path must carry typed source roles for `lhs` and `rhs`, an
i32 accumulator input role, an i32 output role, source vector config
`SEW16 LMUL mf2`, accumulator/result vector config `SEW32 LMUL m1`, signed
widening multiply-accumulate operation kind, accumulator layout, unit-stride
memory form, tail/mask policy, runtime `n` / AVL, route planning, generated
artifact emission, and real `ssh rvv` correctness evidence for representative
counts.

This is one bounded mixed-width contraction-supporting arithmetic slice. It is
not a broad contraction framework, dtype/LMUL matrix, frontend lowering task,
source-front-door route, descriptor route, or performance task.

## Direction Source

- Hermes Direction Brief: `Stage2 RVV widening multiply-accumulate executable slice`.
- Module owner: RVV plugin-owned mixed-width widening multiply-accumulate path
  for signed i16 SEW16 LMUL mf2 `lhs`/`rhs` plus signed i32 SEW32 LMUL m1
  accumulator/result.
- Repo root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean.
- Initial HEAD: `0a460373 rvv: add i16-to-i32 widening conversion slice`.
- No `.trellis/.current-task` existed, so this task was created from the
  Hermes brief before source edits.

## Current Repository Facts

- Current HEAD already supports a bounded signed `i16mf2 -> i32m1` widening
  conversion route through typed body/config/runtime facts, route planning,
  provider emission, generated-bundle dry-run, and real `ssh rvv` evidence.
- Current HEAD already supports a bounded signed i32 `macc_add` route with
  structural accumulator/result layout facts on `tcrv_rvv.macc`, selected-body
  realization, route planning/provider support, generated-bundle dry-run, and
  real `ssh rvv` evidence.
- The production gap is the mixed-width composition: two i16mf2 source loads,
  one i32m1 accumulator load, signed widening multiplication, i32
  accumulation, and i32 store must be represented as typed RVV facts and
  consumed by the RVV plugin before common EmitC/export.
- Dtype, SEW, LMUL, signedness, multiply-accumulate semantics, accumulator
  layout, memory form, policy, and intrinsic mapping must be validated or
  derived from typed body/config/runtime facts. They must not come from route
  ids, ABI strings, artifact names, test names, descriptors, exact intrinsic
  spelling, or common EmitC/export code.
- `tcrv.exec` declares ABI/runtime roles only. RVV compute/config, selected-body
  realization, legality, intrinsic mapping, and fail-closed diagnostics remain
  RVV plugin-owned.

## Requirements

1. Keep scope to one signed `i16mf2 x i16mf2 + i32m1 -> i32m1` unit-stride
   widening macc route with `lhs`, `rhs`, `acc`, `out`, and runtime `n/AVL`
   ABI roles.
2. The selected/pre-realized body or explicit typed body must structurally
   carry source buffer roles, accumulator buffer role, destination buffer role,
   source element type/config, accumulator/result element type/config, signed
   widening multiply-accumulate kind, accumulator layout, memory form, policy,
   runtime `n/AVL`, and ABI role order.
3. `RVVSelectedBodyRealization` must materialize only legal generic typed
   structure for this slice if starting from a pre-realized selected body:
   destination-config `setvl/with_vl`, i16mf2 lhs load, i16mf2 rhs load, i32m1
   accumulator load, generic widening macc compute, and i32m1 output store.
4. RVV route planning must derive ABI order, source/accumulator/result C pointer
   types, vector C types, setvl/load/store/compute intrinsic or intrinsic-leaf
   payloads, route metadata mirrors, and targeted diagnostics from typed
   body/config/runtime facts.
5. Provider/materializer must consume the provider-built plan. Common
   materialization must not infer dtype, widening, multiply-accumulate kind,
   accumulator layout, memory form, policy, intrinsic choice, or ABI meaning.
6. Unsupported macc kind, invalid SEW/LMUL relation, missing source,
   accumulator, or result config, source/accumulator dtype mismatch, missing
   accumulator role, missing `n/AVL`, wrong ABI role/order, stale route-id
   authority, and incomplete typed body structure must fail closed with targeted
   diagnostics.
7. Generated-bundle evidence must use positive and negative i16 operands,
   nonzero i32 accumulator values, and sentinel-filled output storage so
   expected output proves signed widening multiplication, accumulator addition,
   and tail preservation.
8. Real `ssh rvv` evidence is required for any executable correctness claim.

## Acceptance Criteria

- [x] PRD, implement/check context, and task metadata describe this bounded
      mixed-width widening macc slice.
- [x] Positive selected/pre-realized or explicit typed RVV body structurally
      carries lhs/rhs i16mf2 source facts, i32m1 accumulator/result facts,
      signed widening macc kind, accumulator layout, unit-stride memory form,
      policy, runtime `n/AVL`, and ABI roles.
- [x] `RVVSelectedBodyRealization` materializes the bounded slice into explicit
      typed `setvl/with_vl/load/load/load/widening_macc/store` or equivalent
      structure without changing computation semantics.
- [x] RVVEmitCRoutePlanning derives source/accumulator/result ABI types, vector
      types, compute leaf payload, route metadata, and runtime ABI order from
      typed body/config facts.
- [x] Positive route/materialization tests prove typed i16mf2 inputs and i32m1
      accumulator/result facts reach `TCRVEmitCLowerableRoute` and
      provider-owned route metadata.
- [x] Negative fail-closed tests cover unsupported macc kind, invalid SEW/LMUL
      pair, missing/mismatched source or accumulator/result config, missing
      accumulator role, missing AVL/runtime role, wrong memory form, stale
      route-id authority, and incomplete typed body structure.
- [x] Generated-bundle dry-run passes for the widening macc slice at counts
      `7,16,23`.
- [x] Real `ssh rvv` correctness passes for counts `7,16,23` if executable
      correctness is claimed, with expected output proving negative and
      positive i16 products, nonzero accumulator addition, and tail sentinels.
- [x] Focused build, lit, C++ unit, script, and artifact checks for touched RVV
      dialect/config, runtime ABI if touched, construction protocol,
      selected-body realization, route planning/provider, materializer/export,
      and generated-bundle paths pass.
- [x] Active-authority scan confirms no positive legacy/source/descriptor or
      common/export RVV semantic authority is reintroduced.
- [x] `git diff --check`, Trellis task validation, task status update/archive,
      and one coherent production-code commit are completed if the task is
      finished.

## Non-Goals

- No broad contraction framework, matmul lowering, reductions, segmented macc,
  masked macc, unsigned variants, dtype/LMUL clone batches, high-level
  Linalg/Vector/StableHLO lowering, source-front-door positive route,
  one-intrinsic wrapper dialect, dashboard, report-only inventory, helper-only
  refactor, or performance claim.
- No Scalar, IME, Offload, TensorExt, future plugin work, global autotuning,
  readiness state machine, source-shape clone batch, or Template/Toy examples.
- No descriptor-driven computation, direct-C/source-export route restoration,
  route-id authority, artifact-name authority, or compatibility wrapper
  preserving old i32 authority.
- No new dtype-prefixed helper op families such as `tcrv_rvv.i32_macc`.
- No Python implementation of compiler core, dialects, passes, plugin registry,
  capability model, lowering, or emission. Python changes, if any, are limited
  to generated-bundle tooling/evidence.

## Validation Plan

1. Validate and start the Trellis task.
2. Build focused targets expected to include `tcrv-opt`, `tcrv-translate`,
   `tianchenrv-rvv-dialect-test`, `tianchenrv-rvv-extension-plugin-test`,
   `tianchenrv-construction-protocol-common-test`, and
   `tianchenrv-target-artifact-export-test`.
3. Run focused RVV dialect/verifier lit tests for positive and negative
   widening macc structure.
4. Run focused selected-body realization, route planning/provider, EmitC, and
   target artifact lit/FileCheck tests.
5. Run touched C++ tests where route/provider/export helpers are covered.
6. Run `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py` and
   `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test` if the script
   changes or generated-bundle evidence is claimed.
7. Run generated-bundle dry-run for the new widening macc op at counts
   `7,16,23`.
8. Run real `ssh rvv` correctness for the new widening macc op at counts
   `7,16,23` if executable correctness is claimed.
9. Run active-authority scans over active RVV include/lib/script/test paths.
10. Run `git diff --check`.
11. Run broader `check-tianchenrv` if shared compiler behavior changes.

## Technical Notes

Relevant specs read:

- `.trellis/spec/index.md`
- `.trellis/spec/extension-plugins/rvv-plugin.md`
- `.trellis/spec/lowering-runtime/emitc-route.md`
- `.trellis/spec/plugin-protocol/interfaces-and-registry.md`
- `.trellis/spec/variant-pipeline/generation-selection-tuning.md`
- `.trellis/spec/testing/mlir-testing-contract.md`

Prior task context read:

- `.trellis/tasks/archive/2026-05/05-20-stage2-rvv-widening-int-conversion/prd.md`
- `.trellis/tasks/archive/2026-05/05-20-05-20-stage2-rvv-macc-executable-slice/prd.md`
- `.trellis/tasks/archive/2026-05/05-19-05-19-stage2-rvv-selected-body-macc-add/prd.md`
- `.trellis/workspace/codex/journal-12.md`

Initial implementation surface to inspect:

- `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`
- `include/TianChenRV/Dialect/RVV/IR/RVVConfigContract.h`
- `include/TianChenRV/Support/RuntimeABI.h`
- `lib/Dialect/RVV/IR/RVVConfigContract.cpp`
- `lib/Dialect/RVV/IR/RVVDialect.cpp`
- `lib/Support/RuntimeABIContract.cpp`
- `lib/Plugin/RVV/Construction/RVVConstructionProtocol.cpp`
- `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp`
- `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`
- `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`
- `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
- `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
- `scripts/rvv_generated_bundle_abi_e2e.py`
- Existing macc, widening conversion, and generated-bundle tests.

## Definition Of Done

- One bounded mixed-width widening macc route structurally carries source,
  accumulator, result, operation, policy, runtime, and ABI facts through typed
  RVV body structure into RVV route planning/provider metadata and generated
  artifacts.
- Fresh focused positive/negative/generated-bundle/runtime evidence is current
  to this task when claimed.
- No legacy i32/source/descriptor/common-export authority is reintroduced.
- The task is finished/archived only if production code/tests/evidence are
  present in this round; otherwise it remains open with an exact continuation
  point.
