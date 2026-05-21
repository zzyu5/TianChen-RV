# Stage2 RVV multiply-add accumulator executable slice

## Goal

Implement one bounded Stage2 RVV executable slice for a typed signed-i32 /
SEW32 / LMUL m1 multiply-add accumulator dataflow:

```text
out_i32[i] = acc_i32[i] + lhs_i32[i] * rhs_i32[i]
```

for runtime `n`, with tail sentinels beyond `n` preserved. The important
boundary for this task is that `acc` is an explicit input vector/mem_window
and `out` is an explicit output mem_window. This is not a high-level matmul,
Linalg, contraction framework, or one-intrinsic wrapper, and it must not be
implemented by reviving legacy i32 route authority.

## Direction Source

- Direction title: `Stage2 RVV multiply-add accumulator executable slice`.
- Module owner: RVV plugin-owned vector multiply-add accumulator route for one
  bounded typed e32m1 path.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean.
- Initial HEAD: `f3381661 rvv: add computed-mask byte-strided store slice`.
- No `.trellis/.current-task` existed, so this task was created from the
  Direction Brief before source edits.

## Current Repository Facts

- The last archived computed-mask byte-strided-store task completed a typed
  selected/pre-realized RVV body through selected-body realization, route
  planning/provider, generated artifact, and real `ssh rvv` evidence at counts
  `7,16,23` and destination byte strides `4,8,12`.
- Archived macc work completed a bounded `macc_add` path where the output
  buffer also serves as the accumulator input:

  ```text
  out[i] = out[i] + lhs[i] * rhs[i]
  ```

  That is useful prior evidence but is not the complete target behavior here.
- This task must add or repair the compiler-owned path so the typed RVV body
  structurally carries a distinct accumulator input and an output destination:

  ```text
  lhs mem_window, rhs mem_window, acc mem_window, out mem_window, n/AVL
  ```

- `tcrv.exec.mem_window` and `tcrv.exec.runtime_param` declare ABI/runtime
  roles only. Multiply-add operation kind, accumulator semantics, dtype, SEW,
  LMUL, policy, VL placement, memory form, ABI order, route support, and
  executable acceptance must come from the typed `tcrv_rvv` body plus RVV
  plugin legality/realization/provider facts.
- Common EmitC/export may materialize provider-supplied neutral mechanics. It
  must not infer RVV semantics, accumulator layout, dtype/config, ABI order,
  intrinsic choice, or route support from names, artifacts, route ids, C ABI
  strings, descriptors, or tests.
- Stage1/Stage2 guardrails remain active: no positive `RVVI32M1`,
  `rvv-i32m1`, finite positive `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`,
  descriptor/direct-C/source-export, source-front-door, public exact intrinsic
  route authority, route-id authority, artifact-name authority, or
  common/export RVV semantic authority.

## Requirements

1. Keep scope to one bounded signed i32 / SEW32 / LMUL m1 / unit-stride
   vector multiply-add accumulator route.
2. The selected or pre-realized RVV body must structurally carry:
   - `lhs` vector input mem_window;
   - `rhs` vector input mem_window;
   - `acc` accumulator vector input mem_window;
   - `out` output mem_window;
   - runtime `n` / AVL value use;
   - e32m1 vector config and legal tail policy;
   - multiply-add operation kind;
   - accumulator layout where accumulator data is read from `acc`;
   - result layout where the multiply-add result is stored to `out`;
   - unit-load/unit-store memory form;
   - runtime ABI order.
3. RVV selected-body realization must materialize legal generic typed
   `setvl` / `with_vl`, `lhs` load, `rhs` load, `acc` load, typed
   multiply-add dataflow, and `out` store. It must not change computation
   semantics, dtype semantics, ABI roles, selected variant origin, required
   capabilities, dispatch/fallback behavior, or runtime `n` / AVL values.
4. RVV route planning/provider must derive vector/result C types, target
   leaves, headers, ABI order, route mirrors, artifact mirrors, accumulator
   layout, result layout, and diagnostics from typed body/config/runtime facts.
5. Generated artifact execution must prove:
   - runtime `n` controls the active element count;
   - mixed positive/negative `lhs`, `rhs`, and `acc` values compute
     `acc + lhs * rhs` exactly for signed i32;
   - output writes occur only for active lanes;
   - tail sentinels beyond `n` are preserved;
   - multi-VL behavior is covered where count exceeds one hardware VL.
6. Missing accumulator, missing lhs/rhs/output, operand shape or config
   mismatch, missing `n` / AVL, invalid tail policy/config, unsupported
   accumulator/result layout, stale fixed-VL or route-id authority, incomplete
   typed body, descriptor/direct-C/source-front-door authority, and
   common/export semantic inference must fail closed with targeted diagnostics.

## Acceptance Criteria

- [x] PRD, implement/check context, and task metadata describe this bounded
      multiply-add accumulator slice.
- [x] A selected/pre-realized body structurally carries lhs, rhs, accumulator,
      output mem_window, runtime `n` / AVL, e32m1 config, multiply-add
      operation kind, accumulator layout, result layout, tail policy, memory
      forms, and ABI order.
- [x] RVV selected-body realization materializes the bounded pre-realized slice
      into legal typed `tcrv_rvv` setvl/with_vl/load/multiply-add/store
      structure if this slice uses a pre-realized fixture path.
- [x] RVV route planning/provider derive ABI order, vector C type, accumulator
      and result layout mirrors, target leaves, headers, artifact mirrors, and
      diagnostics from typed body/config/runtime facts.
- [x] Positive route/materialization/generated-artifact tests prove the
      accumulator input, lhs/rhs inputs, output destination, runtime `n`,
      multiply-add kind, layout facts, and provider-owned metadata reach the
      production `TCRVEmitCLowerableRoute` path.
- [x] Negative fail-closed tests cover missing accumulator, missing lhs/rhs,
      missing output, operand shape/config mismatch, missing `n` / AVL,
      invalid tail policy/config, unsupported accumulator/result layout, stale
      fixed-VL or route-id authority, incomplete typed body,
      descriptor/direct-C/source-front-door authority, and common/export
      semantic inference where meaningful.
- [x] Generated-bundle dry-run passes for counts `7,16,23` with mixed
      positive/negative lhs/rhs/accumulator values and tail sentinels.
- [x] Real `ssh rvv` generated-bundle run passes for the same bounded coverage.
- [x] Active-authority scan confirms no positive `RVVI32M1`, `rvv-i32m1`,
      finite positive `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`,
      descriptor/direct-C/source-export, source-front-door, public exact
      intrinsic route authority, or common/export RVV semantic authority is
      introduced.
- [x] Focused build/lit/C++/script checks, `check-tianchenrv`,
      `git diff --check`, task validation, finish/archive, clean git status,
      and one coherent commit are completed if the task finishes.

## Non-Goals

- No high-level matmul, Linalg, Vector, StableHLO, frontend lowering, or broad
  contraction framework.
- No reductions, masked multiply-add, widening/dtype/LMUL clone batches,
  source-front-door positive routes, dashboards, report-only work, helper-only
  cleanup, or performance claim.
- No new dtype-prefixed `tcrv_rvv.i32_*` helper namespace and no
  compatibility wrapper preserving old i32 authority.
- No descriptor-driven computation, direct-C/source-export route restoration,
  route-id authority, artifact-name authority, exact intrinsic spelling as
  public route authority, or common/export RVV semantic authority.
- No Python implementation of compiler core, dialects, passes, plugin
  registry, capability model, lowering, route planning, or emission.

## Validation Plan

1. Validate and start the Trellis task.
2. Inspect current RVV dialect/config/runtime ABI, selected-body realization,
   route planning/provider, construction, target support, generated-bundle
   script, and focused tests.
3. Build focused targets expected to include `tcrv-opt`, `tcrv-translate`,
   `tianchenrv-rvv-dialect-test`,
   `tianchenrv-rvv-extension-plugin-test`,
   `tianchenrv-construction-protocol-common-test`, and
   `tianchenrv-target-artifact-export-test`.
4. Run focused lit/FileCheck tests for dialect/verifier negatives,
   selected-body realization, route plan/provider mirrors, target artifact
   output, EmitC materialization, and generated-bundle dry-run.
5. Run touched C++ tests for RVV dialect/plugin/construction/export behavior.
6. Run `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py` and
   `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test` if script
   behavior changes or generated-bundle evidence is claimed.
7. Run generated-bundle dry-run for counts `7,16,23`.
8. Run real `ssh rvv` generated-bundle correctness for counts `7,16,23`.
9. Run active-authority scans over active RVV include/lib/script/test paths.
10. Run `git diff --check`.
11. Run `cmake --build build --target check-tianchenrv -j2`.

## Technical Notes

Relevant specs read:

- `.trellis/spec/index.md`
- `.trellis/spec/extension-plugins/rvv-plugin.md`
- `.trellis/spec/lowering-runtime/emitc-route.md`
- `.trellis/spec/testing/mlir-testing-contract.md`

Prior task context read:

- `.trellis/tasks/archive/2026-05/05-21-05-21-stage2-rvv-computed-mask-byte-strided-store/prd.md`
- `.trellis/tasks/archive/2026-05/05-20-05-20-stage2-rvv-macc-executable-slice/prd.md`
- `.trellis/tasks/archive/2026-05/05-21-stage2-rvv-standalone-reduction-executable-slice/prd.md`

Initial code surface to inspect:

- `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`
- `include/TianChenRV/Dialect/RVV/IR/RVVConfigContract.h`
- `include/TianChenRV/Support/RuntimeABI.h`
- `lib/Support/RuntimeABIContract.cpp`
- `lib/Dialect/RVV/IR/RVVDialect.cpp`
- `lib/Dialect/RVV/IR/RVVConfigContract.cpp`
- `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp`
- `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`
- `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
- `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
- `lib/Plugin/RVV/Construction/RVVConstructionProtocol.cpp`
- `lib/Target/RVV/RVVTargetSupportBundle.cpp`
- `scripts/rvv_generated_bundle_abi_e2e.py`

## Definition Of Done

- One bounded multiply-add accumulator route is implemented on the typed RVV
  selected-body/provider path with distinct accumulator input and output
  destination.
- Route-supported evidence and executable `ssh rvv` evidence are current to
  this task.
- No legacy/source/descriptor/common-export authority is reintroduced.
- The Trellis task is truthful, finished, archived, and committed when
  complete; otherwise it remains open with an exact continuation point.

## Completion Evidence

- Implemented explicit `lhs,rhs,acc,out,n` runtime ABI order and a distinct
  `accumulator-input-buffer` for bounded `macc_add`.
- Updated selected/pre-realized RVV body realization to materialize `setvl`,
  `with_vl`, lhs/rhs/acc loads, `tcrv_rvv.macc`, and output store.
- Updated route planning/provider/construction/target/script surfaces so the
  route derives ABI parameters, accumulator/result layout mirrors, generated
  header prototype, and executable bundle facts from typed RVV body/config and
  runtime ABI bindings.
- Self-repaired a provider bug where the macc accumulator load still used the
  output ABI. The focused EmitC test now structurally checks the accumulator
  load comes from the `acc` argument and the store writes `out`.
- Generated-bundle dry-run passed for explicit and pre-realized `macc_add` at
  counts `7,16,23`.
- Real `ssh rvv` PASS for explicit and pre-realized `macc_add`, counts
  `7,16,23`, with explicit accumulator, signed positive/negative products, and
  tail sentinel preservation.
- `check-tianchenrv` passed: 259/259.
- `git diff --check` passed.
- Diff-level active-authority scan found no newly introduced positive legacy
  `RVVI32M1`, `rvv-i32m1`, finite `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`,
  descriptor/direct-C/source-export/source-front-door, or public exact intrinsic
  authority.
