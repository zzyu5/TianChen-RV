# Stage2 RVV widening dot-product reduction executable slice

## Goal

Implement one bounded Stage2 RVV signed widening dot-product reduction
executable slice:

```text
out_i32[0] = acc_seed_i32 + sum_i sign_extend(lhs_i16[i]) * sign_extend(rhs_i16[i])
```

The selected RVV path must carry typed i16 source roles for `lhs` and `rhs`, an
i32 scalar accumulator seed or one-element accumulator input role, an i32 scalar
result/output role, source vector config `SEW16 LMUL mf2`, reduction/result
config `SEW32 LMUL m1`, signed widening dot-product reduction operation kind,
accumulator/result boundary layout, unit-stride memory form, tail/mask policy,
runtime `n` / AVL, route planning, generated artifact emission, and real
`ssh rvv` correctness evidence for representative counts.

This is one bounded low-level RVV contraction-supporting scalar reduction
slice. It is not matmul lowering, a reduction matrix, a broad contraction
framework, source-front-door work, descriptor/direct-C/source-export work, or a
performance task.

## Direction Source

- Hermes Direction Brief: `Stage2 RVV widening dot-product reduction executable slice`.
- Module owner: RVV plugin-owned horizontal widening dot-product reduction path
  for one bounded signed i16 SEW16 LMUL mf2 lhs/rhs pair plus signed i32 scalar
  accumulator seed/result.
- Repo root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean.
- Initial HEAD: `8228b984 rvv: add widening macc executable slice`.
- No `.trellis/.current-task` existed, so this task was created from the
  Hermes brief before source edits.

## Current Repository Facts

- Current HEAD already supports signed i16mf2-to-i32m1 widening conversion.
- Current HEAD already supports signed i16mf2 x i16mf2 plus i32m1 per-lane
  widening multiply-accumulate through typed body/config/runtime facts,
  selected-body realization, route planning/provider, generated bundle support,
  and real `ssh rvv` evidence.
- Current HEAD already supports a bounded i32 reduce-add route with structural
  accumulator/result layout facts, but that route reduces one i32 vector input
  with a vector seed layout; it is not the full signed i16 widening
  dot-product scalar boundary required here.
- The production gap is the cross-class composition: two i16 source loads,
  signed widening products, horizontal i32 accumulation across runtime `n`,
  scalar seed addition, and scalar output store must be represented as typed RVV
  facts and consumed by the RVV plugin before common EmitC/export.
- Dtype, SEW, LMUL, signedness, dot-product kind, reduction identity/seed,
  scalar result role, policy, and intrinsic mapping must be validated or
  derived from typed body/config/runtime facts. They must not come from route
  ids, ABI strings, artifact names, test names, descriptors, exact intrinsic
  spelling, helper names, or common EmitC/export code.
- `tcrv.exec` declares ABI/runtime roles only. RVV compute/config, selected-body
  realization, legality, intrinsic mapping, and fail-closed diagnostics remain
  RVV plugin-owned.

## Requirements

1. Keep scope to one signed `i16mf2 x i16mf2 -> i32 scalar` unit-stride
   widening dot-product reduction route with `lhs`, `rhs`, `acc_seed` or
   `acc`, `out`, and runtime `n/AVL` ABI roles.
2. The selected/pre-realized body or explicit typed body must structurally
   carry source buffer roles, scalar accumulator seed/result boundary roles,
   source element type/config, reduction/result element type/config, signed
   widening dot-product reduction kind, accumulator seed/layout, scalar result
   layout, memory form, policy, runtime `n/AVL`, and ABI role order.
3. `RVVSelectedBodyRealization` must materialize only legal generic typed
   structure for this slice if starting from a pre-realized selected body:
   destination-config `setvl/with_vl`, i16mf2 lhs load, i16mf2 rhs load,
   explicit accumulator seed/result boundary, signed widening product plus
   horizontal reduction structure or equivalent typed RVV dot-reduction op, and
   scalar i32 output store boundary.
4. RVV route planning must derive ABI order, i16 input C pointer types, i32
   scalar seed/result C types, vector C types, widening product and reduction
   intrinsic leaves, load/store leaves, route metadata mirrors, and targeted
   diagnostics from typed body/config/runtime facts.
5. Provider/materializer must consume the provider-built plan. Common
   materialization must not infer dtype, widening, dot-product kind, reduction
   layout, scalar result role, memory form, policy, intrinsic choice, or ABI
   meaning.
6. Unsupported dot-product/reduction kind, invalid SEW/LMUL relation, missing
   lhs/rhs/seed/result config, missing accumulator seed role, missing scalar
   output role, source/result dtype mismatch, missing `n/AVL`, wrong ABI
   role/order, stale route-id authority, and incomplete typed body structure
   must fail closed with targeted diagnostics.
7. Generated-bundle evidence must use positive and negative i16 operands plus a
   nonzero i32 seed so expected output proves signed widening products,
   horizontal accumulation across all runtime `n`, seed addition, scalar result
   storage, and no extra output writes.
8. Real `ssh rvv` evidence is required for any executable correctness claim.

## Acceptance Criteria

- [x] PRD, implement/check context, and task metadata describe this bounded
      signed i16 widening dot-product reduction slice.
- [x] Positive selected/pre-realized or explicit typed RVV body structurally
      carries lhs/rhs i16mf2 source facts, i32 scalar accumulator seed/result
      facts, signed widening dot-product reduction kind, accumulator/result
      layout, unit-stride memory form, policy, runtime `n/AVL`, and ABI roles.
- [x] `RVVSelectedBodyRealization` materializes the bounded slice into explicit
      typed low-level RVV structure before provider route construction.
- [x] RVVEmitCRoutePlanning derives source/seed/result ABI types, vector types,
      widening product/reduction leaves, route metadata, and runtime ABI order
      from typed body/config facts.
- [x] Positive route/materialization tests prove typed i16mf2 inputs and i32
      scalar accumulator/result facts reach `TCRVEmitCLowerableRoute` and
      provider-owned route metadata.
- [x] Negative fail-closed tests cover unsupported reduction kind, invalid
      SEW/LMUL pair, missing/mismatched source or accumulator/result config,
      missing accumulator seed role, missing scalar result role, missing
      `n/AVL`, stale route-id authority, and incomplete typed body structure.
- [x] Generated-bundle dry-run passes for the widening dot-product reduction
      slice at counts `7,16,23`.
- [x] Real `ssh rvv` correctness passes for counts `7,16,23` if executable
      correctness is claimed, with expected output proving negative and
      positive signed products, horizontal full-`n` accumulation, nonzero seed
      addition, and scalar output/tail sentinel preservation.
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

- No matmul lowering, high-level Linalg/Vector/StableHLO lowering, broad
  contraction framework, segmented reductions, masked reductions, unsigned
  variants, dtype/LMUL clone batches, source-front-door positive route,
  one-intrinsic wrapper dialect, dashboard, report-only inventory, helper-only
  refactor, or performance claim.
- No Scalar, IME, Offload, TensorExt, future plugin work, global autotuning,
  readiness state machine, source-shape clone batch, or Template/Toy examples.
- No descriptor-driven computation, direct-C/source-export route restoration,
  route-id authority, artifact-name authority, or compatibility wrapper
  preserving old i32 authority.
- No new dtype-prefixed helper op families such as `tcrv_rvv.i32_reduction_*`,
  `tcrv_rvv.i32_accumulator_*`, or `tcrv_rvv.i32_macc`.
- No Python implementation of compiler core, dialects, passes, plugin
  registry, capability model, lowering, or emission. Python changes, if any,
  are limited to generated-bundle tooling/evidence.

## Validation Plan

1. Validate and start the Trellis task.
2. Build focused targets expected to include `tcrv-opt`, `tcrv-translate`,
   `tianchenrv-rvv-dialect-test`, `tianchenrv-rvv-extension-plugin-test`,
   `tianchenrv-construction-protocol-common-test`, and
   `tianchenrv-target-artifact-export-test`.
3. Run focused RVV dialect/verifier lit tests for positive and negative
   widening dot-product reduction structure.
4. Run focused selected-body realization, route planning/provider, EmitC, and
   target artifact lit/FileCheck tests.
5. Run touched C++ tests where route/provider/export helpers are covered.
6. Run `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py` and
   `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test` if the script
   changes or generated-bundle evidence is claimed.
7. Run generated-bundle dry-run for the new widening dot-product reduction op
   at counts `7,16,23`.
8. Run real `ssh rvv` correctness for the new widening dot-product reduction
   op at counts `7,16,23` if executable correctness is claimed.
9. Run active-authority scans over active RVV include/lib/script/test paths.
10. Run `git diff --check`.
11. Run broader `check-tianchenrv` if shared compiler behavior changes.

## Technical Notes

Relevant specs read:

- `.trellis/spec/index.md`
- `.trellis/spec/extension-plugins/rvv-plugin.md`
- `.trellis/spec/lowering-runtime/emitc-route.md`
- `.trellis/spec/variant-pipeline/generation-selection-tuning.md`
- `.trellis/spec/testing/mlir-testing-contract.md`
- `.trellis/spec/core-dialect/tcrv-exec-contract.md`

Prior task context read:

- `.trellis/tasks/archive/2026-05/05-20-stage2-rvv-widening-macc-executable-slice/prd.md`
- `.trellis/tasks/archive/2026-05/05-20-stage2-rvv-reduce-add-production-executable-slice/prd.md`
- `.trellis/tasks/archive/2026-05/05-20-stage2-rvv-widening-conversion/prd.md`
- `.trellis/tasks/archive/2026-05/05-20-stage2-rvv-runtime-scalar-broadcast/prd.md`
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
- `lib/Target/RVV/RVVTargetSupportBundle.cpp`
- `scripts/rvv_generated_bundle_abi_e2e.py`
- Existing reduce-add, widening conversion, widening macc, and generated-bundle
  tests.

## Definition Of Done

- One bounded signed i16 widening dot-product reduction route structurally
  carries source, accumulator seed, scalar result, operation, policy, runtime,
  and ABI facts through typed RVV body structure into RVV route
  planning/provider metadata and generated artifacts.
- Fresh focused positive, negative, generated-bundle, and runtime evidence is
  current to this task when claimed.
- No legacy i32/source/descriptor/common-export authority is reintroduced.
- The task is finished/archived only if production code/tests/evidence are
  present in this round; otherwise it remains open with an exact continuation
  point.

## Completion Evidence

- Focused build passed for `tcrv-opt`, `tcrv-translate`,
  `tianchenrv-rvv-dialect-test`, `tianchenrv-rvv-extension-plugin-test`,
  `tianchenrv-construction-protocol-common-test`, and
  `tianchenrv-target-artifact-export-test`.
- Focused C++ tests passed:
  `build/bin/tianchenrv-rvv-dialect-test`,
  `build/bin/tianchenrv-rvv-extension-plugin-test`,
  `build/bin/tianchenrv-construction-protocol-common-test`, and
  `build/bin/tianchenrv-target-artifact-export-test`.
- New pre-realized dot-product reduction artifact test passed all manual RUN
  commands for realization, emission-plan metadata, and header export.
- `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py` passed.
- `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test` passed.
- Generated-bundle dry-run passed for `widening_dot_reduce_add` at counts
  `7,16,23`.
- Real `ssh rvv` generated-bundle harness passed for counts `7,16,23`, with
  output:
  `signed_horizontal_dot seed_added scalar_output tail_preserved`.
- `cmake --build build --target check-tianchenrv -j2` passed `237/237`.
- Active-authority scan found no new positive `RVVI32M1`, `rvv-i32m1`,
  finite positive `tcrv_rvv.i32_reduction_*`, `tcrv_rvv.i32_accumulator_*`,
  `tcrv_rvv.i32_macc`, source-seed, descriptor, direct-C, or source-export
  authority in this slice; remaining hits are existing fail-closed/negative
  tests or explicit unsupported-mode diagnostics.
- `git diff --check` passed.
