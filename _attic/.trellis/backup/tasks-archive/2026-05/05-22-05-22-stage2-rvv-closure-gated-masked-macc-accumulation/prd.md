# Stage2 RVV Closure-Gated Masked Multiply-Add Accumulation

## Goal

Implement one bounded Stage2 RVV contraction-supporting arithmetic owner:
computed-mask masked multiply-add accumulation on the corrected typed
`tcrv_rvv` surface, enforced by RVV plugin-owned legality, selected-body
realization, route planning/provider logic, and
`RVVRouteOperandBindingPlan` closure.

The concrete bounded behavior is signed i32 / SEW32 / LMUL m1 unit-stride
masked vector multiply-add with an explicit accumulator input and output:

```text
if cmp_lhs[i] < cmp_rhs[i]:
  out[i] = acc[i] + lhs[i] * rhs[i]
else:
  out[i] = acc[i]
tail and sentinel lanes remain preserved
```

This route must be selected from `tcrv.exec` RVV variant structure and typed
low-level `tcrv_rvv` facts. It must not be inferred from route ids, helper
strings, artifact names, metadata mirrors, descriptors, source-front-door
markers, direct-C/source exports, separate mul/add fallback, masked_move
fallback, or common EmitC/export semantics.

## Direction Source

- Direction title: `Stage2 RVV closure-gated masked multiply-add accumulation`.
- Module owner: RVV plugin-owned route-supported masked vector multiply-add
  accumulation on the corrected `tcrv_rvv` surface, enforced by
  `RouteOperandBindingPlan` closure.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean.
- Initial HEAD: `6fca5a6c rvv: add closure-gated segment2 masked store`.
- No `.trellis/.current-task` existed, so this task was created from the
  provided Direction Brief before source edits.

## Current Inventory

Inventory is from current RVV dialect/config/construction/selected-body
realization/route planning/provider/generated-bundle surfaces and closest
archived tasks.

- Existing `macc_add` is the nearest accumulator route. It already carries
  `lhs`, `rhs`, explicit `acc`, `out`, runtime `n` / AVL, `tcrv_rvv.macc`,
  accumulator/result layout mirrors, `RVVRouteOperandBindingPlan` ID
  `rvv-route-operand-binding:macc_add.v1`, explicit and pre-realized fixtures,
  generated-bundle evidence, and prior real `ssh rvv` evidence. It is
  unmasked.
- Existing `masked_add` / `masked_sub` / `masked_mul` are elementwise masked
  arithmetic routes using compare-produced mask and inactive passthrough. They
  are not accumulator-carrying multiply-add routes and must not be claimed as
  fused/accumulator evidence.
- Existing computed-mask standalone reduction and computed-mask widening dot
  routes provide mask provenance, compare predicate, accumulator handling,
  neutral/merge behavior, route metadata, generated-bundle patterns, and
  fail-closed diagnostics for masked accumulation-style routes.
- The latest segment2 masked-store task provides the freshest closure-gated
  operand binding pattern for compare lhs/rhs, payloads, destination,
  runtime `n`, header mirrors, generated-bundle dry-runs, real `ssh rvv`, and
  active-authority scan discipline.
- The current gap is the coherent route family for masked macc accumulation:
  no active operation kind such as `computed_masked_macc_add`, no typed
  `tcrv_rvv.masked_macc` dataflow op, no pre-realized computed-mask macc body,
  no provider-owned masked macc route plan, and no generated-bundle evidence
  proving false-lane accumulator passthrough rather than add-only, multiply-only,
  masked elementwise, or unmasked macc behavior.

## Requirements

1. Keep scope to one coherent signed i32 / SEW32 / LMUL m1 route family. This
   PRD uses `computed_masked_macc_add`; use repository-local naming if current
   implementation evidence requires an equally bounded name.
2. Selected RVV bodies must structurally carry:
   - compare lhs and compare rhs buffers producing the mask;
   - lhs and rhs vector payload buffers;
   - accumulator input buffer and accumulator passthrough for inactive lanes;
   - output buffer for the accumulator result;
   - multiply-add operation kind;
   - compare predicate and mask source/form facts;
   - accumulator/result layout facts;
   - runtime `n` / AVL;
   - dtype/config facts including SEW, LMUL, tail policy, and mask policy.
3. If a pre-realized body is used, RVV selected-body realization must consume
   the pre-realized facts into explicit typed `tcrv_rvv` body structure before
   route construction. It must not change computation semantics, dtype
   semantics, parameter roles, selected variant origin, required capabilities,
   dispatch/fallback behavior, inactive-lane accumulator passthrough policy, or
   runtime `n` / AVL.
4. Route planning/provider must derive route support, ABI order, materialized
   operands, target leaf/profile, headers, mirrors, diagnostics, and runtime
   behavior from typed body/config/capability/runtime facts plus the
   closure-gated `RVVRouteOperandBindingPlan`.
5. Unsupported missing accumulator, missing lhs/rhs payload, missing compare
   mask, missing output, missing runtime role, dtype/config mismatch, stale
   route id, wrong binding plan id, missing materialized use, mirror/header
   mismatch, separate mul/add fallback, masked elementwise fallback,
   `masked_move` fallback, descriptor/direct-C/source-front-door authority, and
   common/export semantic inference must fail closed with targeted diagnostics.
6. Generated-bundle evidence must use value-distinguishing compare inputs,
   lhs/rhs payloads, accumulator values, mixed true/false mask lanes, false-lane
   accumulator passthrough preservation, active-lane `acc + lhs * rhs`, tail and
   sentinel preservation, and runtime `n`/AVL variation.
7. Runtime/correctness claims require real `ssh rvv` PASS evidence.

## Acceptance Criteria

- [x] PRD, implement/check context, and task metadata describe this bounded
      masked multiply-add accumulation owner.
- [x] RVV dialect/ODS/verifier and selected-body realization represent the
      route with typed masked macc facts, not positive route-id/helper/metadata
      authority.
- [x] RVV route planning/provider recognizes the typed structure and derives
      ABI order, compare lhs/rhs, lhs/rhs payloads, accumulator input,
      accumulator passthrough, output, runtime `n`/AVL, mask/predicate facts,
      vector/mask C types, target leaf/profile, headers, materialized operands,
      and mirrors from body/config/runtime facts.
- [x] `RVVRouteOperandBindingPlan` closure includes compare lhs/rhs, lhs/rhs
      payloads, accumulator, output, and runtime `n`/AVL with materialized uses
      that prove masked macc operands rather than mirror-only authority.
- [x] Positive structural/FileCheck tests prove explicit selected-body and, if
      feasible in this bounded round, pre-realized computed-mask routes carry
      compare-produced mask, lhs/rhs payloads, accumulator input/passthrough,
      accumulator result, multiply-add kind, runtime `n`/AVL, dtype/config,
      binding operands, header mirrors, and provider-supported mirrors.
- [x] Negative fail-closed tests cover missing lhs/rhs payload, missing
      accumulator, missing mask/compare, missing runtime role, dtype/config
      mismatch, stale or wrong plan id, materialized-use mismatch, mirror/header
      mismatch, separate mul/add route claimed as fused macc, masked elementwise
      or masked_move fallback, route-id/helper-string fallback,
      descriptor/direct-C/source-front-door authority, and common/export
      semantic inference where expressible.
- [x] Generated-bundle dry-runs pass for representative explicit and
      pre-realized computed-mask masked macc routes at runtime counts `7,16,23`.
- [x] Real `ssh rvv` PASS evidence covers representative new routes and proves
      active-lane macc arithmetic, false-lane accumulator passthrough, runtime
      `n`/AVL handling, output tail/sentinel preservation, and value patterns
      that distinguish macc from add-only or multiply-only behavior.
- [x] Active-authority scan over touched RVV/plugin/export/script/test paths
      finds no new positive `RVVI32M1`, `rvv-i32m1`, finite
      `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`, descriptor/direct-C/source-export,
      source-front-door, public exact intrinsic route authority,
      artifact-name authority, or common/export RVV semantic authority.
- [x] Focused checks, `git diff --check`, `check-tianchenrv`, truthful finish/
      archive state, clean git status, and one coherent commit complete if this
      task finishes.

## Non-Goals

- No redo of segmented, indexed, strided, contiguous masked memory movement,
  compare/select, standalone reduction, widening dot reduction, or existing
  unmasked `macc_add` except as regression anchors.
- No high-level matmul, Linalg/Vector/StableHLO frontend lowering, tensor
  contraction ops, full GEMM kernels, broad contraction scheduling, reductions
  beyond accumulator passthrough evidence, dtype/LMUL clone batches,
  source-front-door positive routes, dashboards, report-only work, helper-only
  cleanup, or future plugin work.
- No legacy `RVVI32M1`, `rvv-i32m1`, finite `tcrv_rvv.i32_*`, exact intrinsic
  spelling authority, descriptor-driven direct C, or common EmitC/export
  semantic inference.
- No route through separate multiply plus add as the claimed fused/accumulator
  route.
- No ad-hoc provider fallback around the closure gate.

## Validation Plan

1. Validate and start the Trellis task.
2. Build focused targets needed for `tcrv-opt`, `tcrv-translate`, RVV dialect,
   RVV plugin, construction, route provider, and target artifact checks.
3. Run focused RVV dialect/verifier tests for positive and negative typed
   masked macc structure.
4. Run focused selected-body realization, route planning/provider, EmitC, and
   target artifact FileCheck tests.
5. Run touched C++ tests where route/provider/export helpers are covered.
6. Run `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py` and
   `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test` if the script
   changes or generated-bundle evidence is claimed.
7. Run generated-bundle dry-runs for representative explicit and pre-realized
   masked macc routes at counts `7,16,23`.
8. Run real `ssh rvv` correctness for representative new routes.
9. Run active-authority scans over touched RVV include/lib/script/test paths.
10. Run `git diff --check`.
11. Run `cmake --build build --target check-tianchenrv -j2`.

## Initial Code Surface

- `include/TianChenRV/Dialect/RVV/IR/RVVConfigContract.h`
- `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`
- `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`
- `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`
- `lib/Dialect/RVV/IR/RVVConfigContract.cpp`
- `lib/Dialect/RVV/IR/RVVDialect.cpp`
- `lib/Plugin/RVV/Construction/RVVConstructionProtocol.cpp`
- `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp`
- `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
- `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
- `lib/Target/RVV/RVVTargetSupportBundle.cpp`
- `scripts/rvv_generated_bundle_abi_e2e.py`
- Existing `macc_add`, masked elementwise, computed-mask standalone reduction,
  computed-mask widening dot, compare/select, and generated-bundle tests as
  directly relevant anchors.

## Technical Notes

Relevant specs read:

- `.trellis/spec/index.md`
- `.trellis/spec/extension-plugins/rvv-plugin.md`
- `.trellis/spec/lowering-runtime/emitc-route.md`
- `.trellis/spec/testing/index.md`
- `.trellis/spec/testing/mlir-testing-contract.md`
- `.trellis/spec/guides/plugin-locality-review-guide.md`
- `.trellis/spec/guides/compute-boundary-review-guide.md`

Prior task context read:

- `.trellis/tasks/archive/2026-05/05-22-stage2-rvv-closure-gated-two-field-segmented-masked-store-movement/prd.md`
- `.trellis/tasks/archive/2026-05/05-21-stage2-rvv-multiply-add-accumulator-executable-slice/prd.md`
- `.trellis/tasks/archive/2026-05/05-21-stage2-rvv-operand-binding-contract-closure-gate/prd.md`
- `.trellis/tasks/archive/2026-05/05-21-05-21-stage2-rvv-closure-gated-reduction-accumulation/prd.md`

## Completion Evidence

- Production owners changed:
  `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`,
  `include/TianChenRV/Dialect/RVV/IR/RVVConfigContract.h`,
  `lib/Dialect/RVV/IR/RVVConfigContract.cpp`,
  `lib/Dialect/RVV/IR/RVVDialect.cpp`,
  `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`,
  `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`,
  `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp`,
  `lib/Plugin/RVV/Construction/RVVConstructionProtocol.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`,
  `lib/Target/RVV/RVVTargetSupportBundle.cpp`, target export test ownership,
  and `scripts/rvv_generated_bundle_abi_e2e.py`.
- Added route family:
  `computed_masked_macc_add`, memory form
  `computed-mask-unit-stride-macc`, runtime ABI order
  `cmp_lhs,cmp_rhs,lhs,rhs,acc,out,n`, typed op
  `tcrv_rvv.masked_macc`, and pre-realized op
  `tcrv_rvv.typed_computed_mask_macc_pre_realized_body`.
- Selected-body realization evidence:
  pre-realized computed-mask macc is rewritten into explicit `setvl`, compare
  lhs/rhs loads, payload lhs/rhs loads, accumulator load, compare-produced mask,
  `tcrv_rvv.masked_macc`, and output store. FileCheck verifies the pre-realized
  op is consumed and no separate `tcrv_rvv.macc`, masked_move, descriptor, or
  route-id/helper-string authority is used.
- Binding closure evidence:
  `rvv-route-operand-binding:computed_masked_macc_add.v1` binds compare
  lhs/rhs, payload lhs/rhs, accumulator input plus inactive-lane passthrough,
  output store, runtime `n`/AVL, materialized operands, and header mirrors.
  Provider materialization consumes that binding and fails closed on missing or
  mismatched materialized roles.
- Runtime evidence:
  explicit and pre-realized generated bundles both passed real `ssh rvv` for
  counts `7,16,23`. The harness proved active-lane `acc + lhs * rhs`,
  false-lane accumulator preservation, add-only and multiply-only
  distinguishing values, runtime n/AVL variation, and tail/sentinel
  preservation.
- Focused self-repair:
  shortened the route operand binding mirror to stay under the metadata length
  cap, updated construction protocol conformance for the new generic
  `tcrv_rvv.masked_macc` route, made root evidence JSON checks order
  insensitive, and updated fail-closed expected op lists after adding the new
  typed compute op.
- Checks passed:
  focused dialect/verifier tests, explicit and pre-realized target FileCheck
  tests, generated-bundle dry-runs, script py_compile and self-test, real
  `ssh rvv` explicit/pre-realized runs, `git diff --check`, active-authority
  scans, and `ninja -C build check-tianchenrv` with 322/322 tests.
- Routes intentionally not converted:
  segmented/indexed/strided/contiguous masked memory movement, compare/select
  expansion, standalone reductions, widening dot reductions, dtype/LMUL clone
  batches, frontend lowering, source-front-door routes, high-level matmul/GEMM,
  dashboards, and future plugin work were left untouched by design.

## Definition Of Done

- One bounded masked multiply-add accumulation route is implemented on the
  typed RVV selected-body/provider path with compare-produced mask, explicit
  accumulator input, inactive-lane accumulator passthrough, and output
  destination.
- Route-supported evidence and executable `ssh rvv` evidence are current to
  this task.
- No legacy/source/descriptor/common-export authority is reintroduced.
- The Trellis task is truthful, finished, archived, and committed when
  complete; otherwise it remains open with an exact continuation point.
