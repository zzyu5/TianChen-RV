# Stage2 RVV Closure-Gated Computed-Mask Composition Boundary

## Goal

Implement one bounded Stage 2 RVV route-supported runtime scalar dual-compare
mask-composition select/store boundary on the corrected typed `tcrv_rvv`
surface:

```text
mask_a_i = cmp_lhs_a_i32[i] <= rhs_scalar_a_i32
mask_b_i = cmp_lhs_b_i32[i] <= rhs_scalar_b_i32
mask_i   = mask_a_i && mask_b_i
out_i32[i] = mask_i ? true_value_i32[i] : false_value_i32[i]
```

The selected RVV body must structurally carry both threshold compare inputs,
both non-AVL runtime scalar thresholds, both compare predicates, both produced
mask values, a typed mask-and composition op, true/false payload vectors,
output store/result, runtime `n`/AVL, and typed SEW/LMUL/policy facts. Route
authority must stay in typed `tcrv_rvv` body/config/runtime facts, RVV
plugin-owned selected-body realization, route planning/provider, and
`RVVRouteOperandBindingPlan` closure. Common EmitC/export may only materialize
provider-built payloads and mirrors.

## Direction Source

- Direction title: `Stage2 RVV closure-gated computed-mask composition boundary`.
- Module owner: RVV plugin-owned route-supported mask composition on the
  corrected `tcrv_rvv` surface, enforced by `RouteOperandBindingPlan` closure.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean.
- Initial HEAD: `12dd54c5 rvv: add runtime scalar computed mask reduction`.
- No `.trellis/.current-task` existed, so this task was created from the
  provided Hermes Direction Brief before source edits.

## Current Inventory

Inventory is from current RVV dialect/config/construction/selected-body
realization/route planning/provider/generated-bundle owners and directly
relevant archived Stage 2 tasks, not status words or artifact names.

- `runtime_scalar_cmp_select` is complete for one runtime scalar threshold
  compare feeding `tcrv_rvv.select` and store. It proves non-AVL
  `rhs_scalar` import, `tcrv_rvv.splat`, compare-produced mask, explicit
  true/false payloads, output store, binding closure, generated-bundle
  dry-runs, and real `ssh rvv` evidence.
- `computed_mask_select` is complete for vector-vs-vector compare-produced
  masks feeding `tcrv_rvv.select`, but it has one compare mask and no
  runtime scalar threshold.
- `runtime_scalar_cmp_masked_store`,
  `runtime_scalar_cmp_masked_load_store`,
  `runtime_scalar_cmp_masked_macc_add`, and
  `runtime_scalar_cmp_masked_standalone_reduce_add` are complete runtime
  scalar computed-mask consumers. They prove one threshold-produced mask can
  drive masked store, load-merge, macc, and horizontal reduction consumers
  through typed body facts, binding closure, explicit/pre-realized routes,
  generated-bundle dry-runs, and real `ssh rvv` evidence.
- The current dialect surface has typed `tcrv_rvv.compare` mask production and
  mask-consuming ops such as `select`, `masked_store`, `masked_load`,
  `masked_macc`, and `masked_standalone_reduce`. It does not currently expose
  a typed `tcrv_rvv` mask composition op such as `mask_and`, and no production
  route family proves two threshold-produced masks are composed before driving
  downstream select/store behavior.
- Reusing a single-compare route as evidence would be incorrect. This task's
  missing production consumer is specifically the composition boundary:
  two runtime scalar threshold compares produce two typed masks, a typed
  mask-and op produces the controlling mask, and that composed mask is the
  operand consumed by `tcrv_rvv.select` before store.

## Scope

Add or repair one coherent route family using repository naming conventions.
Use `runtime_scalar_dual_cmp_mask_and_select` unless current code reveals a
nearer local name.

The bounded semantic boundary is:

- `cmp_lhs_a`: `lhs-input-buffer`, `const int32_t *`, unit-stride compare A lhs.
- `rhs_scalar_a`: `rhs-scalar-value`, `int32_t`, non-AVL runtime threshold A,
  materialized by `tcrv_rvv.splat`.
- `cmp_lhs_b`: `rhs-input-buffer` or nearest repository-local second-compare
  input role, `const int32_t *`, unit-stride compare B lhs.
- `rhs_scalar_b`: second runtime scalar threshold, `int32_t`, non-AVL runtime
  scalar ABI value. If the ABI role enum has no distinct second scalar role,
  add the minimal plugin-owned role needed for closure rather than reusing
  helper strings or constants.
- `true_value`: `true-value-input-buffer`, `const int32_t *`, selected when
  the composed mask is true.
- `false_value`: `false-value-input-buffer`, `const int32_t *`, selected when
  the composed mask is false.
- `out`: `output-buffer`, `int32_t *`, unit-stride result store destination.
- `n`: `runtime-element-count`, `size_t`, runtime AVL for `setvl`.
- typed config: signed i32, SEW32, LMUL m1, explicit tail/mask policy.
- body structure: `runtime_abi_value`, `setvl`, `with_vl`, load
  `cmp_lhs_a`, splat `rhs_scalar_a`, compare A `sle`, load `cmp_lhs_b`,
  splat `rhs_scalar_b`, compare B `sle`, typed `mask_and`, load true/false
  payloads, `select`, and `store`.

## Requirements

1. Keep route authority in typed body/config/runtime facts and RVV plugin-owned
   realization/planning/provider code.
2. Do not route through legacy `RVVI32M1`, `rvv-i32m1`, finite positive
   `tcrv_rvv.i32_*`, descriptor/direct-C/source-front-door, artifact-name,
   status/mirror-only, route-id, or helper-string authority.
3. Add a typed `tcrv_rvv` mask composition surface only as needed for this
   route. It must consume typed mask SSA values and produce a typed mask SSA
   value under the selected VL scope; it must not be a high-level predicate
   framework or one-op-per-intrinsic wrapper batch.
4. Add selected-body and generated-bundle support for the bounded dual runtime
   scalar compare plus mask-and select/store route.
5. Add pre-realized selected-body support in the same round unless repository
   evidence shows explicit mode is the only coherent completed production
   consumer; if not completed, record the exact continuation.
6. `RVVRouteOperandBindingPlan` closure must require materialized uses and
   header mirrors for `cmp_lhs_a`, `rhs_scalar_a`, `cmp_lhs_b`,
   `rhs_scalar_b`, `true_value`, `false_value`, `out`, and `n`, including both
   compare lhs loads, both scalar splats/compare RHS values, both compare calls,
   mask-and input/output use, select true/false operands, output store, and
   loop-control roles.
7. Unsupported missing compare A/B input, missing runtime scalar A/B, missing
   typed mask-and, wrong mask op, wrong predicate, missing true/false payload,
   missing output, missing runtime n/AVL, bad dtype/config, stale or wrong plan
   id, helper-string selected mask, constant-threshold substitution,
   mirror-only authority, materialized-use mismatch, single-compare fallback,
   compare-select fallback without mask composition, masked-store/load/reduction
   /macc fallback claimed as mask composition, route-id/helper-string fallback,
   descriptor/direct-C/source-front-door authority, and common/export semantic
   inference must fail closed where expressible.
8. Runtime/correctness evidence must use value-distinguishing compare and
   payload vectors, positive and negative threshold pairs, counts `7`, `16`,
   and `23`, active lanes equal to the intersection of both compare masks,
   false lanes selected/preserved correctly, runtime n/AVL variation, and
   tail/sentinel preservation.

## Acceptance Criteria

- [ ] PRD and task context are truthful and point to the RVV/EmitC/testing
      specs plus directly relevant prior task evidence.
- [ ] Current compare, mask, select, and runtime scalar computed-mask inventory
      is recorded with exact already-supported and missing surfaces.
- [ ] The selected RVV body structurally carries compare lhs A, runtime scalar
      threshold A, compare lhs B, runtime scalar threshold B, both compare
      predicates, two produced masks, a typed mask-and composition, true/false
      payload loads, output store, runtime n/AVL, and typed SEW/LMUL/policy
      facts.
- [ ] RVV selected-body realization supports the pre-realized dual runtime
      scalar compare mask-and select/store body, or the PRD records why
      explicit mode is the completed route for this round.
- [ ] RVV planning/provider derive route support, materialized operands,
      headers, mirrors, target leaf profile, intrinsic spelling, and target
      artifact metadata from typed body/config/runtime facts plus
      `RVVRouteOperandBindingPlan` closure.
- [ ] Positive structural tests prove materialized operand and header mirror
      closure for `cmp_lhs_a`, `rhs_scalar_a`, `cmp_lhs_b`, `rhs_scalar_b`,
      `true_value`, `false_value`, `out`, and `n`.
- [ ] Negative fail-closed tests cover missing compare A/B, missing scalar A/B,
      missing mask-and, wrong mask op, wrong predicate, missing payload/output
      /runtime, bad dtype/config, stale or wrong plan id, materialized-use
      mismatch, mirror/header mismatch, single-compare/compare-select fallback,
      masked-store/load/reduction/macc fallback claimed as mask composition,
      route-id/helper-string fallback, descriptor/direct-C/source-front-door
      authority, and common/export semantic inference where expressible.
- [ ] Generated-bundle dry-runs pass for explicit and, if implemented,
      pre-realized `runtime_scalar_dual_cmp_mask_and_select` with counts
      `7,16,23` and threshold pairs that exercise empty, partial, and full
      mask intersections.
- [ ] Real `ssh rvv` generated-bundle runs pass for completed route modes,
      proving active-lane intersection behavior, false-lane selection, runtime
      n/AVL handling, and tail/sentinel preservation.
- [ ] Active-authority scan over touched RVV/plugin/export/script/test paths
      finds no new positive legacy/source/descriptor/common-export route
      authority.
- [ ] Focused checks, `git diff --check`, `check-tianchenrv`, truthful task
      finish/archive state, clean final git status, and one coherent commit
      complete if this task finishes.

## Non-Goals

- No redo of `runtime_scalar_cmp_masked_standalone_reduce_add`,
  `runtime_scalar_cmp_masked_macc_add`,
  `runtime_scalar_cmp_masked_load_store`,
  `runtime_scalar_cmp_masked_store`,
  `runtime_scalar_cmp_select`, scalar broadcast/splat-store, widening
  conversion, indexed/segmented/strided/contiguous memory matrices, broad
  reduction operation matrices, OR/XOR/NOT mask algebra batches, predicate
  clones, dtype/LMUL clone batches, or high-level frontend lowering except as
  regression anchors.
- No high-level Linalg/Vector/StableHLO frontend lowering, source-front-door
  positive routes, dashboards, report-only work, helper-only cleanup, global
  autotuning, or future plugin work.
- No common EmitC/export ownership of compare, mask composition, select
  semantics, scalar threshold derivation, predicate choice, dtype/config,
  runtime roles, result roles, or intrinsic choices.

## Validation Plan

1. Validate and start the Trellis task.
2. Build focused targets expected to include `tcrv-opt`, `tcrv-translate`, and
   `tianchenrv-rvv-extension-plugin-test`.
3. Run focused FileCheck/lit tests for explicit and pre-realized target/header
   artifacts plus negative fail-closed surfaces.
4. Run `build/bin/tianchenrv-rvv-extension-plugin-test` if route planning or
   provider helpers are changed.
5. Run `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py` and
   `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test` if the script
   changes or generated-bundle evidence is claimed.
6. Run generated-bundle dry-runs for counts `7,16,23` and representative
   threshold-pair cases.
7. Run real `ssh rvv` generated-bundle execution for completed route modes and
   the same representative count/threshold-pair set.
8. Run active-authority scan, `git diff --check`, and
   `cmake --build build --target check-tianchenrv -j2`.

## Technical Notes

Relevant specs read:

- `.trellis/spec/index.md`
- `.trellis/spec/extension-plugins/rvv-plugin.md`
- `.trellis/spec/lowering-runtime/emitc-route.md`

Prior task context read:

- `.trellis/tasks/archive/2026-05/05-23-05-23-stage2-rvv-runtime-scalar-computed-mask-horizontal-reduction-boundary/prd.md`
- `.trellis/tasks/archive/2026-05/05-23-stage2-rvv-runtime-scalar-computed-mask-macc-boundary/prd.md`
- `.trellis/tasks/archive/2026-05/05-22-05-22-stage2-rvv-runtime-scalar-computed-mask-load-merge-boundary/prd.md`
- `.trellis/tasks/archive/2026-05/05-22-stage2-rvv-runtime-scalar-computed-mask-store-boundary/prd.md`
- `.trellis/tasks/archive/2026-05/05-22-05-22-stage2-rvv-runtime-scalar-compare-select-boundary/prd.md`
- `.trellis/tasks/archive/2026-05/05-21-stage2-rvv-operand-binding-contract-closure-gate/prd.md`
- `.trellis/tasks/archive/2026-05/05-21-05-21-stage2-rvv-computed-mask-vector-select-executable-slice/prd.md`

Initial implementation surface to inspect:

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
- Existing compare/select and computed-mask tests as directly relevant anchors.

## Definition Of Done

- One bounded dual runtime scalar threshold compare mask-and select/store route
  is represented, verified, route-supported, materialized through the
  production RVV provider path, dry-run validated, and runtime-validated on
  `ssh rvv`.
- Existing runtime scalar computed-mask store/load/macc/reduction,
  compare-select, computed-mask select, masked memory, macc, conversion, and
  movement Stage 2 routes remain intact.
- The final report distinguishes route-supported evidence, generated-bundle
  dry-run evidence, and executable `ssh rvv` evidence.
- No legacy i32/source/descriptor/common-export authority is reintroduced.
- The Trellis task is truthful, finished, archived, and committed when
  complete; otherwise it remains open with an exact continuation point.
