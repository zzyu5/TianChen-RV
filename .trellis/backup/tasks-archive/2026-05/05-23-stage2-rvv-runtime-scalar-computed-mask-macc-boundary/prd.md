# Stage2 RVV Runtime Scalar Computed-Mask MAcc Boundary

## Goal

Implement one bounded Stage 2 RVV route-supported runtime scalar threshold
computed-mask multiply-add accumulation/store boundary on the corrected typed
`tcrv_rvv` surface:

```text
mask_i = cmp_lhs_i32[i] <= rhs_scalar_i32
out_i32[i] = mask_i ? acc_i32[i] + lhs_i32[i] * rhs_i32[i] : acc_i32[i]
tail lanes remain sentinel-preserved
```

The route must carry compare lhs vector, non-AVL runtime scalar threshold,
compare predicate, compare-produced mask, multiplicand inputs, explicit
accumulator input/passthrough, output store, runtime `n` / AVL, element type,
SEW/LMUL/policy, materialized operands, header mirrors, inactive-lane
accumulator preservation, and artifact/runtime evidence through selected-body
realization, RVV route planning/provider, common EmitC materialization,
generated-bundle execution, and real `ssh rvv` execution.

## Direction Source

- Direction title: `Stage2 RVV closure-gated runtime scalar computed-mask
  multiply-add accumulation boundary`.
- Module owner: RVV plugin-owned route-supported runtime scalar threshold
  compare feeding masked multiply-add accumulation/store on the corrected
  `tcrv_rvv` surface, enforced by `RouteOperandBindingPlan` closure.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean.
- Initial HEAD: `e18a8d89 rvv: add runtime scalar computed mask load store`.
- No `.trellis/.current-task` existed, so this task was created from the
  provided Direction Brief before source edits.

## Current Inventory

Inventory comes from current RVV dialect/config/construction/selected-body
realization/route planning/provider/generated-bundle owners and directly
relevant archived Stage 2 tasks.

- `computed_masked_macc_add` is complete for vector-vs-vector compare:
  compare lhs/rhs buffers produce a mask that feeds `tcrv_rvv.masked_macc`;
  payload lhs/rhs, accumulator input, output, runtime `n`, config facts,
  selected-body realization, binding closure, generated-bundle evidence, and
  real `ssh rvv` evidence already exist.
- `runtime_scalar_cmp_select`, `runtime_scalar_cmp_masked_store`, and
  `runtime_scalar_cmp_masked_load_store` are complete runtime-threshold
  computed-mask consumers. They prove non-AVL `rhs_scalar` ABI import,
  `tcrv_rvv.splat`, compare-produced masks, materialized operand closure,
  explicit/pre-realized selected-body paths, generated artifacts, and real
  `ssh rvv` evidence.
- `macc_add` is complete for unmasked multiply-add accumulation with explicit
  accumulator input and distinct output destination. It is not a computed-mask
  route and must not be claimed as masked accumulation evidence.
- `masked_add`/`masked_sub`/`masked_mul`, compare-select, masked-store, and
  masked-load routes are regression anchors only. They do not carry a fused
  multiply-add accumulator operation.
- The current missing production consumer is a coherent route family where a
  runtime scalar threshold produces the mask that controls `tcrv_rvv.masked_macc`
  and false lanes preserve the explicit accumulator value.

## Scope

Add or repair one coherent route family using repository naming conventions.
If no existing name fits, use `runtime_scalar_cmp_masked_macc_add`
consistently. The bounded semantic boundary is:

- `cmp_lhs`: `lhs-input-buffer`, `const int32_t *`, unit-stride compare lhs
  load.
- `rhs_scalar`: `rhs-scalar-value`, `int32_t`, non-AVL runtime threshold,
  materialized by `tcrv_rvv.splat`.
- `lhs`: `macc-lhs-input-buffer` or nearest repository-local role, `const
  int32_t *`, unit-stride multiplicand lhs load.
- `rhs`: `macc-rhs-input-buffer` or nearest repository-local role, `const
  int32_t *`, unit-stride multiplicand rhs load.
- `acc`: `accumulator-input-buffer`, `const int32_t *`, unit-stride
  accumulator load and inactive-lane passthrough.
- `out`: `output-buffer`, `int32_t *`, unit-stride result store destination.
- `n`: `runtime-element-count`, `size_t`, runtime AVL for `setvl`.
- typed config: signed i32, SEW32, LMUL m1, explicit tail/mask policy.
- body structure: `runtime_abi_value`, `setvl`, `with_vl`, load `cmp_lhs`,
  splat `rhs_scalar`, compare `sle`, load payload `lhs`, load payload `rhs`,
  load `acc`, `masked_macc`, and store `out`.
- inactive-lane semantics: false lanes preserve `acc[i]`; tail sentinel
  remains preserved.

## Requirements

1. Keep route authority in typed `tcrv_rvv` body/config/runtime facts and RVV
   plugin-owned realization/planning/provider code.
2. Do not route through legacy `RVVI32M1`, `rvv-i32m1`, finite positive
   `tcrv_rvv.i32_*`, descriptor/direct-C/source-front-door, artifact-name,
   status/mirror-only, route-id, or helper-string authority.
3. Add selected-body and generated-bundle support for the bounded runtime
   scalar computed-mask masked macc route.
4. Add pre-realized selected-body support in the same round unless repository
   evidence shows explicit mode is the only feasible production consumer; if
   not completed, record the exact continuation.
5. `RouteOperandBindingPlan` closure must require materialized uses and header
   mirrors for `cmp_lhs`, `rhs_scalar`, payload `lhs`, payload `rhs`, `acc`,
   `out`, and `n`, including compare lhs, scalar splat/compare rhs, macc lhs,
   macc rhs, accumulator passthrough, result store, and loop-control roles.
6. Unsupported missing threshold lhs, missing rhs scalar, missing multiplicand,
   missing accumulator/output, missing runtime n/AVL, bad dtype/config, wrong
   predicate, wrong op kind, stale or wrong plan id, helper-string selected
   macc, constant-threshold substitution, mirror-only authority,
   materialized-use mismatch, compare-select fallback, masked-store fallback,
   masked-load fallback, unmasked macc fallback, route-id/helper fallback,
   descriptor/direct-C/source-front-door, and common/export semantic inference
   must fail closed where expressible.
7. Runtime/correctness evidence must use value-distinguishing threshold lhs,
   payload lhs/rhs, accumulator, positive and negative `rhs_scalar` thresholds,
   counts `7`, `16`, and `23`, both accumulated and preserved false lanes,
   runtime n/AVL variation, and tail sentinel preservation.

## Acceptance Criteria

- [ ] PRD and task context are truthful and point to the RVV/EmitC/testing
      specs plus directly relevant prior task evidence.
- [ ] Current computed-mask, runtime scalar threshold, and masked
      macc/accumulation inventory is recorded with exact already-supported and
      missing surfaces.
- [ ] The selected RVV body structurally carries threshold lhs load,
      `rhs_scalar` splat, compare predicate, produced mask, payload lhs/rhs
      loads, accumulator load/passthrough, output store, runtime n/AVL, and
      typed SEW/LMUL/policy facts.
- [ ] RVV selected-body realization supports the pre-realized runtime scalar
      computed-mask masked macc body, or the PRD records why explicit mode is
      the completed route for this round.
- [ ] RVV planning/provider derive route support, materialized operands,
      headers, mirrors, target leaf profile, intrinsic spelling, and target
      artifact metadata from typed body/config/runtime facts plus
      `RouteOperandBindingPlan` closure.
- [ ] Positive structural tests prove materialized operand and header mirror
      closure for `cmp_lhs`, `rhs_scalar`, payload `lhs`, payload `rhs`,
      `acc`, `out`, and `n`.
- [ ] Negative fail-closed tests cover missing threshold lhs/rhs scalar,
      missing multiplicands, missing accumulator/output/n, unsupported
      predicate, wrong op kind, bad dtype/config, stale or wrong plan id,
      materialized-use mismatch, mirror/header mismatch, compare-select,
      masked-store, masked-load, or unmasked macc fallback claimed as runtime
      scalar computed-mask macc, route-id/helper-string fallback,
      descriptor/direct-C/source-front-door authority, and common/export
      semantic inference where expressible.
- [ ] Generated-bundle dry-runs pass for explicit and, if implemented,
      pre-realized `runtime_scalar_cmp_masked_macc_add` with counts `7,16,23`
      and `rhs_scalar` values `-37,91`.
- [ ] Real `ssh rvv` generated-bundle runs pass for completed route modes,
      proving active-lane `acc + lhs * rhs`, false-lane accumulator
      passthrough preservation, runtime n/AVL handling, and tail preservation.
- [ ] Active-authority scan over touched RVV/plugin/export/script/test paths
      finds no new positive legacy/source/descriptor/common-export route
      authority.
- [ ] Focused checks, `git diff --check`, `check-tianchenrv`, truthful task
      finish/archive state, clean final git status, and one coherent commit
      complete if this task finishes.

## Non-Goals

- No redo of `runtime_scalar_cmp_masked_load_store`,
  `runtime_scalar_cmp_masked_store`, `runtime_scalar_cmp_select`,
  scalar broadcast add/sub/mul, `runtime_i32_splat_store`, widening
  conversion, indexed/segmented/strided/contiguous memory matrices, broad
  reduction work, or predicate/dtype/LMUL clone batches except as regression
  anchors.
- No high-level Linalg/Vector/StableHLO frontend lowering, source-front-door
  positive routes, dashboards, report-only work, helper-only cleanup, global
  autotuning, or future plugin work.
- No common EmitC/export ownership of compare, mask, multiply-add,
  accumulator, dtype/config, runtime roles, side-effect roles, result roles, or
  intrinsic choices.

## Validation Plan

1. Validate and start the Trellis task.
2. Build focused targets expected to include `tcrv-opt`, `tcrv-translate`, and
   `tianchenrv-rvv-extension-plugin-test`.
3. Run focused FileCheck/lit tests for explicit and pre-realized target/header
   artifacts plus negative fail-closed surfaces.
4. Run `build/bin/tianchenrv-rvv-extension-plugin-test` if route planning or
   provider helpers are changed.
5. Run `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py` and
   `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test` if the
   script changes or generated-bundle evidence is claimed.
6. Run generated-bundle dry-runs for counts `7,16,23` and
   `rhs_scalar=-37,91`.
7. Run real `ssh rvv` generated-bundle execution for completed route modes and
   the same representative count/threshold set.
8. Run active-authority scan, `git diff --check`, and
   `cmake --build build --target check-tianchenrv -j2`.

## Technical Notes

Relevant specs read:

- `.trellis/spec/index.md`
- `.trellis/spec/extension-plugins/rvv-plugin.md`
- `.trellis/spec/lowering-runtime/emitc-route.md`
- `.trellis/spec/testing/index.md`
- `.trellis/spec/testing/mlir-testing-contract.md`
- `.trellis/spec/variant-pipeline/index.md`

Prior task context read:

- `.trellis/tasks/archive/2026-05/05-22-05-22-stage2-rvv-runtime-scalar-computed-mask-load-merge-boundary/prd.md`
- `.trellis/tasks/archive/2026-05/05-22-stage2-rvv-runtime-scalar-computed-mask-store-boundary/prd.md`
- `.trellis/tasks/archive/2026-05/05-22-05-22-stage2-rvv-runtime-scalar-compare-select-boundary/prd.md`
- `.trellis/tasks/archive/2026-05/05-22-05-22-stage2-rvv-closure-gated-masked-macc-accumulation/prd.md`
- `.trellis/tasks/archive/2026-05/05-21-stage2-rvv-multiply-add-accumulator-executable-slice/prd.md`
- `.trellis/tasks/archive/2026-05/05-21-stage2-rvv-operand-binding-contract-closure-gate/prd.md`

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
- Existing runtime scalar compare-select, runtime scalar computed-mask
  store/load-store, computed-mask masked macc, unmasked macc, and binding-plan
  tests as directly relevant anchors.

## Definition Of Done

- One bounded runtime scalar threshold computed-mask masked multiply-add
  accumulation/store route is represented, verified, route-supported,
  materialized through the production RVV provider path, dry-run validated, and
  runtime-validated on `ssh rvv`.
- Existing runtime scalar compare-select, runtime scalar computed-mask memory,
  computed-mask masked macc, unmasked macc, reductions, conversion, and
  movement Stage2 routes remain intact.
- The final report distinguishes route-supported evidence, generated-bundle
  dry-run evidence, and executable `ssh rvv` evidence.
- No legacy i32/source/descriptor/common-export authority is reintroduced.
- The Trellis task is truthful, finished, archived, and committed when
  complete; otherwise it remains open with an exact continuation point.
