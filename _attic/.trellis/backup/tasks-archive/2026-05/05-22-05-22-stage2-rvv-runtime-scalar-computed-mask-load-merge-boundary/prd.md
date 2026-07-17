# Stage2 RVV Runtime Scalar Computed-Mask Masked Load-Merge Boundary

## Goal

Implement one bounded Stage 2 RVV route-supported runtime scalar threshold
computed-mask contiguous masked load-merge/store boundary on the corrected
typed `tcrv_rvv` surface:

```text
mask_i = lhs_i32[i] <= rhs_scalar_i32
merged_i = mask_i ? src_i32[i] : old_dst_i32[i]
dst_i32[i] = merged_i
tail lanes remain sentinel-preserved
```

The route must carry lhs vector payload, non-AVL runtime scalar threshold,
compare predicate, compare-produced mask, source memory load, old-destination
passthrough/merge value, output store, runtime `n` / AVL, element type,
SEW/LMUL/policy, materialized operands, header mirrors, inactive-lane
preservation, and artifact/runtime evidence through selected-body realization,
RVV route planning/provider, common EmitC materialization, generated-bundle
execution, and real `ssh rvv` execution.

## Direction Source

- Direction title: `Stage2 RVV closure-gated runtime scalar computed-mask
  load-merge boundary`.
- Module owner: RVV plugin-owned route-supported runtime scalar threshold
  compare feeding a computed-mask masked contiguous load with
  passthrough/merge and store on the corrected `tcrv_rvv` surface, enforced by
  `RouteOperandBindingPlan` closure.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean.
- Initial HEAD: `d919e67d rvv: add runtime scalar computed mask store`.
- No `.trellis/.current-task` existed, so this task was created from the
  provided Direction Brief before source edits.

## Current Inventory

Inventory comes from current RVV dialect/config/construction/selected-body
realization/route planning/provider/generated-bundle owners and directly
relevant archived Stage 2 tasks.

- `runtime_scalar_cmp_masked_store` is complete for runtime scalar threshold
  compare-produced masks controlling masked store side effects. It binds
  `lhs`, `rhs_scalar`, `src`, `dst`, and `n`; realizes `load lhs -> splat
  rhs_scalar -> compare sle -> load src -> masked_store`; closure-gates
  materialized uses and mirrors; and has generated-bundle plus real `ssh rvv`
  evidence.
- `runtime_scalar_cmp_select` is complete for runtime scalar threshold compare
  feeding `tcrv_rvv.select` and store. It proves non-AVL scalar ABI import,
  scalar splat, compare-produced mask, materialized operand closure, generated
  artifacts, and `ssh rvv` evidence for positive and negative thresholds.
- `masked_unit_load_store` is complete for runtime mask-buffer controlled
  contiguous masked data load with old-destination passthrough and unit store.
  It realizes `mask_load`, old destination load, `tcrv_rvv.masked_load`, and
  `store`.
- `computed_masked_unit_load_store` is complete for vector-vs-vector
  compare-produced masks controlling contiguous masked data load with
  passthrough and store. It proves the typed `tcrv_rvv.masked_load` surface and
  RouteOperandBindingPlan closure for compare-produced masks, but its compare
  RHS is a vector buffer, not a runtime scalar threshold.
- The missing production consumer for this round is a contiguous masked data
  load-merge where the mask is produced in the selected typed RVV body by
  comparing `lhs` to a runtime scalar threshold, then consumed directly by
  `tcrv_rvv.masked_load` with old-destination passthrough and a following
  store.

## Scope

Add or repair one coherent route family using repository naming conventions.
If no existing name fits, use `runtime_scalar_cmp_masked_load_store`
consistently. The bounded semantic boundary is:

- `lhs`: `lhs-input-buffer`, `const int32_t *`, unit-stride compare lhs load.
- `rhs_scalar`: `rhs-scalar-value`, `int32_t`, non-AVL runtime threshold,
  materialized by `tcrv_rvv.splat`.
- `src`: `source-input-buffer`, `const int32_t *`, unit-stride source memory
  for active masked-load lanes.
- `dst`: `output-buffer`, `int32_t *`, unit-stride passthrough source and
  store destination; false lanes preserve old destination values.
- `n`: `runtime-element-count`, `size_t`, runtime AVL for `setvl`.
- typed config: signed i32, SEW32, LMUL m1, explicit tail/mask policy.
- body structure: `runtime_abi_value`, `setvl`, `with_vl`, `load lhs`,
  `splat rhs_scalar`, `compare sle`, `load old dst passthrough`,
  `masked_load src, mask, passthrough`, `store`.
- inactive-lane semantics: false lanes preserve old destination values and tail
  sentinel remains preserved.

## Requirements

1. Keep route authority in typed `tcrv_rvv` body/config/runtime facts and RVV
   plugin-owned realization/planning/provider code.
2. Do not route through legacy `RVVI32M1`, `rvv-i32m1`, finite positive
   `tcrv_rvv.i32_*`, descriptor/direct-C/source-front-door, artifact-name, or
   helper-string authority.
3. Add selected-body and generated-bundle support for the bounded runtime
   scalar computed-mask masked load-merge route.
4. Add pre-realized selected-body support in the same round unless repository
   evidence shows explicit mode is the only feasible production consumer; if
   not completed, record the exact continuation.
5. `RouteOperandBindingPlan` closure must require materialized uses and header
   mirrors for `lhs`, `rhs_scalar`, `src`, `dst`, and `n`, including
   masked-load source, passthrough, and store roles.
6. Unsupported missing lhs, missing rhs scalar, missing source, missing
   passthrough/destination, missing runtime n/AVL, bad dtype/config, wrong
   predicate, stale plan id, helper-string selected mask, constant-threshold
   substitution, mirror-only authority, materialized-use mismatch,
   compare-select fallback, masked-store fallback, unmasked-load fallback,
   route-id/helper fallback, descriptor/direct-C/source-front-door, and
   common/export semantic inference must fail closed where expressible.
7. Runtime/correctness evidence must use value-distinguishing lhs/source/dst
   buffers, positive and negative `rhs_scalar` thresholds, counts `7`, `16`,
   and `23`, both loaded and preserved false lanes, runtime n/AVL variation,
   and tail sentinel preservation.

## Acceptance Criteria

- [ ] PRD and task context are truthful and point to the RVV/EmitC/testing
      specs plus directly relevant prior task evidence.
- [ ] Current computed-mask, runtime scalar compare, and contiguous masked-load
      inventory is recorded with exact already-supported and missing surfaces.
- [ ] The selected RVV body structurally carries lhs load, rhs scalar splat,
      compare predicate, produced mask, source masked-load, old destination
      passthrough, output store, runtime n/AVL, and typed SEW/LMUL/policy facts.
- [ ] RVV selected-body realization supports the pre-realized runtime scalar
      computed-mask masked load-merge body, or the PRD records why explicit
      mode is the completed route for this round.
- [ ] RVV planning/provider derive route support, materialized operands,
      headers, mirrors, target leaf profile, and target artifact metadata from
      typed body/config/runtime facts plus `RouteOperandBindingPlan` closure.
- [ ] Positive structural tests prove materialized operand and header mirror
      closure for `lhs`, `rhs_scalar`, `src`, `dst`, and `n`, including
      `masked_load` source/passthrough/store roles.
- [ ] Negative fail-closed tests cover missing lhs/rhs scalar/source/dst/n,
      missing passthrough, unsupported predicate, bad dtype/config, stale plan
      id, materialized-use mismatch, compare-select or masked-store fallback,
      unmasked-load fallback, route-id/helper-string fallback,
      descriptor/direct-C/source-front-door authority, and common/export
      semantic inference where expressible.
- [ ] Generated-bundle dry-runs pass for explicit and, if implemented,
      pre-realized `runtime_scalar_cmp_masked_load_store` with counts
      `7,16,23` and `rhs_scalar` values `-37,91`.
- [ ] Real `ssh rvv` generated-bundle runs pass for completed route modes,
      proving active masked loads, false-lane passthrough preservation, store
      results, and tail preservation.
- [ ] Active-authority scan over touched RVV/plugin/export/script/test paths
      finds no new positive legacy/source/descriptor/common-export route
      authority.
- [ ] Focused checks, `git diff --check`, `check-tianchenrv`, truthful task
      finish/archive state, clean final git status, and one coherent commit
      complete if this task finishes.

## Non-Goals

- No redo of `runtime_scalar_cmp_masked_store`,
  `runtime_scalar_cmp_select`, scalar broadcast add/sub/mul,
  `runtime_i32_splat_store`, widening conversion, reductions, macc,
  indexed/segmented/strided memory routes, or broad masked-memory matrices
  except as regression anchors.
- No scalar add/sub/mul clone batch, broad compare predicate matrix, dtype/LMUL
  clone batch, high-level Linalg/Vector/StableHLO frontend lowering,
  source-front-door positive route, dashboard, report-only work, or helper-only
  cleanup.
- No common EmitC/export ownership of compare, masked-load, passthrough,
  threshold, predicate, mask/source/destination semantics, dtype/config,
  runtime roles, merge roles, or result roles.

## Validation Plan

1. Validate and start the Trellis task.
2. Build focused targets expected to include `tcrv-opt`, `tcrv-translate`, and
   `tianchenrv-rvv-extension-plugin-test`.
3. Run focused FileCheck/lit tests for explicit and pre-realized target/header
   artifacts plus negative fail-closed surfaces.
4. Run `build/bin/tianchenrv-rvv-extension-plugin-test` if route planning or
   provider helpers are changed.
5. Run generated-bundle dry-runs for counts `7,16,23` and
   `rhs_scalar=-37,91`.
6. Run real `ssh rvv` generated-bundle execution for completed route modes and
   the same representative count/threshold set.
7. Run active-authority scan, `git diff --check`, and
   `cmake --build build --target check-tianchenrv -j2`.

## Technical Notes

Relevant specs read:

- `.trellis/spec/index.md`
- `.trellis/spec/extension-plugins/rvv-plugin.md`
- `.trellis/spec/lowering-runtime/emitc-route.md`
- `.trellis/spec/testing/mlir-testing-contract.md`

Prior task context read:

- `.trellis/tasks/archive/2026-05/05-22-stage2-rvv-runtime-scalar-computed-mask-store-boundary/prd.md`
- `.trellis/tasks/archive/2026-05/05-22-stage2-rvv-masked-memory-movement-coverage/prd.md`
- `.trellis/tasks/archive/2026-05/05-22-05-22-stage2-rvv-runtime-scalar-compare-select-boundary/prd.md`
- `.trellis/tasks/archive/2026-05/05-21-stage2-rvv-route-operand-abi-binding/prd.md`

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
- Existing runtime scalar compare-select, runtime scalar computed-mask store,
  computed-mask unit load-store, masked-load, and binding-plan tests.

## Definition Of Done

- One bounded runtime scalar threshold computed-mask contiguous masked
  load-merge/store route is represented, verified, route-supported,
  materialized through the production RVV provider path, dry-run validated, and
  runtime-validated on `ssh rvv`.
- Existing runtime scalar compare-select, runtime scalar computed-mask store,
  masked unit load-store, computed-mask unit load-store, reductions, macc,
  conversion, and movement Stage2 routes remain intact.
- The final report distinguishes route-supported evidence, generated-bundle
  dry-run evidence, and executable `ssh rvv` evidence.
- No legacy i32/source/descriptor/common-export authority is reintroduced.
- The Trellis task is truthful, finished, archived, and committed when
  complete; otherwise it remains open with an exact continuation point.
