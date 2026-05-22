# Stage2 RVV Closure-Gated Runtime Scalar Compare-Select Boundary

## Goal

Implement one bounded Stage 2 RVV runtime scalar threshold compare-select/store
boundary on the corrected typed `tcrv_rvv` surface:

```text
out_i32[i] = (lhs_i32[i] <= rhs_scalar_i32) ? true_value_i32[i] : false_value_i32[i]
```

The route must start from selected RVV body structure, carry the non-AVL
runtime scalar threshold through `tcrv_rvv.splat`, produce a compare mask from
typed body facts, consume that mask in `tcrv_rvv.select`, store the selected
value, and use `RouteOperandBindingPlan` closure before artifact export or
runtime evidence is claimed.

## Direction Source

- Direction title: `Stage2 RVV closure-gated runtime scalar compare-select
  boundary`.
- Module owner: RVV plugin-owned route-supported runtime i32 scalar threshold
  compare plus select/store on the corrected `tcrv_rvv` surface.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean.
- Initial HEAD: `32d2697f rvv: add closure-gated scalar vector add boundary`.
- No `.trellis/.current-task` existed, so this task was created from the
  provided Direction Brief before source edits.

## Current Inventory

Inventory is from the current RVV dialect/config/construction/selected-body
realization/route planning/provider/generated-bundle owners plus the archived
runtime scalar-vector add and computed-mask vector select tasks.

- `computed_mask_select` is route-supported for vector-vs-vector compare:
  `cmp_lhs`, `cmp_rhs`, `true_value`, `false_value`, `out`, and `n` flow
  through typed loads, `tcrv_rvv.compare`, `tcrv_rvv.select`, `tcrv_rvv.store`,
  route planning/provider, generated artifacts, and `ssh rvv` evidence.
- `scalar_broadcast_add` and `runtime_i32_splat_store` prove that a
  non-AVL `rhs_scalar` runtime ABI value can be imported, splatted through
  `tcrv_rvv.splat`, closure-gated by `RouteOperandBindingPlan`, materialized
  into generated artifacts, and executed on `ssh rvv`.
- Current planning explicitly rejects compare/select with a RHS broadcast or
  scalar splat. That is the intended missing consumer for this task.
- Current `TypedCompareSelectPreRealizedBodyOp` supports only vector RHS
  compare/select, and `TypedComputedMaskSelectPreRealizedBodyOp` supports only
  vector compare operands with explicit true/false payloads. Neither models a
  runtime scalar threshold.

## Scope

Add or repair one coherent route family named using repository conventions, with
the expected semantic boundary:

- `lhs`: `lhs-input-buffer`, `const int32_t *`, unit-stride compare lhs vector
  load.
- `rhs_scalar`: `rhs-scalar-value`, `int32_t`, non-AVL runtime threshold,
  materialized by `tcrv_rvv.splat`.
- `true_value`: `true-value-input-buffer`, `const int32_t *`, selected when the
  compare predicate is true.
- `false_value`: `false-value-input-buffer`, `const int32_t *`, selected when
  the compare predicate is false.
- `out`: `output-buffer`, `int32_t *`, unit-stride store destination.
- `n`: `runtime-element-count`, `size_t`, runtime AVL for `setvl`.
- typed config: signed i32, SEW32, LMUL m1, tail agnostic / mask agnostic.
- body structure: `runtime_abi_value`, `setvl`, `with_vl`, `load lhs`,
  `splat rhs_scalar`, `compare`, `load true_value`, `load false_value`,
  `select`, `store`.
- bounded predicate: `sle`, so runtime tests can distinguish positive and
  negative thresholds while exercising true and false lanes.

## Requirements

1. Keep route authority in typed `tcrv_rvv` body/config/runtime facts and RVV
   plugin-owned realization/planning/provider code.
2. Do not route through legacy `RVVI32M1`, `rvv-i32m1`, finite positive
   `tcrv_rvv.i32_*`, descriptor/direct-C/source-front-door, artifact-name, or
   helper-string authority.
3. Add explicit selected-body artifact/generated-bundle support for the runtime
   scalar compare-select route.
4. Add pre-realized selected-body support if it can be completed coherently in
   this round; otherwise leave explicit support as the completed production
   boundary and document the continuation point.
5. `RouteOperandBindingPlan` closure must require materialized uses and header
   mirrors for `lhs`, `rhs_scalar`, `true_value`, `false_value`, `out`, and
   `n`.
6. Unsupported missing lhs, missing rhs scalar, missing true/false payload,
   missing output, missing runtime n/AVL, bad dtype/config, unsupported
   predicate, stale plan id, scalar constant substitution, mirror-only
   authority, route-id/helper fallback, descriptor/source-front-door/direct-C,
   and common/export semantic inference must fail closed.
7. Runtime/correctness evidence must use value-distinguishing lhs vectors,
   positive and negative `rhs_scalar` thresholds, counts `7`, `16`, and `23`,
   true and false select lanes, runtime n/AVL variation, and tail sentinel
   preservation.

## Acceptance Criteria

- [x] PRD and task context are truthful and point to the RVV/EmitC specs and
      directly relevant prior task evidence.
- [x] Current compare/select plus scalar-runtime inventory is recorded with
      exact already supported and missing surfaces.
- [x] The selected RVV body structurally carries lhs load, rhs scalar splat,
      compare predicate, produced mask, true/false payload loads, output store,
      runtime n/AVL, and typed SEW/LMUL/policy facts.
- [x] RVV selected-body realization supports the pre-realized runtime scalar
      compare-select body or the PRD records why explicit mode is the bounded
      completed route for this round.
- [x] RVV planning/provider derive route support, materialized operands,
      headers, mirrors, target leaf profile, and target artifact metadata from
      typed body/config/runtime facts plus `RouteOperandBindingPlan` closure.
- [x] Positive structural tests prove materialized operand and header mirror
      closure for all six operands.
- [x] Negative fail-closed tests cover missing/wrong scalar threshold binding,
      missing lhs/true/false/out/n, unsupported predicate, bad dtype/config,
      stale plan id, materialized-use mismatch, and attempts to claim
      compare-select through scalar add/splat-only, route id, helper string,
      descriptor/direct-C/source-front-door, or common/export inference.
- [x] Generated-bundle dry-runs pass for explicit and, if implemented,
      pre-realized runtime scalar compare-select with counts `7,16,23` and
      `rhs_scalar` values `-37,91`.
- [x] Real `ssh rvv` generated-bundle runs pass for the completed route modes,
      proving true/false lane behavior and tail sentinel preservation.
- [x] Active-authority scan over touched RVV/plugin/export/script/test paths
      finds no new positive legacy/source/descriptor/common-export route
      authority.
- [x] Focused checks, `git diff --check`, `check-tianchenrv`, truthful task
      finish/archive state, clean final git status, and one coherent commit
      complete if this task finishes.

## Completion Evidence

- Production path added `runtime_scalar_cmp_select` as an RVV selected-body
  operation and runtime-scalar-compare-select memory form, with ABI order
  `lhs,rhs_scalar,true_value,false_value,out,n`.
- Explicit selected-body and pre-realized selected-body routes both materialize
  `load lhs -> splat rhs_scalar -> load true/false -> compare sle -> select ->
  store`, with closure-gated `RouteOperandBindingPlan` metadata and header
  mirrors.
- Self-repair 1: route sequence verification initially passed
  `tcrv_rvv.compare` as the RHS source for this route; it now passes the
  materialized `tcrv_rvv.splat` source, matching the route plan.
- Self-repair 2: the first route operand binding mirror exceeded the 512-byte
  artifact metadata limit; the route now exports a compact mirror while keeping
  full materialized-use names inside the provider binding plan.
- Generated-bundle dry-runs passed for explicit and pre-realized
  `runtime_scalar_cmp_select`, counts `7,16,23`, and `rhs_scalar` values
  `-37,91`.
- Real `ssh rvv` runs passed for explicit and pre-realized
  `runtime_scalar_cmp_select`; every count/threshold case reported both
  true/false lane behavior and tail preservation.
- `check-tianchenrv` passed `334/334`.

## Non-Goals

- No redo of `runtime_i32_splat_store`, `scalar_broadcast_add`, computed-mask
  vector select, reductions, macc, widening conversion, or memory movement
  except as regression anchors.
- No scalar add/sub/mul clone batch, broad compare predicate matrix,
  dtype/LMUL clone batch, high-level Linalg/Vector/StableHLO frontend lowering,
  source-front-door positive route, dashboard, report-only work, or helper-only
  cleanup.
- No common EmitC/export ownership of compare/select semantics, scalar
  threshold derivation, predicate choice, dtype/config, runtime roles, or result
  roles.

## Validation Plan

1. Validate the task metadata.
2. Build focused tools/tests:
   `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-rvv-extension-plugin-test -j2`.
3. Run `build/bin/tianchenrv-rvv-extension-plugin-test`.
4. Run focused FileCheck for target/header artifact, conversion/materialization,
   pre-realized negative, and script dry-run tests.
5. Run generated-bundle dry-run and real `ssh rvv` for counts `7,16,23` and
   `rhs_scalar=-37,91`.
6. Run active-authority scan, `git diff --check`, and
   `cmake --build build --target check-tianchenrv -j2`.
