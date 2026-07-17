# Stage2 RVV Runtime Scalar Computed-Mask Horizontal Reduction Boundary

## Goal

Implement one bounded Stage 2 RVV route-supported runtime scalar threshold
computed-mask horizontal signed i32 sum reduction boundary on the corrected
typed `tcrv_rvv` surface:

```text
mask_i = cmp_lhs_i32[i] <= rhs_scalar_i32
out_scalar_i32[0] = seed_i32 + sum(payload_i32[i] where mask_i is true)
inactive lanes are excluded from the sum
tail lanes remain sentinel-preserved
```

This must be a real scalar-output horizontal reduction consumer of the
runtime scalar computed-mask path. Route authority must stay in typed
`tcrv_rvv` body/config/runtime facts, RVV plugin-owned selected-body
realization, route planning/provider, and `RVVRouteOperandBindingPlan` closure.
Common EmitC/export may only materialize provider-built payloads and mirrors.

## Direction Source

- Direction title: `Stage2 RVV closure-gated runtime scalar computed-mask
  horizontal reduction boundary`.
- Module owner: RVV plugin-owned runtime scalar threshold compare feeding
  masked horizontal i32 sum reduction with scalar output/runtime evidence.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean.
- Initial HEAD: `6e8f65ee rvv: add runtime scalar computed mask macc`.
- No `.trellis/.current-task` existed, so this task was created from the
  provided Hermes Direction Brief before source edits.

## Current Inventory

Inventory is from current RVV dialect/config/construction/selected-body
realization/route planning/provider/generated-bundle owners and directly
relevant archived Stage 2 tasks, not status words.

- `computed_mask_standalone_reduce_add` exists and is complete for vector-vs-
  vector compare-produced masks:
  `cmp_lhs[i] <= cmp_rhs[i]` controls `tcrv_rvv.masked_standalone_reduce`.
  It binds `cmp_lhs`, `cmp_rhs`, `src`, `acc`, `out`, and `n`.
- `standalone_reduce_add` exists for scalar-output horizontal reduction without
  a computed mask.
- `runtime_scalar_cmp_select`, `runtime_scalar_cmp_masked_store`,
  `runtime_scalar_cmp_masked_load_store`, and
  `runtime_scalar_cmp_masked_macc_add` exist for runtime scalar threshold
  computed-mask consumers. They prove non-AVL `rhs_scalar` import, `splat`,
  `compare sle`, materialized operand closure, explicit/pre-realized selected
  body paths, generated bundle runs, and real `ssh rvv` evidence.
- The missing production consumer is the combination of those two families:
  `rhs_scalar` threshold compare-produced mask feeding
  `tcrv_rvv.masked_standalone_reduce` with scalar seed/result ABI binding.
- Reusing the old vector-threshold `computed_mask_standalone_reduce_add` as
  evidence would be incorrect because its compare RHS is an input buffer, not a
  runtime scalar threshold.

## Scope

Add or repair one coherent route family using repository naming conventions.
Use `runtime_scalar_cmp_masked_standalone_reduce_add` unless current code
requires an equivalent name.

The bounded semantic boundary is:

- `cmp_lhs`: `lhs-input-buffer`, `const int32_t *`, unit-stride compare lhs.
- `rhs_scalar`: `rhs-scalar-value`, `int32_t`, non-AVL runtime threshold,
  materialized by `tcrv_rvv.splat`.
- `src`: `source-input-buffer`, `const int32_t *`, unit-stride reduction
  payload.
- `acc`: `accumulator-input-buffer`, `const int32_t *`, scalar seed input.
- `out`: `output-buffer`, `int32_t *`, scalar output/result binding.
- `n`: `runtime-element-count`, `size_t`, runtime AVL for `setvl`.
- typed config: signed i32, SEW32, LMUL m1, explicit tail/mask policy.
- body structure: `runtime_abi_value`, `setvl`, `with_vl`, load `cmp_lhs`,
  splat `rhs_scalar`, compare `sle`, load `src`,
  `masked_standalone_reduce`, and scalar-output store.

## Requirements

1. Keep route authority in typed body/config/runtime facts and RVV plugin-owned
   realization/planning/provider code.
2. Do not route through legacy `RVVI32M1`, `rvv-i32m1`, finite positive
   `tcrv_rvv.i32_*`, descriptor/direct-C/source-front-door, artifact-name,
   status/mirror-only, route-id, or helper-string authority.
3. Add selected-body and generated-bundle support for the bounded runtime
   scalar computed-mask standalone reduction route.
4. Add pre-realized selected-body support in the same round unless repository
   evidence shows explicit mode is the only feasible production consumer; if
   not completed, record the exact continuation.
5. `RouteOperandBindingPlan` closure must require materialized uses and header
   mirrors for `cmp_lhs`, `rhs_scalar`, `src`, `acc`, `out`, and `n`, including
   compare lhs, scalar splat/compare RHS, reduction payload, accumulator seed,
   scalar result state/store, and loop-control roles.
6. Unsupported missing threshold lhs, missing rhs scalar, missing payload,
   missing scalar output/result, missing accumulator/initial value, missing
   runtime n/AVL, bad dtype/config, wrong predicate, wrong reduction kind,
   stale or wrong plan id, helper-string selected reduction, constant-threshold
   substitution, mirror-only authority, materialized-use mismatch,
   compare-select fallback, masked-store/load fallback, unmasked reduction
   fallback, macc fallback, route-id/helper fallback, descriptor/direct-C/
   source-front-door, and common/export semantic inference must fail closed
   where expressible.
7. Runtime/correctness evidence must use value-distinguishing threshold and
   payload vectors, positive and negative `rhs_scalar` thresholds, counts `7`,
   `16`, and `23`, scalar result equal to the sum of active masked lanes plus
   seed, inactive-lane exclusion, runtime n/AVL variation, and tail/sentinel
   preservation.

## Acceptance Criteria

- [x] PRD and task context are truthful and point to the RVV/EmitC/testing
      specs plus directly relevant prior task evidence.
- [x] Current computed-mask reduction and runtime scalar computed-mask
      consumer inventory is recorded with exact already-supported and missing
      surfaces.
- [x] The selected RVV body structurally carries threshold lhs load,
      `rhs_scalar` splat, compare predicate, produced mask, reduction payload
      load, scalar seed/result binding, runtime n/AVL, and typed
      SEW/LMUL/policy facts.
- [x] RVV selected-body realization supports the pre-realized runtime scalar
      computed-mask standalone reduction body, or the PRD records why explicit
      mode is the completed route for this round.
- [x] RVV planning/provider derive route support, materialized operands,
      headers, mirrors, target leaf profile, intrinsic spelling, and target
      artifact metadata from typed body/config/runtime facts plus
      `RouteOperandBindingPlan` closure.
- [x] Positive structural tests prove materialized operand and header mirror
      closure for `cmp_lhs`, `rhs_scalar`, `src`, `acc`, `out`, and `n`.
- [x] Negative fail-closed tests cover missing lhs/rhs scalar/payload/acc/out/n,
      unsupported predicate, wrong reduction kind, bad dtype/config, stale or
      wrong plan id, materialized-use mismatch, mirror/header mismatch,
      compare-select, masked-store/load, unmasked reduction, or macc fallback
      claimed as runtime scalar computed-mask reduction, route-id/helper-string
      fallback, descriptor/direct-C/source-front-door authority, and
      common/export semantic inference where expressible.
- [x] Generated-bundle dry-runs pass for explicit and, if implemented,
      pre-realized `runtime_scalar_cmp_masked_standalone_reduce_add` with
      counts `7,16,23` and `rhs_scalar=-37,91`.
- [x] Real `ssh rvv` generated-bundle runs pass for completed route modes,
      proving active-lane scalar accumulation, false-lane exclusion, runtime
      n/AVL handling, scalar output correctness, and tail preservation.
- [x] Active-authority scan over touched RVV/plugin/export/script/test paths
      finds no new positive legacy/source/descriptor/common-export route
      authority.
- [x] Focused checks, `git diff --check`, `check-tianchenrv`, truthful task
      finish/archive state, clean final git status, and one coherent commit
      complete if this task finishes.

## Non-Goals

- No redo of `runtime_scalar_cmp_masked_macc_add`,
  `runtime_scalar_cmp_masked_load_store`,
  `runtime_scalar_cmp_masked_store`, `runtime_scalar_cmp_select`,
  scalar broadcast add/sub/mul, `runtime_i32_splat_store`, widening
  conversion, indexed/segmented/strided/contiguous memory matrices, broad
  reduction operation matrices, predicate clones, or dtype/LMUL clone batches
  except as regression anchors.
- No high-level Linalg/Vector/StableHLO frontend lowering, source-front-door
  positive routes, dashboards, report-only work, helper-only cleanup, global
  autotuning, or future plugin work.
- No common EmitC/export ownership of compare, mask, reduction, accumulator,
  scalar result ABI, dtype/config, runtime roles, result roles, or intrinsic
  choices.

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

- `.trellis/tasks/archive/2026-05/05-21-05-21-stage2-rvv-closure-gated-reduction-accumulation/prd.md`
- `.trellis/tasks/archive/2026-05/05-22-05-22-stage2-rvv-closure-gated-masked-horizontal-reduce-sum-accumulation/prd.md`
- `.trellis/tasks/archive/2026-05/05-22-05-22-stage2-rvv-runtime-scalar-computed-mask-load-merge-boundary/prd.md`
- `.trellis/tasks/archive/2026-05/05-22-stage2-rvv-runtime-scalar-computed-mask-store-boundary/prd.md`
- `.trellis/tasks/archive/2026-05/05-23-stage2-rvv-runtime-scalar-computed-mask-macc-boundary/prd.md`

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
- Existing computed-mask standalone reduction, runtime scalar computed-mask,
  and macc tests as directly relevant anchors.

## Definition Of Done

- One bounded runtime scalar threshold computed-mask scalar-output add
  reduction route is represented, verified, route-supported, materialized
  through the production RVV provider path, dry-run validated, and
  runtime-validated on `ssh rvv`.
- Existing computed-mask standalone reduction, runtime scalar computed-mask
  store/load/macc, compare-select, masked memory, macc, conversion, and
  movement Stage2 routes remain intact.
- The final report distinguishes route-supported evidence, generated-bundle
  dry-run evidence, and executable `ssh rvv` evidence.
- No legacy i32/source/descriptor/common-export authority is reintroduced.
- The Trellis task is truthful, finished, archived, and committed when
  complete; otherwise it remains open with an exact continuation point.
