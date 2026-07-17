# Stage2 RVV closure-gated strided masked store movement

## Goal

Repair the bounded Stage2 RVV strided masked-store movement owner so
`computed_masked_strided_store` is route-supported from typed `tcrv_rvv` body
facts through `RVVRouteOperandBindingPlan` closure, provider-built
`TCRVEmitCLowerableRoute`, generated bundle evidence, and real `ssh rvv`
correctness.

The concrete route remains the already-established signed i32 / SEW32 / LMUL
m1 computed-mask plus runtime destination byte-strided store slice:

```text
if cmp_lhs[i] < cmp_rhs[i]:
  dst at byte offset i * dst_stride_bytes = src[i]
else:
  dst at byte offset i * dst_stride_bytes remains unchanged
tail and non-strided gaps remain sentinel-preserved
```

This round must move the production/default path from the old merged-vector
shape:

```text
compare + source load + old-destination strided load
  -> masked_move
  -> strided_store
```

to a masked strided-store movement shape where the typed body and RVV plugin
carry destination memory form, runtime byte stride, mask producer, source
payload, inactive-lane no-write policy, runtime n/AVL, dtype/config, materialized
operands, headers, mirrors, and diagnostics.

## Direction Source

- Direction title: `Stage2 RVV closure-gated strided masked store movement`.
- Module owner: RVV plugin-owned route-supported strided masked store movement
  on the corrected `tcrv_rvv` surface, enforced by
  `RouteOperandBindingPlan` closure.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean.
- Initial HEAD: `cc5dde46 rvv: add closure-gated masked load movement`.
- No `.trellis/.current-task` existed, so this task was created from the
  provided Direction Brief before source edits.

## Current Inventory

Inventory is from current RVV dialect, construction, selected-body realization,
route planning/provider, target artifact, generated-bundle tests, and archived
immediate predecessor tasks.

- `masked_unit_load_store` and `computed_masked_unit_load_store` were just
  repaired to use typed `tcrv_rvv.masked_load` facts, old-destination
  passthrough, route operand binding closure, header mirrors, generated-bundle
  evidence, and real `ssh rvv` evidence.
- `masked_unit_store` is a route-supported unit-stride masked-store sibling
  using `tcrv_rvv.masked_store`, explicit runtime mask import, payload source,
  destination output buffer, inactive-lane output preservation, route operand
  binding closure, generated-bundle evidence, and real `ssh rvv` evidence.
- `unit_load_strided_store` and prior computed-mask strided-store tasks
  established runtime destination byte stride, destination byte-stride ABI
  role, strided destination preservation, generated-bundle dry-runs, and
  real `ssh rvv` evidence.
- Current `TypedComputedMaskStridedStorePreRealizedBodyOp` structurally binds
  compare lhs/rhs, source, destination, runtime n/AVL, destination byte stride,
  predicate, mask role/source/form, inactive-lane policy, SEW/LMUL, and policy.
  Its documentation and realization still say the RVV plugin realizes it into
  `load/load/load/strided_load/compare/masked_move/strided_store`.
- Current route planning accepts `computed_masked_strided_store` as a masked
  memory movement route, but its structural checks require one old-destination
  `tcrv_rvv.strided_load`, one `tcrv_rvv.masked_move`, and one
  `tcrv_rvv.strided_store`.
- Current binding plan for `computed_masked_strided_store` is closure-gated and
  includes `cmp_lhs`, `cmp_rhs`, `src`, `dst`, `n`, and `dst_stride_bytes`, but
  materialized uses still reflect `old-dst-load`, `old-dst-stride`,
  `active-source`, and `strided-store` rather than masked strided-store
  suppression.
- `tcrv_rvv.masked_store` is currently unit-stride only:
  `memory_form = "masked-unit-store"` and four operands
  `(buffer, mask, value, vl)`. It does not yet carry a destination stride
  operand or strided masked-store memory form.
- Common EmitC/export must remain neutral; the RVV plugin must derive any
  masked strided-store target leaf, ABI order, header mirrors, and materialized
  operands from typed body/config/runtime facts.

## Requirements

1. Keep scope to the bounded strided masked-store family represented by
   `computed_masked_strided_store`; do not broaden into indexed, segmented,
   gather/scatter, strided loads, reductions, contractions, conversions, or
   dtype/LMUL clone batches.
2. Repair or extend the typed RVV body surface so a valid route structurally
   carries:
   - compare-produced mask or an equivalent explicit typed mask producer for
     this computed-mask route;
   - destination `mem_window` / output buffer role;
   - runtime destination byte stride role;
   - vector source payload role;
   - inactive-lane no-write / destination preservation policy;
   - tail preservation policy;
   - runtime n/AVL;
   - e32m1 dtype/config facts;
   - materialized memory operands and header mirrors.
3. Rewire production/default selected-body realization for
   `computed_masked_strided_store` away from `masked_move + strided_store`
   route authority and toward a typed masked strided-store movement structure.
4. Route planning/provider must derive route support, materialized operands,
   target leaf/profile, headers, mirrors, diagnostics, and runtime ABI order
   from typed body/config/runtime facts and the closure-gated
   `RVVRouteOperandBindingPlan`.
5. The route must fail closed for unsupported indexed/segmented/gather/scatter
   variants, missing compare/mask, missing destination memory window, missing
   destination byte stride, missing vector source, missing runtime n/AVL,
   invalid or stale stride unit, dtype/config mismatch, stale route id,
   old masked-move fallback, mirror/header mismatch, materialized-use mismatch,
   descriptor/direct-C/source-front-door authority, route-id/helper-string
   fallback, and common/export RVV semantic inference.
6. Generated-bundle evidence must use value-distinguishing compare inputs,
   source payloads, destination byte strides greater than one element, mixed
   true/false mask lanes, sentinel-filled destination storage, runtime n/AVL
   variation, and tail/gap preservation checks.
7. Runtime or correctness claims require real `ssh rvv` PASS evidence.

## Acceptance Criteria

- [x] PRD, implement/check context, and task metadata describe this bounded
      strided masked-store movement owner.
- [x] RVV dialect/ODS/verifier and selected-body realization represent
      `computed_masked_strided_store` with typed masked strided-store movement
      facts instead of positive `masked_move + strided_store` authority.
- [x] RVV route planning/provider recognizes the repaired typed structure,
      derives ABI order, byte-stride handling, mask type, vector C type,
      target leaf/profile, headers, materialized operands, and mirrors from
      typed body/config/runtime facts.
- [x] `RVVRouteOperandBindingPlan` closure for
      `computed_masked_strided_store` includes compare lhs/rhs, source payload,
      destination, runtime n/AVL, and destination byte stride with materialized
      uses that prove masked strided-store operands rather than mirror-only
      authority.
- [x] Positive structural/FileCheck tests prove pre-realized and, if current
      surfaces support it without widening scope, explicit selected-body routes
      carry compare-produced mask, source payload, destination, byte stride,
      inactive-lane no-write policy, tail policy, runtime n/AVL, dtype/config,
      binding operands, and header mirrors.
- [x] Negative fail-closed tests cover missing mask/compare, missing source,
      missing destination, missing destination byte stride, wrong stride role or
      stride unit, unsupported memory form, missing runtime n/AVL, mask/vector
      mismatch, dtype/config mismatch, stale or wrong plan id, mirror/header
      mismatch, materialized-use mismatch, old `masked_move` fallback,
      route-id/helper-string fallback, descriptor/direct-C/source-front-door
      authority, and common/export semantic inference where expressible.
- [x] Generated-bundle dry-runs pass for representative
      `computed_masked_strided_store` routes at counts `7,16,23` and
      destination byte strides `4,8,12`.
- [x] Real `ssh rvv` runs pass for the same representative route set or a
      truthful narrower route set if implementation evidence shows an explicit
      route is not yet supported; evidence must prove true-lane strided writes,
      false-lane no-write preservation, untouched-gap preservation,
      runtime n/AVL handling, and tail sentinel preservation.
- [x] Active-authority scan over touched RVV/plugin/export/script/test paths
      finds no new positive `RVVI32M1`, `rvv-i32m1`, finite
      `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`, descriptor/direct-C/source-export,
      source-front-door, public exact intrinsic route authority,
      artifact-name authority, or common/export RVV semantic authority.
- [x] Focused checks, `git diff --check`, `check-tianchenrv`, truthful task
      finish/archive state, clean git status, and one coherent commit complete
      if the task finishes.

## Non-Goals

- No redo of contiguous `masked_load`, `masked_unit_load_store`,
  `computed_masked_unit_load_store`, or `masked_unit_store` except as
  regression anchors.
- No indexed or segmented movement, gather/scatter matrix, strided load family,
  reductions, compare/select expansion, contractions, conversions,
  dtype/LMUL clone batches, scalar fallback, IME, Offload, TensorExt,
  Template/Toy, or future plugin side quest.
- No high-level Linalg/Vector/StableHLO frontend lowering.
- No source-front-door positive route, descriptor-driven computation,
  direct-C/source-export route, dashboard, report-only work, or helper-only
  cleanup as the main result.
- No legacy `RVVI32M1`, `rvv-i32m1`, finite `tcrv_rvv.i32_*`, or
  dtype-prefixed helper route authority.
- No ad-hoc provider fallback around the closure gate and no RVV semantics
  chosen by common EmitC/export.

## Validation Plan

1. Validate and start the Trellis task.
2. Build focused targets needed for `tcrv-opt`, `tcrv-translate`, RVV dialect,
   RVV plugin, construction, route provider, and target artifact checks.
3. Run focused RVV dialect/verifier tests for positive and negative
   computed-mask strided masked-store structure.
4. Run focused selected-body realization, route planning/provider, EmitC, and
   target artifact FileCheck tests.
5. Run touched C++ tests where route/provider/export helpers are covered.
6. Run `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py` and
   `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test` if the script
   changes or generated-bundle evidence is claimed.
7. Run generated-bundle dry-runs for `computed_masked_strided_store` at counts
   `7,16,23` and destination byte strides `4,8,12`.
8. Run real `ssh rvv` correctness for the representative new/repaired routes.
9. Run active-authority scans over touched RVV include/lib/script/test paths.
10. Run `git diff --check`.
11. Run `cmake --build build --target check-tianchenrv -j2`.

## Initial Code Surface

- `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`
- `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`
- `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`
- `lib/Dialect/RVV/IR/RVVDialect.cpp`
- `lib/Plugin/RVV/Construction/RVVConstructionProtocol.cpp`
- `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp`
- `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
- `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
- `lib/Target/RVV/RVVTargetSupportBundle.cpp`
- `scripts/rvv_generated_bundle_abi_e2e.py`
- `test/Dialect/RVV/masked-strided-store-dataflow.mlir`
- `test/Dialect/RVV/pre-realized-computed-mask-strided-store-memory-negative.mlir`
- `test/Target/RVV/explicit-selected-body-artifact-computed-masked-strided-store.mlir`
- `test/Target/RVV/pre-realized-selected-body-artifact-computed-masked-strided-store.mlir`
- `test/Scripts/rvv-generated-bundle-abi-e2e-pre-realized-computed-masked-strided-store-dry-run.test`

## Round Outcome

Implemented and validated in this round:

- Added `tcrv_rvv.masked_strided_store` as the bounded typed destination
  movement op for computed-mask byte-strided stores.
- Rewired pre-realized `computed_masked_strided_store` realization from the old
  `strided_load -> masked_move -> strided_store` shape to
  `compare -> masked_strided_store`.
- Updated RVV route planning/provider to require closure-gated materialized
  uses for compare lhs/rhs, source payload, destination base, runtime n/AVL, and
  destination byte stride.
- Added explicit and pre-realized target fixtures, direct dialect negative
  tests, C++ route/construction test updates, and generated-bundle evidence.
- Real `ssh rvv` PASS covered counts `7,16,23`, destination byte strides
  `4,8,12`, active writes, false-lane no-write behavior, untouched gaps, and
  tail/sentinel preservation.
- Diff-level authority scan found no new positive `RVVI32M1`, `rvv-i32m1`,
  finite `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`, descriptor/direct-C/source-export,
  or source-front-door authority. The only exact intrinsic diff hit is the
  provider-owned target leaf `__riscv_vsse32_v_i32m1_m`, selected after typed
  body/config/runtime closure rather than used as route authority.

## Definition Of Done

- `computed_masked_strided_store` is represented, verified, route-supported,
  materialized, dry-run validated, and runtime-validated on `ssh rvv` through
  typed masked strided-store movement facts and closure-gated operand bindings.
- Existing contiguous masked memory and strided-store slices remain intact.
- The report distinguishes route-supported, generated-bundle dry-run, and
  executable `ssh rvv` evidence.
- No legacy i32/source/descriptor/common-export authority is reintroduced.
- The Trellis task is truthful, finished, archived, and committed when
  complete; otherwise it remains open with an exact continuation point.
