# Stage2 RVV Closure-Gated Runtime Scalar Broadcast Boundary

## Goal

Implement one bounded Stage 2 RVV runtime scalar-to-vector broadcast-store
owner on the corrected typed `tcrv_rvv` surface:

```text
out_i32[i] = rhs_scalar_i32
```

for active lanes under runtime `n` / AVL, with tail sentinel preservation.
The repository-facing route name for this round is `runtime_i32_splat_store`
unless current code reveals a stricter local naming convention during
implementation.

This is not the existing `scalar_broadcast_add/sub/mul` elementwise family.
Those routes already cover `lhs vector load + rhs_scalar splat + binary +
store`. This task covers the missing boundary where the runtime scalar itself
is splatted to a vector and stored, with no lhs load and no binary fallback.

## Direction Source

- Direction title: `Stage2 RVV closure-gated runtime scalar broadcast
  boundary`.
- Module owner: RVV plugin-owned route-supported runtime i32 scalar-to-vector
  broadcast on the corrected `tcrv_rvv` surface, enforced by
  `RouteOperandBindingPlan` closure.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean.
- Initial HEAD: `300f5bc3 rvv: add explicit widening conversion evidence`.
- No `.trellis/.current-task` existed, so this task was created from the
  provided Direction Brief before source edits.

## Current Inventory

Inventory is from current RVV dialect/config/construction/selected-body
realization/route planning/provider/generated-bundle surfaces, plus archived
scalar-broadcast and widening conversion tasks.

- Current runtime scalar support includes:
  - runtime ABI role `rhs-scalar-value`;
  - generic typed `tcrv_rvv.splat`;
  - pre-realized binary memory form `rhs-scalar-broadcast`;
  - selected-body realization to `setvl/with_vl/load/splat/binary/store`;
  - route family `scalar_broadcast_add`, `scalar_broadcast_sub`, and
    `scalar_broadcast_mul`;
  - generated-bundle dry-run and real `ssh rvv` support for scalar-broadcast
    elementwise routes.
- Current scalar-broadcast elementwise route support structurally requires an
  lhs load and a binary compute op. It is not a pure scalar splat-store
  boundary.
- Current explicit `tcrv_rvv.splat` verifier requires the scalar operand to be
  an explicit `rhs-scalar-value` runtime ABI binding and already rejects buffer
  role misuse.
- The current route planner has closure support through
  `RVVRouteOperandBindingPlan`, but no dedicated plan for `rhs_scalar,out,n`
  scalar splat-store.
- The current generated-bundle harness accepts runtime scalar addends for
  elementwise scalar broadcast; it needs a bounded splat-store expectation and
  runtime checks proving every active lane equals the runtime scalar.

## Scope

Convert and prove one coherent route family:

- `runtime_i32_splat_store`
  - `rhs_scalar`: `rhs-scalar-value`, `int32_t`, explicit runtime scalar
    value, materialized `tcrv_rvv.splat` operand, header mirror.
  - `out`: `output-buffer`, `int32_t *`, unit-stride vector store,
    materialized output memory, header mirror.
  - `n`: `runtime-element-count`, `size_t`, setvl AVL, loop control, header
    mirror.
  - typed config: signed i32, SEW32, LMUL m1, tail agnostic / mask agnostic.
  - body structure: `runtime_abi_value`, `setvl`, `with_vl`, `splat`,
    `store`.

## Requirements

1. Add or repair one production RVV route whose selected RVV bodies
   structurally carry scalar runtime value, output memory, vector broadcast
   result, runtime `n` / AVL, element type, SEW, LMUL, and policy facts.
2. Provide explicit typed selected-body support from `tcrv_rvv.runtime_abi_value`
   through `tcrv_rvv.setvl`, `tcrv_rvv.splat`, and `tcrv_rvv.store`.
3. Provide pre-realized selected-body support if the repository's current
   realization surface has a local fit. If it requires a new pre-realized body
   op, the op must be RVV plugin-local and must realize into the same explicit
   typed body before route construction.
4. Route planning/provider must derive route support, runtime ABI order,
   materialized operands, target leaf/profile, headers, mirrors, diagnostics,
   and runtime behavior from typed body/config/runtime facts plus
   `RVVRouteOperandBindingPlan` closure.
5. Common EmitC/export must remain neutral: it may materialize the
   provider-built route payload, but it must not infer scalar broadcast
   semantics, scalar value, dtype/config, runtime roles, or result roles.
6. Unsupported missing scalar runtime, missing output, missing `n`/AVL, wrong
   dtype/config, stale route id, helper-string selected broadcast, scalar
   constant substitution, binary fallback claimed as splat-store,
   mirror/header-only authority, descriptor/direct-C/source-front-door
   authority, and common/export semantic inference must fail closed where the
   current surfaces can express them.
7. Generated-bundle evidence must use positive and negative runtime scalar
   values, runtime counts `7`, `16`, and `23`, active-lane equality to the
   runtime scalar, and tail/sentinel preservation.
8. Runtime/correctness claims require real `ssh rvv` PASS evidence.

## Acceptance Criteria

- [x] Current scalar-broadcast inventory is recorded with exact active,
      completed, and missing bounded routes.
- [x] `runtime_i32_splat_store` has explicit selected-body target artifact
      coverage proving runtime scalar value, output memory, vector splat
      result, SEW32/LMUL m1/policy facts, runtime `n` / AVL, route operand
      binding plan, binding summary, provider-supported mirrors, and generated
      header mirrors.
- [x] If added, pre-realized `runtime_i32_splat_store` support realizes into
      `setvl/with_vl/splat/store` before route construction.
- [x] Provider emission consumes `RVVRouteOperandBindingPlan` for
      `rhs_scalar`, `out`, and `n` materialized operands and fails closed on
      missing/wrong plan, output/scalar role swaps, materialized-use mismatch,
      runtime ABI mismatch, config mismatch, or stale mirror/header metadata.
- [x] Negative coverage proves scalar constants, helper strings, stale route
      ids, missing runtime scalar/output/n, bad dtype/config, binary fallback,
      descriptor/direct-C/source-front-door authority, and common/export
      semantic inference cannot become `runtime_i32_splat_store` authority.
- [x] Generated-bundle dry-runs pass for explicit and, if completed,
      pre-realized `runtime_i32_splat_store`, counts `7,16,23`, with at least
      two runtime scalar values including a negative value.
- [x] Real `ssh rvv` generated-bundle runs pass for explicit and, if
      completed, pre-realized `runtime_i32_splat_store`, proving active lanes
      equal the runtime scalar, runtime `n`/AVL variation, and tail/sentinel
      preservation.
- [x] Active-authority scan over touched RVV include/lib/script/test paths
      finds no new positive `RVVI32M1`, `rvv-i32m1`, finite positive
      `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`, descriptor/direct-C/source-export,
      source-front-door, public exact intrinsic route authority,
      artifact-name authority, or common/export RVV semantic authority.
- [x] Focused checks, `git diff --check`, `check-tianchenrv`, truthful task
      finish/archive state, clean git status, and one coherent commit complete
      if this task finishes.

## Completion Evidence

- Added explicit and pre-realized `runtime_i32_splat_store` surfaces carrying
  `rhs_scalar`, `out`, `n`, SEW32, LMUL m1, agnostic policy, vector splat
  result, and unit-stride store through typed RVV body structure.
- Added RouteOperandBindingPlan closure for `rhs_scalar,out,n` with
  materialized-use mirrors:
  `runtime-scalar-splat-call`, `materialized-store-base`, `setvl-avl`,
  `loop-control`, and generated header mirrors.
- Added provider/materialization support that emits the splat/store route
  without lhs load or binary fallback and leaves common EmitC/export neutral.
- Added negative fail-closed coverage for wrong op/config, stale authority
  metadata, missing or swapped scalar/output/runtime roles, binary fallback,
  constants, stale route ids, descriptor/direct-C/source-front-door residue,
  materialized-use mismatch, and mirror/header mismatch.
- Generated-bundle dry-runs and real `ssh rvv` runs passed for explicit and
  pre-realized selected bodies with runtime counts `7,16,23` and scalar values
  `-37,91`; every active lane equaled `rhs_scalar` and tail sentinels were
  preserved.

## Non-Goals

- No redo of existing scalar-broadcast add/sub/mul, widening conversion,
  reduction, masked macc, memory movement, compare/select, elementwise
  arithmetic, or contraction routes except as regression anchors.
- No broad broadcast matrix, dtype/LMUL clone batch, masked broadcast family,
  source-front-door positive route, high-level Linalg/Vector/StableHLO
  frontend lowering, dashboard, report-only work, or helper-only cleanup.
- No route through legacy `RVVI32M1`, `rvv-i32m1`, finite `tcrv_rvv.i32_*`,
  exact intrinsic spelling authority, descriptor-driven direct C, or common
  EmitC/export semantic inference.

## Validation Plan

1. Validate and start this Trellis task.
2. Run focused dialect/verifier coverage for `tcrv_rvv.splat` and any new
   pre-realized splat-store body surface.
3. Add and run focused explicit selected-body target/header checks for
   `runtime_i32_splat_store`.
4. Add and run focused pre-realized target/header checks if the pre-realized
   surface is implemented in this round.
5. Run focused route materialization and negative fail-closed FileCheck
   coverage.
6. Run focused C++ route operand binding plan validation if the changed
   behavior is not fully visible in lit/FileCheck.
7. Run `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py` and
   `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test` if the script
   changes.
8. Run generated-bundle dry-runs for explicit and, if completed, pre-realized
   `runtime_i32_splat_store`, counts `7,16,23`, scalar values `-37,91`.
9. Run real `ssh rvv` for the same representative cases after dry-runs pass.
10. Run active-authority scans over touched RVV include/lib/script/test paths.
11. Run `git diff --check`.
12. Run `cmake --build build --target check-tianchenrv -j2`.

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
- Existing scalar-broadcast/splat and widening conversion target, dialect,
  generated-bundle, and provider tests.

## Technical Notes

Relevant specs read:

- `.trellis/spec/index.md`
- `.trellis/spec/extension-plugins/rvv-plugin.md`
- `.trellis/spec/lowering-runtime/emitc-route.md`
- `.trellis/spec/testing/mlir-testing-contract.md`
- `.trellis/spec/variant-pipeline/index.md`
- `.trellis/spec/plugin-protocol/index.md`

Prior task context read:

- `.trellis/tasks/archive/2026-05/05-22-05-22-stage2-rvv-closure-gated-typed-widening-conversion-boundary/prd.md`
- `.trellis/tasks/archive/2026-05/05-20-stage2-rvv-runtime-scalar-broadcast/prd.md`
- `.trellis/tasks/archive/2026-05/05-20-stage2-rvv-vector-scalar-broadcast-executable-path/prd.md`
- `.trellis/tasks/archive/2026-05/05-21-stage2-rvv-runtime-scalar-broadcast-add-executable-slice/prd.md`
- `.trellis/tasks/archive/2026-05/05-21-05-21-stage2-rvv-runtime-scalar-broadcast-add-production/prd.md`
- `.trellis/tasks/archive/2026-05/05-21-05-21-stage2-rvv-scalar-broadcast-elementwise-family-consolidation/prd.md`
