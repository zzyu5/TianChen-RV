# Stage2 RVV Closure-Gated Typed Widening Conversion Boundary

## Goal

Implement one bounded Stage 2 RVV signed widening conversion owner on the
corrected typed `tcrv_rvv` surface, with route support, materialized operands,
metadata mirrors, generated bundle behavior, and runtime evidence derived from
typed body/config/runtime facts plus `RVVRouteOperandBindingPlan` closure.

The concrete bounded owner for this round is `widen_i32_to_i64`:

```text
out_i64[i] = sign_extend_i64(lhs_i32[i])
```

with unit-stride source and destination memory, runtime `n` / AVL,
source `i32` SEW32 LMUL m1, destination `i64` SEW64 LMUL m2, tail/mask
agnostic policy, and signed widening relation
`signed-i32m1-to-i64m2`.

## Direction Source

- Direction title: `Stage2 RVV closure-gated typed widening conversion
  boundary`.
- Module owner: RVV plugin-owned route-supported signed widening conversion
  on the corrected `tcrv_rvv` surface, enforced by
  `RouteOperandBindingPlan` closure.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean.
- Initial HEAD: `e1ff72f5 rvv: validate masked horizontal reduction closure`.
- No `.trellis/.current-task` existed, so this task was created from the
  provided Direction Brief before source edits.

## Current Inventory

Inventory is from current RVV dialect/config/construction/selected-body
realization/route planning/provider/generated-bundle surfaces and archived
conversion / operand-binding tasks.

- Current conversion operation inventory has two active route-supported and
  executable signed widening conversion routes:
  - `widen_i32_to_i64`: source `i32/m1`, destination `i64/m2`, relation
    `signed-i32m1-to-i64m2`, runtime ABI order `lhs,out,n`.
  - `widen_i16_to_i32`: source `i16/mf2`, destination `i32/m1`, relation
    `signed-i16mf2-to-i32m1`, runtime ABI order `lhs,out,n`.
- Current pre-realized conversion route support already uses
  `tcrv_rvv.typed_widening_conversion_pre_realized_body`, realizes it to
  `setvl/with_vl/load/widening_convert/store`, derives
  `rvv-route-operand-binding:widen_i32_to_i64.v1` or
  `rvv-route-operand-binding:widen_i16_to_i32.v1`, and has generated-bundle
  dry-run plus prior `ssh rvv` evidence.
- Current explicit typed `tcrv_rvv.widening_convert` dialect coverage exists in
  `test/Dialect/RVV/generic-widening-conversion-dataflow.mlir`, but there is
  no explicit selected-body target artifact fixture or generated-bundle
  evidence for `widen_i32_to_i64`. That is the bounded production/evidence gap
  this round will close.
- The RVV provider already owns conversion route mapping and consumes
  `RVVRouteOperandBindingPlan` for conversion source load, convert source,
  destination store, source/result config mirrors, relation mirrors, runtime
  loop/AVL, and header mirrors. This round must verify that current owner and
  repair only current-head gaps.

## Scope

Convert and prove this bounded submodule:

- `widen_i32_to_i64`
  - `lhs`: `lhs-input-buffer`, `const int32_t *`, materialized unit-stride
    source load, conversion source, source config mirror `src-i32m1`, signed
    relation mirror, header mirror.
  - `out`: `output-buffer`, `int64_t *`, materialized unit-stride destination
    store, conversion result, destination config mirror `res-i64m2`, signed
    relation mirror, header mirror.
  - `n`: `runtime-element-count`, `size_t`, setvl AVL, loop control, header
    mirror.

`widen_i16_to_i32` remains an existing active conversion route and regression
anchor, but it is not the owner expanded in this round.

## Requirements

1. Keep scope to one coherent signed widening conversion route:
   `widen_i32_to_i64`.
2. Selected RVV bodies must structurally carry source payload, destination
   result, source and destination element/config facts, conversion kind,
   conversion relation, unit-stride memory form, runtime `n` / AVL, and policy.
3. Explicit selected-body route support must start from typed
   `tcrv_rvv.runtime_abi_value`, `tcrv_rvv.setvl`, `tcrv_rvv.load`,
   `tcrv_rvv.widening_convert`, and `tcrv_rvv.store`, not from route ids,
   helper strings, descriptors, source-front-door markers, artifact names, or
   common EmitC/export inference.
4. Pre-realized support must remain valid and must realize into the same
   typed explicit body structure before route construction.
5. Route planning/provider must derive route support, ABI order, materialized
   operands, target leaf/profile, headers, mirrors, diagnostics, and runtime
   behavior from typed body/config/capability/runtime facts plus
   `RVVRouteOperandBindingPlan` closure.
6. Unsupported narrowing, unsigned relation, missing source, missing result,
   missing runtime role, source/result dtype/config mismatch, stale route id,
   helper-string selected conversion, binary/elementwise fallback claimed as
   conversion, mirror-only authority, descriptor/direct-C/source-front-door
   authority, and common/export semantic inference must fail closed where the
   current surfaces can express them.
7. Generated-bundle evidence must use value-distinguishing positive and
   negative i32 inputs, i64 outputs, runtime `n`/AVL variation, and tail/
   sentinel preservation that distinguishes signed widening from zero
   extension, truncation, or pass-through behavior.
8. Runtime/correctness claims require real `ssh rvv` PASS evidence.

## Acceptance Criteria

- [x] Current conversion inventory is recorded with exact active, bounded, and
      deferred routes.
- [x] `widen_i32_to_i64` has explicit selected-body target artifact coverage
      proving typed source payload, destination result, source/destination
      SEW+LMUL facts, signed conversion relation, runtime `n` / AVL,
      route operand binding plan, binding summary, provider-supported mirrors,
      and generated header mirrors.
- [x] Existing pre-realized `widen_i32_to_i64` coverage remains passing and
      proves realization into `setvl/with_vl/load/widening_convert/store`
      before route construction.
- [x] Provider emission consumes `RVVRouteOperandBindingPlan` for `lhs`, `out`,
      and `n` materialized operands and fails closed on missing/wrong plan,
      source/destination role swaps, materialized-use mismatch, runtime ABI
      mismatch, config/relation mismatch, or stale mirror/header metadata.
- [x] Negative coverage proves unsupported signedness/narrowing or helper
      metadata cannot become conversion route authority; binary/elementwise
      fallback cannot masquerade as `widen_i32_to_i64`.
- [x] Generated-bundle dry-runs pass for representative explicit and
      pre-realized `widen_i32_to_i64`, counts `7,16,23`.
- [x] Real `ssh rvv` generated-bundle runs pass for representative explicit
      and pre-realized `widen_i32_to_i64`, proving signed widening,
      runtime `n`/AVL variation, and tail/sentinel preservation.
- [x] Active-authority scan over touched RVV include/lib/script/test paths
      finds no new positive `RVVI32M1`, `rvv-i32m1`, finite positive
      `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`, descriptor/direct-C/source-export,
      source-front-door, public exact intrinsic route authority,
      artifact-name authority, or common/export RVV semantic authority.
- [x] Focused checks, `git diff --check`, `check-tianchenrv`, truthful task
      finish/archive state, clean git status, and one coherent commit complete
      if this task finishes.

## Non-Goals

- No broad conversion matrix and no new conversion variants beyond
  `widen_i32_to_i64`.
- No new unsigned, narrowing, floating, vector-to-scalar, mask/passthrough, or
  dtype/LMUL clone batch conversion support.
- No redo of reduction, masked macc, segmented/indexed/strided/contiguous
  memory movement, compare/select, elementwise arithmetic, contraction, or dot
  reduction except as regression anchors.
- No high-level Linalg/Vector/StableHLO/frontend lowering, source-front-door
  positive route, tensor contraction, matmul/GEMM, dashboards, report-only
  work, helper-only cleanup, or future plugin work.
- No route through legacy `RVVI32M1`, `rvv-i32m1`, finite `tcrv_rvv.i32_*`,
  exact intrinsic spelling authority, descriptor-driven direct C, or common
  EmitC/export semantic inference.

## Validation Plan

1. Validate and start this Trellis task.
2. Run focused dialect/verifier coverage for generic widening conversion.
3. Add and run focused explicit selected-body target/header checks for
   `widen_i32_to_i64`.
4. Run existing pre-realized `widen_i32_to_i64` target/header checks.
5. Run focused C++ route operand binding plan validation.
6. Run `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py` and
   `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test` if the script
   changes.
7. Run generated-bundle dry-runs for explicit and pre-realized
   `widen_i32_to_i64`, counts `7,16,23`.
8. Run real `ssh rvv` for representative explicit and pre-realized
   `widen_i32_to_i64` after dry-runs pass.
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
- Existing conversion target, dialect, generated-bundle, and provider tests.

## Technical Notes

Relevant specs read:

- `.trellis/spec/index.md`
- `.trellis/spec/extension-plugins/rvv-plugin.md`
- `.trellis/spec/lowering-runtime/emitc-route.md`

Prior task context read:

- `.trellis/tasks/archive/2026-05/05-21-05-21-stage2-rvv-conversion-operand-binding/prd.md`
- `.trellis/tasks/archive/2026-05/05-21-stage2-rvv-operand-binding-contract-closure-gate/prd.md`
- `.trellis/tasks/archive/2026-05/05-21-stage2-rvv-route-operand-abi-binding/prd.md`
- `.trellis/tasks/archive/2026-05/05-22-05-22-stage2-rvv-closure-gated-masked-horizontal-reduce-sum-accumulation/prd.md`
- `.trellis/tasks/archive/2026-05/05-22-05-22-stage2-rvv-closure-gated-masked-macc-accumulation/prd.md`

## Completion Evidence

### Conversion Inventory

- Completed bounded owner:
  - `widen_i32_to_i64`: active route-supported/executable signed widening
    conversion, source `i32/m1`, destination `i64/m2`, relation
    `signed-i32m1-to-i64m2`, ABI order `lhs,out,n`, closure plan
    `rvv-route-operand-binding:widen_i32_to_i64.v1`.
- Existing active conversion route retained as regression/deferred surface:
  - `widen_i16_to_i32`: source `i16/mf2`, destination `i32/m1`, relation
    `signed-i16mf2-to-i32m1`, ABI order `lhs,out,n`, closure plan
    `rvv-route-operand-binding:widen_i16_to_i32.v1`.
- No unsigned, narrowing, floating, mask/passthrough, frontend, source
  front-door, or dtype/LMUL clone conversion route was added.

### Module Behavior Completed

- Added explicit selected-body target artifact coverage for `widen_i32_to_i64`
  in `test/Target/RVV/explicit-selected-body-artifact-widen-i32-to-i64.mlir`.
  The fixture starts from typed `tcrv_rvv.runtime_abi_value`,
  `tcrv_rvv.setvl`, `tcrv_rvv.load`, `tcrv_rvv.widening_convert`, and
  `tcrv_rvv.store`, then proves PLAN and HEADER mirrors for:
  - operation `widen_i32_to_i64`;
  - typed compute op `tcrv_rvv.widening_convert`;
  - config `SEW64 LMUL m2`, source `SEW32 LMUL m1`;
  - relation `signed-i32m1-to-i64m2`;
  - runtime ABI order `lhs,out,n`;
  - route binding plan and operand summary.
- Added explicit selected-body generated-bundle support for
  `widen_i32_to_i64` in `scripts/rvv_generated_bundle_abi_e2e.py`, using
  value-distinguishing signed i32 inputs and i64 outputs.
- Added `test/Scripts/rvv-generated-bundle-abi-e2e-widen-i32-to-i64-dry-run.test`
  to prove explicit evidence JSON and harness output carry the conversion
  route, typed compute op, route binding plan, source/destination config,
  signed relation, and explicit selected-body front door.
- RVV provider/planning C++ was not changed because current HEAD already
  derives conversion materialized operands and mirrors from
  `RVVRouteOperandBindingPlan`; this round verified that owner and closed the
  missing explicit selected-body evidence boundary instead of duplicating route
  logic.

### Checks

- [OK] `build/bin/tcrv-opt test/Target/RVV/explicit-selected-body-artifact-widen-i32-to-i64.mlir --tcrv-materialize-emission-plans | /usr/lib/llvm-20/bin/FileCheck test/Target/RVV/explicit-selected-body-artifact-widen-i32-to-i64.mlir --check-prefix=PLAN`.
- [OK] `build/bin/tcrv-opt test/Target/RVV/explicit-selected-body-artifact-widen-i32-to-i64.mlir --tcrv-materialize-emission-plans | build/bin/tcrv-translate --tcrv-export-target-header-artifact | /usr/lib/llvm-20/bin/FileCheck test/Target/RVV/explicit-selected-body-artifact-widen-i32-to-i64.mlir --check-prefix=HEADER`.
- [OK] `build/bin/tcrv-opt test/Dialect/RVV/generic-widening-conversion-dataflow.mlir --split-input-file --verify-diagnostics | /usr/lib/llvm-20/bin/FileCheck test/Dialect/RVV/generic-widening-conversion-dataflow.mlir`.
- [OK] `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`.
- [OK] `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`.
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`.
- [OK] Lit focused explicit tests via `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -v . --filter='rvv-generated-bundle-abi-e2e-widen-i32-to-i64-dry-run|explicit-selected-body-artifact-widen-i32-to-i64'` from `build/test`: 2/2 passed.
- [OK] Lit focused pre-realized regression via `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -v . --filter='pre-realized-selected-body-artifact-widen-i32-to-i64|rvv-generated-bundle-abi-e2e-pre-realized-widen-i32-to-i64-dry-run'` from `build/test`: 2/2 passed.
- [OK] Explicit generated-bundle dry-run:
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/20260522-explicit-widen-i32-to-i64-dry`.
- [OK] Pre-realized generated-bundle dry-run:
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/20260522-pre-realized-widen-i32-to-i64-dry`.
- [OK] Real `ssh rvv` explicit generated-bundle run:
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/20260522-explicit-widen-i32-to-i64-ssh`,
  PASS for counts `7,16,23`.
- [OK] Real `ssh rvv` pre-realized generated-bundle run:
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/20260522-pre-realized-widen-i32-to-i64-ssh`,
  PASS for counts `7,16,23`.
- [OK] Diff-level active-authority scan over added/changed lines found only
  negative-boundary wording / `implicit-check-not` assertions for
  descriptor/direct-C/source-export/source-front-door / `tcrv_rvv.i32_`; no
  new positive legacy/source/descriptor/common-export route authority.
- [OK] `git diff --check`.
- [OK] `cmake --build build --target check-tianchenrv -j2`: 324/324 passed.

### Spec Update Review

No `.trellis/spec/**` update is needed. The existing RVV plugin, unified
EmitC route, and MLIR testing contracts already require typed body/config
authority, plugin-owned conversion route support, common EmitC neutrality, and
real `ssh rvv` evidence for runtime/correctness claims.
