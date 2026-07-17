# Stage2 RVV Typed Compare-Select Route-Family Derivation

## Task Source

Hermes Direction Brief: `Switch: Stage2 RVV typed compare-select route-family derivation`.

Repository facts at task creation:

- Root: `/home/kingdom/phdworks/TianchenRV`.
- Initial worktree: clean.
- Initial HEAD: `4a095938 rvv: derive typed masked elementwise route facts`.
- No `.trellis/.current-task` existed, so this task was created from the brief.
- The immediately preceding archived task `05-28-05-28-stage2-rvv-typed-masked-elementwise-route-family-derivation` completed typed masked elementwise route derivation for `masked_i64_add` and `masked_lmul_m2_sub`, including provider-derived vector/mask facts and `ssh rvv` evidence.

## Problem

Typed masked elementwise routes now prove that non-i32m1 vector and mask facts can be derived from selected `tcrv_rvv` body/config/runtime facts. Plain compare/select still has positive fixtures centered on `cmp_select` / `cmp_select_sle` with SEW32 LMUL m1, and repo inspection shows the plain compare/select pre-realized verifier and selected-body realization still hard-code SEW32 LMUL m1. That keeps compare/select itself under-proven as a typed mask/vector route-family consumer.

This round must make plain compare/select consume typed `tcrv_rvv` body/config/runtime facts for bounded non-i32m1 witnesses rather than deriving route authority from fixture names, route ids, ABI strings, artifact names, exact intrinsic spellings, or common EmitC behavior.

## Goal

Implement one coherent production module for bounded typed plain compare/select route-family derivation:

```text
selected tcrv.exec RVV variant
  -> typed compare/select tcrv_rvv pre-realized body
  -> RVV plugin-local selected-body realization
  -> realized setvl/with_vl/load/compare/select/store body
  -> route planning/provider validates typed vector, mask, predicate, select, runtime, and capability facts
  -> TCRVEmitCLowerableRoute
  -> neutral common EmitC materialization
  -> target artifact validates provider-derived mirrors
  -> generated RVV C artifact and ssh rvv evidence for representative witnesses
```

The representative positive scope is one i64 m1 compare/select case and one i32 m2 compare/select case, both still ordinary instances of the generic typed compare/select surface.

## Requirements

- Plain compare/select pre-realized verifier and selected-body realization accept only the bounded typed configs `SEW32 LMUL m1`, `SEW32 LMUL m2`, and `SEW64 LMUL m1`, preserving tail/mask agnostic policy and existing predicate/layout/memory-form constraints.
- Selected-body realization must derive `setvl`, `with_vl`, load vector types, compare mask type, select vector type, and store type from the body `sew`/`lmul` attrs, not from hard-coded i32m1 constants.
- Route planning/provider must derive compare intrinsic, select intrinsic, vector C type, mask C type, runtime ABI order, runtime n/AVL/VL loop facts, route-control mirrors, operand-binding facts, and plain compare/select route-family plan from typed body/config/runtime facts.
- i64 m1 plain compare/select must derive `vint64m1_t`, `vbool64_t`, `__riscv_vsetvl_e64m1`, `__riscv_vle64_v_i64m1`, `__riscv_vms*_vv_i64m1_b64`, `__riscv_vmerge_vvm_i64m1`, and `__riscv_vse64_v_i64m1`.
- i32 m2 plain compare/select must derive `vint32m2_t`, `vbool16_t`, `__riscv_vsetvl_e32m2`, `__riscv_vle32_v_i32m2`, `__riscv_vms*_vv_i32m2_b16`, `__riscv_vmerge_vvm_i32m2`, and `__riscv_vse32_v_i32m2`.
- The realized body and provider route must preserve compare predicate, select true/false operand binding, output binding, element type, SEW, LMUL, mask type, mask source, runtime n/AVL/VL relation, setvl placement, memory roles, required capabilities, runtime ABI order, and mirror-only metadata.
- Unsupported or inconsistent compare/select bodies, wrong predicate, wrong vector/mask type, wrong SEW/LMUL relation, wrong operand binding, stale ABI/route metadata, missing capability, exact-intrinsic-as-authority, script/artifact-name authority, or common EmitC semantic choice must fail closed with targeted diagnostics or bounded negative checks.

## Acceptance Criteria

- Production diff includes RVV dialect verifier, selected-body realization, route planning/provider facts, and only directly necessary generated-bundle/script/test movement.
- New typed compare/select target fixtures cover:
  - i64 m1 plain compare/select with signed predicate and typed `vbool64_t` mask facts.
  - i32 m2 plain compare/select with typed `vbool16_t` mask facts.
- Generated-bundle dry-run validates materialized selected-body MLIR, emitted RVV C/C++ compare/select leaves, route metadata, header/prototype ABI, provider-supported mirrors, route-control mirrors, route operand-binding mirrors, and mirror-only artifact metadata for runtime counts `0`, `1`, `16`, `23`, and `257`.
- `ssh rvv` generated-bundle compile/run/correctness passes for the new typed compare/select witnesses over runtime counts `0`, `1`, `16`, `23`, and `257`, or the exact external blocker is reported without claiming runtime correctness.
- Existing plain compare/select `cmp_select` and `cmp_select_sle`, completed typed masked elementwise `masked_i64_add` and `masked_lmul_m2_sub`, and plain typed elementwise `i64_add` / `lmul_m2_add` remain non-regressed.
- Focused fail-closed coverage proves at least one unsupported or stale typed compare/select config is rejected before provider/common EmitC route construction.
- `git diff --check` passes.
- `ninja -C build check-tianchenrv` passes, or an exact non-task blocker is reported with focused checks passing.
- Bounded authority scan over touched production/test/script files shows no new central ad hoc, name-derived, metadata-derived, descriptor-derived, ABI-string-derived, script-derived, artifact-name-derived, common-EmitC-derived, source-front-door-derived, route-id-derived, exact-intrinsic-derived, direct-route-entry-only, pre-realized-fixture-only, or legacy-i32-derived route authority.

## Non-Goals

- No computed-mask select expansion.
- No runtime-scalar compare/select expansion.
- No masked elementwise clone batch.
- No reductions, standalone reductions, MAcc/contraction, conversion, segment2, memory movement, or high-level frontend lowering.
- No one-intrinsic wrapper dialect or dtype/LMUL clone table.
- No positive direct pre-realized route-entry support for compare/select; this remains selected-boundary-only.
- No source-front-door positive RVV routes, descriptor-driven compute, route-id-derived authority, exact-intrinsic-derived authority, or common-EmitC semantic invention.

## Technical Notes

Specs read before implementation:

- `.trellis/spec/index.md`
- `.trellis/spec/extension-plugins/rvv-plugin.md`
- `.trellis/spec/core-dialect/tcrv-exec-contract.md`
- `.trellis/spec/lowering-runtime/emitc-route.md`
- Archived task `.trellis/tasks/archive/2026-05/05-28-05-28-stage2-rvv-typed-masked-elementwise-route-family-derivation`

Likely production owners:

- `lib/Dialect/RVV/IR/RVVDialect.cpp`
- `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp`
- `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
- `lib/Target/RVV/RVVTargetSupportBundle.cpp` only if target-side compare/select validation needs typed mirror repair.

Likely direct evidence consumers:

- `scripts/rvv_generated_bundle_abi_e2e.py`
- `test/Target/RVV/pre-realized-selected-body-artifact-cmp-select*.mlir`
- `test/Scripts/rvv-generated-bundle-abi-e2e-*cmp-select*-dry-run.test`
- focused non-regression generated-bundle invocations for completed typed elementwise and typed masked elementwise witnesses.
