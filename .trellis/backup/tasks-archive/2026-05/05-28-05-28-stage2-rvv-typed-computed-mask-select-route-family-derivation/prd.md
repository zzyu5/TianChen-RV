# Stage2 RVV Typed Computed-Mask Select Route-Family Derivation

## Task Source

Hermes Direction Brief: `Switch: Stage2 RVV typed computed-mask select route-family derivation`.

Repository facts at task creation:

- Root: `/home/kingdom/phdworks/TianchenRV`.
- Initial worktree: clean.
- Initial HEAD: `c277dc8c rvv: derive typed compare select route facts`.
- No `.trellis/.current-task` existed, so this task was created from the brief.
- The immediately preceding archived task `05-28-stage2-rvv-typed-compare-select-route-family-derivation` completed typed plain compare/select derivation for i64 m1 and i32 m2 witnesses, with generated-bundle and `ssh rvv` evidence.
- Current code inspection shows `TypedComputedMaskSelectPreRealizedBodyOp` and `validatePreRealizedRVVSelectedComputedMaskSelectBody(...)` still restrict computed-mask select to SEW32 LMUL m1, while route planning/provider already has computed-mask select family, route-control, mask/tail-policy, operand-binding, and compare/select statement-plan surfaces.

## Problem

Computed-mask select is the adjacent compare/select selected-body family that combines a computed compare mask with separate true/false vector value bindings. Existing selected-boundary executable evidence covers `computed_mask_select` and `computed_mask_select_sle`, but it remains centered on SEW32 LMUL m1.

This round must make bounded computed-mask select consume typed `tcrv_rvv` body/config/runtime facts for representative non-i32m1 witnesses, rather than deriving route authority from fixture names, route ids, ABI strings, artifact names, exact intrinsic spellings, direct route-entry mode, pre-realized fixture names, or common EmitC behavior.

## Goal

Implement one coherent production module for bounded typed computed-mask select route-family derivation:

```text
selected tcrv.exec RVV variant
  -> typed computed-mask-select tcrv_rvv pre-realized body
  -> RVV plugin-local selected-body realization
  -> realized setvl/with_vl/load/compare/select/store body
  -> route planning/provider validates typed vector, mask, predicate, true/false binding, runtime, route-control, mask/tail, and capability facts
  -> TCRVEmitCLowerableRoute
  -> neutral common EmitC materialization
  -> target artifact validates provider-derived mirrors
  -> generated RVV C artifact and ssh rvv evidence for representative witnesses
```

The representative positive scope is one i64 m1 computed-mask select and one i32 m2 computed-mask select, or the closest existing predicate variants required by current op-kind naming. Both must remain ordinary instances of the generic typed computed-mask select surface.

## Requirements

- Computed-mask select pre-realized verifier and selected-body realization accept only bounded typed configs `SEW32 LMUL m1`, `SEW32 LMUL m2`, and `SEW64 LMUL m1`, preserving existing predicate, memory form, mask role/source/form, select layout, and tail/mask agnostic policy constraints.
- Selected-body realization must derive `setvl`, `with_vl`, compare input loads, true/false value loads, compare mask type, select vector type, and store type from `sew`/`lmul` typed body/config facts.
- Route planning/provider must derive vector C type, mask C type, compare intrinsic, select intrinsic, runtime ABI order, runtime n/AVL/VL loop facts, route-control mirrors, mask/tail-policy mirrors, operand-binding facts, and computed-mask select route-family plan from typed body/config/runtime/capability facts.
- i64 m1 computed-mask select must derive `vint64m1_t`, `vbool64_t`, `__riscv_vsetvl_e64m1`, `__riscv_vle64_v_i64m1`, `__riscv_vms*_vv_i64m1_b64`, `__riscv_vmerge_vvm_i64m1`, and `__riscv_vse64_v_i64m1`.
- i32 m2 computed-mask select must derive `vint32m2_t`, `vbool16_t`, `__riscv_vsetvl_e32m2`, `__riscv_vle32_v_i32m2`, `__riscv_vms*_vv_i32m2_b16`, `__riscv_vmerge_vvm_i32m2`, and `__riscv_vse32_v_i32m2`.
- The realized body and provider route must preserve computed mask producer/use, predicate kind, true/false value operand binding, output binding, element type, SEW, LMUL, mask type, mask role/source/form, runtime n/AVL/VL relation, setvl placement, memory roles, required capabilities, runtime ABI order, and mirror-only metadata.
- Unsupported or inconsistent computed-mask select bodies, wrong predicate, wrong vector or mask type, wrong SEW/LMUL relation, wrong true/false binding, stale ABI/route metadata, missing capability, exact-intrinsic-as-authority, script/artifact-name authority, direct-route-entry-only authority, or common EmitC semantic choice must fail closed with targeted diagnostics or bounded negative checks.

## Acceptance Criteria

- Production diff includes RVV dialect verifier, selected-body realization, route planning/provider facts, and directly necessary construction/script/target/test consumers.
- New typed computed-mask select target fixtures cover:
  - i64 m1 computed-mask select with signed predicate and typed `vbool64_t` mask facts.
  - i32 m2 computed-mask select with typed `vbool16_t` mask facts.
- Generated-bundle dry-run validates selected-boundary realization, materialized selected-body MLIR, emitted RVV C/C++ compare/select leaves, route metadata, header/prototype ABI, provider-supported mirrors, route-control mirrors, mask/tail-policy mirrors, route operand-binding mirrors, mirror-only artifact metadata, and `route_entry_realization = false` for runtime counts `0`, `1`, `16`, `23`, and `257`.
- Direct pre-realized route-entry requests for computed-mask select remain fail-closed; no positive direct route-entry support is added.
- `ssh rvv` generated-bundle compile/run/correctness passes for the new typed computed-mask select witnesses over runtime counts `0`, `1`, `16`, `23`, and `257`, covering both true and false predicate lanes, or the exact external blocker is reported without claiming runtime correctness.
- Existing plain compare/select typed witnesses `cmp_select_i64` and `cmp_select_lmul_m2`, existing SEW32 computed-mask select witnesses `computed_mask_select` and `computed_mask_select_sle`, and completed typed masked elementwise witnesses `masked_i64_add` and `masked_lmul_m2_sub` remain non-regressed.
- Focused fail-closed coverage proves at least one unsupported or stale typed computed-mask select config is rejected before provider/common EmitC route construction.
- `git diff --check` passes.
- `ninja -C build check-tianchenrv` passes, or an exact non-task blocker is reported with focused checks passing.
- Bounded authority scan over touched production/test/script files shows no new central ad hoc, name-derived, metadata-derived, descriptor-derived, ABI-string-derived, script-derived, artifact-name-derived, common-EmitC-derived, source-front-door-derived, route-id-derived, exact-intrinsic-derived, direct-route-entry-only, pre-realized-fixture-only, or legacy-i32-derived route authority.

## Non-Goals

- No runtime scalar compare/select, runtime scalar dual compare-mask-and-select, or runtime scalar masked memory expansion.
- No computed-mask arithmetic expansion, reductions, standalone reductions, MAcc/contraction, conversion, segment2, base memory movement, or high-level Linalg/frontend lowering.
- No broad clone batch beyond the representative typed computed-mask select witnesses.
- No one-intrinsic wrapper dialect, source-front-door positive route, descriptor-driven compute, route-id-derived authority, exact-intrinsic-derived authority, or common-EmitC semantic invention.
- No positive direct pre-realized route-entry support for computed-mask select.

## Technical Notes

Specs read before implementation:

- `.trellis/spec/index.md`
- `.trellis/spec/extension-plugins/index.md`
- `.trellis/spec/extension-plugins/rvv-plugin.md`
- `.trellis/spec/core-dialect/index.md`
- `.trellis/spec/core-dialect/tcrv-exec-contract.md`
- `.trellis/spec/lowering-runtime/index.md`
- `.trellis/spec/lowering-runtime/emitc-route.md`
- `.trellis/spec/testing/index.md`
- `.trellis/spec/testing/mlir-testing-contract.md`
- `.trellis/spec/variant-pipeline/index.md`
- `.trellis/spec/guides/index.md`
- `.trellis/spec/guides/capability-first-design-guide.md`
- `.trellis/spec/guides/plugin-locality-review-guide.md`
- `.trellis/spec/guides/compute-boundary-review-guide.md`
- Archived task `.trellis/tasks/archive/2026-05/05-28-stage2-rvv-typed-compare-select-route-family-derivation`
- Archived task `.trellis/tasks/archive/2026-05/05-25-stage2-rvv-computed-mask-select-route-control-provider-plan`

Likely production owners:

- `lib/Dialect/RVV/IR/RVVDialect.cpp`
- `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp`
- `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
- `lib/Plugin/RVV/Construction/RVVConstructionProtocol.cpp` if construction mirrors need typed computed-mask repair.
- `lib/Target/RVV/RVVTargetSupportBundle.cpp` if target-side computed-mask select mirror validation needs typed repair.

Likely direct evidence consumers:

- `scripts/rvv_generated_bundle_abi_e2e.py`
- `test/Target/RVV/pre-realized-selected-body-artifact-computed-mask-select*.mlir`
- `test/Scripts/rvv-generated-bundle-abi-e2e-pre-realized-computed-mask-select*.test`
- focused non-regression generated-bundle invocations for typed plain compare/select and typed masked elementwise witnesses.
