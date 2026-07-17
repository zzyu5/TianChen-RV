# Stage2 RVV Typed Masked Elementwise Route-Family Derivation

## Task Source

Hermes Direction Brief: `Expand: Stage2 RVV typed masked elementwise route-family derivation`.

Repository facts at task creation:

- Root: `/home/kingdom/phdworks/TianchenRV`.
- Initial worktree: clean.
- Initial HEAD: `76843488 rvv: derive typed elementwise route facts`.
- No `.trellis/.current-task` existed, so this task was created from the brief.
- The immediately preceding archived task `05-28-stage2-rvv-typed-dtype-lmul-elementwise-route-family-derivation` completed typed plain elementwise `i64_add` and `lmul_m2_add` route derivation through planning/provider/target artifact validation and RVV evidence.

## Problem

Plain elementwise arithmetic now carries non-i32m1 dtype/SEW/LMUL facts through the generic typed RVV route-family path. Masked elementwise arithmetic is the adjacent consumer class, but current positive masked witnesses are still centered on bounded `masked_add`/`masked_sub`/`masked_mul` i32m1 fixtures. This leaves the route-family boundary under-proven for the masked case where both vector value types and mask types must be derived from typed `tcrv_rvv` body/config/capability/runtime facts.

This round must make masked elementwise arithmetic a real typed route-family consumer rather than a route-id, e32m1, metadata, fixture, script, artifact-name, exact-intrinsic, or common-EmitC-derived special case.

## Goal

Implement one coherent production module for typed masked elementwise route-family derivation:

```text
selected tcrv.exec RVV variant
  -> typed masked elementwise tcrv_rvv body/config/capability/runtime facts
  -> RVV route planning derives element/mask/config/runtime facts
  -> RVV provider validates materialization, residual binding, route-control, and statement-plan facts
  -> TCRVEmitCLowerableRoute
  -> neutral common EmitC materialization
  -> target artifact validates provider-derived mirrors
  -> generated RVV C artifact and ssh rvv evidence for representative witnesses
```

The representative set must prove generic derivation across non-i32m1 dtype/LMUL and masked op-kind behavior without creating a broad dtype/LMUL clone table.

## Requirements

- Route planning/provider for masked elementwise arithmetic derives element type, signed C type, SEW, LMUL, vector type, vector C type, mask type, mask C type, tail/mask policy, setvl/load/compare/masked arithmetic/merge/store leaves, runtime ABI order, runtime n/AVL/VL loop, provider-supported mirrors, and selected capability facts from typed `tcrv_rvv` body/config/capability/runtime facts.
- Masked elementwise provider construction fails closed before creating `TCRVEmitCLowerableRoute` when typed materialization facts, mask type facts, residual operand-binding facts, route-control facts, statement-plan facts, runtime ABI order/roles, selected capability facts, or provider mirror labels disagree.
- Target artifact consumers rebuild the provider route and validate masked elementwise payload facts: headers, C type mappings, mask mappings, runtime ABI mappings, statement callees, runtime loop facts, inactive-lane/tail policy mirrors, selected-body provenance, and explicit mirror-only metadata.
- Generated-bundle consumers cover masked elementwise artifacts with non-i32m1 dtype/LMUL witnesses and verify inactive-lane/tail preservation for runtime counts `0`, `1`, `16`, `23`, and `257`.
- Unsupported combinations, stale e32m1 metadata, stale unmasked/masked metadata, missing or wrong mask facts, mismatched element/mask SEW/LMUL, wrong policy, wrong ABI role/order, missing capability/config, stale provider mirror, direct route-id or artifact-name authority, exact intrinsic authority, source-front-door residue, descriptor residue, and common-EmitC semantic invention must fail closed with targeted diagnostics or bounded negative checks.
- Existing plain typed elementwise `i64_add` and `lmul_m2_add` remain non-regressed.

## Acceptance Criteria

- Production diff includes RVV route planning/provider and target artifact validation movement, plus only directly necessary generated-bundle consumers and focused tests.
- At least one non-i32m1 dtype masked elementwise witness proves provider-derived element/mask facts, e.g. an `i64` masked arithmetic selected body with `SEW=64`, `LMUL=m1`, signed C type, `vint64m1_t`, `vbool64_t`, e64m1 arithmetic/compare/merge/store leaves, runtime ABI order, runtime n/AVL/VL facts, and explicit provider mirror labels.
- At least one non-i32m1 LMUL masked elementwise witness proves provider-derived element/mask facts, e.g. an `i32` `LMUL=m2` masked arithmetic selected body with `SEW=32`, `LMUL=m2`, signed C type, `vint32m2_t`, `vbool16_t`, e32m2 arithmetic/compare/merge/store leaves, runtime ABI order, runtime n/AVL/VL facts, and explicit provider mirror labels.
- Masked op-kind behavior is covered beyond only `masked_add`; a minimal representative such as `masked_sub` or `masked_mul` must pass through the same typed route-family path.
- Dry-run generated artifacts expose `mask_tail_policy_boundary`, typed config, mask type/C type mapping, provider-supported mirror labels, route-control mirrors, route operand-binding mirrors, statement-plan callees, and mirror-only artifact metadata for the new witnesses.
- `ssh rvv` compile/run/correctness passes for the new masked witnesses over runtime counts `0`, `1`, `16`, `23`, and `257`, including inactive-lane/tail preservation, or the exact external blocker is reported without claiming runtime correctness.
- Focused non-regression passes for plain typed elementwise `i64_add` and `lmul_m2_add`.
- `git diff --check` passes.
- `ninja -C build check-tianchenrv` passes, or an exact non-task blocker is reported with focused checks passing.
- Bounded authority scan over touched production/test/script files shows no new central ad hoc, name-derived, metadata-derived, descriptor-derived, ABI-string-derived, script-derived, artifact-name-derived, common-EmitC-derived, source-front-door-derived, route-id-derived, exact-intrinsic-derived, direct-route-entry-only, pre-realized-fixture-only, or legacy-i32-derived route authority.

## Non-Goals

- No strided elementwise expansion.
- No scalar broadcast expansion.
- No reductions, standalone reductions, MAcc/contraction, conversion, segment2, compare/select expansion, or computed-mask memory work.
- No high-level Linalg/Vector/StableHLO/frontend lowering.
- No one-intrinsic wrapper dialect or dtype/LMUL clone batch.
- No dashboard/report-only/evidence-only round.
- No rework of completed plain `i64_add`/`lmul_m2_add` except direct non-regression required by this module.
- No source-front-door positive RVV routes, descriptor-driven compute, route-id-derived authority, exact-intrinsic-derived authority, or common-EmitC semantic invention.

## Technical Notes

Specs read before implementation:

- `.trellis/spec/index.md`
- `.trellis/spec/extension-plugins/rvv-plugin.md`
- `.trellis/spec/core-dialect/tcrv-exec-contract.md`
- `.trellis/spec/lowering-runtime/emitc-route.md`
- `.trellis/spec/testing/mlir-testing-contract.md`
- `.trellis/spec/guides/{capability-first-design-guide,plugin-locality-review-guide,compute-boundary-review-guide}.md`

Likely production owners:

- `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`
- `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
- `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
- `lib/Target/RVV/RVVTargetSupportBundle.cpp`

Likely direct evidence consumers:

- `scripts/rvv_generated_bundle_abi_e2e.py`
- `test/Target/RVV/*masked*`
- `test/Scripts/rvv-generated-bundle-abi-e2e-*masked*-dry-run.test`
- Focused non-regression tests for `i64_add` and `lmul_m2_add`
