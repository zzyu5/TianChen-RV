# Stage2 RVV Typed Runtime-Scalar Dual Compare-Mask/Select Route-Family Derivation

## Task Source

Hermes Direction Brief: `Switch: Stage2 RVV typed runtime-scalar dual compare-mask/select route-family derivation`.

Repository facts at task creation:

- Root: `/home/kingdom/phdworks/TianchenRV`.
- Initial worktree: clean.
- Initial HEAD: `e03ab753 rvv: derive typed runtime scalar cmp select route facts`.
- No `.trellis/.current-task` existed, so this task was created from the brief.
- The immediately preceding archived task
  `.trellis/tasks/archive/2026-05/05-28-05-28-stage2-rvv-typed-runtime-scalar-cmp-select-route-family-derivation`
  completed typed `runtime_scalar_cmp_select` derivation for i64 m1 and i32 m2
  witnesses with direct pre-realized route-entry fail-closed evidence.

## Problem

`runtime_scalar_dual_cmp_mask_and_select` is the adjacent compare/select
route family that imports two runtime scalar thresholds, splats both under the
same runtime VL, compares two loaded vectors, composes both masks with
`mask_and`, selects true/false payloads, and stores the result.

Current repository inspection shows this path is still narrower than the typed
single runtime-scalar compare/select path:

- `TypedRuntimeScalarDualCompareMaskAndSelectPreRealizedBodyOp::verify()` is
  bounded to SEW32 LMUL m1 and fixed `int32_t` ABI C types.
- `getRVVSelectedBodyRuntimeScalarDualCompareMaskAndSelectRuntimeABIParameters()`
  returns only fixed `int32_t` pointer/scalar parameters.
- `validateRVVSelectedBodyComputedMaskSelectRouteFamilyPlan()` and
  `deriveRVVSelectedBodyComputedMaskSelectRouteFamilyPlan()` fail-close dual
  runtime-scalar select to e32m1 only.
- Dual route mirrors still expose e32m1-specific target leaf and C type summary
  constants, while the single runtime-scalar path already uses typed generic
  summaries.
- Script and target fixtures currently cover only the baseline dual witness.

This round must move production authority to typed body/config/runtime facts
for the dual route, not merely add more proof-only coverage.

## Goal

Implement one coherent production owner for bounded typed
`runtime_scalar_dual_cmp_mask_and_select` route-family derivation:

```text
selected tcrv.exec RVV variant
  -> typed pre-realized dual runtime-scalar compare/mask/select body
  -> RVV selected-body realization
  -> realized setvl/with_vl/load/splat/load/splat/compare/compare/mask_and/select/store body
  -> route-family planning from typed body/config/runtime facts
  -> provider-derived TCRVEmitCLowerableRoute facts
  -> neutral common EmitC
  -> generated target artifact
  -> ssh rvv evidence for non-baseline typed witnesses
```

Representative positive witnesses:

- `runtime_scalar_dual_cmp_mask_and_select_i64`: SEW64 LMUL m1.
- `runtime_scalar_dual_cmp_mask_and_select_lmul_m2`: SEW32 LMUL m2.

Both witnesses are ordinary instances of the generic typed dual runtime-scalar
compare/mask/select route-family surface. They must not introduce dtype- or
LMUL-cloned route authority.

## Requirements

- The RVV dialect/config contract must accept bounded typed dual witnesses
  needed for this round while preserving selected-boundary-only constraints,
  `sle` predicates, `and` mask composition, runtime scalar roles, select layout,
  tail/mask policy, and fail-closed stale authority attributes.
- Runtime ABI parameter helpers must derive dual compare lhs pointer types,
  scalar threshold types, true/false payload pointer types, output pointer
  types, and runtime `n` from element C type facts where typed witnesses need
  them.
- Selected-body realization must derive vector/mask/scalar types, `setvl`,
  both loads, both scalar splats, both compares, `mask_and`, true/false loads,
  select, store, AVL/VL, and source provenance from structural
  `sew`/`lmul`/policy/runtime ABI facts.
- Route planning/provider facts must derive element type, SEW, LMUL, vector C
  type, mask C type, scalar C type, compare predicates, secondary compare
  predicate, two scalar splat leaves, both compare leaves, mask-and leaf,
  select/merge leaf, store leaf, route-control facts, runtime ABI order,
  operand-binding plan, mask/tail policy mirrors, header/type/intrinsic mirrors,
  target leaf profile, and provider-supported mirror from realized typed body
  and target/config facts.
- Target/script generated-bundle boundary must validate i64 and LMUL m2 dual
  witnesses through selected-boundary realization with
  `route_entry_realization = false`, and direct pre-realized route-entry must
  remain fail-closed.
- Fail-closed diagnostics must cover stale or inconsistent element type,
  SEW/LMUL/mask relation, scalar ABI type, missing or swapped runtime scalar
  bindings, missing second compare, missing mask_and, wrong select operands,
  wrong AVL/VL, stale route id/mirror metadata, exact-intrinsic-as-authority,
  and common-EmitC semantic invention before executable route construction.

## Acceptance Criteria

- Production diff includes owner movement in the RVV dialect/config contract,
  construction protocol, selected-body realization, route planning/provider,
  and generated-bundle target/script boundary as needed. Tests alone are not
  sufficient.
- Typed selected-boundary fixtures or generated witnesses cover
  `runtime_scalar_dual_cmp_mask_and_select_i64` and
  `runtime_scalar_dual_cmp_mask_and_select_lmul_m2`.
- Generated-bundle dry-run validates selected-boundary realization,
  materialized selected-body MLIR, two runtime scalar splats, two compares,
  mask-and/select/store leaves, emitted RVV C/C++ leaves, provider metadata
  mirrors, header/prototype ABI, route-control mirrors, mask/tail-policy
  mirrors, operand-binding mirrors, and `route_entry_realization = false`.
- Direct pre-realized route-entry requests for baseline/i64/m2 dual witnesses
  remain fail-closed; no positive direct route-entry support is reintroduced.
- Real `ssh rvv` generated-bundle compile/run/correctness passes for the two
  typed dual witnesses over counts `0`, `1`, `16`, `23`, and `257` with
  threshold pairs that exercise both masks, one-mask-only lanes, both-true
  lanes, false lanes, selected payloads, and tail preservation, or the exact
  external blocker is recorded without claiming runtime correctness.
- Non-regression coverage for completed `runtime_scalar_cmp_select` and
  `computed_mask_select` typed witnesses passes.
- Focused lit/FileCheck or C++ tests covering changed behavior pass, including
  `tianchenrv-rvv-extension-plugin-test` if plugin C++ owner code changes.
- Python tooling touched by the task passes `python3 -m py_compile`; script
  self-test is rerun when script behavior changes.
- `git diff --check` passes.
- `ninja -C build check-tianchenrv` passes, or an exact blocker is reported
  after focused checks pass.
- Bounded authority scan over touched production/test/script files shows no new
  central ad hoc, name-derived, metadata-derived, descriptor-derived,
  ABI-string-derived, script-derived, artifact-name-derived,
  common-EmitC-derived, source-front-door-derived, route-id-derived,
  exact-intrinsic-derived, direct-route-entry-only,
  pre-realized-fixture-only, or legacy-i32-derived route authority.

## Completion Evidence

Completed in this round as one bounded Stage2
`runtime_scalar_dual_cmp_mask_and_select` route-family owner.

- Production movement: RVV config/runtime ABI helpers now build dual
  compare-mask/select ABI parameters from element C type facts; the RVV
  verifier accepts only the bounded SEW32 LMUL m1, SEW32 LMUL m2, and SEW64
  LMUL m1 configs with matching scalar SSA/C ABI types; selected-body
  realization and route planning consume typed body/config/runtime facts before
  provider route construction.
- Positive witnesses: `runtime_scalar_dual_cmp_mask_and_select_i64` covers
  SEW64 LMUL m1 with `int64_t`, `vint64m1_t`, and `vbool64_t`;
  `runtime_scalar_dual_cmp_mask_and_select_lmul_m2` covers SEW32 LMUL m2 with
  `int32_t`, `vint32m2_t`, and `vbool16_t`.
- Selected-boundary evidence: generated-bundle dry-run succeeded for the two
  typed witnesses with `route_entry_realization = false`, consumed
  pre-realized bodies, two runtime scalar splats, two compares, mask-and,
  select/store leaves, typed provider mirrors, header ABI, and RVV C/C++
  intrinsic leaves.
- Direct route-entry evidence: direct pre-realized route-entry failed closed
  for baseline, i64, and LMUL m2 dual compare-mask/select witnesses with the
  selected-boundary-only diagnostic; no positive direct route-entry shortcut
  was introduced.
- Real hardware evidence: `ssh rvv` compile/run/correctness passed for both
  typed witnesses with counts `0,1,16,23,257` and threshold pairs
  `-37/-37`, `-37/91`, `91/-37`, and `91/91`; harness output covered both
  masks, composed true/false lanes, single-mask-only lanes, selected payloads,
  and tail preservation.
- Non-regression evidence: typed `runtime_scalar_cmp_select_i64`,
  `runtime_scalar_cmp_select_lmul_m2`, `computed_mask_select_i64`, and
  `computed_mask_select_lmul_m2` generated-bundle dry-runs passed.
- Checks: `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`;
  `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`;
  `ninja -C build tcrv-opt tcrv-translate
  tianchenrv-rvv-extension-plugin-test`;
  `build/bin/tianchenrv-rvv-extension-plugin-test`;
  `git diff --check`; single final `ninja -C build check-tianchenrv`
  passed `429/429`.
- Self-repair note: accidental concurrent generated-bundle/check invocations
  caused transient shared artifact collisions; clean single reruns passed and
  are the evidence used for closeout.
- Spec-update judgment: no `.trellis/spec/**` edit was needed because the
  existing RVV plugin, `tcrv.exec`, EmitC route, and testing specs already
  encode selected-body realization, typed body authority, mirror-only metadata,
  and common EmitC neutrality. This task instantiates those contracts for a
  bounded dual runtime-scalar compare-mask/select family.

## Non-Goals

- No work on `runtime_scalar_cmp_masked_store`,
  `runtime_scalar_cmp_masked_load_store`, computed-mask memory movement,
  contraction, widening conversion, segment2, high-level frontend/Linalg
  lowering, dashboards, smoke matrices, or evidence-only tasks.
- No additional proof-only coverage for the completed
  `runtime_scalar_cmp_select` route-family except focused non-regression.
- No dtype-prefixed op namespaces, one-intrinsic wrapper dialects, descriptor
  compute paths, source-front-door positive routes, or route-entry restoration.

## Technical Notes

Read before implementation:

- `.trellis/spec/index.md`
- `.trellis/spec/extension-plugins/rvv-plugin.md`
- `.trellis/spec/core-dialect/tcrv-exec-contract.md`
- `.trellis/spec/lowering-runtime/emitc-route.md`
- `.trellis/tasks/archive/2026-05/05-28-05-28-stage2-rvv-typed-runtime-scalar-cmp-select-route-family-derivation/prd.md`
- `.trellis/workspace/codex/journal-17.md`

Likely production owners:

- `include/TianChenRV/Dialect/RVV/IR/RVVConfigContract.h`
- `lib/Dialect/RVV/IR/RVVConfigContract.cpp`
- `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`
- `lib/Dialect/RVV/IR/RVVDialect.cpp`
- `lib/Plugin/RVV/Construction/RVVConstructionProtocol.cpp`
- `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp`
- `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`
- `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
- `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
- `scripts/rvv_generated_bundle_abi_e2e.py`

Likely focused evidence:

- `test/Dialect/RVV/runtime-scalar-dual-compare-mask-and-select-dataflow.mlir`
- `test/Target/RVV/pre-realized-selected-body-artifact-runtime-scalar-dual-cmp-mask-and-select.mlir`
- New i64/m2 target or script fixtures for the typed dual witnesses.
- Direct pre-realized route-entry fail-closed script test for baseline/i64/m2.
