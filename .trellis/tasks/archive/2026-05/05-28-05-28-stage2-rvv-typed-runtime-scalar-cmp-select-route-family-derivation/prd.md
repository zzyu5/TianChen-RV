# Stage2 RVV Typed Runtime-Scalar Compare-Select Route-Family Derivation

## Task Source

Hermes Direction Brief: `Continue: Stage2 RVV typed runtime-scalar compare-select route-family derivation`.

Repository facts at task creation:

- Root: `/home/kingdom/phdworks/TianchenRV`.
- Initial worktree: clean.
- Initial HEAD: `bf1930b0 rvv: derive typed computed-mask select route facts`.
- No `.trellis/.current-task` existed, so this task was created from the brief.
- The immediately preceding archived task
  `.trellis/tasks/archive/2026-05/05-28-05-28-stage2-rvv-typed-computed-mask-select-route-family-derivation`
  completed provider-derived typed computed-mask select route-family derivation
  for `computed_mask_select_i64` and `computed_mask_select_lmul_m2`.
- Earlier session evidence closed baseline `runtime_scalar_cmp_select`
  selected-boundary behavior, two-RHS generated-bundle harness coverage, and
  direct pre-realized route-entry fail-closed behavior, but the current target
  fixtures still need non-baseline typed witnesses.

## Problem

`runtime_scalar_cmp_select` is a sibling in the computed-mask/select
route-family. It differs from vector computed-mask select because the compare
RHS is a runtime scalar ABI value that must be imported, splatted under the
same runtime VL, compared against the loaded vector LHS, and then used to
select between typed true/false vector values.

The current bounded witness is still narrower than the typed route-family
coverage just completed for vector computed-mask select. This round must make
the runtime scalar RHS path derive element type, SEW, LMUL, scalar C/MLIR type,
vector C type, mask type, predicate, runtime ABI order, select operand binding,
target leaf profile, and intrinsic mirrors from typed body/config/runtime facts.
It must not derive route support from route ids, ABI strings, artifact names,
script conventions, exact intrinsic spellings, common EmitC behavior, direct
route-entry shortcuts, descriptor residue, source-front-door markers, or legacy
i32 helper authority.

## Goal

Implement one coherent production module for bounded typed
`runtime_scalar_cmp_select` route-family derivation:

```text
selected tcrv.exec RVV variant
  -> typed pre-realized tcrv_rvv runtime-scalar compare-select body
  -> RVV selected-body realization
  -> realized setvl/with_vl/load/splat/compare/select/store facts
  -> computed-mask/select route-family planning from typed body/config/runtime facts
  -> TCRVEmitCLowerableRoute
  -> neutral common EmitC
  -> generated target artifact
  -> ssh rvv evidence for non-baseline typed witnesses
```

Representative positive witnesses are:

- `runtime_scalar_cmp_select_i64`: SEW64 LMUL m1 with signed scalar threshold.
- `runtime_scalar_cmp_select_lmul_m2`: SEW32 LMUL m2 with signed scalar
  threshold.

Both witnesses must remain ordinary instances of the generic typed
runtime-scalar compare-select route-family surface.

## Requirements

- The RVV dialect/config contract must accept only bounded typed
  `runtime_scalar_cmp_select` configurations needed for this round, preserving
  existing predicate, memory form, runtime scalar RHS role, select layout,
  tail/mask policy, and selected-boundary-only constraints.
- Selected-body realization must derive `setvl`, `with_vl`, LHS load, runtime
  scalar splat, compare mask type, true/false value loads, select vector type,
  and store type from structural `sew`/`lmul`/runtime ABI facts.
- Route planning/provider must derive vector C type, mask C type, scalar RHS C
  type, scalar RHS MLIR type, compare intrinsic, splat/broadcast intrinsic,
  select/merge intrinsic, store intrinsic, runtime ABI order, runtime n/AVL/VL
  loop facts, route-control mirrors, mask/tail-policy mirrors, route-family
  plan facts, and operand-binding facts from the realized typed body and runtime
  imports.

## Completion Evidence

Completed in this round as a single runtime-scalar compare-select owner.

- Production movement: RVV verifier/config contracts now accept bounded typed
  runtime-scalar compare-select witnesses while rejecting stale scalar type and
  C type facts; selected-body realization preserves runtime scalar binding,
  predicate, dtype, SEW/LMUL, policy, AVL/VL, and ABI order; route planning
  derives typed vector/mask/scalar C facts and intrinsic mirrors from structural
  typed body/config/runtime facts.
- Positive witnesses: `runtime_scalar_cmp_select_i64` covers SEW64 LMUL m1;
  `runtime_scalar_cmp_select_lmul_m2` covers SEW32 LMUL m2.
- Generated-bundle evidence:
  `artifacts/tmp/runtime-scalar-cmp-select-dry-run/pre-realized-typed-runtime-scalar-cmp-select`
  and
  `artifacts/tmp/runtime-scalar-cmp-select-ssh-rvv/pre-realized-typed-runtime-scalar-cmp-select-rvv`.
- Real hardware evidence: `ssh rvv` passed for both witnesses with runtime
  counts `0,1,16,23,257` and signed RHS scalar values `-37,91`; both reported
  true/false lane coverage and tail preservation.
- Fail-closed evidence: direct pre-realized route-entry is rejected for
  `runtime_scalar_cmp_select`, `runtime_scalar_cmp_select_i64`, and
  `runtime_scalar_cmp_select_lmul_m2`.
- Non-regression: typed computed-mask select `i64` and `lmul_m2` generated
  bundle dry-run still passes.
- Checks: focused verifier/FileCheck/manual lit-equivalent checks, generated
  artifact FileChecks, `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`,
  `ninja -C build tcrv-opt tcrv-translate tianchenrv-rvv-extension-plugin-test`,
  `build/bin/tianchenrv-rvv-extension-plugin-test`,
  `ninja -C build check-tianchenrv` with 426/426 passing tests,
  `git diff --check`, and bounded touched-file authority scans.
- Authority result: positive route/executable evidence depends on typed
  `tcrv_rvv` body/config/runtime facts consumed by the RVV provider and common
  EmitC remains neutral; direct-route-entry, descriptor, source-front-door,
  artifact-name, route-id, ABI-string, script-derived, and common-EmitC-derived
  authority remain negative or mirror-only.
- i64 m1 runtime-scalar compare-select must derive `int64_t`, `vint64m1_t`,
  `vbool64_t`, e64/m1 setvl/load/splat/compare/select/store leaves, and the
  corresponding scalar RHS ABI binding without relying on fixture or artifact
  names.
- i32 m2 runtime-scalar compare-select must derive `int32_t`, `vint32m2_t`,
  `vbool16_t`, e32/m2 setvl/load/splat/compare/select/store leaves, and the
  corresponding scalar RHS ABI binding without relying on fixture or artifact
  names.
- The generated bundle ABI must keep `lhs`, scalar RHS threshold, true value,
  false value, output, and runtime `n` in the provider-derived order expected
  by the selected body. The artifact metadata may mirror this order only after
  provider route construction.
- Unsupported or inconsistent bodies must fail closed before provider/common
  EmitC route construction, including scalar/vector element mismatch, stale
  scalar ABI type, invalid SEW/LMUL/mask relation, wrong predicate, wrong
  select layout, missing runtime n/AVL/VL, wrong runtime ABI order, stale route
  metadata, direct-route-entry-only authority, artifact-name/script-derived
  authority, exact-intrinsic-as-authority, and common-EmitC semantic invention.

## Acceptance Criteria

- Production diff includes owner movement only where needed in the RVV dialect
  verifier/config contract, construction protocol, selected-body realization,
  route planning/provider, and generated-bundle or target artifact boundary.
  Tests alone are not sufficient.
- Typed target fixtures or script-generated selected-boundary fixtures cover
  `runtime_scalar_cmp_select_i64` and `runtime_scalar_cmp_select_lmul_m2`.
- Generated-bundle dry-run validates selected-boundary realization,
  materialized selected-body MLIR, runtime scalar splat/compare/select leaves,
  emitted RVV C/C++ leaves, route metadata mirrors, header/prototype ABI,
  provider-supported mirrors, route-control mirrors, mask/tail-policy mirrors,
  operand-binding mirrors, and `route_entry_realization = false`.
- Direct pre-realized route-entry requests for `runtime_scalar_cmp_select` and
  the new typed witnesses remain fail-closed; no positive direct route-entry
  support is reintroduced.
- Real `ssh rvv` generated-bundle compile/run/correctness passes for both
  typed witnesses over counts `0`, `1`, an exact-vector case, a tail case, and
  a stress case, with signed input values and at least two scalar thresholds,
  or the exact external blocker is recorded without claiming runtime
  correctness.
- Non-regression dry-runs for completed `computed_mask_select_i64` and
  `computed_mask_select_lmul_m2` pass.
- Focused fail-closed tests cover at least one stale or inconsistent typed
  runtime-scalar compare-select body/config/runtime ABI mismatch before common
  EmitC.
- Python tooling touched by the task passes `python3 -m py_compile`; script
  self-test is rerun when script behavior changes.
- Focused lit/FileCheck or C++ tests covering changed behavior pass, including
  `tianchenrv-rvv-extension-plugin-test` if plugin C++ owner code changes.
- `git diff --check` passes.
- `ninja -C build check-tianchenrv` passes, or an exact blocker is reported
  after focused checks pass.
- Bounded authority scan over touched production/test/script files shows no new
  central ad hoc, name-derived, metadata-derived, descriptor-derived,
  ABI-string-derived, script-derived, artifact-name-derived,
  common-EmitC-derived, source-front-door-derived, route-id-derived,
  exact-intrinsic-derived, direct-route-entry-only,
  pre-realized-fixture-only, or legacy-i32-derived route authority.

## Non-Goals

- No `runtime_scalar_dual_cmp_mask_and_select` work.
- No masked arithmetic, computed-mask memory movement, segment2,
  reduction/contraction, standalone reduction, widening conversion, MAcc, or
  high-level Linalg/frontend lowering.
- No one-intrinsic wrapper dialects, descriptor-driven compute,
  source-front-door positive routes, dashboards, broad smoke matrices, or
  evidence-only tasks.
- No positive direct pre-realized route-entry support for
  `runtime_scalar_cmp_select`.
- No broad dtype/LMUL clone batch beyond the two bounded typed witnesses.

## Technical Notes

Specs and prior context read before implementation:

- `.trellis/spec/index.md`
- `.trellis/spec/extension-plugins/index.md`
- `.trellis/spec/extension-plugins/rvv-plugin.md`
- `.trellis/spec/core-dialect/tcrv-exec-contract.md`
- `.trellis/spec/lowering-runtime/emitc-route.md`
- `.trellis/spec/testing/mlir-testing-contract.md`
- `.trellis/spec/guides/plugin-locality-review-guide.md`
- `.trellis/tasks/archive/2026-05/05-28-05-28-stage2-rvv-typed-computed-mask-select-route-family-derivation/prd.md`
- `.trellis/workspace/codex/journal-17.md` sessions 276 and 283

Likely production owners:

- `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`
- `lib/Dialect/RVV/IR/RVVConfigContract.cpp`
- `lib/Dialect/RVV/IR/RVVDialect.cpp`
- `lib/Plugin/RVV/Construction/RVVConstructionProtocol.cpp`
- `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp`
- `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
- `lib/Target/RVV/RVVTargetSupportBundle.cpp` if target-side mirror validation
  needs typed runtime-scalar repair.

Likely evidence consumers:

- `scripts/rvv_generated_bundle_abi_e2e.py`
- `test/Target/RVV/pre-realized-selected-body-artifact-runtime-scalar-cmp-select.mlir`
- `test/Scripts/rvv-generated-bundle-abi-e2e-pre-realized-runtime-scalar-cmp-select-dry-run.test`
- `test/Scripts/rvv-generated-bundle-abi-e2e-direct-pre-realized-runtime-scalar-cmp-select-fail-closed.test`
- New typed runtime-scalar compare-select script/lit fixtures if needed.
