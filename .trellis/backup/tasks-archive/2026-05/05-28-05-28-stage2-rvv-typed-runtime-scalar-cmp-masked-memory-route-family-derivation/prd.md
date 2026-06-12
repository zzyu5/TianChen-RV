# Stage2 RVV Typed Runtime-Scalar Compare-Masked Memory Route-Family Derivation

## Task Source

Hermes Direction Brief:
`Switch: Stage2 RVV typed runtime-scalar compare-masked memory route-family derivation`.

Repository facts at task creation:

- Root: `/home/kingdom/phdworks/TianchenRV`.
- Initial worktree: clean.
- Initial HEAD:
  `38713fa4 rvv: derive typed runtime scalar dual cmp select route facts`.
- No `.trellis/.current-task` existed, so this task was created from the brief.
- The immediately preceding archived task
  `.trellis/tasks/archive/2026-05/05-28-05-28-stage2-rvv-typed-runtime-scalar-dual-cmp-mask-select-route-family-derivation`
  completed typed `runtime_scalar_dual_cmp_mask_and_select` derivation for
  SEW64 LMUL m1 and SEW32 LMUL m2 witnesses.
- The older archived masked-memory boundary task
  `.trellis/tasks/archive/2026-05/05-28-stage2-rvv-runtime-scalar-cmp-masked-memory-side-effect-boundary`
  closed baseline selected-boundary runtime/correctness evidence for
  `runtime_scalar_cmp_masked_store` and
  `runtime_scalar_cmp_masked_load_store`, but did not make the bounded family
  typed across i64 or LMUL m2 witnesses.

## Problem

`runtime_scalar_cmp_masked_store` and
`runtime_scalar_cmp_masked_load_store` are the adjacent runtime-scalar
compare-derived masked memory routes after typed runtime-scalar compare/select
and dual compare-mask/select. They import a runtime scalar threshold, splat it
under the same runtime VL as a loaded compare input, derive a predicate mask,
then feed RVV masked memory side effects.

Current inspection shows the family has the baseline selected-boundary path and
runtime evidence, but the next production bottleneck is typed derivation across
the bounded family:

- runtime ABI helpers for computed-mask memory still expose fixed baseline
  signatures for `lhs,rhs_scalar,src,dst,n`;
- verifier and selected-body realization code must prove that scalar SSA/C ABI
  type, vector type, mask type, SEW, LMUL, policy, and memory form are derived
  from typed `tcrv_rvv` body/config/runtime facts;
- route planning/provider metadata must mirror typed computed-mask memory
  facts after provider construction, not infer them from op names, route ids,
  fixture names, ABI strings, artifact names, or exact intrinsic spellings;
- generated-bundle fixtures currently prove the baseline family and the typed
  compare/select pattern, but this route family still needs i64 m1 and i32 m2
  selected-boundary witnesses plus direct route-entry fail-closed coverage.

## Goal

Implement one coherent production owner for bounded typed
runtime-scalar compare-masked memory route-family derivation:

```text
selected tcrv.exec RVV variant
  -> typed pre-realized runtime-scalar compare masked-memory body
  -> RVV selected-body realization
  -> realized setvl/with_vl/load/splat/compare plus masked_store or masked_load/store body
  -> computed-mask memory route-family plan from typed body/config/runtime facts
  -> provider-derived TCRVEmitCLowerableRoute facts
  -> neutral common EmitC
  -> generated target artifact
  -> ssh rvv evidence for typed witnesses
```

The production module owns both operations together:

- `runtime_scalar_cmp_masked_store`
- `runtime_scalar_cmp_masked_load_store`

Representative positive witnesses:

- baseline SEW32 LMUL m1 non-regression;
- SEW64 LMUL m1 typed witness;
- SEW32 LMUL m2 typed witness.

If existing verifier/config constraints block one typed member, the task may
close a smaller coherent subset only with a precise production blocker and no
claim that the blocked member is supported.

## Requirements

- The RVV dialect/config contract must accept only bounded supported typed
  computed-mask memory witnesses needed for this round while preserving
  selected-boundary-only behavior, `sle` predicate facts, compare-produced mask
  provenance, mask/tail policy, memory form, inactive-lane policy, and
  fail-closed stale authority attributes.
- Runtime ABI parameter helpers must derive pointer and scalar ABI C types
  from element type facts for typed witnesses:
  - masked store ABI order: `lhs,rhs_scalar,src,dst,n`;
  - masked load-store ABI order: `lhs,rhs_scalar,src,dst,n`;
  - scalar threshold C type must match the element C type.
- Selected-body realization must derive vector/mask/scalar types, `setvl`,
  compare input load, runtime scalar splat, compare mask, source payload load,
  masked store, optional old-destination passthrough load, masked load, final
  store, AVL/VL, source provenance, and memory side-effect role from structural
  `sew`/`lmul`/policy/runtime ABI facts.
- Route planning/provider facts must derive element type, SEW, LMUL, vector C
  type, mask C type, scalar C type, compare predicate, splat/compare leaf,
  masked-store or masked-load/store leaves, route-control facts, runtime ABI
  order, operand-binding plan, mask/tail policy mirrors, header/type/intrinsic
  mirrors, target leaf profile, provider-supported mirror, and statement-plan
  leaves from realized typed body and target/config facts.
- Direct pre-realized route-entry must remain fail-closed for baseline and
  typed witnesses. This task must not add positive direct route-entry support.
- Generated-bundle boundary and target fixtures must validate selected-body
  realization with `route_entry_realization = false` and
  `pre_realized_body_consumed = true`.
- Unsupported dtype/config/policy/runtime combinations must fail closed before
  executable route construction with targeted diagnostics.

## Acceptance Criteria

- Production diff includes owner movement in at least one relevant production
  boundary as needed: RVV dialect/config contract, construction protocol,
  selected-body realization, route planning/provider, target artifact boundary,
  or generated-bundle ABI boundary. Tests alone are not sufficient unless
  inspection proves production already derives the typed facts and only missing
  executable evidence is being added.
- Typed selected-boundary fixtures or generated witnesses cover i64 m1 and
  i32 m2 forms for both masked-memory op kinds, or document an exact production
  blocker for a narrowed subset.
- Generated-bundle dry-runs validate selected-boundary realization,
  materialized selected-body MLIR, runtime scalar splat, compare mask,
  masked_store or masked_load plus store leaves, emitted RVV C/C++ leaves,
  provider metadata mirrors, header/prototype ABI, route-control mirrors,
  mask/tail-policy mirrors, operand-binding mirrors, and
  `route_entry_realization = false`.
- Direct pre-realized route-entry requests for baseline/i64/m2 masked-memory
  witnesses remain fail-closed; no positive direct route-entry shortcut is
  reintroduced.
- Real `ssh rvv` generated-bundle compile/run/correctness passes for the typed
  witnesses over counts `0`, `1`, an exact-vector count, a tail count, and a
  stress count with signed i32/i64 threshold patterns that exercise all-masked,
  none-masked, mixed active lanes, inactive lane preservation, old destination
  passthrough for load-store, source preservation, and tail preservation.
- Focused non-regression coverage for completed
  `runtime_scalar_dual_cmp_mask_and_select` and `runtime_scalar_cmp_select`
  typed families passes.
- Focused lit/FileCheck or C++ tests covering changed behavior pass, including
  `tianchenrv-rvv-extension-plugin-test` if plugin C++ owner code changes.
- Python tooling touched by the task passes `python3 -m py_compile`; script
  self-test is rerun when script behavior changes.
- `git diff --check` passes.
- `cmake --build build --target check-tianchenrv -j2` passes, or an exact
  blocker is recorded after focused checks pass.
- Bounded authority scan over touched production/test/script files shows no new
  central ad hoc, name-derived, metadata-derived, descriptor-derived,
  ABI-string-derived, script-derived, artifact-name-derived,
  common-EmitC-derived, source-front-door-derived, route-id-derived,
  exact-intrinsic-derived, direct-route-entry-only,
  pre-realized-fixture-only, or legacy-i32-derived route authority.

## Definition Of Done

- Trellis PRD/context are truthful and current.
- Both masked-memory routes either derive typed config/runtime/provider facts
  across the bounded subset, or the exact blocker is recorded without claiming
  support.
- Direct route-entry remains fail-closed.
- Focused checks, ssh rvv evidence, non-regression, authority scan, and final
  status are recorded.
- The task is finished/archived only after acceptance is met.
- One coherent commit records the completed round.

## Non-Goals

- No reductions, MAcc, widening dot, segment2, conversion expansion,
  high-level Linalg/frontend lowering, source-front-door work, dashboards,
  broad smoke matrices, or report-only work.
- No direct pre-realized route-entry positive support for either masked memory
  op kind.
- No dtype-prefixed helper op families, one-intrinsic wrapper dialects,
  descriptor compute paths, source-front-door positive routes, or route-id
  authority.
- No re-proving completed runtime-scalar dual compare/select work except as
  focused non-regression.

## Technical Notes

Read before implementation:

- `.trellis/spec/index.md`
- `.trellis/spec/extension-plugins/rvv-plugin.md`
- `.trellis/spec/core-dialect/tcrv-exec-contract.md`
- `.trellis/spec/lowering-runtime/emitc-route.md`
- `.trellis/tasks/archive/2026-05/05-28-05-28-stage2-rvv-typed-runtime-scalar-dual-cmp-mask-select-route-family-derivation/prd.md`
- `.trellis/tasks/archive/2026-05/05-28-stage2-rvv-runtime-scalar-cmp-masked-memory-side-effect-boundary/prd.md`
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
- `lib/Target/RVV/RVVTargetSupportBundle.cpp`
- `scripts/rvv_generated_bundle_abi_e2e.py`

Likely focused evidence:

- `test/Dialect/RVV/runtime-scalar-computed-mask-store-dataflow.mlir`
- `test/Dialect/RVV/runtime-scalar-computed-mask-load-store-dataflow.mlir`
- `test/Target/RVV/pre-realized-selected-body-artifact-runtime-scalar-cmp-masked-store.mlir`
- `test/Target/RVV/pre-realized-selected-body-artifact-runtime-scalar-cmp-masked-load-store.mlir`
- Generated-bundle dry-run and direct fail-closed Script tests for baseline,
  i64 m1, and i32 m2 masked-memory witnesses.
