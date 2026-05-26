# Stage2 RVV Computed-Mask Segment2 Store Route-Entry Owner

## Goal

Promote the bounded `computed_masked_segment2_store_unit_load`
representative from selected-boundary-only support to RVV plugin-owned direct
route-entry support. The production path for this task is:

```text
selected tcrv.exec RVV variant
  -> typed pre-realized computed-mask segment2 store tcrv_rvv body
  -> RVV segment2 memory selected-body realization owner
  -> realized setvl/with_vl/compare-produced mask/unit field loads/
     masked_segment2_store body
  -> RVV computed-mask memory family facts plus segment2 statement plan
  -> route-control plan, memory operand-binding facts, provider facts
  -> provider-built TCRVEmitCLowerableRoute
  -> common EmitC
  -> generated RVV artifact and ssh rvv evidence
```

This is the complementary store direction to the archived
`computed_masked_segment2_load_unit_store` route-entry task. It is not a broad
segment2, dtype, LMUL, frontend, or source-front-door expansion.

## Direction Source

- Direction title: `Stage2 RVV computed-mask segment2 store route-entry owner`.
- Module owner: RVV plugin-owned selected-body realization and
  provider/planning boundary for the bounded computed-mask segment2 store
  representative.
- Exact representative after bounded code/spec inspection:
  `computed_masked_segment2_store_unit_load`.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean.
- Initial HEAD: `fc1273f0 rvv: route computed-mask segment2 entry owner`.
- `.trellis/.current-task` was absent, so this task was created from the
  Hermes brief before source edits.

## Current Repository Facts

- Specs require RVV route support to derive from typed `tcrv_rvv`
  body/config/runtime facts, RVV plugin-owned selected-body realization,
  provider-owned route facts, and a provider-built
  `TCRVEmitCLowerableRoute`. Common EmitC/export may only materialize provider
  output and mirror facts after route construction.
- `rvv-plugin.md` currently records that direct segment2 route-entry support
  includes plain segment2 deinterleave, plain segment2 interleave, and bounded
  computed-mask segment2 load/unit-store, while
  `computed_masked_segment2_store_unit_load` remains selected-boundary-only
  until an explicit owner task adds matching facts and evidence.
- `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp` already validates and
  realizes `TypedComputedMaskSegment2StorePreRealizedBodyOp` through the
  segment2 memory selected-body owner, producing compare loads, field payload
  loads, compare mask, and `masked_segment2_store`.
- The direct route-entry predicate explicitly returns false for
  `TypedComputedMaskSegment2StorePreRealizedBodyOp`; this is the bounded
  production blocker for the representative.
- `RVVEmitCRoutePlanning.cpp` already has computed-mask segment2 store
  route-family facts, memory operand bindings, route-control facts, and
  segment2 memory statement-plan support. The task should wire the
  representative into route-entry ownership and evidence, not rebuild route
  semantics in common EmitC or scripts.
- `scripts/rvv_generated_bundle_abi_e2e.py` already knows the selected-boundary
  store representative and harness oracle, but direct pre-realized route-entry
  mode currently allows only the computed-mask segment2 load direction.

## Requirements

1. Keep the executable representative to
   `computed_masked_segment2_store_unit_load` with `cmp_lhs`, `cmp_rhs`,
   `src0`, `src1`, `dst`, and runtime `n`.
2. The `segment2 memory` selected-body realization owner remains one owner for
   plain and computed-mask segment2 pre-realized bodies. Its direct route-entry
   predicate must include the bounded computed-mask segment2 store/unit-load
   shape only when typed facts match the explicit store representative.
3. Direct route-entry realization must consume the typed pre-realized store
   body before provider facts are collected, producing `setvl`, `with_vl`,
   compare LHS/RHS loads, field0/field1 source loads, same-VL compare mask,
   and one `masked_segment2_store`.
4. Provider route construction must still go through the RVV-owned
   computed-mask memory family facts, segment2 memory statement plan,
   route-control plan, memory operand-binding facts, materialization facts,
   and selected target capability facts. No route semantics may move into
   common EmitC/export, scripts, metadata, ABI strings, artifact names,
   route ids, descriptors, exact intrinsic spellings, or legacy i32 helpers.
5. Fail-closed diagnostics must remain targeted for wrong mask producer/source,
   stale mask metadata, same-VL mismatch, missing inactive-destination
   preservation policy, unsupported segment factor, field role/order mismatch,
   dtype/config mismatch, memory-form mismatch, runtime ABI role/order
   mismatch, unsupported policy, missing route-control facts, stale
   materialization/operand-binding facts, and stale mirror metadata.
6. Generated-bundle evidence must distinguish true masked segment2 store from
   unmasked store, masked load, inverted mask, stream swap, concatenation,
   unit copy, passthrough-only behavior, wrong stride, stale constant data,
   inactive-lane destination clobbering, tail clobbering, and
   metadata-derived behavior.
7. Recent computed-mask segment2 load, segment2 deinterleave/interleave,
   masked elementwise, reduction, scalar-broadcast, conversion,
   runtime-scalar, compare/select, MAcc, contraction, and base memory route
   owners must not regress.

## Acceptance Criteria

- [x] `segment2 memory` has an owner-scoped route-entry predicate that accepts
      the bounded `TypedComputedMaskSegment2StorePreRealizedBodyOp` shape for
      `computed_masked_segment2_store_unit_load`.
- [x] C++ route-entry owner tests prove the representative is owned by the
      `segment2 memory` realization family, is direct route-entry eligible,
      consumes the pre-realized body, realizes compare/mask/source-load/
      masked segment2 store structure, reaches the computed-mask memory and
      segment2 statement-plan provider boundaries, and builds a provider route.
- [x] C++ fail-closed tests cover at least wrong mask source, missing
      inactive-lane destination preservation policy, unsupported segment
      factor, wrong source/destination role order or ABI role, dtype/config or
      memory-form mismatch, runtime `n` ABI mismatch, unsupported policy, and
      stale metadata attempting to authorize another shape before provider
      construction.
- [x] `rvv_generated_bundle_abi_e2e.py --direct-pre-realized-route-entry`
      accepts the bounded `computed_masked_segment2_store_unit_load`
      representative and records route-entry realization evidence.
- [x] Generated-bundle dry-run passes for direct pre-realized
      `computed_masked_segment2_store_unit_load` over counts
      `0,7,16,23,257`.
- [x] Real `ssh rvv` generated-bundle run passes for direct pre-realized
      `computed_masked_segment2_store_unit_load` over counts
      `0,7,16,23,257`, proving active and inactive lanes, inactive
      destination preservation, field order, source preservation, and tail
      preservation.
- [x] Focused non-regression covers recent representative route owners:
      computed-mask segment2 load, plain segment2 deinterleave, plain
      segment2 interleave, masked elementwise, standalone reduction,
      scalar-broadcast, conversion, runtime-scalar MAcc, compare/select,
      MAcc, contraction, and base memory.
- [x] `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`,
      focused C++ tests, focused generated-bundle checks, bounded authority
      scans, `git diff --check`, and `check-tianchenrv` pass or exact blockers
      are recorded.
- [x] Trellis task state, journal/archive, and one coherent commit are
      truthful if complete.

## Out Of Scope

- Broad segment width expansion, dtype/LMUL clone batches, gather/scatter
  expansion, high-level frontend/Linalg/Vector routes, source-front-door
  positives, dashboards, global tuning databases, broad smoke matrices, or
  evidence-only changes.
- Moving mask/segment semantics, dtype/config, route support, intrinsic
  choices, ABI role authority, or inactive-lane semantics into common EmitC,
  scripts, metadata, route ids, artifact names, descriptors, ABI strings,
  exact intrinsic spellings, or legacy i32 helpers.

## Technical Approach

1. Extend the segment2 route-entry predicate in
   `RVVSelectedBodyRealization.cpp` to recognize only the bounded
   computed-mask segment2 store/unit-load typed op by op class, `op_kind`,
   `memory_form`, segment count, mask source/form, inactive-lane policy,
   SEW/LMUL, policy, source/destination memory forms, and field roles.
2. Reuse the existing selected-boundary realization and planning/provider
   surfaces; do not add a provider/common semantic fallback.
3. Update focused C++ route-entry owner coverage so store/unit-load becomes a
   positive direct route-entry case and stale-metadata negative coverage still
   proves metadata cannot authorize an invalid shape.
4. Allow the representative in generated-bundle direct pre-realized
   route-entry mode and add a focused lit dry-run for the direct store path.
5. Run direct pre-realized dry-run and `ssh rvv` evidence with counts
   `0,7,16,23,257`; run focused non-regression and quality gates.

## Implementation Result

- Extended the `segment2 memory` selected-body realization owner direct
  route-entry predicate in `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp` so
  the bounded `TypedComputedMaskSegment2StorePreRealizedBodyOp` representative
  is route-entry eligible only when typed facts match
  `computed_masked_segment2_store_unit_load`, segment count 2,
  `computed-mask-unit-load-segment2-store`, compare-produced same-VL mask
  facts, field0/field1 source roles, unit source memory forms, interleaved
  destination form, inactive destination preservation policy, SEW32/LMUL m1,
  and agnostic tail/mask policy.
- Reused the existing segment2 memory owner realization hook. The direct
  route-entry path consumes the pre-realized body into `setvl`, `with_vl`,
  compare LHS/RHS loads, field0/field1 payload loads, compare, and one
  `masked_segment2_store` before provider facts are collected.
- Kept route construction in existing RVV-owned planning/provider boundaries:
  computed-mask memory family facts, segment2 memory statement plan,
  route-control provider plan, memory operand-binding facts,
  materialization facts, selected target capability facts, and provider-built
  `TCRVEmitCLowerableRoute`.
- Updated generated-bundle evidence tooling so
  `--pre-realized-selected-body --direct-pre-realized-route-entry` accepts the
  bounded `computed_masked_segment2_store_unit_load` representative and still
  verifies that pre-realized body ops are consumed before materialized evidence.
- Added focused lit dry-run coverage for direct pre-realized computed-mask
  segment2 store evidence, including route-entry realization mirrors,
  provider-derived route facts, mask/source/field metadata checks,
  inactive-lane contract, field order contract, and tail/source preservation
  harness checks.
- Updated `.trellis/spec/extension-plugins/rvv-plugin.md` so the durable
  route-entry contract now includes bounded computed-mask segment2
  store/unit-load, while preserving the requirement that all other segment2
  bodies remain selected-boundary-only until explicit owner work adds matching
  facts and evidence.

## Validation Evidence

- `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py` passed.
- `cmake --build build --target tianchenrv-rvv-extension-plugin-test -j2`
  passed.
- `./build/bin/tianchenrv-rvv-extension-plugin-test` passed with
  `RVV extension plugin smoke test passed`.
- `cmake --build build --target tcrv-opt tcrv-translate -j2` passed.
- Direct pre-realized generated-bundle dry-run:
  `artifacts/tmp/stage2_computed_mask_segment2_store_route_entry_owner/direct-pre-realized-computed-masked-segment2-store`
  - Result: `rvv_generated_bundle_abi_e2e: dry_run_success`.
  - Counts: `0,7,16,23,257`.
  - Evidence includes `materializer:
    rvv-route-entry-selected-body-realization`,
    `route_entry_realization: true`, no
    `--tcrv-materialize-selected-lowering-boundaries` in the direct pipeline,
    `pre_realized_body_consumed: true`, provider route
    `rvv-generic-computed-masked-segment2-store-unit-load-emitc-route`,
    runtime ABI order `cmp_lhs,cmp_rhs,src0,src1,dst,n`, route operand
    binding plan
    `rvv-route-operand-binding:computed_masked_segment2_store_unit_load.v1`,
    compare-produced same-VL mask facts, segment2 field roles, and
    `provider_supported_mirror:rvv-computed-mask-segment2-store-plan-validated`.
- Real `ssh rvv` direct pre-realized generated-bundle run:
  `artifacts/tmp/stage2_computed_mask_segment2_store_route_entry_owner_ssh/direct-pre-realized-computed-masked-segment2-store-ssh-rvv`
  - Result: `PASS op=computed_masked_segment2_store_unit_load
    counts=0,7,16,23,257`.
  - Output reported active/inactive lanes for `n=7`, `16`, `23`, and `257`,
    inactive destination preservation for all false lanes, field-distinguishing
    lanes for all active counts, `source_preserved`, and `tail_preserved`.
- Focused direct route-entry non-regression dry-run:
  `artifacts/tmp/stage2_computed_mask_segment2_store_route_entry_owner_nonregression/direct-pre-realized-route-entry-nonregression`
  - Covered `computed_masked_segment2_store_unit_load`,
    `computed_masked_segment2_load_unit_store`,
    `segment2_deinterleave_unit_store`, `segment2_interleave_unit_load`,
    `scalar_broadcast_add`, `standalone_reduce_add`, `widen_i16_to_i32`,
    `runtime_scalar_cmp_masked_macc_add`, `cmp_select`,
    `computed_mask_select`, `macc_add`, `scalar_broadcast_macc_add`,
    `computed_masked_macc_add`, `widening_macc_add`,
    `widening_dot_reduce_add`, `strided_input_widening_dot_reduce_add`,
    `computed_masked_widening_dot_reduce_add`,
    `computed_masked_strided_input_widening_dot_reduce_add`, and
    `strided_load_unit_store`.
- Masked elementwise non-regression dry-run:
  `artifacts/tmp/stage2_computed_mask_segment2_store_route_entry_owner_nonregression/pre-realized-masked-add-nonregression`
  - Covered `masked_add` over counts `0,7,16,23,257`.
- Added-line authority scan over touched production, script, test, spec, and
  task files found no new positive legacy i32, source-front-door, descriptor,
  direct-C/source-export, route-id-derived, artifact-name-derived,
  ABI-string-derived, exact-intrinsic-derived, common-EmitC-derived, script-
  or metadata-derived route authority. Matches were limited to PRD non-goals
  and red lines, lit `CHECK-NOT` guards, or provider-derived mirror fields
  after route construction.
- `git diff --check` passed.
- `cmake --build build --target check-tianchenrv -j2` passed: 388/388 tests.
- `clang-format` could not be run because `clang-format` is not installed in
  the local environment; C++ formatting was checked against surrounding style
  and the full build/test suite passed.
