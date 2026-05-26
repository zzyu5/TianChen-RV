# Stage2 RVV Computed-Mask Segment2 Memory Route-Entry Owner

## Goal

Promote one bounded computed-mask segment2 RVV memory representative,
`computed_masked_segment2_load_unit_store`, from selected-boundary executable
support to RVV plugin-owned direct route-entry support. The production path is:

```text
selected tcrv.exec RVV variant
  -> typed pre-realized computed-mask segment2 tcrv_rvv body
  -> RVV segment2 memory selected-body realization owner
  -> realized setvl/with_vl/compare-produced mask/masked_segment2_load/stores body
  -> RVV computed-mask memory family facts plus segment2 statement plan
  -> route-control plan, memory operand-binding facts, provider facts
  -> provider-built TCRVEmitCLowerableRoute
  -> common EmitC
  -> generated RVV artifact and ssh rvv evidence
```

This is not a broad segment2 matrix. It intentionally keeps
`computed_masked_segment2_store_unit_load`, wider segment factors, dtype/LMUL
clones, gather/scatter expansion, Linalg/frontend work, and source-front-door
positive routes out of scope.

## Direction Source

- Direction title: `Stage2 RVV computed-mask segment2 memory route-entry owner`.
- Module owner: RVV plugin-owned selected-body realization and
  provider/planning boundary for one computed-mask segment2 memory
  representative.
- Exact representative after bounded inspection:
  `computed_masked_segment2_load_unit_store`.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean.
- Initial HEAD: `358a590c rvv: route segment2 deinterleave entry owner`.
- `.trellis/.current-task` was absent, so this task was created from the
  Hermes brief before source edits.

## Current Repository Facts

- Specs require RVV route support to derive from typed `tcrv_rvv`
  body/config/runtime facts, RVV plugin-owned selected-body realization,
  provider-owned route facts, and a provider-built `TCRVEmitCLowerableRoute`.
  Common EmitC/export may only materialize provider output and mirror facts
  after route construction.
- `rvv-plugin.md` states that computed-mask segment2 memory must not use the
  non-segment computed-mask statement-plan boundary; it must continue through
  the segment2 memory statement-plan owner.
- Plain segment2 interleave and deinterleave direct route-entry owners have
  landed. The current segment2 realization owner is route-entry capable only
  for `TypedSegment2InterleaveMemoryPreRealizedBodyOp` and
  `TypedSegment2DeinterleaveMemoryPreRealizedBodyOp`.
- `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp` already owns validation and
  selected-boundary realization for
  `TypedComputedMaskSegment2LoadPreRealizedBodyOp` and
  `TypedComputedMaskSegment2StorePreRealizedBodyOp`.
- Existing selected-boundary generated-bundle coverage and lit fixtures
  already know `computed_masked_segment2_load_unit_store`; direct
  `--direct-pre-realized-route-entry` mode does not yet exercise it as a
  route-entry consumer.
- `RVVEmitCRoutePlanning.cpp` already has computed-mask segment2 load/store
  route-family and segment2 statement-plan surfaces. This round should wire the
  representative into route-entry ownership and strengthen targeted evidence,
  not rebuild route semantics in common EmitC or scripts.

## Requirements

1. Keep the executable representative to
   `computed_masked_segment2_load_unit_store` with `cmp_lhs`, `cmp_rhs`,
   `src`, `out0`, `out1`, and runtime `n`.
2. The `segment2 memory` selected-body realization owner remains one owner for
   plain and computed-mask segment2 pre-realized bodies. Its direct
   route-entry predicate must explicitly include only the bounded
   computed-mask segment2 load/unit-store representative added in this task,
   plus the already landed plain segment2 interleave/deinterleave members.
3. Direct route-entry realization must consume the typed computed-mask segment2
   pre-realized body before provider facts are collected, producing
   `setvl`, `with_vl`, compare LHS/RHS loads, old output passthrough loads,
   same-VL compare mask, `masked_segment2_load`, and two field stores.
4. Provider route construction must still go through the RVV-owned
   computed-mask memory family facts, segment2 memory statement plan,
   route-control plan, memory operand-binding facts, materialization facts,
   and selected target capability facts. No route semantics may move into
   common EmitC/export, scripts, metadata, ABI strings, artifact names, route
   ids, descriptors, exact intrinsic spellings, or legacy i32 helpers.
5. Fail-closed diagnostics must remain targeted for wrong mask producer/source,
   stale mask metadata, same-VL mismatch, missing passthrough/inactive-lane
   facts, unsupported segment factor, field role/order mismatch, dtype/config
   mismatch, memory-form mismatch, runtime ABI role/order mismatch, unsupported
   policy, missing route-control facts, stale materialization/operand-binding
   facts, and stale mirror metadata.
6. Generated-bundle evidence must distinguish true masked segment2 deinterleave
   from unmasked segment2, inverted mask, stream swap, concatenation, unit copy,
   passthrough-only behavior, wrong stride, stale constant data,
   inactive-lane clobbering, tail clobbering, and metadata-derived behavior.
7. Recent segment2 deinterleave/interleave, masked elementwise, reduction,
   scalar-broadcast, conversion, runtime-scalar, compare/select, MAcc,
   contraction, and base memory route owners must not regress.

## Acceptance Criteria

- [x] `segment2 memory` has an owner-scoped route-entry predicate that accepts
      the bounded `TypedComputedMaskSegment2LoadPreRealizedBodyOp` shape for
      `computed_masked_segment2_load_unit_store`.
- [x] `TypedComputedMaskSegment2StorePreRealizedBodyOp` remains outside the
      direct route-entry subset unless implementation evidence proves it is
      unavoidable for the load representative.
- [x] C++ route-entry owner tests prove the representative is owned by the
      `segment2 memory` realization family, is direct route-entry eligible,
      consumes the pre-realized body, realizes compare/mask/passthrough/
      masked segment2 load/store structure, reaches
      `rvv-segment2-memory-route-family-plan.v1`, and builds a provider route.
- [x] C++ fail-closed tests cover at least wrong mask source, missing
      passthrough/inactive-lane policy, unsupported segment factor, wrong
      source/destination role order or ABI role, dtype/config or memory-form
      mismatch, runtime `n` ABI mismatch, unsupported policy, and stale
      selected-body route-entry authority before provider construction.
- [x] `rvv_generated_bundle_abi_e2e.py --direct-pre-realized-route-entry`
      accepts the bounded `computed_masked_segment2_load_unit_store`
      representative and records route-entry realization evidence.
- [x] Generated-bundle dry-run passes for direct pre-realized
      `computed_masked_segment2_load_unit_store` over counts
      `0,7,16,23,257`.
- [x] Real `ssh rvv` generated-bundle run passes for direct pre-realized
      `computed_masked_segment2_load_unit_store` over counts
      `0,7,16,23,257`, proving true/false mask lanes, inactive-lane
      preservation, field-order distinction, source preservation, and tail
      preservation.
- [x] Focused non-regression covers recent representative route owners:
      plain segment2 deinterleave, plain segment2 interleave, masked
      elementwise, standalone reduction, scalar-broadcast, conversion,
      runtime-scalar MAcc, compare/select, MAcc, contraction, and base memory.
- [x] `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`,
      focused C++ tests, focused generated-bundle checks, bounded authority
      scans, `git diff --check`, and `check-tianchenrv` pass or exact blockers
      are recorded.
- [x] Trellis task state, journal/archive, and one coherent commit are
      truthful if complete.

## Out Of Scope

- `computed_masked_segment2_store_unit_load` route-entry promotion unless a
  bounded compile-time dependency proves it is required for the load
  representative.
- Segment width expansion, dtype/LMUL clone batches, gather/scatter expansion,
  high-level frontend/Linalg/Vector routes, source-front-door positives,
  dashboards, global tuning databases, broad smoke matrices, or evidence-only
  changes.
- Moving mask/segment semantics, dtype/config, route support, intrinsic
  choices, ABI role authority, or inactive-lane semantics into common EmitC,
  scripts, metadata, route ids, artifact names, descriptors, ABI strings,
  exact intrinsic spellings, or legacy i32 helpers.

## Technical Approach

1. Extend the segment2 route-entry predicate in
   `RVVSelectedBodyRealization.cpp` to recognize only the bounded computed-mask
   segment2 load/unit-store typed op by op class, `op_kind`, `memory_form`,
   segment count, mask source/form, inactive-lane policy, SEW/LMUL, and policy.
2. Keep selected-boundary realization and provider route construction in the
   existing RVV-owned realization/planning/provider boundaries; do not add a
   provider/common semantic fallback.
3. Add focused C++ route-entry owner coverage for the direct representative and
   owner-local fail-closed diagnostics for stale/wrong typed facts.
4. Allow the representative in generated-bundle direct pre-realized
   route-entry mode and make evidence explicitly report route-entry
   realization for the computed-mask segment2 path.
5. Run direct pre-realized dry-run and `ssh rvv` evidence with counts
   `0,7,16,23,257`; run focused non-regression and quality gates.

## Implementation Result

- Extended the `segment2 memory` selected-body realization owner route-entry
  predicate in `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp` so the bounded
  `TypedComputedMaskSegment2LoadPreRealizedBodyOp` representative is direct
  route-entry eligible only when typed facts match
  `computed_masked_segment2_load_unit_store`, segment count 2,
  `computed-mask-segment2-load-unit-store`, compare-produced same-VL mask
  facts, field0/field1 output roles, interleaved source form, unit-store
  destination form, passthrough inactive-lane policy, SEW32/LMUL m1, and
  agnostic tail/mask policy.
- Kept `TypedComputedMaskSegment2StorePreRealizedBodyOp` outside the direct
  route-entry subset. A stale `route_id`/metadata mutation on the store path
  remains unable to authorize route-entry realization.
- Reused the existing segment2 memory owner realization hook. The direct
  route-entry path consumes the pre-realized body into `setvl`, `with_vl`,
  compare LHS/RHS loads, old field passthrough loads, compare, one
  `masked_segment2_load`, and two field stores before provider facts are
  collected.
- Updated generated-bundle evidence tooling so
  `--pre-realized-selected-body --direct-pre-realized-route-entry` accepts only
  the bounded `computed_masked_segment2_load_unit_store` representative for
  this computed-mask segment2 task. Store/unit-load and broader segment2
  expansion remain unsupported in direct route-entry mode.
- Added a focused lit dry-run for direct pre-realized computed-mask segment2
  load evidence, including route-entry realization mirrors, provider-derived
  route facts, mask/source/field metadata checks, inactive-lane contract, field
  order contract, and tail/source preservation harness checks.
- Updated `.trellis/spec/extension-plugins/rvv-plugin.md` to record the
  durable route-entry contract: segment2 memory direct route-entry now includes
  bounded computed-mask segment2 load/unit-store, while computed-mask
  segment2 store/unit-load remains selected-boundary-only until an explicit
  owner adds matching facts and evidence.

## Validation Evidence

- `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py` passed.
- `cmake --build build --target tianchenrv-rvv-extension-plugin-test -j2`
  passed.
- `./build/bin/tianchenrv-rvv-extension-plugin-test` passed with
  `RVV extension plugin smoke test passed`.
- `cmake --build build --target tcrv-opt tcrv-translate -j2` passed.
- Direct pre-realized generated-bundle dry-run:
  `artifacts/tmp/stage2_computed_mask_segment2_route_entry_owner/direct-pre-realized-computed-masked-segment2-load-dry-run`
  - Result: `rvv_generated_bundle_abi_e2e: dry_run_success`.
  - Counts: `0,7,16,23,257`.
  - Evidence includes `materializer:
    rvv-route-entry-selected-body-realization`,
    `route_entry_realization: true`, no
    `--tcrv-materialize-selected-lowering-boundaries` in the direct pipeline,
    `pre_realized_body_consumed: true`, provider route
    `rvv-generic-computed-masked-segment2-load-unit-store-emitc-route`,
    runtime ABI order `cmp_lhs,cmp_rhs,src,out0,out1,n`,
    route operand binding plan
    `rvv-route-operand-binding:computed_masked_segment2_load_unit_store.v1`,
    compare-produced same-VL mask facts, segment2 field roles, and
    `provider_supported_mirror:rvv-computed-mask-segment2-load-plan-validated`.
- Real `ssh rvv` direct pre-realized generated-bundle run:
  `artifacts/tmp/stage2_computed_mask_segment2_route_entry_owner_ssh/direct-pre-realized-computed-masked-segment2-load-ssh-rvv`
  - Result: `PASS op=computed_masked_segment2_load_unit_store
    counts=0,7,16,23,257`.
  - Output reported active/inactive lanes for `n=7`, `16`, `23`, and `257`,
    inactive passthrough preservation for all false lanes, field-distinguishing
    lanes for all active counts, `source_preserved`, and `tail_preserved`.
- Focused direct route-entry non-regression dry-run:
  `artifacts/tmp/stage2_computed_mask_segment2_route_entry_owner_nonregression/direct-pre-realized-route-entry-nonregression`
  - Covered `computed_masked_segment2_load_unit_store`,
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
  `artifacts/tmp/stage2_computed_mask_segment2_route_entry_owner_nonregression/pre-realized-masked-add-nonregression`
  - Covered `masked_add` over counts `0,7,16,23,257`.
- Added-line authority scan over touched production, script, test, and spec
  files found no new positive legacy i32, source-front-door, descriptor,
  direct-C/source-export, route-id-derived, artifact-name-derived,
  ABI-string-derived, exact-intrinsic-derived, common-EmitC-derived, script- or
  metadata-derived route authority. New matches were limited to targeted
  negative tests for `metadata-*`/`route_id`, `implicit-check-not` guards, or
  provider-derived intrinsic mirrors after route construction.
- `git diff --check` passed before archive.
- `cmake --build build --target check-tianchenrv -j2` passed
  `387/387` tests.

## Spec Update Judgment

Spec update was required because this task promoted a new direct route-entry
member in an existing RVV selected-body realization owner. The durable contract
belongs in `.trellis/spec/extension-plugins/rvv-plugin.md`: computed-mask
segment2 load/unit-store is now a bounded `segment2 memory` route-entry member
with owner-scoped typed-fact validation, while computed-mask segment2
store/unit-load remains selected-boundary-only.

## Validation Plan

1. `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-26-05-26-stage2-rvv-computed-mask-segment2-memory-route-entry-owner`
2. `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
3. `cmake --build build --target tianchenrv-rvv-extension-plugin-test -j2`
4. `./build/bin/tianchenrv-rvv-extension-plugin-test`
5. Direct pre-realized generated-bundle dry-run for
   `computed_masked_segment2_load_unit_store`, counts `0,7,16,23,257`.
6. Direct pre-realized generated-bundle `ssh rvv` run for
   `computed_masked_segment2_load_unit_store`, counts `0,7,16,23,257`.
7. Focused direct route-entry non-regression dry-run for recent representative
   route owners, including landed plain segment2 deinterleave/interleave and
   masked elementwise.
8. Bounded authority scans over touched production, script, and test files.
9. `git diff --check`
10. `cmake --build build --target check-tianchenrv -j2`

## Technical Notes

- Specs read: `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`, and
  `.trellis/spec/lowering-runtime/emitc-route.md`.
- Previous task context read:
  `.trellis/tasks/archive/2026-05/05-26-stage2-rvv-segment2-deinterleave-route-entry-owner/prd.md`,
  `.trellis/tasks/archive/2026-05/05-26-05-26-stage2-rvv-segment2-interleave-memory-route-family-owner/prd.md`,
  and
  `.trellis/tasks/archive/2026-05/05-26-stage2-rvv-masked-elementwise-route-family-owner/prd.md`.
- Initial production files:
  `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp`,
  `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`,
  and `scripts/rvv_generated_bundle_abi_e2e.py`.
- Focused tests:
  `test/Plugin/RVVExtensionPluginTest.cpp`, computed-mask segment2 generated
  bundle tests under `test/Scripts/`, and directly related segment2, masked
  elementwise, memory, math, and generated-bundle non-regression paths.
