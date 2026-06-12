# Stage2 RVV Segment2 Deinterleave Route-Entry Owner

## Goal

Promote the bounded typed RVV `segment2_deinterleave_unit_store`
representative from selected-boundary-only support to RVV plugin-owned direct
route-entry support. The production path is:

```text
selected tcrv.exec RVV variant
  -> typed pre-realized tcrv_rvv.segment2_deinterleave body
  -> RVV segment2 memory selected-body realization owner
  -> realized setvl/with_vl/segment2_load/move/store/store body
  -> RVV segment2 family plan, route-control plan, memory operand binding,
     and statement-plan boundaries
  -> provider-built TCRVEmitCLowerableRoute
  -> common EmitC
  -> generated RVV artifact and ssh rvv evidence
```

This is the complementary direct route-entry owner for the segment2
deinterleave direction after `segment2_interleave_unit_load` landed. It is not
a segment-width matrix, dtype/LMUL clone batch, computed-mask segment2 task,
source-front-door task, high-level frontend route, or evidence-only round.

## Direction Source

- Direction title: `Stage2 RVV segment2 deinterleave route-entry owner`.
- Module owner: RVV plugin-owned direct route-entry realization and
  provider/planning boundary for the bounded plain
  `segment2_deinterleave_unit_store` representative.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean.
- Initial HEAD: `ea85a9a7 rvv: route segment2 interleave entry owner`.
- `.trellis/.current-task` was absent, so this task was created from the Hermes
  brief before source edits.

## Current Repository Facts

- Specs require route support to come from typed `tcrv_rvv`
  body/config/runtime facts, selected target capability facts, RVV
  plugin-owned realization, and a provider-built
  `TCRVEmitCLowerableRoute`. Common EmitC/export may only materialize provider
  output and mirror facts after route construction.
- The archived interleave owner task made `segment2_interleave_unit_load`
  direct route-entry capable and left plain deinterleave explicitly outside
  the direct route-entry subset.
- Current `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp` already has a
  segment2 memory owner and selected-boundary realization for both
  `TypedSegment2DeinterleaveMemoryPreRealizedBodyOp` and
  `TypedSegment2InterleaveMemoryPreRealizedBodyOp`.
- The exact production gap is bounded: `isPreRealizedRVVSegment2MemoryRouteEntryOp`
  accepts only `TypedSegment2InterleaveMemoryPreRealizedBodyOp`, so a direct
  route-entry path still rejects the existing typed deinterleave body before
  provider facts can be collected.
- `test/Plugin/RVVExtensionPluginTest.cpp` currently contains a positive
  direct route-entry case for segment2 interleave and a negative case proving
  segment2 deinterleave is not route-entry capable.
- `scripts/rvv_generated_bundle_abi_e2e.py` already has explicit and
  pre-realized selected-boundary expectations, generated artifact checks, and
  runtime harness logic for `segment2_deinterleave_unit_store`; its direct
  pre-realized route-entry allowlist still excludes that op.

## Requirements

1. Keep the executable representative to one plain
   `segment2_deinterleave_unit_store` route with `src`, `out0`, `out1`, and
   runtime `n`.
2. The segment2 memory realization owner remains one owner for all segment2
   pre-realized bodies. Its route-entry eligibility must be owner-scoped and
   must explicitly include bounded plain deinterleave and bounded plain
   interleave, while keeping unrelated computed-mask segment2 bodies outside
   the direct route-entry subset.
3. Direct route-entry realization must consume the typed deinterleave
   pre-realized body into `setvl`, `with_vl`, `segment2_load`, two `move`
   operations, and two unit stores before route facts are collected.
4. Provider facts must still be validated through the existing RVV-owned
   segment2 family plan, route-control plan, memory operand-binding facts, and
   segment2 statement-plan boundary. The direct route-entry hook must not
   bypass or duplicate those planning/provider checks.
5. Generated-bundle direct route-entry evidence must expose route-entry
   realization and prove the generated artifact carries `src`, `out0`, `out1`,
   runtime `n`, segment count 2, field order, interleaved source form,
   unit-store destination form, SEW32/LMUL m1, policy, route family plan,
   route-control facts, binding plan, provider mirrors, and segment-load/field
   output facts as mirrors after provider route construction.
6. Runtime evidence must distinguish true deinterleave from interleave,
   concatenation, stream swap, unit copy, wrong stride, stale constant data,
   inactive route behavior, and tail clobbering. Counts must include zero,
   small, exact/tail cases, and a non-one-vector stress case.
7. Recent segment2 interleave, masked elementwise, reduction,
   scalar-broadcast, conversion, runtime-scalar, compare/select, MAcc,
   contraction, and base memory route owners must not regress.
8. No source-front-door, descriptor, direct-C/source-export, route-id,
   artifact-name, ABI-string, exact-intrinsic, script, common-EmitC, metadata,
   status, or legacy-i32 surface may become route authority.

## Acceptance Criteria

- [x] `segment2 memory` has an owner-scoped direct route-entry predicate that
      accepts the bounded `TypedSegment2DeinterleaveMemoryPreRealizedBodyOp`
      shape and still accepts the landed bounded interleave shape.
- [x] Computed-mask segment2 bodies and malformed deinterleave bodies remain
      fail-closed before route construction through owner/local diagnostics.
- [x] C++ route-entry owner tests prove `segment2_deinterleave_unit_store`
      reaches route-entry realization through the owner registry, consumes the
      pre-realized body, realizes `segment2_load` plus two moves and two
      stores, reaches `rvv-segment2-memory-route-family-plan.v1`, and builds a
      provider route.
- [x] The same focused tests keep `segment2_interleave_unit_load` positive and
      keep unrelated route-entry owner coverage intact.
- [x] `rvv_generated_bundle_abi_e2e.py --direct-pre-realized-route-entry`
      supports the bounded `segment2_deinterleave_unit_store` representative
      and records route-entry realization evidence.
- [x] Generated-bundle dry-run passes for direct pre-realized
      `segment2_deinterleave_unit_store` over counts `0,7,16,23,257`.
- [x] Real `ssh rvv` generated-bundle run passes for direct pre-realized
      `segment2_deinterleave_unit_store` over counts `0,7,16,23,257`, with
      output proving field-order distinction and tail preservation.
- [x] Focused non-regression covers recent representative route owners:
      segment2 interleave, masked elementwise, standalone reduction,
      scalar-broadcast, conversion, runtime-scalar MAcc, compare/select, MAcc,
      contraction, and base memory.
- [x] `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`,
      focused C++ test target, focused generated-bundle checks, bounded
      authority scans, `git diff --check`, and `check-tianchenrv` pass or exact
      blockers are recorded.
- [x] Trellis task state, journal/archive, and one coherent commit are
      truthful if complete.

## Out Of Scope

- Segment3/segment4, dtype/LMUL clone batches, gather/scatter expansion,
  computed-mask segment2 expansion, high-level Linalg/Vector frontend work,
  source-front-door positives, dashboards, global tuning DBs, or broad smoke
  matrices.
- Moving segment semantics, dtype/config, route support, intrinsic choices, or
  ABI role authority into common EmitC/export, scripts, metadata, route ids,
  artifact names, descriptors, ABI strings, or legacy i32 helpers.
- Reworking existing segment2 family, route-control, memory operand-binding,
  or statement-plan boundaries unless direct route-entry integration reveals a
  correctness bug.

## Technical Approach

1. Extend the segment2 route-entry predicate in
   `RVVSelectedBodyRealization.cpp` to recognize bounded plain deinterleave by
   typed op class, op kind, and memory form.
2. Update the route-entry diagnostic text to name segment2 deinterleave and
   interleave memory as supported direct route-entry families.
3. Convert the C++ route-entry owner test from deinterleave-negative to
   deinterleave-positive, with realized-body shape checks tailored to the
   deinterleave sequence.
4. Allow `segment2_deinterleave_unit_store` in generated-bundle
   `--direct-pre-realized-route-entry` mode and update the help/diagnostic
   text.
5. Run direct pre-realized generated-bundle dry-run and `ssh rvv` evidence with
   counts `0,7,16,23,257`; run focused non-regression and final quality gates.

## Implementation Result

- Extended `isPreRealizedRVVSegment2MemoryRouteEntryOp` so the existing
  `segment2 memory` realization owner accepts bounded plain
  `TypedSegment2DeinterleaveMemoryPreRealizedBodyOp` route-entry bodies by
  typed op class, `op_kind = segment2_deinterleave_unit_store`, and
  `memory_form = segment2-load-unit-store`.
- Kept the segment2 owner shared with the landed
  `segment2_interleave_unit_load` path and left computed-mask segment2 bodies
  outside the direct route-entry subset.
- Updated the route-entry diagnostic to name segment2
  deinterleave/interleave memory as supported direct route-entry families.
- Converted the focused C++ owner test from deinterleave-negative to
  deinterleave-positive and asserted that direct route-entry realization
  consumes the pre-realized body into one `segment2_load`, two `move` ops, and
  two `store` ops before provider route facts are collected.
- Extended `rvv_generated_bundle_abi_e2e.py` so
  `--pre-realized-selected-body --direct-pre-realized-route-entry` accepts the
  bounded `segment2_deinterleave_unit_store` representative.
- Updated `.trellis/spec/extension-plugins/rvv-plugin.md` to preserve the
  durable contract: direct segment2 memory route-entry support is
  owner-scoped, currently bounded to plain deinterleave and plain interleave,
  and still requires realized typed RVV structure to feed the segment2 family,
  route-control, operand-binding, and statement-plan boundaries.

## Validation Evidence

- `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-26-stage2-rvv-segment2-deinterleave-route-entry-owner`
  passed.
- `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py` passed.
- `git diff --check` passed.
- `cmake --build build --target tianchenrv-rvv-extension-plugin-test -j2`
  passed.
- `cmake --build build --target tcrv-opt tcrv-translate -j2` passed.
- `./build/bin/tianchenrv-rvv-extension-plugin-test` passed with
  `RVV extension plugin smoke test passed`.
- Direct pre-realized generated-bundle dry-run:
  `artifacts/tmp/stage2_segment2_deinterleave_route_entry_owner/direct-pre-realized-segment2-deinterleave-dry-run`
  - Result: `rvv_generated_bundle_abi_e2e: dry_run_success`.
  - Counts: `0,7,16,23,257`.
  - Evidence includes direct route-entry materializer
    `rvv-route-entry-selected-body-realization`, `route_entry_realization:
    true`, no selected-lowering-boundary materializer in the actual pipeline,
    realized `segment2_load` plus field moves and stores, runtime ABI order
    `src,out0,out1,n`, route operand-binding plan
    `rvv-route-operand-binding:segment2_deinterleave_unit_store.v1`,
    `rvv-segment2-memory-route-family-plan.v1`, and
    `provider_supported_mirror:rvv-segment2-deinterleave-plan-validated`.
- Real `ssh rvv` direct pre-realized generated-bundle run:
  `artifacts/tmp/stage2_segment2_deinterleave_route_entry_owner_ssh/direct-pre-realized-segment2-deinterleave-ssh-rvv`
  - Result: `PASS op=segment2_deinterleave_unit_store counts=0,7,16,23,257`.
  - Output reported field-order distinguishing lanes for `n=7`, `16`, `23`,
    and `257`, and `tail_preserved` for every count including `0`.
- Focused direct route-entry non-regression dry-run:
  `artifacts/tmp/stage2_segment2_deinterleave_route_entry_owner_nonregression/direct-pre-realized-route-entry-nonregression`
  - Covered `segment2_interleave_unit_load`, `scalar_broadcast_add`,
    `standalone_reduce_add`, `widen_i16_to_i32`,
    `runtime_scalar_cmp_masked_macc_add`, `cmp_select`,
    `computed_mask_select`, `macc_add`, `scalar_broadcast_macc_add`,
    `computed_masked_macc_add`, `widening_macc_add`,
    `widening_dot_reduce_add`, `strided_load_unit_store`, and
    `computed_masked_widening_dot_reduce_add`.
- Masked elementwise non-regression dry-run:
  `artifacts/tmp/stage2_segment2_deinterleave_route_entry_owner_nonregression/pre-realized-masked-add-nonregression`
  - Covered `masked_add` over counts `0,7,16,23,257`.
- Added-line authority scan over touched production, script, test, and spec
  files found no new positive legacy i32, source-front-door, descriptor,
  direct-C/source-export, route-id-derived, artifact-name-derived,
  ABI-string-derived, exact-intrinsic-derived, common-EmitC-derived, script- or
  metadata-derived route authority. The only match was the spec statement that
  provider-built routes lower through common EmitC.
- `cmake --build build --target check-tianchenrv -j2` passed
  `386/386` tests.

## Spec Update Judgment

Spec update was required because this task promoted a new direct route-entry
member of an existing selected-body realization owner. The update belongs in
`.trellis/spec/extension-plugins/rvv-plugin.md` because it changes the durable
RVV selected-body route-entry contract, not just one test fixture.

## Validation Plan

1. `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-26-stage2-rvv-segment2-deinterleave-route-entry-owner`
2. `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
3. `cmake --build build --target tianchenrv-rvv-extension-plugin-test -j2`
4. `./build/bin/tianchenrv-rvv-extension-plugin-test`
5. Direct pre-realized generated-bundle dry-run for
   `segment2_deinterleave_unit_store`, counts `0,7,16,23,257`.
6. Direct pre-realized generated-bundle `ssh rvv` run for
   `segment2_deinterleave_unit_store`, counts `0,7,16,23,257`.
7. Focused direct route-entry non-regression dry-run for recent representative
   route owners, including the landed segment2 interleave owner.
8. Bounded authority scans over touched production, script, and test files.
9. `git diff --check`
10. `cmake --build build --target check-tianchenrv -j2`

## Technical Notes

- Specs read: `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/index.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  `.trellis/spec/testing/index.md`,
  `.trellis/spec/testing/mlir-testing-contract.md`, and shared guides.
- Previous task context read:
  `.trellis/tasks/archive/2026-05/05-26-05-26-stage2-rvv-segment2-interleave-memory-route-family-owner/prd.md`.
- Initial implementation surface:
  `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp`,
  `test/Plugin/RVVExtensionPluginTest.cpp`, and
  `scripts/rvv_generated_bundle_abi_e2e.py`.
