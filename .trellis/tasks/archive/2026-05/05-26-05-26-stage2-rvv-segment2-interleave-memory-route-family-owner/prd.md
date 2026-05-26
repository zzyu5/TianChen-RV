# Stage2 RVV Segment2 Interleave Memory Route-Family Owner

## Goal

Make the existing Stage2 RVV `segment2_interleave_unit_load` representative
route executable through the production selected-body direct route-entry
boundary. The bounded path is:

```text
selected tcrv.exec RVV variant
  -> typed pre-realized tcrv_rvv.segment2_interleave body
  -> RVV segment2 memory selected-body realization owner
  -> realized setvl/with_vl/load/load/segment2_store body
  -> RVV segment2 family plan, route-control plan, memory operand binding,
     and statement-plan boundaries
  -> provider-built TCRVEmitCLowerableRoute
  -> common EmitC
  -> generated RVV artifact and ssh rvv evidence
```

This is a route-entry ownership and evidence strengthening task for one
segment2 interleave representative. It is not a new segment width, dtype, LMUL,
gather/scatter, masked segment, or frontend matrix.

## Direction Source

- Direction title: `Stage2 RVV segment/interleave memory movement route-family owner`.
- Module owner: RVV plugin-owned route-entry realization and
  provider/planning boundary for segment/interleaved memory movement, centered
  on one `segment2_interleave_unit_load` representative.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean.
- Initial HEAD: `3ea38342 rvv: route masked elementwise control owner`.
- `.trellis/.current-task` was absent, so this task was created from the Hermes
  brief before source edits.

## Current Repository Facts

- Specs require route support to come from typed `tcrv_rvv` body/config/runtime
  facts, selected target capability facts, RVV plugin-owned realization, and a
  provider-built `TCRVEmitCLowerableRoute`. Common EmitC/export may only
  materialize provider output and mirror facts after route construction.
- Existing plain segment2 work already provides:
  `RVVSelectedBodySegment2MemoryRouteFamilyPlan`,
  `controlsSegment2Memory`, memory operand-binding facts,
  `RVVSelectedBodySegment2MemoryRouteStatementPlan`, generated-bundle tooling,
  and prior `ssh rvv` evidence for selected-lowering-boundary mode.
- The exact production gap in current HEAD is narrower: the `segment2 memory`
  selected-body realization owner exists, but its `isRouteEntryConsumer` hook
  is `nullptr`, so the direct route-entry bridge cannot consume a
  pre-realized segment2 body before provider route construction.
- `scripts/rvv_generated_bundle_abi_e2e.py` already knows
  `segment2_interleave_unit_load` for explicit and pre-realized
  selected-lowering-boundary evidence, but `--direct-pre-realized-route-entry`
  rejects it.
- Existing dialect/lit coverage already proves many fail-closed segment2
  interleave surfaces: unsupported segment count, duplicated/swapped field
  roles, wrong source/destination ABI roles, invalid memory forms, missing AVL,
  mismatched field config, and stale authority metadata.

## Requirements

1. Keep the executable representative to one plain
   `segment2_interleave_unit_load` route with `src0`, `src1`, `dst`, and
   runtime `n`.
2. The segment2 memory realization owner must remain one owner for all segment2
   pre-realized bodies, but route-entry eligibility must be explicitly narrower
   and must include only the bounded plain interleave representative unless a
   code-local reason requires a different exact subset.
3. Direct route-entry realization must consume the typed pre-realized body into
   `setvl`, `with_vl`, two unit-stride loads, and one `segment2_store` before
   route facts are collected.
4. Provider facts must still be validated through the existing RVV-owned
   segment2 family plan, route-control plan, memory operand-binding facts, and
   segment2 statement-plan boundary. The direct route-entry hook must not
   bypass those planning/provider checks.
5. The generated-bundle direct route-entry evidence must expose the
   route-entry realization boundary and prove the generated artifact carries
   `src0`, `src1`, `dst`, runtime `n`, segment count 2, field order, unit-load
   source forms, interleaved destination form, SEW32/LMUL m1, policy, route
   family plan, route-control facts, binding plan, provider mirrors, and
   segment-store artifact facts as mirrors after provider route construction.
6. Runtime evidence must distinguish true interleave from concatenation, stream
   swap, unit copy, wrong stride, stale constant data, inactive route behavior,
   and tail clobbering. Counts must include zero, small, exact/tail cases, and
   a non-one-vector stress case.
7. Recent masked elementwise, reduction, scalar-broadcast, conversion,
   runtime-scalar, compare/select, MAcc, contraction, and base memory route
   owners must not regress.
8. No source-front-door, descriptor, direct-C/source-export, route-id,
   artifact-name, ABI-string, exact-intrinsic, script, common-EmitC, metadata,
   status, or legacy-i32 surface may become route authority.

## Acceptance Criteria

- [x] `segment2 memory` has an owner-scoped direct route-entry predicate that
      accepts `TypedSegment2InterleaveMemoryPreRealizedBodyOp` for the bounded
      `segment2_interleave_unit_load` shape and does not make unrelated
      segment2 routes route-entry capable.
- [x] C++ route-entry owner tests prove `segment2_interleave_unit_load`
      reaches route-entry realization through the owner registry, consumes the
      pre-realized body, realizes exactly the expected load/load/segment2_store
      structure, reaches the segment2 provider plan, and builds a route through
      the provider.
- [x] The same tests prove non-route-entry segment2 bodies remain outside the
      direct route-entry subset or fail closed through the existing owner
      diagnostics.
- [x] `rvv_generated_bundle_abi_e2e.py --direct-pre-realized-route-entry`
      supports the bounded `segment2_interleave_unit_load` representative and
      records route-entry realization evidence.
- [x] Generated-bundle dry-run passes for direct pre-realized
      `segment2_interleave_unit_load` over counts `0,7,16,23,257`.
- [x] Real `ssh rvv` generated-bundle run passes for direct pre-realized
      `segment2_interleave_unit_load` over counts `0,7,16,23,257`, with output
      proving field-order distinction and tail preservation.
- [x] Focused non-regression covers representative recent route-entry owners:
      masked elementwise, standalone reduction, scalar-broadcast, conversion,
      runtime-scalar MAcc, compare/select, MAcc, contraction, and base memory.
- [x] `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`,
      focused C++ test target, focused generated-bundle checks, bounded
      authority scans, `git diff --check`, and `check-tianchenrv` pass or exact
      blockers are recorded.
- [x] Trellis task state, journal/archive, and one coherent commit are
      truthful if complete.

## Implementation Result

- Added `isPreRealizedRVVSegment2MemoryRouteEntryOp` and wired it into the
  existing `segment2 memory` selected-body realization owner. The owner still
  covers all segment2 pre-realized bodies for selected-boundary realization,
  but direct route-entry eligibility is now explicitly limited to the bounded
  plain `segment2_interleave_unit_load` shape.
- Updated the route-entry diagnostic to name segment2 interleave memory as a
  supported direct route-entry family.
- Extended `RVVExtensionPluginTest.cpp` so the production route-entry path
  proves `segment2_interleave_unit_load` consumes its pre-realized body before
  route facts, realizes two loads plus one `segment2_store`, reaches
  `rvv-segment2-memory-route-family-plan.v1`, and builds a provider route.
- Added C++ coverage proving plain segment2 deinterleave remains outside the
  direct route-entry subset and fails closed through the route-entry bridge
  rather than silently becoming route-entry capable.
- Extended `rvv_generated_bundle_abi_e2e.py` so
  `--direct-pre-realized-route-entry` accepts
  `segment2_interleave_unit_load` and records route-entry realization evidence.
- Updated `.trellis/spec/extension-plugins/rvv-plugin.md` to record the durable
  contract: segment2 memory route-entry support is owner-scoped and currently
  bounded to plain interleave; other segment2 bodies require explicit future
  owner work before route-entry support.

## Validation Evidence

- `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-26-05-26-stage2-rvv-segment2-interleave-memory-route-family-owner`
  passed.
- `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py` passed.
- `cmake --build build --target tianchenrv-rvv-extension-plugin-test -j2`
  passed.
- `./build/bin/tianchenrv-rvv-extension-plugin-test` passed with
  `RVV extension plugin smoke test passed`.
- `cmake --build build --target tcrv-opt tcrv-translate -j2` passed.
- Direct pre-realized generated-bundle dry-run:
  `artifacts/tmp/stage2_segment2_interleave_route_family_owner/direct-pre-realized-segment2-interleave-dry-run`
  - Result: `rvv_generated_bundle_abi_e2e: dry_run_success`.
  - Counts: `0,7,16,23,257`.
  - Evidence includes `route_entry_realization: true`,
    `materializer: rvv-route-entry-selected-body-realization`,
    route operand binding plan
    `rvv-route-operand-binding:segment2_interleave_unit_load.v1`,
    `rvv-segment2-memory-route-family-plan.v1`,
    runtime ABI order `src0,src1,dst,n`, provider-supported mirror, segment
    count 2, tuple type, source/destination memory forms, and generated object
    plus header paths.
- Real `ssh rvv` direct pre-realized generated-bundle run:
  `artifacts/tmp/stage2_segment2_interleave_route_family_owner_ssh/direct-pre-realized-segment2-interleave-ssh-rvv`
  - Result: `PASS op=segment2_interleave_unit_load counts=0,7,16,23,257`.
  - Output reported field-order distinguishing lanes `7,16,23,257` for
    nonzero counts and `tail_preserved` for every count including zero.
- Focused direct route-entry non-regression dry-run:
  `artifacts/tmp/stage2_segment2_interleave_route_family_owner_nonregression/direct-pre-realized-route-entry-nonregression`
  - Covered `scalar_broadcast_add`, `standalone_reduce_add`,
    `widen_i16_to_i32`, `runtime_scalar_cmp_masked_macc_add`, `cmp_select`,
    `computed_mask_select`, `macc_add`, `scalar_broadcast_macc_add`,
    `widening_macc_add`, `widening_dot_reduce_add`,
    `strided_load_unit_store`, and
    `computed_masked_widening_dot_reduce_add`.
- Masked elementwise non-regression dry-run:
  `artifacts/tmp/stage2_segment2_interleave_route_family_owner_nonregression/pre-realized-masked-add-nonregression`
  - Covered `masked_add` over counts `0,7,16,23,257`.
- Bounded added-line authority scan over touched production, script, test, and
  PRD/spec files found no new positive legacy i32, source-front-door,
  descriptor, direct-C/source-export, route-id-derived, artifact-name-derived,
  ABI-string-derived, exact-intrinsic-derived, common-EmitC-derived, script- or
  metadata-derived route authority.
- `git diff --check` passed.
- `cmake --build build --target check-tianchenrv -j2` passed twice after code
  changes; the final run passed 386/386 tests.

## Spec Update Judgment

Spec update was required because this task changed the durable selected-body
route-entry contract. The update in
`.trellis/spec/extension-plugins/rvv-plugin.md` states that the segment2 memory
realization owner may expose a narrower route-entry subset and that the current
direct route-entry representative is bounded plain
`segment2_interleave_unit_load`; other segment2 bodies remain
selected-boundary-only until explicitly promoted.

## Out Of Scope

- Segment3/segment4, dtype/LMUL clone batches, gather/scatter expansion,
  masked segment2 expansion, high-level Linalg/Vector frontend work, source
  front-door positives, dashboards, global tuning DBs, or broad smoke matrices.
- Moving segment semantics, dtype/config, route support, intrinsic choices, or
  ABI role authority into common EmitC/export, scripts, metadata, route ids,
  artifact names, descriptors, ABI strings, or legacy i32 helpers.
- Reworking existing segment2 family, route-control, or statement-plan
  boundaries unless direct route-entry integration reveals a correctness bug.

## Technical Approach

1. Add a segment2 route-entry predicate in `RVVSelectedBodyRealization.cpp`
   that makes only the bounded plain interleave pre-realized body direct
   route-entry capable.
2. Register that predicate on the existing `segment2 memory` realization owner
   and update the route-entry diagnostic allowlist text.
3. Extend C++ route-entry owner tests with a `segment2_interleave_unit_load`
   pre-realized fixture and provider-plan assertion.
4. Allow `segment2_interleave_unit_load` in generated-bundle
   `--direct-pre-realized-route-entry` mode and update the help/diagnostic
   text.
5. Run direct pre-realized generated-bundle dry-run and `ssh rvv` evidence with
   counts `0,7,16,23,257`; run focused non-regression and final quality gates.

## Validation Plan

1. `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-26-05-26-stage2-rvv-segment2-interleave-memory-route-family-owner`
2. `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
3. `cmake --build build --target tianchenrv-rvv-extension-plugin-test -j2`
4. `./build/bin/tianchenrv-rvv-extension-plugin-test`
5. Direct pre-realized generated-bundle dry-run for
   `segment2_interleave_unit_load`, counts `0,7,16,23,257`.
6. Direct pre-realized generated-bundle `ssh rvv` run for
   `segment2_interleave_unit_load`, counts `0,7,16,23,257`.
7. Focused direct route-entry non-regression dry-run for recent representative
   route owners.
8. Bounded authority scans over touched production, script, and test files.
9. `git diff --check`
10. `cmake --build build --target check-tianchenrv -j2`

## Technical Notes

- Specs read: `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  `.trellis/spec/testing/index.md`,
  `.trellis/spec/testing/mlir-testing-contract.md`,
  `.trellis/spec/variant-pipeline/index.md`, and the shared guides.
- Previous task context read:
  `.trellis/tasks/archive/2026-05/05-26-stage2-rvv-masked-elementwise-route-family-owner/prd.md`,
  `.trellis/tasks/archive/2026-05/05-25-stage2-rvv-segment2-route-control/prd.md`,
  `.trellis/tasks/archive/2026-05/05-24-stage2-rvv-segment2-memory-statement-plan-ownership/prd.md`,
  `.trellis/tasks/archive/2026-05/05-23-05-23-stage2-rvv-plain-segment2-memory-route-family/prd.md`,
  and
  `.trellis/tasks/archive/2026-05/05-20-05-20-stage2-rvv-segment2-interleave-memory-executable-slice/prd.md`.
- Initial implementation surface:
  `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp`,
  `test/Plugin/RVVExtensionPluginTest.cpp`, and
  `scripts/rvv_generated_bundle_abi_e2e.py`.
