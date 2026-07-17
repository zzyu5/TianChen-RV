# Stage2 RVV selected-body contraction realization owner

## Goal

Make the RVV selected-body realization owner the explicit production owner for
pre-realized widening dot/reduction contraction bodies before route planning.
The production route-entry path must be able to consume pre-realized
contraction shorthand, materialize typed low-level `tcrv_rvv` setvl/load/
compare/mask/strided-load/widening-dot-reduce/store structure, and then let the
consolidated direct contraction provider owner build the
`TCRVEmitCLowerableRoute`.

This round is upstream of the direct contraction provider-plan consolidation
from `ea5b3db3`. It does not add new contraction coverage or move semantics
into common EmitC.

## Direction Source

- Direction title: `Stage2 RVV selected-body contraction realization owner`.
- Module owner: RVV plugin-local selected-body realization owner for
  widening-dot/reduction contraction bodies.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean.
- Initial HEAD: `ea5b3db3 rvv: consolidate direct contraction provider plan`.
- `.trellis/.current-task` was absent, so this task was created from the
  supplied Hermes/User brief before source edits.
- Serial worker constraint: no subagents, spawned agents, parallel agents, or
  multi-agent workflows.

## Current Repository Facts

- `RVVSelectedBodyRealization.cpp` already has a `contraction` realization
  owner, validation functions for widening MAcc / widening dot-reduce /
  computed-mask / strided-input contraction pre-realized bodies, and a local
  `RVVSelectedBodyContractionRealizationPlan`.
- That realization currently produces typed low-level `tcrv_rvv` structure:
  `setvl`, `with_vl`, unit or strided loads, optional compare-produced mask,
  `widening_macc`, `widening_dot_reduce` or
  `masked_widening_dot_reduce`, and `store`.
- The owner registry marks contraction as non-route-entry-capable. Therefore
  `RVVExtensionPlugin::buildVariantEmissionPlan(...)` and
  `buildVariantEmitCLowerableRoute(...)` cannot directly realize a
  pre-realized contraction body through the route-entry bridge; existing target
  tests rely on running `--tcrv-materialize-selected-lowering-boundaries`
  before emission planning.
- The archived task
  `.trellis/tasks/archive/2026-05/05-26-stage2-rvv-contraction-route-owner-consolidation`
  added `RVVSelectedBodyDirectContractionRouteProviderPlan` and rewired the
  provider to obtain it before `TCRVEmitCLowerableRoute` construction.
- Focused direct-provider tests already cover realized contraction route
  materialization facts, route-control plan, math operand-binding facts, direct
  provider-plan consumption, and fail-closed diagnostics after realization.

## Requirements

1. Make the RVV selected-body `contraction` owner route-entry-capable for the
   bounded active pre-realized contraction bodies that already have validated
   realization logic:
   - `typed_widening_dot_reduce_pre_realized_body`;
   - `typed_computed_mask_widening_dot_reduce_pre_realized_body`;
   - `typed_computed_mask_strided_input_widening_dot_reduce_pre_realized_body`;
   - preserve existing widening MAcc / strided-input widening dot-reduce
     behavior without expanding coverage.
2. Route-entry realization must dispatch through the owner registry and the
   contraction owner hook. It must not use a separate allowlist, route id,
   artifact name, ABI string, script, test name, descriptor, metadata, or common
   EmitC branch as route authority.
3. Realized IR must structurally carry:
   - operation kind and contraction relation;
   - source dtype/SEW/LMUL and result/accumulator dtype/SEW/LMUL;
   - runtime `n`/AVL/VL policy;
   - unit or strided payload memory form and lhs/rhs stride runtime ABI values
     where present;
   - compare-produced same-VL mask source/predicate for computed-mask cases;
   - accumulator seed/layout and scalar result/store layout;
   - selected variant/kernel identity in `with_vl` attrs.
4. After route-entry realization, the consolidated direct contraction provider
   plan must be reachable and must consume realized facts before route
   construction.
5. Unsupported or inconsistent pre-realized contraction bodies must fail closed
   in the realization owner before route planning, with targeted diagnostics for
   mask, stride ABI, dtype/relation, accumulator/result, runtime `n`, or
   unsupported shape.
6. Existing explicit contraction routes and existing materialize-boundary-first
   pre-realized target tests must keep their behavior.
7. Common EmitC/export, scripts, route ids, ABI strings, artifact names,
   descriptor residue, source-front-door fixtures, mirror metadata, and legacy
   i32 helper names must remain non-authoritative.

## Acceptance Criteria

- [x] `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp` marks the contraction
      realization owner route-entry-capable through a structural owner predicate
      and keeps route-entry dispatch owner-registry based.
- [x] The route-entry diagnostic for unsupported pre-realized families is
      updated so contraction is no longer described as unsupported.
- [x] Focused C++ tests prove `buildVariantEmissionPlan(...)` or
      `buildVariantEmitCLowerableRoute(...)` directly realizes at least:
      `widening_dot_reduce_add`,
      `computed_masked_widening_dot_reduce_add`, and
      `computed_masked_strided_input_widening_dot_reduce_add`.
- [x] The same tests prove the pre-realized op is consumed and the realized IR
      contains the expected typed `tcrv_rvv` setvl/with_vl/load or
      strided_load/compare/masked_widening_dot_reduce or
      widening_dot_reduce/store facts.
- [x] The same tests prove the realized contraction route reaches
      `rvv-contraction-route-family-plan.v1` and the direct contraction
      provider plan before route construction.
- [x] Focused fail-closed C++ coverage exercises representative owner
      diagnostics for inconsistent mask source, wrong stride ABI role,
      stale dtype/relation, accumulator/result layout mismatch, wrong runtime
      `n` role, and unsupported owner mismatch or shape.
- [x] Representative explicit and pre-realized contraction non-regression
      remains covered by focused C++ and/or existing target lit tests. Real
      `ssh rvv` is rerun only if emitted route/artifact/runtime facts change.
- [x] Bounded authority scan over touched production/test files finds no new
      positive route authority from legacy i32, source-front-door, descriptor,
      ABI-string, script, artifact-name, common-EmitC, metadata-only,
      status-only, or route-id-derived sources.
- [x] `git diff --check` passes.
- [x] Focused RVV plugin test and `check-tianchenrv` pass, or an exact blocker
      is recorded.

## Out Of Scope

- No new contraction operation coverage, dtype/LMUL clone batches, unsigned or
  floating variants, high-level Linalg/Vector/StableHLO frontend lowering,
  source-front-door positive route, dashboard/report work, broad smoke matrix,
  or evidence-only task.
- No change to the direct contraction provider-plan semantics from `ea5b3db3`.
- No migration of RVV semantics into `tcrv.exec`, common EmitC/export, target
  metadata, route ids, ABI strings, artifact names, scripts, descriptors,
  source-front-door fixtures, test names, or legacy i32 helper names.
- No generated artifact or runtime correctness claim unless emitted code,
  ABI, artifact, or runtime facts change and are evidenced.

## Technical Approach

1. Validate and start the Trellis task.
2. Reuse the existing contraction owner-local realization hook and structural
   consumer predicate as the route-entry owner predicate.
3. Update the route-entry unsupported-family diagnostic to list contraction as
   supported.
4. Extend focused C++ tests in `test/Plugin/RVVExtensionPluginTest.cpp` with
   route-entry pre-realized contraction cases.
5. Add targeted fail-closed owner diagnostics by mutating typed body attrs/ABI
   roles before route-entry realization.
6. Run focused build/test, authority scan, `git diff --check`, and
   `check-tianchenrv`.

## Validation Plan

1. `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-26-05-26-stage2-rvv-selected-body-contraction-realization-owner`
2. `cmake --build build --target tianchenrv-rvv-extension-plugin-test -j2`
3. `./build/bin/tianchenrv-rvv-extension-plugin-test`
4. Existing focused lit for pre-realized and explicit contraction target
   artifacts if emitted route facts are unchanged; generated-bundle `ssh rvv`
   only if route/artifact/runtime facts change.
5. Bounded authority scan over touched RVV realization/test/spec files.
6. `git diff --check`
7. `cmake --build build --target check-tianchenrv -j2`

## Technical Notes

- Specs read:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`, and
  `.trellis/spec/lowering-runtime/emitc-route.md`.
- Archived task read:
  `.trellis/tasks/archive/2026-05/05-26-stage2-rvv-contraction-route-owner-consolidation/prd.md`.
- Workspace journal read:
  `.trellis/workspace/codex/journal-15.md`, especially sessions 224, 226,
  227, 235, 237, and 238.
- Initial implementation surfaces inspected:
  `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp`,
  `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`,
  `lib/Plugin/RVV/RVVExtensionPlugin.cpp`, and
  `test/Plugin/RVVExtensionPluginTest.cpp`.

## Definition Of Done

Pre-realized contraction selected bodies can enter the production RVV
route-entry path, are realized by the RVV contraction owner into typed
low-level `tcrv_rvv` structure, and then reach the direct contraction provider
plan before `TCRVEmitCLowerableRoute` construction. Focused positive and
fail-closed coverage passes, no non-authoritative route source is introduced,
Trellis state is truthful, final checks pass, and one coherent commit records
the work if all acceptance criteria are met.

## Implementation Summary

- Made the `contraction` `RVVSelectedBodyRealizationOwner` route-entry-capable
  by wiring its structural owner predicate into `isRouteEntryConsumer`.
- Updated the route-entry unsupported-family diagnostic to include
  contraction.
- Added focused C++ route-entry coverage for pre-realized plain
  `widening_dot_reduce_add`, computed-mask
  `computed_masked_widening_dot_reduce_add`, and computed-mask strided
  `computed_masked_strided_input_widening_dot_reduce_add`.
- The C++ test now proves each pre-realized contraction body is consumed into
  realized typed `tcrv_rvv` ops, reaches the contraction route-family plan, and
  obtains the direct contraction provider plan.
- Added owner-local fail-closed coverage for mask-source mismatch, stale
  dtype/relation, result-layout mismatch, unsupported op kind, wrong runtime
  `n` role, and wrong rhs-stride ABI role.
- Updated the RVV plugin spec to record contraction as a supported route-entry
  selected-body realization family.
- Updated the generated-bundle evidence script's direct route-entry allowlist
  and help text so it can exercise the new production contraction route-entry
  bridge without treating the script as route authority.

## Verification Results

- `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-26-05-26-stage2-rvv-selected-body-contraction-realization-owner`
  passed.
- `cmake --build build --target tianchenrv-rvv-extension-plugin-test -j2`
  passed.
- `./build/bin/tianchenrv-rvv-extension-plugin-test` passed.
- Focused lit from `build/test` for three explicit and three pre-realized
  widening-dot contraction target artifacts passed: 6/6.
- `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py` passed.
- `cmake --build build --target tcrv-opt tcrv-translate -j2` passed.
- Direct route-entry dry-run passed:
  `python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run --pre-realized-selected-body --direct-pre-realized-route-entry --op-kind widening_dot_reduce_add --op-kind computed_masked_widening_dot_reduce_add --op-kind computed_masked_strided_input_widening_dot_reduce_add --runtime-count 0 --runtime-count 7 --runtime-count 16 --runtime-count 23 --artifact-root artifacts/tmp/stage2_rvv_selected_body_contraction_realization_owner --run-id direct-pre-realized-contraction-route-entry-dry --overwrite --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate`.
- Added-line authority scan over touched production/test/spec/script files
  found no new positive legacy-i32, source-front-door, descriptor, route-id,
  ABI-string, artifact-name, common-EmitC, metadata-only, or status-only route
  authority. Matches were spec good-case wording, capability availability in
  test MLIR, a provider-supported test assertion after realized provider route
  construction, and an unsupported-shape negative string.
- `git diff --check` passed.
- `cmake --build build --target check-tianchenrv -j2` passed with 381/381
  tests.
- Real `ssh rvv` was not rerun because emitted route, ABI, artifact, and
  runtime semantics did not change; the new behavior is route-entry
  realization reachability for already-supported contraction routes.
