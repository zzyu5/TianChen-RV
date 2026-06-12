# Stage2 RVV Route-Family Artifact ABI Consumer Owner

## Goal

Make the RVV target artifact/runtime ABI consumer boundary explicitly consume
provider-built segment2 `TCRVEmitCLowerableRoute` facts before generated RVV C
artifact claims. The production target/export path must use the rebuilt
provider route as authority for segment2 headers, type mappings, ABI order,
source provenance, statement-plan provenance, and provider mirror agreement.
It must reject stale metadata, script-derived claims, ABI-string-derived route
claims, route-id-derived claims, artifact-name-derived claims, and missing
provider payload facts before object/header export.

The primary consumer is `computed_masked_segment2_update_unit_load`. One
adjacent segment2 family, preferably `segment2_interleave_unit_load` or the
full active segment2 set, must prove this is a route-family artifact consumer
path rather than a single-case patch.

## Direction Source

- Direction title: `Stage2 RVV route-family artifact ABI consumer owner`.
- Module owner: RVV target artifact/runtime ABI consumer boundary.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean.
- Initial HEAD: `52c4adef rvv: construct segment2 routes from provider plans`.
- No `.trellis/.current-task` existed, so this task was created from the
  supplied Direction Brief before source edits.

## What I Already Know

- The current RVV authority chain is selected `tcrv.exec` RVV variant -> typed
  low-level `tcrv_rvv` body -> RVV plugin legality / selected-body realization
  / route provider -> `TCRVEmitCLowerableRoute` -> neutral common EmitC ->
  target artifact -> `ssh rvv` evidence for runtime/correctness claims.
- The archived predecessor task
  `.trellis/tasks/archive/2026-05/05-26-05-26-stage2-rvv-segment2-provider-route-construction-owner`
  rewired `RVVEmitCRouteProvider` so active segment2 route construction
  consumes `RVVSelectedBodySegment2RouteFamilyProviderPlan` for route id,
  headers, type mappings, ABI mappings, and statement plans.
- `lib/Target/RVV/RVVTargetSupportBundle.cpp` already validates RVV selected
  artifact candidates by resolving the selected `tcrv.exec.variant`, rebuilding
  the provider route through `describeRVVSelectedBodyEmitCRoute(request,
  &route)`, verifying the route, and comparing candidate metadata mirrors and
  runtime ABI parameters against the provider-derived description.
- The remaining bottleneck is to make the artifact/runtime ABI consumer
  boundary explicitly validate segment2 route payload facts from the rebuilt
  provider route: required headers, type mappings, ABI order/value mappings,
  source provenance, call/loop statement provenance, and segment2 family
  mirrors. This should live in the target/export boundary, not in scripts or
  common EmitC.
- `scripts/rvv_generated_bundle_abi_e2e.py` already has segment2 expectations
  and generated-bundle evidence, but scripts must remain consumers/verifiers
  and not define support. Any script change in this task should demote stale
  artifact/ABI claims or add consumer-side self-checks only.
- Common EmitC and generic target export must remain neutral. They must not
  infer RVV dtype, policy, memory form, route support, intrinsic spelling,
  ABI order, or segment2 family from metadata or route ids.

## Requirements

1. Add focused production validation in `lib/Target/RVV/RVVTargetSupportBundle.cpp`
   so the RVV target artifact bridge consumes the rebuilt provider route
   payload for segment2 route families before accepting generated object/header
   artifact candidates.
2. The primary segment2 path must be
   `computed_masked_segment2_update_unit_load`; at least one adjacent active
   segment2 family must pass through the same consumer validation. Cover all
   active segment2 families if the implementation remains local.
3. The target consumer must validate, from provider-built route facts and
   provider-derived description rather than scripts or names:
   route id, required headers, VL/vector/mask type mappings, ordered ABI
   mappings, selected source provenance, provider-supported mirror labels,
   route operand binding mirrors, segment2 family plan mirror, runtime
   n/AVL/VL relation mirrors, memory form, mask relation, arithmetic kind for
   update, and statement-plan provenance.
4. Unsupported or inconsistent route family, ABI order, operand binding, memory
   form, mask/VL relation, runtime `n`/AVL, type mapping, header requirement,
   provider mirror, route id, artifact kind/name, or script-only claim must fail
   closed before executable artifact claims.
5. If inspection proves part of the production boundary is already correct,
   preserve it and strengthen the precise missing proof rather than duplicating
   provider logic or rebuilding segment2 semantics in target/export.
6. Do not add new RVV operations, segment widths, dtype/LMUL clone batches,
   high-level frontend/Linalg routes, common EmitC RVV semantics, dashboards,
   broad smoke matrices, or evidence-only packaging.

## Acceptance Criteria

- [x] Production changes touch `lib/Target/RVV/RVVTargetSupportBundle.cpp` and
      only narrow interfaces/tests/scripts when needed.
- [x] `computed_masked_segment2_update_unit_load` and at least one adjacent
      segment2 family prove target artifact/runtime ABI validation consumes
      provider-built route headers, type mappings, ABI mappings, source
      provenance, statement provenance, and mirrors after route construction.
- [x] Fail-closed coverage rejects stale route id, stale provider mirror, wrong
      ABI order, missing operand binding, missing or wrong type/header mapping,
      wrong segment2 family plan, wrong memory form, wrong mask/runtime AVL
      relation, artifact-name-derived support, and script-derived authority
      attempts where the current test surface exposes them.
- [x] Generated-bundle dry-run covers computed-mask segment2 update and one
      adjacent segment2 family.
- [x] `ssh rvv` evidence covers computed-mask segment2 update or the migrated
      representative with counts including 0, small, exact, tail, and stress.
- [x] Focused non-regression covers computed-mask segment2 store/load,
      segment2 deinterleave/interleave, masked elementwise, reduction,
      scalar-broadcast, conversion, runtime-scalar MAcc, compare/select, MAcc,
      contraction, and base memory route-entry paths.
- [x] Bounded touched-file authority scan finds no new central ad hoc,
      name-derived, metadata-derived, descriptor-derived, ABI-string-derived,
      script-derived, artifact-name-derived, common-EmitC-derived,
      source-front-door-derived, route-id-derived, exact-intrinsic-derived, or
      legacy-i32-derived executable authority.
- [x] `git diff --check` passes.
- [x] Focused build/test targets pass, and `check-tianchenrv` passes or an
      exact blocker is recorded and the task remains open.
- [x] Trellis task status, journal, archive, and one coherent commit are
      completed if all acceptance criteria are satisfied.

## Validation Plan

1. Start the Trellis task after PRD/context setup.
2. Implement focused target/export validation over rebuilt provider route
   payloads for segment2 candidates.
3. Add or update focused C++ and/or generated-bundle self-tests that exercise
   the artifact/runtime ABI consumer boundary, not provider route construction
   alone.
4. Build and run `tianchenrv-rvv-extension-plugin-test` if C++ plugin tests are
   touched.
5. Run generated-bundle dry-runs for direct pre-realized
   `computed_masked_segment2_update_unit_load` and one adjacent segment2
   family.
6. Run direct route-entry and selected-boundary non-regression dry-runs for the
   listed active owners.
7. Run representative `ssh rvv` evidence as required by the changed segment2
   artifact path.
8. Run bounded authority scan, `git diff --check`, and `check-tianchenrv`.

## Out Of Scope

- New operation support or broad Stage2 coverage expansion.
- New segment factor beyond segment2.
- New dtype/LMUL clone batches.
- Linalg/Vector/StableHLO frontend lowering.
- Source-front-door positive routes.
- Descriptor/direct-C/source-export route resurrection.
- Common EmitC/export RVV semantic inference.
- Dashboard/report-only/helper-only changes.
- Weakening existing owners for computed-mask segment2 update/store/load,
  plain segment2 deinterleave/interleave, masked elementwise, reduction,
  scalar-broadcast, conversion, runtime-scalar MAcc, compare/select, MAcc,
  contraction, or base memory.

## Technical Notes

- Specs read:
  - `.trellis/spec/index.md`
  - `.trellis/spec/extension-plugins/rvv-plugin.md`
  - `.trellis/spec/lowering-runtime/emitc-route.md`
  - `.trellis/spec/core-dialect/tcrv-exec-contract.md`
- Predecessor context read:
  - `.trellis/tasks/archive/2026-05/05-26-05-26-stage2-rvv-segment2-provider-route-construction-owner/prd.md`
  - `.trellis/tasks/archive/2026-05/05-26-05-26-stage2-rvv-segment2-provider-route-construction-owner/implement.jsonl`
  - `.trellis/tasks/archive/2026-05/05-26-05-26-stage2-rvv-segment2-provider-route-construction-owner/check.jsonl`
- Initial code surface:
  - `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`
  - `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`
  - `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
  - `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
  - `lib/Target/RVV/RVVTargetSupportBundle.cpp`
  - `scripts/rvv_generated_bundle_abi_e2e.py`
  - `test/Plugin/RVVExtensionPluginTest.cpp`

## Definition Of Done

The RVV target artifact/runtime ABI consumer boundary actively validates
provider-built segment2 route payloads for computed-mask segment2 update and an
adjacent segment2 family, consumer-side fail-closed tests and generated-bundle
evidence pass, Trellis state is truthful, and one coherent commit records the
completed work.

## Completion Evidence

- Production owner changed:
  `lib/Target/RVV/RVVTargetSupportBundle.cpp`.
- Consumer tests changed:
  `test/Target/RVV/pre-realized-selected-body-artifact-computed-masked-segment2-update.mlir`
  and
  `test/Target/RVV/pre-realized-selected-body-artifact-segment2-interleave-unit-load.mlir`.
- Spec update:
  `.trellis/spec/extension-plugins/rvv-plugin.md` documents the segment2
  target export consumer contract.
- Focused build:
  `cmake --build build --target TianChenRVRVVTarget tcrv-translate tcrv-opt -j2`.
- Focused lit:
  `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv
  Target/RVV/pre-realized-selected-body-artifact-computed-masked-segment2-update.mlir
  Target/RVV/pre-realized-selected-body-artifact-segment2-interleave-unit-load.mlir`
  passed 2/2.
- Focused non-regression lit:
  `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter=...`
  passed 70 selected tests.
- Generated bundle dry-run:
  `scripts/rvv_generated_bundle_abi_e2e.py --dry-run
  --pre-realized-selected-body --direct-pre-realized-route-entry --op-kind
  computed_masked_segment2_update_unit_load --op-kind
  segment2_interleave_unit_load --run-id dry-run-final` passed.
- Real RVV evidence:
  `scripts/rvv_generated_bundle_abi_e2e.py --pre-realized-selected-body
  --direct-pre-realized-route-entry --op-kind
  computed_masked_segment2_update_unit_load --runtime-count
  0 --runtime-count 7 --runtime-count 16 --runtime-count 23 --runtime-count
  257 --run-id ssh-rvv-computed-update-final` passed on `ssh rvv`.
- Authority scan:
  added production lines in `lib/Target/RVV/RVVTargetSupportBundle.cpp` had no
  new descriptor/direct-C/source-export/script-derived/artifact-name/route-id/
  legacy-i32 executable authority matches.
- Final quality:
  `git diff --check`, task context validation, and
  `cmake --build build --target check-tianchenrv -j2` passed; full lit result
  was 390/390.
