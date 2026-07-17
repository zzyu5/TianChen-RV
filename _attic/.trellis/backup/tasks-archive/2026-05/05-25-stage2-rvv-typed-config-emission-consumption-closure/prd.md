# Stage2 RVV typed config emission consumption closure

## Goal

Close the provider/emission side of the typed RVV dtype/config route facts
boundary added by the previous task. Existing route-supported selected-body
families must consume typed config facts as the authority for vector C types,
headers, intrinsic leaves, statement-plan inputs, and target artifact mirrors,
then fail closed before route or artifact export when dtype/SEW/LMUL/policy
facts are missing, stale, or inconsistent.

This is a bounded Stage2 RVV ownership task. It is not new route-family
coverage, not descriptor/source export, and not a runtime/performance claim.

## Direction Source

- Direction title: `Stage2 RVV typed config emission consumption closure`.
- Module owner: RVV plugin-owned consumption of typed dtype/config route facts
  in provider route construction, intrinsic spelling, EmitC lowering inputs,
  and RVV artifact mirrors for existing route-supported families.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean before task creation.
- Initial HEAD: `7156b6cd rvv: add typed config route facts boundary`.
- `.trellis/.current-task` was absent, so this task was created from the
  Hermes Direction Brief before source edits.

## What I Already Know

- `.trellis/spec/index.md` requires the current RVV-first authority chain:
  selected `tcrv.exec` RVV variant -> typed low-level `tcrv_rvv` body -> RVV
  plugin-owned legality / selected-body realization / route provider ->
  `TCRVEmitCLowerableRoute` -> common EmitC -> target artifact.
- `.trellis/spec/extension-plugins/rvv-plugin.md` requires route
  materialization facts to be RVV-local provider input derived from verified
  typed body/config/runtime facts and family plans; common EmitC and target
  export may consume mirrors only after provider route construction.
- `.trellis/spec/lowering-runtime/emitc-route.md` requires common EmitC/export
  to consume provider-built route payloads without inferring RVV dtype, SEW,
  LMUL, mask/tail policy, operation kind, schedules, ABI, headers, or
  intrinsic names from names or metadata.
- The previous archived task
  `.trellis/tasks/archive/2026-05/05-25-stage2-rvv-typed-dtype-config-derivation-boundary/prd.md`
  added `RVVSelectedBodyTypedConfigFacts`, derives it during selected-body
  route analysis, mirrors it through
  `RVVSelectedBodyRouteMaterializationFacts`, and added focused arithmetic,
  memory, and stale-fact tests.
- Current code evidence:
  - `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h` carries
    `RVVSelectedBodyTypedConfigFacts` in both route analysis and
    materialization facts.
  - `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp` derives those facts from
    realized `setvl` / `with_vl` config and provider config profile, and
    verifies they mirror the provider-derived route description before
    materialization facts are returned.
  - `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp` already obtains
    materialization facts before route construction and uses their type/leaves
    for much of route statement construction, but the central provider and
    target artifact path still need an explicit consumption closure so typed
    facts are the documented provider/export authority rather than a side
    mirror.
  - `lib/Target/RVV/RVVTargetSupportBundle.cpp` rebuilds the provider route
    before accepting a target artifact candidate, but its metadata mirror
    validation currently checks only selected route/operation/predicate/binding
    mirrors, not the full typed config mirror set.

## Requirements

1. Provider route construction must explicitly consume
   `RVVSelectedBodyRouteMaterializationFacts::typedConfigFacts` as the typed
   dtype/config authority for generic result vector type, mask/index types,
   VL C type, vector C type, setvl/load/store intrinsic leaves, and headers
   for existing route-supported families.
2. Statement-plan paths for at least one arithmetic or compare/select family
   and one memory family must receive the same materialization facts boundary
   and must not derive dtype/config from operation names, route ids, helper
   names, ABI strings, artifact names, or common EmitC/export code.
3. Target artifact metadata validation must fail closed when candidate
   `tcrv_rvv.*` dtype/config mirror fields disagree with the provider-rebuilt
   selected-body route description. Mirrors remain mirrors; they do not become
   route authority.
4. Required headers, vector/mask/index C type mappings, and intrinsic leaves
   must be accepted only after typed facts and provider/family plans agree.
5. Unsupported or inconsistent typed fact combinations must produce targeted
   diagnostics before provider route construction or target artifact export.
6. Retained i32 routes remain ordinary generic typed `tcrv_rvv` instances. Do
   not introduce legacy `RVVI32M1*`, `rvv-i32m1-*`, `tcrv_rvv.i32_*`,
   `!tcrv_rvv.i32m*`, descriptor, source-front-door, or exact intrinsic-name
   authority.
7. Common EmitC and target export remain neutral consumers; they may validate
   provider mirrors but must not choose RVV dtype/config/intrinsics.

## Acceptance Criteria

- [x] PRD and context files reference the RVV plugin, EmitC route, testing,
      implementation-stack specs, and previous typed derivation task.
- [x] Provider construction uses typed config facts from materialization facts
      as the explicit source for vector C type, VL type, generic load/store
      intrinsic leaves, and header/type-map inputs for ordinary elementwise or
      compare/select plus one memory route.
- [x] Focused C++ tests prove typed config facts drive route construction and
      emitted route payload fields for one arithmetic or compare/select case
      and one memory case.
- [x] Focused fail-closed tests cover stale typed config facts before provider
      construction and stale target artifact dtype/config mirror metadata
      before artifact export.
- [x] Representative lit/FileCheck target fixtures show provider-derived
      dtype/config mirrors for affected arithmetic or compare/select and memory
      artifacts.
- [x] Bounded scans over touched RVV planning/provider/target/test files show
      no new positive legacy i32/helper/name/metadata/source-front-door or
      descriptor authority.
- [x] `git diff --check` passes.
- [x] Focused RVV plugin/target tests pass, and `check-tianchenrv` passes or an
      exact blocker is documented.
- [x] Trellis task status, journal, archive, and commit are truthful if the
      task completes.

## Non-Goals

- No broad new operation-family coverage, reductions, contractions, high-level
  frontend lowering, source-front-door positive routes, descriptor/direct-C/
  source-export paths, dashboards, broad smoke matrices, or runtime/performance
  claims.
- No one-op-per-intrinsic wrappers, dtype/LMUL clone batches, or new
  dtype-prefixed helper ops.
- No movement of RVV semantic choices into common EmitC/export.
- No Python implementation of compiler core, dialect, legality, route
  planning, provider, lowering, or emission behavior.

## Technical Approach

1. Start and validate this Trellis task.
2. Inspect current typed facts derivation/materialization, provider type-map
   and statement construction, target metadata mirror validation, and affected
   tests.
3. Add a small provider-side typed-config consumption helper or equivalent
   validation so provider construction consumes the typed facts fields directly
   and fails closed if the provider description/family plan payload disagrees.
4. Extend target artifact candidate mirror validation for typed config fields
   such as element type, SEW, LMUL, tail/mask policy, config contract, vector
   C type, setvl/load/store intrinsic, and related type mirrors that are
   present in metadata.
5. Add focused positive and fail-closed C++ tests and representative
   lit/FileCheck checks.
6. Run focused builds/tests, bounded authority scans, `git diff --check`, and
   `check-tianchenrv` if feasible.

## Validation Plan

1. `rtk python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-25-stage2-rvv-typed-config-emission-consumption-closure`
2. `rtk cmake --build build --target tianchenrv-rvv-extension-plugin-test -j2`
3. `rtk build/bin/tianchenrv-rvv-extension-plugin-test`
4. Focused lit/FileCheck for affected selected-body RVV target fixtures.
5. Focused target support test if available, or C++ plugin coverage for target
   mirror validation helper if target support has no standalone binary.
6. Bounded authority scan over touched RVV planning/provider/target/test files.
7. `rtk git diff --check`
8. `rtk cmake --build build --target check-tianchenrv -j2`, unless an exact
   blocker is encountered and recorded.

## Definition Of Done

Provider/emission and target artifact export consume typed config facts as the
only dtype/config authority for existing route-supported selected-body
families; stale provider or artifact mirrors fail closed; focused and full
checks pass or an exact blocker is documented; the task is finished/archived;
and one coherent commit records the work.

## Implementation Result

- `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp` now initializes provider
  materialization defaults from `RVVSelectedBodyTypedConfigFacts` for VL C
  type, result vector type/C type, setvl leaf, typed vector load/store leaves,
  and mask type mirrors when present. Existing family plans may still provide
  family-specialized leaves, but they are checked against the typed facts for
  the generic typed fields they consume.
- `getRVVSelectedBodyRouteMaterializationFacts` now fails closed when provider
  materialization facts disagree with typed config facts for VL C type, result
  vector type, result vector C type, setvl intrinsic, mask type/C type, and
  generic typed vector load/store leaves used by elementwise/select or base
  memory consumers.
- `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp` now reads
  `materializationFacts.typedConfigFacts` directly for route type mappings:
  `!tcrv_rvv.vl`, the result vector type/C type, and index vector type/C type
  when present.
- Focused plugin tests prove the provider route maps typed config facts for an
  arithmetic LMUL m2 route and a base memory route, and reject a stale base
  memory plan that tries to carry `metadata_i32_vec` instead of `vint32m1_t`.
- Focused target artifact tests prove stale `tcrv_rvv.element_type` mirror
  metadata is rejected before artifact export. Live code already rebuilt the
  provider route and validated the broader typed mirror set; this round added
  direct stale element-type coverage rather than duplicating that target
  implementation.

## Validation Result

- [x] `rtk python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-25-stage2-rvv-typed-config-emission-consumption-closure`
- [x] `rtk python3 ./.trellis/scripts/task.py validate .trellis/tasks/archive/2026-05/05-25-stage2-rvv-typed-config-emission-consumption-closure`
- [x] `rtk cmake --build build --target tianchenrv-rvv-extension-plugin-test -j2`
- [x] `rtk build/bin/tianchenrv-rvv-extension-plugin-test`
- [x] `rtk cmake --build build --target tianchenrv-target-artifact-export-test -j2`
- [x] `rtk build/bin/tianchenrv-target-artifact-export-test`
- [x] `rtk cmake --build build --target tcrv-opt tcrv-translate -j2`
- [x] Focused lit/FileCheck filter for
      `pre-realized-selected-body-artifact-(cmp-select|strided-load-unit-store|i64-add)`
      from `build/test`; the first attempt overlapped with relinking and hit a
      transient `Permission denied` process-spawn error, the rerun after build
      completion passed.
- [x] Bounded authority scan over touched RVV planning/provider/target/test
      files. Hits were existing negative/fail-closed fixtures or target
      descriptor/source-export rejection tests; no new positive legacy i32,
      helper-name, metadata, source-front-door, descriptor, direct-C, or
      source-export authority was introduced.
- [x] `rtk git diff --check`
- [x] `rtk cmake --build build --target check-tianchenrv -j2`

## Spec Sync

No `.trellis/spec/**` change was required. This round implemented the existing
RVV plugin typed-config route-materialization contract and EmitC route
neutrality contract without adding a new API signature, metadata key, command,
or cross-layer contract.
