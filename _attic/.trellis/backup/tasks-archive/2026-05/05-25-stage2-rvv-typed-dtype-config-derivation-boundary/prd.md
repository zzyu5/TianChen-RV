# Stage2 RVV typed dtype/config derivation boundary

## Goal

Make RVV dtype/config derivation a first-class RVV plugin-owned boundary for
existing route-supported selected-body families. Route analysis must derive
element type, SEW, LMUL, policy, vector C type, header, intrinsic spelling, and
artifact mirrors from realized typed `tcrv_rvv` body/config/capability/runtime
facts, then fail closed for unsupported or inconsistent combinations before
route or artifact authority can be claimed.

This is production RVV compiler ownership work after the previous ssh execution
closure. It is not new operation-family coverage and not another evidence-only
round.

## Direction Source

- Direction title: `Stage2 RVV typed dtype/config derivation boundary`.
- Module owner: RVV plugin-owned dtype, SEW, LMUL, and policy derivation from
  realized typed `tcrv_rvv` bodies into route facts, statement plans, provider
  route construction, and emitted artifact metadata for existing
  route-supported families.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean before task creation.
- Initial HEAD: `bc2658c9 rvv: record pre-realized route-entry ssh execution closure`.
- `.trellis/.current-task` was absent, so this task was created from the Hermes
  Direction Brief before source edits.

## What I Already Know

- `.trellis/spec/index.md` requires the RVV-first authority chain:
  selected `tcrv.exec` RVV variant -> typed low-level `tcrv_rvv` body -> RVV
  plugin-owned legality / selected-body realization / route provider ->
  `TCRVEmitCLowerableRoute` -> common EmitC -> target artifact.
- `.trellis/spec/extension-plugins/rvv-plugin.md` requires dtype/config/
  operation facts to be structural in typed values, config, and body structure;
  route ids, artifact names, ABI strings, test names, and intrinsic spellings
  are not authority.
- `.trellis/spec/lowering-runtime/emitc-route.md` requires common EmitC to
  consume provider-built routes without inferring RVV dtype, SEW, LMUL,
  mask/tail policy, operation kind, schedules, or intrinsic names.
- The archived task
  `.trellis/tasks/archive/2026-05/05-25-stage2-rvv-pre-realized-selected-body-executable-closure/prd.md`
  closed the previous bounded direct pre-realized route-entry executable path
  through `ssh rvv`; it made no compiler source changes and should not be used
  as a reason to continue evidence-only work.
- Current code already has a corrected typed RVV surface and route analysis
  path:
  - generic `!tcrv_rvv.vector<elem, lmul>`, `!tcrv_rvv.mask<elem, lmul>`, and
    `!tcrv_rvv.index_vector<elem, lmul>` types in the RVV dialect;
  - realized selected-body route analysis collects `setvl` / `with_vl`, then
    calls `validateRVVSelectedBodyTypedConfigFacts`;
  - `deriveRVVSelectedBodyConfigProfile` maps SEW/LMUL/policy to vector C
    types, headers, and intrinsic leaves;
  - `getRVVSelectedBodyRouteMaterializationFacts` is the existing provider
    boundary consumed by `RVVEmitCRouteProvider`;
  - artifact metadata currently mirrors `tcrv_rvv.sew`, `tcrv_rvv.lmul`,
    tail/mask policy, and route-family facts after route construction.

## Requirements

1. Add one RVV-owned typed config facts boundary for realized selected-body
   route analysis.
2. The boundary must derive, at minimum:
   - element type spelling / element bit width;
   - SEW;
   - LMUL;
   - tail and mask policy;
   - selected config contract id;
   - vector, mask, and optional index vector type names;
   - vector, mask, and optional index vector C types;
   - provider-owned setvl/load/store/compute intrinsic leaves already valid for
     the supported family.
3. Existing i32 selected-body cases must continue only as ordinary instances of
   the generic typed surface. No old finite `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`,
   `RVVI32M1*`, `rvv-i32m1-*`, source-front-door, descriptor, or artifact-name
   authority may be introduced or revived.
4. Route materialization facts and statement-plan inputs must consume the new
   typed config facts boundary instead of relying on route ids, operation names,
   artifact metadata, helper names, ABI names, or common EmitC code to choose
   dtype/config.
5. Unsupported or inconsistent dtype/config combinations must fail closed with
   targeted diagnostics before provider route construction or target artifact
   export.
6. Artifact metadata may mirror dtype/config facts only after provider route
   construction. Mirror names should remain explicit and must not become
   acceptance state.
7. Keep common EmitC and target export neutral: they may consume provider facts
   and mirrors, but must not infer RVV dtype/config/intrinsics.

## Acceptance Criteria

- [x] A named RVV plugin-owned typed config facts boundary exists in planning
      code and is exposed through the route materialization facts consumed by
      the provider.
- [x] Focused C++ plugin/provider tests cover at least one existing
      arithmetic or compare/select route and one existing memory route, proving
      typed config facts are derived from body/config facts and preserved into
      route materialization facts.
- [x] Focused fail-closed coverage rejects at least one unsupported or stale
      dtype/config/policy combination before route/provider facts are accepted.
- [x] Representative lit/FileCheck target coverage for affected selected-body
      artifacts shows dtype/config mirrors and emitted intrinsic/type/header
      choices still come from provider-built routes.
- [x] Existing i32 behavior remains an ordinary typed generic route instance,
      not a legacy i32 authority path.
- [x] Bounded active-authority scans over touched RVV dialect, realization,
      planning, provider, target, and test files show no new legacy i32,
      source-front-door, descriptor/direct-C/source-export, or mirror-only
      authority drift.
- [x] `git diff --check` passes.
- [x] A focused RVV plugin/test target passes, and `check-tianchenrv` passes or
      an exact blocker is documented.
- [x] Trellis task status and workspace journal record the implemented
      boundary, checks, remaining risk, and no-new-runtime-claim status.

## Out Of Scope

- No broad new operation-family coverage.
- No reductions, contractions, high-level frontend lowering, dtype/LMUL clone
  batches, one-op-per-intrinsic wrappers, or source-front-door positive routes.
- No descriptor/direct-C/source-export path.
- No dashboards, broad smoke matrices, or runtime/performance claims.
- No migration to Scalar, IME, Offload, TensorExt, Template/Toy, or future
  plugin work.
- No Python implementation of compiler core, dialect, legality, route planning,
  provider, lowering, or emission behavior.

## Technical Approach

1. Start the task and validate Trellis context.
2. Add a small RVV planning struct/API for typed selected-body config facts.
3. Populate it from the already-collected realized `setvl` / `with_vl` and
   generic typed `tcrv_rvv` values, reusing existing config-profile derivation
   for type/C-type/intrinsic mapping.
4. Thread the facts into `RVVSelectedBodyRouteAnalysis` and
   `RVVSelectedBodyRouteMaterializationFacts`.
5. Add targeted validation so stale description/family-plan/provider facts
   disagreeing with typed config facts fail closed.
6. Add focused C++ and lit/FileCheck checks for an arithmetic or compare/select
   route and a memory route.
7. Run bounded scans and focused build/test checks.

## Validation Plan

1. `rtk python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-25-stage2-rvv-typed-dtype-config-derivation-boundary`
2. Focused C++ plugin test target or focused lit filter covering
   `RVVExtensionPluginTest` materialization/config facts.
3. Focused lit/FileCheck target tests for the affected selected-body artifacts.
4. Bounded active-authority scans over touched files for:
   - `RVVI32M1`;
   - `rvv-i32m1`;
   - positive `tcrv_rvv.i32_`;
   - `!tcrv_rvv.i32m`;
   - descriptor/direct-C/source-export/source-front-door authority;
   - metadata/status/result as route authority.
5. `rtk git diff --check`
6. `rtk cmake --build build --target check-tianchenrv -j2`, unless a focused
   blocker is found and documented exactly.

## Implementation Result

- Added `RVVSelectedBodyTypedConfigFacts` as a named RVV planning/materialization
  boundary carrying element type, element bit width, SEW, LMUL, tail/mask
  policy, config contract, vector/mask/index type names, C types, VL C type,
  and provider-derived setvl/load/store leaves.
- Route analysis now derives those facts after the realized `tcrv_rvv.setvl` /
  `with_vl` config is collected and after the provider config profile has been
  derived. Provider materialization verifies the facts mirror the
  provider-derived route description before statement-plan pointers are exposed.
- Route description verification now requires `elementTypeName` to match the
  selected SEW/config profile, and target artifact metadata mirrors
  `tcrv_rvv.element_type` only after route/provider description construction.
- Focused plugin coverage proves ordinary elementwise add and
  `strided_load_unit_store` memory routes preserve typed config facts into
  materialization facts. A stale `vectorCType` mutation fails closed with a
  targeted diagnostic mentioning `typed config facts vector C type` and the
  provider-derived expected value.
- Representative target FileCheck coverage now checks `tcrv_rvv.element_type`
  in emission-plan metadata and `tianchenrv.rvv.element_type` in generated
  headers for compare/select, memory, and i64 arithmetic artifacts.
- Static RVV dialect metadata tests were updated from 19 to 20 metadata entries
  to include the new `element_type` mirror.

## Validation Result

- `rtk python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-25-stage2-rvv-typed-dtype-config-derivation-boundary`
- `rtk cmake --build build --target tcrv-opt tcrv-translate -j2`
- `rtk /usr/bin/python3.10 /usr/lib/llvm-20/build/utils/lit/lit.py -sv /home/kingdom/phdworks/TianchenRV/build/test --filter 'pre-realized-selected-body-artifact-(cmp-select|strided-load-unit-store|i64-add)'` from `build/test`
- `rtk cmake --build build --target tianchenrv-rvv-extension-plugin-test -j2`
- `rtk build/bin/tianchenrv-rvv-extension-plugin-test`
- `rtk cmake --build build --target tianchenrv-rvv-dialect-test -j2`
- `rtk /usr/bin/python3.10 /usr/lib/llvm-20/build/utils/lit/lit.py -sv /home/kingdom/phdworks/TianchenRV/build/test --filter 'rvv-dialect'` from `build/test`
- `rtk git diff --check`
- Bounded active-authority scan over touched RVV planning/provider/dialect/test
  files found only existing negative fail-closed legacy fixtures and stale-route
  rejection strings; no positive legacy i32, descriptor, source-front-door,
  direct-C/source-export, or metadata-only authority was introduced.
- `rtk cmake --build build --target check-tianchenrv -j2`

No new runtime, correctness, or performance claim was made, so no new `ssh rvv`
evidence was required for this round.
