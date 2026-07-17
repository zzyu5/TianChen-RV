# Stage2 RVV target artifact route-family validator coverage gate

## Goal

Close the RVV target artifact route-family validator registry coverage gap.
Executable selected typed RVV target artifact acceptance must depend on exactly
one target-owned route-family validator. A selected RVV artifact route that
matches no registered family validator, or matches more than one validator,
must fail closed before generic metadata, artifact names, route ids, manifests,
or stale mirrors can act as authority.

## Direction Source

- Direction title: `Switch: Stage2 RVV target artifact route-family validator
  coverage gate`.
- Module owner: RVV target-owned route-family artifact validator registry
  fail-closed coverage for selected typed RVV artifact routes.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean.
- Initial HEAD: `2bcde071 rvv: migrate base memory artifact validation`.
- `.trellis/.current-task` was absent, so this task was created from the
  supplied Hermes direction brief before source edits.
- Serial worker constraint from the brief: no subagents, spawned agents,
  parallel agents, or multi-agent workflows.

## What I Already Know

- The archived base-memory task states the registry already has migrated
  validators for widening-dot, MAcc, standalone reduction/accumulation,
  elementwise arithmetic, runtime-scalar splat-store, conversion dtype-policy,
  compare/select mask, segment2 memory, and base-memory movement.
- `RVVTargetArtifactRouteFamilyValidation.cpp` already owns the registered
  family validators and family-specific provider-fact / candidate-mirror
  checks.
- `RVVTargetSupportBundle.cpp` already remains a neutral bridge for selected
  route rebuild, generic verification, metadata mirror mechanics, and registry
  dispatch.
- Current code rejects ambiguous registry matches in
  `selectRVVTargetArtifactRouteFamilyValidator`, but both public validation
  entrypoints return success when no validator matches.
- Specs require selected RVV artifact export to rebuild provider route facts
  from the selected typed `tcrv_rvv` body and treat candidate metadata only as
  mirrors after provider route construction.
- Focused target artifact validation exposed an existing positive selected-body
  artifact route with `operation = add` and `memory_form = rhs-broadcast-load`
  that previously reached success only because the registry treated a missing
  validator as no-op success. That route must be owned by the elementwise
  family and must carry provider-derived elementwise family facts before target
  artifact acceptance.

## Requirements

1. Make `validateRVVTargetArtifactRouteFamilyProviderFacts` fail closed when
   the selected typed RVV route description matches no registered target-owned
   route-family validator.
2. Make `validateRVVTargetArtifactRouteFamilyCandidateMirrors` fail closed for
   the same unowned-route condition.
3. Preserve the existing ambiguous-match rejection: a selected route matching
   more than one registered validator must keep failing closed with a targeted
   diagnostic.
4. Prove every currently route-supported selected-body operation family that
   reaches target artifact validation is covered by exactly one registered
   validator.
5. Preserve family-specific checks inside
   `RVVTargetArtifactRouteFamilyValidation.cpp`; do not move RVV semantic
   checks into `RVVTargetSupportBundle.cpp` or common EmitC/export.
6. Preserve candidate metadata as mirror-only evidence after provider route
   reconstruction. Do not make route ids, artifact names, metadata support
   fields, manifests, descriptor residue, ABI strings, source-front-door
   markers, or exact intrinsic spelling authoritative.

## Acceptance Criteria

- [x] Production provider-fact validation rejects selected typed RVV route
      descriptions that no registered family validator owns.
- [x] Production candidate-mirror validation rejects selected typed RVV route
      descriptions that no registered family validator owns.
- [x] Ambiguous selected-route ownership continues to fail closed and remains
      tested.
- [x] Focused C++ target artifact tests prove:
      - every currently registered/route-supported selected-body family is
        accepted by exactly one validator;
      - an unowned selected RVV route description fails provider-fact
        validation;
      - an unowned selected RVV route description fails candidate-mirror
        validation;
      - an intentionally ambiguous selected RVV route description fails the
        registry before family-specific validation can succeed.
- [x] `RVVTargetSupportBundle.cpp` remains neutral and does not gain
      family-specific route-family semantic branches.
- [x] Focused build for `TianChenRVRVVTarget` and
      `tianchenrv-target-artifact-export-test` passes.
- [x] `build/bin/tianchenrv-target-artifact-export-test` passes.
- [x] `git diff --check` passes.
- [x] Bounded scan/test evidence shows no unowned selected RVV artifact route
      can return success from route-family validation.
- [x] `check-tianchenrv` passes if focused checks pass, or an exact blocker is
      recorded.
- [x] Trellis task status, journal/archive state, and commit state are truthful
      at the end of the round.

## Out Of Scope

- Adding new RVV operation coverage.
- Adding dtype or LMUL clone batches.
- Frontend/Linalg work or source-front-door positive routes.
- Runtime behavior changes or new `ssh rvv` runtime/correctness/performance
  claims.
- Dashboard/report generation, broad smoke matrices, or helper-only file
  splitting.
- Moving target artifact route-family authority into common EmitC/export,
  descriptor/source export, manifests, artifact names, route ids, or metadata
  support fields.

## Technical Approach

1. Keep the task bounded to the target-owned RVV artifact validator registry.
2. Update the registry selection/entrypoint behavior so "no selected
   validator" is an error, not a no-op success.
3. Add small focused test-only route-description mutations in
   `TargetArtifactExportTest.cpp` to exercise unowned and ambiguous registry
   outcomes without adding production route coverage.
4. Add a coverage loop over representative current selected-body families that
   rebuilds provider route descriptions from existing fixture helpers and
   proves each reaches successful provider-fact and candidate-mirror
   validation.
5. Keep RHS broadcast-load add as an elementwise-family selected typed route by
   deriving elementwise provider facts from the typed body instead of relying on
   metadata, route ids, or artifact names.
6. Run focused target artifact checks first, then full `check-tianchenrv` if
   the focused checks pass.

## Validation Plan

1. `cmake --build build --target TianChenRVRVVTarget tianchenrv-target-artifact-export-test -j2`
2. `build/bin/tianchenrv-target-artifact-export-test`
3. Focused `Target/RVV` lit only if artifact fixtures are affected.
4. `git diff --check`
5. Bounded scan over:
   - `lib/Target/RVV/RVVTargetSupportBundle.cpp`
   - `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`
   - `test/Target/TargetArtifactExportTest.cpp`
6. `cmake --build build --target check-tianchenrv -j2`

## Technical Notes

Required specs and prior context:

- `.trellis/spec/index.md`
- `.trellis/spec/extension-plugins/rvv-plugin.md`
- `.trellis/spec/lowering-runtime/emitc-route.md`
- `.trellis/spec/plugin-protocol/interfaces-and-registry.md`
- `.trellis/spec/testing/mlir-testing-contract.md`
- `.trellis/spec/variant-pipeline/generation-selection-tuning.md`
- `.trellis/tasks/archive/2026-05/05-29-stage2-rvv-base-memory-movement-artifact-validator-migration/prd.md`
- `.trellis/tasks/archive/2026-05/05-28-stage2-rvv-artifact-route-family-validator-boundary/prd.md`

Primary production files:

- `include/TianChenRV/Target/RVV/RVVTargetArtifactRouteFamilyValidation.h`
- `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
- `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`
- `lib/Target/RVV/RVVTargetSupportBundle.cpp`

Focused tests:

- `test/Target/TargetArtifactExportTest.cpp`

## Completion Notes

- Provider-fact and candidate-mirror validation now fail closed when no target
  artifact route-family validator owns a selected typed RVV route description.
- Registry coverage now includes target-owned `vector-reduction` and
  `widening-macc-contraction` validators for the route-supported
  `reduce_add` and `widening_macc_add` selected typed bodies.
- Runtime-scalar computed-mask store/load-store artifact routes are owned by
  the compare/select mask family instead of bypassing family validation.
- RHS broadcast-load `add` is treated as an ordinary elementwise arithmetic
  route-family consumer with provider-derived elementwise facts.
- Focused self-repair:
  - removed an over-strict `runtimeControlPlanID` requirement from the
    `reduce_add` target validator because current provider facts expose
    runtime AVL/VL through setvl/runtime ABI facts and mirrors, not that field;
  - updated the plugin negative unit test so `Add + RHSBroadcastLoad` is no
    longer expected to be a non-consumer after this route became covered.
- Checks passed:
  - `cmake --build build --target TianChenRVRVVTarget tianchenrv-target-artifact-export-test -j2`
  - `build/bin/tianchenrv-target-artifact-export-test`
  - `cmake --build build --target TianChenRVRVVTarget tianchenrv-target-artifact-export-test tianchenrv-rvv-extension-plugin-test -j2`
  - `build/bin/tianchenrv-target-artifact-export-test && build/bin/tianchenrv-rvv-extension-plugin-test`
  - `git diff --check`
  - bounded scan for registry no-validator errors and new family ownership
  - `cmake --build build --target check-tianchenrv -j2` (`456/456` passed)
