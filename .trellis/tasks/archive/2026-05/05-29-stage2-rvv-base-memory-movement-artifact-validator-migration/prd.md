# Stage2 RVV base memory movement artifact validator migration

## Goal

Add a production target-owned RVV route-family artifact validator for
base-memory-movement selected-body routes. The new validator lives in
`RVVTargetArtifactRouteFamilyValidation.cpp`, consumes rebuilt provider route
facts plus candidate mirror metadata, and keeps `RVVTargetSupportBundle.cpp`
as a neutral selected-route rebuild, generic bridge, metadata mirror, and
registry-dispatch layer.

## Direction Source

- Direction title: `Switch: Stage2 RVV base memory movement target-artifact
  route-family validator ownership`.
- Module owner: RVV target-owned artifact route-family validator registry for
  base-memory-movement selected-body routes.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean.
- Initial HEAD: `382b5941 rvv: migrate runtime splat-store artifact
  validation`.
- `.trellis/.current-task` was absent, so this task was created from the
  supplied Hermes direction brief before source edits.
- Serial worker constraint: no subagents, spawned agents, parallel agents, or
  multi-agent workflows.

## What I Already Know

- The immediately preceding task moved runtime-scalar splat-store artifact
  validation into the target-owned route-family validator registry and left the
  support bundle neutral.
- `RVVTargetArtifactRouteFamilyValidation.cpp` already contains migrated
  validators for compare-select-mask, conversion-dtype-policy, segment2-memory,
  standalone-reduction-accumulation, elementwise-arithmetic,
  runtime-scalar-splat-store, MAcc, and widening-dot-reduction.
- Base-memory movement route facts are already exposed by the RVV planning and
  provider path: route-family plan id, provider-supported mirror, memory form,
  source/destination memory form, stride/index/mask-related facts, runtime ABI
  order, headers, type mappings, route operand binding plan, runtime control,
  and provider-built statement layout.
- Existing target artifact fixtures already exercise representative base
  memory movement artifact paths, including strided, indexed, masked/unit
  forms.
- Current specs require target artifact acceptance to rebuild the provider
  route from the selected typed RVV body and then validate family-specific
  facts through the registry. Candidate metadata is mirror-only evidence after
  route construction, not support authority.
- `RVVTargetSupportBundle.cpp` already calls the route-family registry through
  `validateRVVTargetArtifactRouteFamilyProviderFacts` and
  `validateRVVTargetArtifactRouteFamilyCandidateMirrors`; this task must not
  add family-specific memory semantics back to the support bundle.

## Requirements

1. Register a production `base-memory-movement` route-family validator in
   `RVVTargetArtifactRouteFamilyValidation.cpp`.
2. The validator must select only true base memory movement selected-body
   descriptions:
   - `StridedLoadUnitStore`;
   - `UnitLoadStridedStore`;
   - `IndexedGatherUnitStore`;
   - `IndexedScatterUnitLoad`;
   - `MaskedUnitLoadStore`;
   - `MaskedUnitStore`.
3. The validator must validate provider-derived facts for:
   - rebuilt route id agreement with the selected provider route;
   - provider support mirror;
   - `base_memory_movement_route_family_plan`;
   - route operand binding plan and operand summary;
   - operation-specific memory form;
   - source and destination memory form;
   - stride, index, and mask bindings where applicable;
   - runtime ABI order and ABI parameter mapping;
   - required headers;
   - C type mapping;
   - runtime control / setvl / VL loop shape;
   - provider-built statement layout for load/store, strided load/store,
     indexed gather/scatter, mask load/masked move, passthrough where needed,
     and store steps;
   - selected typed RVV source provenance on pre-loop and loop statements;
   - target leaf/type/config facts already rebuilt by the provider.
4. The validator must validate candidate metadata only as mirrors of rebuilt
   provider facts, including:
   - `tcrv_rvv.provider_supported_mirror`;
   - `tcrv_rvv.base_memory_movement_route_family_plan`;
   - `tcrv_rvv.route_operand_binding_plan`;
   - `tcrv_rvv.route_operand_binding_operands`;
   - `tcrv_rvv.memory_form`;
   - `tcrv_rvv.source_memory_form`;
   - `tcrv_rvv.destination_memory_form`;
   - `tcrv_rvv.runtime_abi_order`;
   - `tcrv_rvv.required_header_declarations`;
   - `tcrv_rvv.c_type_mapping`;
   - route-specific stride/index/mask metadata mirrors when present.
5. The validator must fail closed on unsupported, ambiguous, missing, or stale
   base-memory facts:
   - wrong route-family plan;
   - stale non-base route-family mirror;
   - wrong memory form;
   - wrong source/destination memory form;
   - wrong stride/index/mask binding;
   - wrong runtime ABI order or mapping;
   - missing provider-supported mirror;
   - missing required headers or type mappings;
   - missing setvl/VL loop facts;
   - missing provider-built base-memory statement facts;
   - route-id-derived or metadata-only support claims;
   - stale artifact metadata.
6. Keep `RVVTargetSupportBundle.cpp` neutral: it may keep generic route
   rebuild, route verification, registry context/dispatch, and generic metadata
   mirror checks, but it must not grow a base-memory semantic helper or direct
   family validator call.
7. Preserve existing RVV runtime behavior, selected-body realization,
   provider route construction, common EmitC materialization, generated C
   emission, and current artifact fixtures.
8. Do not add broad route-family coverage, dashboards, helper-only refactors,
   source-front-door positive routes, descriptor/source-export authority, or
   artifact-name / route-id / exact-intrinsic / metadata-derived acceptance.

## Acceptance Criteria

- [x] `RVVTargetArtifactRouteFamilyValidation.cpp` contains a registered
      `base-memory-movement` route-family validator.
- [x] The registry dispatches representative base memory movement selected
      descriptions to that validator and leaves unrelated route families to
      their own validators or no-op registry behavior.
- [x] Provider-fact validation checks rebuilt base-memory route facts:
      provider support, route-family plan, operand binding, memory forms,
      runtime ABI order/mapping, headers, type mappings, runtime control,
      source provenance, and statement layout.
- [x] Candidate-mirror validation checks base-memory mirrors and rejects stale
      or foreign route-family metadata.
- [x] Positive target-artifact validation coverage includes representative
      strided, indexed, and masked/unit base-memory forms already present in
      `test/Target/RVV`.
- [x] Negative `TargetArtifactExportTest.cpp` coverage proves wrong
      route-family plan, stale non-base family mirrors, wrong memory forms,
      wrong stride/index/mask binding, wrong runtime ABI order, missing
      provider-supported mirror, missing headers/type mappings, metadata-only
      support, and stale artifact metadata fail closed.
- [x] `RVVTargetSupportBundle.cpp` remains neutral and contains no
      base-memory semantic helper or direct family validator call.
- [x] Focused build and tests pass, or an exact blocker is recorded.
- [x] `git diff --check` passes.
- [x] Bounded authority scan over `RVVTargetSupportBundle.cpp` and
      `RVVTargetArtifactRouteFamilyValidation.cpp` confirms no central ad hoc,
      name-derived, metadata-derived, descriptor-derived, ABI-string-derived,
      script-derived, artifact-name-derived, common-EmitC-derived,
      source-front-door-derived, route-id-derived, exact-intrinsic-derived,
      direct-route-entry-only, pre-realized-fixture-only, or legacy-i32-derived
      authority.
- [x] Trellis task status, journal, archive state, and commit state are
      truthful at the end of the round.

## Out Of Scope

- Changing RVV runtime behavior.
- Changing selected-body realization semantics.
- Changing provider route construction semantics.
- Changing generated C emission or common EmitC materialization.
- Adding new runtime/correctness/performance claims or requiring `ssh rvv`
  evidence.
- Adding new frontend lowering, Linalg flows, Scalar, IME, Offload, TensorExt,
  or unrelated route-family work.
- Broad test matrices, dashboards, reports, or helper-only cleanup.

## Technical Approach

1. Keep this Trellis task bounded to the supplied Hermes direction.
2. Inspect migrated route-family validators, especially runtime-scalar
   splat-store and segment2-memory, to reuse existing validator style.
3. Add base-memory operation classification and provider-fact validation in
   `RVVTargetArtifactRouteFamilyValidation.cpp`.
4. Add base-memory candidate-mirror validation and stale foreign family mirror
   rejection in the same target-owned registry file.
5. Register the validator in the static registry table.
6. Strengthen `TargetArtifactExportTest.cpp` with focused positive and
   negative cases using existing route-description/candidate mutation helpers.
7. Run focused target artifact tests, directly related lit tests, provider
   tests if provider facts are touched, `git diff --check`, authority scans,
   and `check-tianchenrv` if focused checks pass.
8. Finish/archive the task and create one coherent commit if acceptance is met.

## Validation Plan

1. Build focused targets:
   `cmake --build build --target TianChenRVRVVTarget tcrv-opt tcrv-translate tianchenrv-target-artifact-export-test tianchenrv-rvv-extension-plugin-test -j2`.
2. Run focused C++ tests:
   `build/bin/tianchenrv-target-artifact-export-test` and
   `build/bin/tianchenrv-rvv-extension-plugin-test`.
3. Run directly related lit tests from `build/test`, at minimum the existing
   base-memory target artifact fixtures for strided, indexed, and masked/unit
   forms.
4. Run `git diff --check`.
5. Run bounded authority scans over:
   - `lib/Target/RVV/RVVTargetSupportBundle.cpp`;
   - `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`.
6. Run `cmake --build build --target check-tianchenrv -j2` when focused
   checks pass, or record the exact blocker.

## Technical Notes

Required specs and prior context:

- `.trellis/spec/index.md`
- `.trellis/spec/extension-plugins/rvv-plugin.md`
- `.trellis/spec/core-dialect/tcrv-exec-contract.md`
- `.trellis/spec/lowering-runtime/emitc-route.md`
- `.trellis/spec/testing/mlir-testing-contract.md`
- `.trellis/tasks/archive/2026-05/05-28-05-28-stage2-rvv-runtime-scalar-splat-store-artifact-validator-migration/prd.md`
- `.trellis/tasks/archive/2026-05/05-25-05-25-stage2-rvv-base-memory-movement-production-family-ownership/prd.md`
- `.trellis/tasks/archive/2026-05/05-24-stage2-rvv-base-memory-statement-plan-ownership/prd.md`
- `.trellis/tasks/archive/2026-05/05-23-05-23-stage2-rvv-base-memory-movement-route-family-ownership/prd.md`

Primary production files:

- `include/TianChenRV/Target/RVV/RVVTargetArtifactRouteFamilyValidation.h`
- `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`
- `lib/Target/RVV/RVVTargetSupportBundle.cpp`

Focused tests:

- `test/Target/TargetArtifactExportTest.cpp`
- existing `test/Target/RVV/*base-memory*` / selected-body memory movement
  target artifact lit fixtures.

## Completion Evidence

- Added a target-owned `base-memory-movement` validator and registry entry in
  `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`.
- Left `lib/Target/RVV/RVVTargetSupportBundle.cpp` unchanged; its role remains
  neutral route rebuild, generic verification, metadata mirror handling, and
  registry dispatch.
- Added positive and negative `TargetArtifactExportTest.cpp` coverage for
  strided, indexed, and masked/unit base-memory movement artifact routes.
- Checks passed:
  - `cmake --build build --target TianChenRVRVVTarget tianchenrv-target-artifact-export-test -j2`
  - `build/bin/tianchenrv-target-artifact-export-test`
  - `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-rvv-extension-plugin-test -j2`
  - `build/bin/tianchenrv-rvv-extension-plugin-test`
  - related `lit` Target/RVV base-memory artifact tests
  - `git diff --check`
  - bounded authority scan over the RVV support bundle and artifact validator
  - `cmake --build build --target check-tianchenrv -j2`
