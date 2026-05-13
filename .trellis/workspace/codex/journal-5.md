# Journal - codex (Part 5)

> Continuation from `journal-4.md` (archived at ~2000 lines)
> Started: 2026-05-13

---



## Session 52: Vector source arithmetic-family adapter registry

**Date**: 2026-05-13
**Task**: Vector source arithmetic-family adapter registry
**Branch**: `main`

### Summary

Added the shared dynamic vector add/sub source adapter path, proved dynamic i32-vsub through direct artifacts, plan-and-export bundle, and ssh rvv counts 7/16/23.

### Main Changes

### Task

Vector source arithmetic-family adapter registry.

### Main Changes

- Added dynamic vector `i32-vsub` support through the neutral `--tcrv-lower-source-rvv-binary-to-exec` route while keeping the deprecated vector i32-vadd adapter vadd-only.
- Extended source-tail metadata validation to accept `mlir-vector-scf-runtime-i32-vsub.v1` through the same runtime `%n` / transfer-tail active-lane contract as dynamic vadd.
- Added marker/body mismatch fail-closed coverage for dynamic vector add/sub in both directions.
- Added dynamic vector i32-vsub direct lowering/source-export lit and plan-and-export bundle lit with source/header/object artifact records.

### Evidence

- Local direct artifacts: `artifacts/tmp/vector_source_arithmetic_family_adapter_registry/direct/vector_dynamic_i32_vsub/`.
- Local bundle artifacts: `artifacts/tmp/vector_source_arithmetic_family_adapter_registry/bundle/vector_dynamic_i32_vsub/`.
- ssh rvv evidence: `artifacts/tmp/vector_source_arithmetic_family_adapter_registry/e2e/20260513T-vector-dynamic-i32-vsub-adapter-registry/evidence.json`.
- Remote source-built and bundle-object runs printed `tcrv_rvv_i32_vsub_microkernel_external_abi_ok counts=7,16,23`.

### Checks

- `cmake --build build --target TianChenRVTarget TianChenRVTransforms tcrv-opt tcrv-translate tianchenrv-target-artifact-export-test -j2`
- `./build/bin/tianchenrv-target-artifact-export-test`
- Focused lit for new vector vsub, marker mismatch, existing vector vadd, vsub bundle/export, and broader `VectorToExec|TargetArtifactBundleExport|RVVMicrokernel|EmissionManifest|LinalgToExec`.
- `python3 scripts/rvv_microkernel_e2e.py --self-test`
- Trellis validate before finish and after archive.


### Git Commits

| Hash | Message |
|------|---------|
| `this commit` | (see git log) |

### Testing

- [OK] Focused build, C++ target-export test, focused lit, e2e self-test, Trellis validation, and ssh rvv evidence passed.

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 58: Template typed-role op construction path

**Date**: 2026-05-13
**Task**: Template typed-role op construction path
**Branch**: `main`

### Summary

Extended the Template construction path from manifest-string generated output
to a C++ typed role/interface realization consumed by proposal, legality,
emission-plan metadata, target artifact preflight, and generated output.

### Main Changes

- Added TemplateTypedRoleInterfaceRealization and TemplateTypedRoleGraphRealization.
- Added typed realization verification against the construction manifest,
  common-interface realization, role-specific interfaces, EmitC role-to-call
  mapping, and evidence profile.
- Template plugin now carries and validates tcrv_template.typed_role_realization
  on materialized variants.
- Template emission planning and target artifact route metadata now require
  template_typed_role_realization selected-plan metadata.
- Generated artifact output now prints typed_role_realization, typed_role
  records, and typed role fields in generated_emitc_step entries.
- Added C++ and lit coverage for typed role success and stale/missing/reordered
  fail-closed cases.
- Updated the plugin-protocol construction-template scenario for typed-role
  realization APIs and tests.
- No core tcrv.exec or lib/Transforms Template semantic branches were added.

### Checks

- `cmake --build build --target TianChenRVTemplatePlugin TianChenRVTemplateTarget tianchenrv-template-extension-plugin-test tcrv-opt tcrv-translate -j2`
- `./build/bin/tianchenrv-template-extension-plugin-test`
- Template route pipeline and export through `tcrv-opt | tcrv-translate`
- `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter='template-extension-plugin|TemplateMetadataArtifact'`
- `./build/bin/tianchenrv-target-artifact-export-test`
- `cmake --build build --target tianchenrv-rvv-extension-plugin-test -j2`
- `./build/bin/tianchenrv-rvv-extension-plugin-test`
- Core-neutrality grep for Template strings in `lib/Transforms` and core exec paths
- `git diff --check`
- Trellis validation before and after archive

### Status

[OK] Completed and archived; commit pending in this session.


## Session 56: Extension manifest target-support route activation

**Date**: 2026-05-13
**Task**: Extension manifest target-support route activation
**Branch**: `main`

### Summary

Activated RVV target-support artifact and translate routes through a generic
extension plugin manifest hook, leaving central built-in route setup as generic
aggregation rather than a direct RVV helper caller.

### Main Changes

- Added default `ExtensionPlugin` target-support hooks for extension bundle
  configuration and target translate route contribution.
- Implemented the RVV hook in `RVVExtensionPlugin`, delegating RVV direct and
  RVV+scalar dispatch route activation to the existing RVV target-support
  bundle.
- Rewired built-in target artifact setup to configure the RVV extension bundle
  through generic plugin manifest aggregation.
- Rewired built-in target translate route registration to iterate enabled
  plugins and invoke their target-support translate hook.
- Added C++ coverage for RVV plugin manifest activation and built-in translate
  aggregation.
- Updated plugin-protocol and lowering-runtime specs for the new durable hook.

### Evidence

- Direct artifacts:
  `artifacts/tmp/extension_manifest_target_support_route_activation/direct/vector_dynamic_i32_vsub/`
  and
  `artifacts/tmp/extension_manifest_target_support_route_activation/direct/vector_dynamic_i32_vadd/`.
- Bundle artifacts:
  `artifacts/tmp/extension_manifest_target_support_route_activation/bundle/vector_dynamic_i32_vsub/`
  and
  `artifacts/tmp/extension_manifest_target_support_route_activation/bundle/vector_dynamic_i32_vadd/`.
- ssh rvv evidence:
  `artifacts/tmp/extension_manifest_target_support_route_activation/e2e/20260513T-extension-manifest-target-support-route-activation-vsub/evidence.json`.
- Remote result: dynamic vector `i32-vsub` succeeded for runtime counts `7`,
  `16`, and `23`.

### Checks

- `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-13-05-13-extension-manifest-target-support-route-activation`
- `cmake --build build --target TianChenRVTransforms TianChenRVTarget tcrv-opt tcrv-translate tianchenrv-target-artifact-export-test -j2`
- `./build/bin/tianchenrv-target-artifact-export-test`
- `cmake --build build --target tianchenrv-plugin-registry-test tianchenrv-rvv-extension-plugin-test tianchenrv-i32-binary-family-registry-test -j2`
- `./build/bin/tianchenrv-plugin-registry-test`
- `./build/bin/tianchenrv-rvv-extension-plugin-test`
- `./build/bin/tianchenrv-i32-binary-family-registry-test`
- Focused lit filter covering dynamic vector add/sub, invalid vector
  diagnostics, fixed-vector vadd, bundle export, RVV microkernel, emission
  manifest, and linalg add/sub with 68 selected tests passed.
- Broader lit filter
  `VectorToExec|LinalgToExec|TargetArtifactBundleExport|RVVMicrokernel|EmissionManifest`
  with 80 selected tests passed.
- Direct and bundle artifact commands recorded in the archived task PRD.
- `git diff --check`, `git diff --cached --check`, Trellis validation before
  finish/archive and after archive.

### Status

[OK] Completed and archived.


## Session 56: RVV target-support artifact bundle extraction

**Date**: 2026-05-13
**Task**: RVV target-support artifact bundle extraction
**Branch**: `main`

### Summary

Moved RVV direct and RVV+scalar dispatch artifact route registration
composition into an RVV target-support bundle, reduced built-in target exporter
and translate-route code to delegation, and preserved dynamic vadd/vsub
direct and bundle behavior with fresh RVV evidence.

### Main Changes

- Added `RVVTargetSupportBundle` as the RVV-owned registration surface for
  direct RVV source/header/object routes plus RVV+scalar dispatch
  source/header/object routes.
- Moved `RVVScalarDispatch.cpp` from `lib/Target/Builtin/` to
  `lib/Target/RVV/` and moved the public dispatch header under
  `include/TianChenRV/Target/RVV/`, keeping a forwarding compatibility header.
- Rewired `BuiltinTargetArtifactExporters.cpp` and
  `BuiltinTargetTranslateRoutes.cpp` to call RVV target-support helpers rather
  than iterating RVV route manifests directly.
- Added C++ registry coverage for the combined RVV target-support bundle,
  scalar dependency gating, direct RVV independence from scalar, and route
  metadata preservation.
- Updated the lowering-runtime spec with the central built-in delegation rule.

### Evidence

- Direct artifacts:
  `artifacts/tmp/rvv_target_support_bundle_extraction/direct/vector_dynamic_i32_vsub/`
  and
  `artifacts/tmp/rvv_target_support_bundle_extraction/direct/vector_dynamic_i32_vadd/`.
- Plan-and-export bundle artifacts:
  `artifacts/tmp/rvv_target_support_bundle_extraction/bundle/vector_dynamic_i32_vsub/`
  and
  `artifacts/tmp/rvv_target_support_bundle_extraction/bundle/vector_dynamic_i32_vadd/`.
- ssh rvv evidence:
  `artifacts/tmp/rvv_target_support_bundle_extraction/e2e/20260513T-rvv-target-support-bundle-extraction-vsub/evidence.json`.
- Remote result: dynamic vector `i32-vsub` succeeded for runtime counts
  `7`, `16`, and `23`.

### Checks

- `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-13-rvv-target-support-artifact-bundle-extraction`
- `cmake --build build --target TianChenRVTransforms TianChenRVTarget tcrv-opt tcrv-translate tianchenrv-target-artifact-export-test -j2`
- `./build/bin/tianchenrv-target-artifact-export-test`
- `cmake --build build --target tianchenrv-i32-binary-family-registry-test -j2`
- `./build/bin/tianchenrv-i32-binary-family-registry-test`
- Focused lit filter covering dynamic vector add/sub, invalid vector
  diagnostics, fixed-vector vadd, bundle export, RVV microkernel, emission
  manifest, and linalg add/sub with 68 selected tests passed.
- Broad focused lit filter
  `VectorToExec|LinalgToExec|TargetArtifactBundleExport|RVVMicrokernel|EmissionManifest`
  with 80 selected tests passed.
- Direct and bundle artifact commands recorded in the task PRD.
- `git diff --check`
- `git diff --cached --check`

### Status

[OK] Completed and ready to archive.

## 2026-05-13 - RVV selected-config fail-closed artifact validation

### Summary

- Created and completed task
  `.trellis/tasks/05-13-rvv-selected-config-fail-closed-artifact-validation`
  from the Hermes Direction Brief.
- Tightened RVV+scalar dispatch bundle component validation to consume the full
  `RVVBinarySelectedConfigContract` for selected vector shape, runtime AVL/VL,
  source runtime extent, typed RVV source metadata, and descriptor-local count
  before bundle metadata export.
- Added direct i32-vsub negative coverage for stale SEW/LMUL/tail/mask,
  runtime VL source/scope, runtime element-count authority, and missing runtime
  AVL source.
- Added vsub dispatch composite source/header/object preflight coverage for
  stale runtime VL metadata and lit fail-closed checks for direct source and
  target artifact bundle export.

### Evidence

- Direct artifacts:
  `artifacts/tmp/rvv_selected_config_fail_closed/direct/vector_dynamic_i32_vsub/`
  and
  `artifacts/tmp/rvv_selected_config_fail_closed/direct/vector_dynamic_i32_vadd/`.
- Bundle artifacts:
  `artifacts/tmp/rvv_selected_config_fail_closed/bundle/vector_dynamic_i32_vsub/`
  and
  `artifacts/tmp/rvv_selected_config_fail_closed/bundle/vector_dynamic_i32_vadd/`.
- ssh rvv evidence:
  `artifacts/tmp/rvv_selected_config_fail_closed/e2e/20260513T-rvv-selected-config-fail-closed-vsub/evidence.json`.
- Remote result: dynamic vector `i32-vsub` succeeded for runtime counts `7`,
  `16`, and `23`.

### Checks

- `cmake --build build --target TianChenRVTransforms TianChenRVTarget tcrv-opt tcrv-translate tianchenrv-target-artifact-export-test -j2`
- `./build/bin/tianchenrv-target-artifact-export-test`
- Focused lit for the two changed dynamic vsub direct/bundle files: 2 selected
  tests passed.
- Focused lit for dynamic vector add/sub, invalid vector diagnostics, bundle
  export, RVV microkernel, emission manifest, fixed-vector vadd, and linalg
  add/sub: 52 selected tests passed.
- Broader focused lit `VectorToExec|LinalgToExec|TargetArtifactBundleExport|RVVMicrokernel|EmissionManifest`: 80 selected tests passed.
- `git diff --check`, `git diff --cached --check`, and Trellis validation.

### Status

[OK] Completed; ready to archive and commit.

---

## Session 53: RVV finite-binary family contract registry

**Date**: 2026-05-13
**Task**: RVV finite-binary family contract registry
**Branch**: `main`

### Summary

Completed the bounded RVV finite-binary family contract registry migration for
the current source-frontdoor add/sub production route. Source linalg/vector
family inference now queries the RVV registry for source arithmetic op,
frontend marker, and dynamic vector source-kind metadata instead of maintaining
an independent support-layer family lookup surface.

### Main Changes

- Added source arithmetic op and dynamic vector source-kind fields to
  `RVVBinaryFamilyDescriptor`.
- Added RVV registry helpers for frontend contract lookup, source arithmetic
  inference, dynamic vector source-family acceptance, and marker formatting.
- Migrated `LowerSourceRVVBinaryToExec.cpp` linalg/vector source inference,
  marker validation, and dynamic source-kind materialization to the RVV
  registry.
- Removed unused support-layer frontend family lookup/format helpers so
  `FiniteBinaryFrontendLowering.h` remains an ABI/source-authority carrier.

### Evidence

- Direct artifacts:
  `artifacts/tmp/rvv_finite_binary_family_contract_registry/direct/vector_dynamic_i32_vadd/`
  and
  `artifacts/tmp/rvv_finite_binary_family_contract_registry/direct/vector_dynamic_i32_vsub/`.
- Plan-and-export bundle artifacts:
  `artifacts/tmp/rvv_finite_binary_family_contract_registry/bundle/vector_dynamic_i32_vadd/`
  and
  `artifacts/tmp/rvv_finite_binary_family_contract_registry/bundle/vector_dynamic_i32_vsub/`.
- ssh rvv evidence:
  `artifacts/tmp/rvv_finite_binary_family_contract_registry/e2e/20260513T-rvv-finite-binary-family-contract-registry-vsub/evidence.json`.
- Remote result: dynamic vector `i32-vsub` succeeded for runtime counts
  `7`, `16`, and `23`.

### Checks

- `cmake --build build --target TianChenRVTransforms TianChenRVTarget tcrv-opt tcrv-translate tianchenrv-target-artifact-export-test -j2`
- `./build/bin/tianchenrv-target-artifact-export-test`
- Focused lit for dynamic vector vadd/vsub, vector invalid diagnostics,
  fixed-vector vadd, linalg add/sub, bundle export, RVV microkernel, and
  emission manifest surfaces.
- Broad focused lit filter:
  `VectorToExec|LinalgToExec|TargetArtifactBundleExport|RVVMicrokernel|EmissionManifest`
  with 80 selected tests passed.
- Direct and bundle artifact commands recorded in the task PRD.
- `git diff --check`
- Trellis validation before finish/archive.

### Status

[OK] **Completed**


## Session 54: RVV selected config and runtime VL boundary

**Date**: 2026-05-13
**Task**: RVV selected config and runtime VL boundary
**Branch**: `main`

### Summary

Exposed dispatch bundle runtime VL boundary metadata from the selected RVV config contract, tightened RVV microkernel materialization policy derivation, and proved dynamic vsub on ssh rvv.

### Main Changes

### Task

RVV selected config and runtime VL boundary.

### Main Changes

- Added `tcrv_rvv.dispatch_contract_runtime_vl_boundary` to RVV+scalar dispatch bundle metadata, derived from `RVVBinarySelectedConfigContract`.
- Tightened RVV binary microkernel materialization so `setvl` / `with_vl` policy metadata is derived from the selected-config contract.
- Strengthened dynamic vector vsub direct checks and dynamic vadd/vsub bundle-index checks for selected config and runtime VL authority.

### Evidence

- Direct artifacts: `artifacts/tmp/rvv_selected_config_runtime_vl_boundary/direct/vector_dynamic_i32_vadd/` and `artifacts/tmp/rvv_selected_config_runtime_vl_boundary/direct/vector_dynamic_i32_vsub/`.
- Bundle artifacts: `artifacts/tmp/rvv_selected_config_runtime_vl_boundary/bundle/vector_dynamic_i32_vadd/` and `artifacts/tmp/rvv_selected_config_runtime_vl_boundary/bundle/vector_dynamic_i32_vsub/`.
- ssh rvv evidence: `artifacts/tmp/rvv_selected_config_runtime_vl_boundary/e2e/20260513T-rvv-selected-config-runtime-vl-boundary-vsub/evidence.json`.
- Remote result: dynamic vector `i32-vsub` succeeded for runtime counts `7`, `16`, and `23`.

### Checks

- `cmake --build build --target TianChenRVTransforms TianChenRVTarget tcrv-opt tcrv-translate tianchenrv-target-artifact-export-test -j2`
- `./build/bin/tianchenrv-target-artifact-export-test`
- Focused lit filter covering dynamic vector add/sub, invalid vector diagnostics, bundle export, RVV microkernel, emission manifest, fixed-vector vadd, and linalg add/sub.
- Broader lit filter `VectorToExec|LinalgToExec|TargetArtifactBundleExport|RVVMicrokernel|EmissionManifest` with 80 selected tests passed.
- `git diff --check`, `git diff --cached --check`, Trellis validation before finish/archive and after archive.

### Status

[OK] Completed and archived.


### Git Commits

| Hash | Message |
|------|---------|
| `this commit` | (see git log) |

### Testing

- [OK] (Add test results)

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 55: RVV plugin-owned artifact route registration

**Date**: 2026-05-13
**Task**: RVV plugin-owned artifact route registration
**Branch**: `main`

### Summary

Moved RVV-primary dispatch artifact route registration under the RVV plugin-owned target exporter bundle with scalar as an explicit dependency, preserved direct and bundle dynamic vadd/vsub behavior, and refreshed bounded RVV evidence.

### Main Changes

### Task

RVV plugin-owned artifact route registration.

### Main Changes

- Added RVV built-in target artifact bundle composition so `rvv-plugin` registers both RVV direct microkernel routes and RVV-primary RVV+scalar dispatch source/header/object routes.
- Changed RVV+scalar dispatch target exporter bundle registration from `scalar-plugin` owner with `rvv-plugin` dependency to `rvv-plugin` owner with `scalar-plugin` dependency.
- Kept scalar standalone fallback target artifact routes under `scalar-plugin` only.
- Moved RVV+scalar dispatch route metadata requirements from the scalar extension bundle to the RVV extension bundle.
- Updated `TargetArtifactExportTest.cpp` coverage for owner/dependency failures, duplicate bundle rejection, extension-bundle metadata ownership, and route metadata preservation.
- Updated the lowering-runtime spec to document RVV-owned dispatch route registration with scalar dependency.

### Evidence

- Direct artifacts: `artifacts/tmp/rvv_plugin_owned_route_registration/direct/vector_dynamic_i32_vsub/` and `artifacts/tmp/rvv_plugin_owned_route_registration/direct/vector_dynamic_i32_vadd/`.
- Bundle artifacts: `artifacts/tmp/rvv_plugin_owned_route_registration/bundle/vector_dynamic_i32_vsub/` and `artifacts/tmp/rvv_plugin_owned_route_registration/bundle/vector_dynamic_i32_vadd/`.
- ssh rvv evidence: `artifacts/tmp/rvv_plugin_owned_route_registration/e2e/20260513T-rvv-plugin-owned-route-registration-vsub/evidence.json`.
- Remote result: dynamic vector `i32-vsub` succeeded for runtime counts `7`, `16`, and `23`.

### Checks

- `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-13-rvv-plugin-owned-artifact-route-registration`
- `cmake --build build --target TianChenRVTransforms TianChenRVTarget tcrv-opt tcrv-translate tianchenrv-target-artifact-export-test -j2`
- `./build/bin/tianchenrv-target-artifact-export-test`
- Focused lit filter covering dynamic vector add/sub, invalid vector diagnostics, fixed-vector vadd, bundle export, RVV microkernel, emission manifest, and linalg add/sub.
- Broader lit filter `VectorToExec|LinalgToExec|TargetArtifactBundleExport|RVVMicrokernel|EmissionManifest` with 80 selected tests passed.
- Direct and bundle artifact commands recorded in the archived task PRD.
- `git diff --check`
- Trellis validation before finish/archive.

### Status

[OK] Completed and archived.


### Git Commits

| Hash | Message |
|------|---------|
| `this commit` | (see git log) |

### Testing

- [OK] (Add test results)

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 57: Executable extension-family construction template

**Date**: 2026-05-13
**Task**: Executable extension-family construction template
**Branch**: `main`

### Summary

Converted the Template extension path into a C++-consumable construction
manifest for the Extension-Family Plugin Construction Protocol and validated
that plugin proposal, legality, emission planning, target artifact validation,
and generated artifact output consume the same manifest.

### Main Changes

- Added `TemplateConstructionProtocol` with protocol version, archetype,
  semantic role graph, family declaration, common-interface realization, EmitC
  route mapping, and evidence profile.
- Attached construction metadata to Template variant proposals and verified it
  during Template variant legality.
- Emitted construction selected-plan metadata from Template emission planning.
- Made the Template target artifact exporter validate and print construction
  protocol fields from the C++ manifest.
- Added a plugin-protocol code-spec scenario for the executable construction
  template manifest contract.
- Removed the stale clarification task and replaced it with the real Trellis
  task for this direction.

### Checks

- `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-13-executable-extension-family-construction-template`
- `cmake --build build --target TianChenRVTemplatePlugin TianChenRVTemplateTarget tianchenrv-template-extension-plugin-test tcrv-opt tcrv-translate -j2`
- `./build/bin/tianchenrv-template-extension-plugin-test`
- Template pipeline/export direct commands with `tcrv-opt` and `tcrv-translate`
- `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter='template-extension-plugin|TemplateMetadataArtifact'` from `build/test`
- `cmake --build build --target tianchenrv-target-artifact-export-test -j2`
- `./build/bin/tianchenrv-target-artifact-export-test`
- `cmake --build build --target tianchenrv-rvv-extension-plugin-test -j2`
- `./build/bin/tianchenrv-rvv-extension-plugin-test`
- `git diff --check`
- `git diff --cached --check`

### Status

[OK] Implementation checks passed; finish/archive and commit pending.


## Session 56: RVV op-owned EmitC artifact production route

**Date**: 2026-05-13
**Task**: RVV op-owned EmitC artifact production route
**Branch**: `main`

### Summary

Rewired RVV finite-binary artifact emission so arithmetic intrinsic callee selection is derived from typed tcrv_rvv compute source-op provenance and selected-family agreement; refreshed direct/bundle artifacts and ssh rvv dynamic i32-vsub evidence.

### Main Changes

### Task

RVV op-owned EmitC artifact production route.

### Main Changes

- Added RVV finite-family lookup by materialized RVV arithmetic op name.
- Rewired RVV arithmetic `emitc.call_opaque` callee selection so intrinsic spelling comes from the typed compute source op plus `TCRVEmitCLowerableOpInterface` provenance.
- Added selected-family/source-family agreement checks before source/header/object emission.
- Preserved descriptor mirrors as compatibility diagnostics only; they no longer independently select arithmetic intrinsic spelling for the production EmitC route.
- Refreshed direct and bundle dynamic i32-vsub/i32-vadd artifacts under `artifacts/tmp/rvv_op_owned_emitc_artifact_route_20260513T070822Z`.

### Evidence

- Direct artifacts: `artifacts/tmp/rvv_op_owned_emitc_artifact_route_20260513T070822Z/direct/vector_dynamic_i32_vsub/` and `artifacts/tmp/rvv_op_owned_emitc_artifact_route_20260513T070822Z/direct/vector_dynamic_i32_vadd/`.
- Bundle artifacts: `artifacts/tmp/rvv_op_owned_emitc_artifact_route_20260513T070822Z/bundle/vector_dynamic_i32_vsub/` and `artifacts/tmp/rvv_op_owned_emitc_artifact_route_20260513T070822Z/bundle/vector_dynamic_i32_vadd/`.
- ssh rvv evidence: `artifacts/tmp/rvv_op_owned_emitc_artifact_route_20260513T070822Z/e2e/20260513T070822Z-rvv-op-owned-emitc-artifact-route-vsub/`.
- Remote result: dynamic vector `i32-vsub` succeeded for runtime counts `7`, `16`, and `23`.

### Checks

- `cmake --build build --target TianChenRVTransforms TianChenRVTarget tcrv-opt tcrv-translate tianchenrv-target-artifact-export-test -j2`
- `./build/bin/tianchenrv-target-artifact-export-test`
- `./build/bin/tianchenrv-i32-binary-family-registry-test`
- `./build/bin/tianchenrv-rvv-binary-planning-test`
- `./build/bin/tianchenrv-rvv-extension-plugin-test`
- `./build/bin/tianchenrv-rvv-selected-lowering-boundary-test`
- `./build/bin/tianchenrv-emitc-lowerable-interface-test`
- Focused lit covering dynamic vsub/vadd, RVV microkernel, emission manifest, target artifact bundle export, fixed-vector vadd, dynamic-tail/runtime-VL authority, plugin route activation, family registry, and linalg compatibility.
- `git diff --check`
- Trellis validation before finish/archive and after archive.

### Git Commits

| Hash | Message |
|------|---------|
| `this commit` | (see git log) |

### Testing

- [OK] Focused build, C++ checks, lit, direct/bundle artifact generation, and `ssh rvv` dynamic i32-vsub evidence passed.

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 57: Template role-graph EmitC route realization

**Date**: 2026-05-13
**Task**: Template role-graph EmitC route realization
**Branch**: `main`

### Summary

Realized Template construction manifest as a generated role-graph-to-EmitC source-like artifact route.

### Main Changes

- Added TemplateGeneratedOutputRoute / TemplateGeneratedOutputStep and buildTemplateGeneratedOutputRoute.
- Template target artifact now validates and prints generated_emitc_step records plus generated_source derived from role-to-EmitC mapping.
- Focused C++ and lit coverage added for positive route output and stale/missing/reordered/mismatched failure cases.
- No core tcrv.exec or lib/Transforms Template semantic branches were added.


### Git Commits

| Hash | Message |
|------|---------|
| `this commit` | (see git log) |

### Testing

- [OK] (Add test results)

### Status

[OK] **Completed**

### Next Steps

- None - task complete
