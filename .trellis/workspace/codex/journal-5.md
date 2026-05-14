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

[OK] Implementation checks and Trellis archive passed; commit pending.


## Session 60: RVV op-owned object artifact evidence closure

**Date**: 2026-05-13
**Task**: RVV op-owned object artifact evidence closure
**Branch**: `main`

### Summary

Closed the bounded RVV i32-vadd source/header/object route by embedding
op-owned selected source/config/runtime ABI provenance into the generated
RISC-V relocatable object and proving the source-built and generated-object
external ABI path on `ssh rvv`.

### Main Changes

- Added an object-only `.rodata.tianchenrv.rvv_artifact` section to the direct
  RVV microkernel object export path.
- The object evidence section is derived from the same `RVVMicrokernelRecord`
  as source/header export: selected source kind, dtype/family/operator,
  microkernel op, EmitC source op/interface, selected vector shape,
  runtime AVL/VL authority, runtime ABI kind/name/glue role, and ordered ABI
  parameters.
- Preserved default source/header as library-style runtime-callable artifacts
  without hidden `main` or self-check harnesses.
- Extended `scripts/rvv_microkernel_e2e.py` so the runner validates the object
  evidence payload before remote external-ABI evidence.
- Updated `rvv-microkernel-object.mlir` to check direct and generic object
  evidence section payloads and to fail closed when selected-boundary source
  identity is stripped before object export.
- Updated the lowering-runtime spec to document the bounded object evidence
  section as compiler-artifact provenance only.
- Confirmed no core `tcrv.exec` or `lib/Transforms` RVV semantic branches were
  added.

### Evidence

- Artifact directory:
  `artifacts/tmp/rvv_op_owned_object_artifact_evidence_closure/20260513T-rvv-op-owned-object-artifact-evidence-closure-i32-vadd/`
- Evidence JSON:
  `artifacts/tmp/rvv_op_owned_object_artifact_evidence_closure/20260513T-rvv-op-owned-object-artifact-evidence-closure-i32-vadd/evidence.json`
- `ssh rvv` source-built external caller:
  `tcrv_rvv_microkernel_external_abi_ok counts=7,16`
- `ssh rvv` generated-object external caller:
  `tcrv_rvv_microkernel_external_abi_ok counts=7,16`
- Object section:
  `.rodata.tianchenrv.rvv_artifact`,
  schema `rvv-op-owned-object-artifact.v1`, 32 validated fields.

### Checks

- `cmake --build build --target TianChenRVRVVTarget tcrv-translate -j2`
- `python3 -m py_compile scripts/rvv_microkernel_e2e.py`
- `python3 scripts/rvv_microkernel_e2e.py --self-test`
- `cmake --build build --target TianChenRVRVVDialect TianChenRVRVVPlugin TianChenRVRVVTarget TianChenRVScalarTarget tianchenrv-rvv-extension-plugin-test tianchenrv-rvv-selected-lowering-boundary-test tianchenrv-rvv-binary-planning-test tianchenrv-rvv-binary-variant-legality-test tianchenrv-target-artifact-export-test tcrv-opt tcrv-translate -j2`
- `./build/bin/tianchenrv-rvv-extension-plugin-test`
- `./build/bin/tianchenrv-rvv-selected-lowering-boundary-test`
- `./build/bin/tianchenrv-rvv-binary-planning-test`
- `./build/bin/tianchenrv-rvv-binary-variant-legality-test`
- `./build/bin/tianchenrv-target-artifact-export-test`
- Focused 9-test lit set for RVV microkernel object/source/header/runtime ABI,
  stale boundary/config, missing policy, descriptor-only quarantine, and
  RVV+scalar ABI role binding.
- Broader lit filter:
  `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter='rvv-microkernel|RVVScalarDispatch|RVVSmokeProbe|LoweringBoundary|rvv-'`
  from `build/test`: 104 selected tests passed.
- Focused `ssh rvv` e2e command:
  `python3 scripts/rvv_microkernel_e2e.py --artifact-root artifacts/tmp/rvv_op_owned_object_artifact_evidence_closure --run-id 20260513T-rvv-op-owned-object-artifact-evidence-closure-i32-vadd --overwrite --expect-selected-kernel rvv_microkernel_manifest`
- `git diff --check`
- Core neutrality scan:
  `git diff -- lib/Transforms include/TianChenRV/Dialect/Exec lib/Dialect/Exec`

### Status

[OK] Implementation checks and `ssh rvv` evidence passed; finish/archive and
commit pending.


## Session 59: RVV selected-boundary extension-op production route

**Date**: 2026-05-13
**Task**: RVV selected-boundary extension-op production route
**Branch**: `main`

### Summary

Completed the bounded RVV selected-boundary production-route migration for the
existing i32 binary microkernel path. The selected boundary now carries typed
RVV source identity into target artifact preflight, and target export validates
that identity against the actual `tcrv_rvv` typed microkernel body before
emitting source/header/object artifacts.

### Main Changes

- Added verifier-backed typed binary source identity attrs to
  `tcrv_rvv.lowering_boundary`: selected source kind, dtype, family, operator,
  executable microkernel op, EmitC source op, and generated EmitC lowerable
  interface.
- Updated RVV selected-boundary materialization to copy plugin-selected binary
  source identity from the selected variant/plan into the materialized
  boundary.
- Updated RVV target artifact preflight to validate selected boundary source
  identity against the typed microkernel body before artifact output.
- Made RVV source-authority candidate validation kernel-local for the selected
  variant/role, while preserving singleton-record constraints for direct
  standalone source/header/object exports.
- Declared direct `TianChenRVTarget` link dependencies for RVV and Scalar
  target libraries.
- Updated RVV microkernel and scalar-dispatch lit tests for the new boundary
  identity attrs and stale source/body fail-closed cases.
- Confirmed no core `tcrv.exec` or generic transform changes were introduced.

### Checks

- `cmake --build build --target TianChenRVRVVDialect TianChenRVRVVPlugin TianChenRVRVVTarget TianChenRVScalarTarget tianchenrv-rvv-extension-plugin-test tianchenrv-rvv-selected-lowering-boundary-test tianchenrv-target-artifact-export-test tcrv-opt tcrv-translate -j2`
- `cmake --build build --target tianchenrv-rvv-binary-variant-legality-test -j2`
- `./build/bin/tianchenrv-rvv-extension-plugin-test`
- `./build/bin/tianchenrv-rvv-selected-lowering-boundary-test`
- `./build/bin/tianchenrv-target-artifact-export-test`
- `./build/bin/tianchenrv-rvv-binary-planning-test`
- `./build/bin/tianchenrv-rvv-binary-variant-legality-test`
- `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv Target/RVVMicrokernel/rvv-microkernel-auto-materialization.mlir Target/RVVScalarDispatch/rvv-scalar-i32-vadd-dispatch-runtime-abi-role-binding.mlir Transforms/VariantMaterialization/plugin-variant-materialization-rvv-selected-shape.mlir` from `build/test`
- `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter='rvv-microkernel-auto-materialization|RVVScalarDispatch|RVVSmokeProbe|LoweringBoundary|rvv-'` from `build/test`
- `git diff --check`
- Core neutrality scan:
  `git diff -- lib/Transforms include/TianChenRV/Dialect/Exec lib/Dialect/Exec`

No `ssh rvv` evidence was run because generated RVV intrinsic body/header/object
semantics were not changed and this session makes no RVV runtime, correctness,
or performance claim.

### Status

[OK] Implementation checks passed; finish/archive and commit pending.


## Session 60: Common construction-protocol artifact route

**Date**: 2026-05-13
**Task**: Common construction-protocol artifact route
**Branch**: `main`

### Summary

Factored the repeated Template/Toy/TensorExtLite construction-protocol
validation and generated-output route into a shared C++ construction protocol
module consumed by all three existing extension families.

### Main Changes

- Added `TianChenRVPluginConstructionProtocol` with common manifest,
  semantic-role, typed-role, role-op interface, EmitC mapping, evidence-profile,
  generated-route, and generated-output emission helpers.
- Converted Template, Toy, and TensorExtLite construction protocol data types
  to aliases of the shared construction data model while preserving their
  extension-local public API names.
- Rewired Template, Toy, and TensorExtLite verifier/build functions to call the
  shared route with extension-local specs.
- Rewired Template/Toy/TensorExtLite target metadata artifacts to emit typed
  role and generated-output records through common helpers.
- Added `tianchenrv-construction-protocol-common-test` to prove the shared
  model and generated-output emitter are consumed by all three families.
- Confirmed no Template/Toy/TensorExtLite semantic branch was added to
  `tcrv.exec`, `lib/Transforms`, or common orchestration passes.

### Checks

- `cmake --build build --target TianChenRVPluginConstructionProtocol TianChenRVTemplatePlugin TianChenRVToyPlugin TianChenRVTensorExtLitePlugin -j2`
- `cmake --build build --target TianChenRVPluginConstructionProtocol TianChenRVTemplateDialect TianChenRVTemplatePlugin TianChenRVTemplateTarget tianchenrv-template-extension-plugin-test TianChenRVToyDialect TianChenRVToyPlugin TianChenRVToyTarget tianchenrv-toy-extension-plugin-test TianChenRVTensorExtLiteDialect TianChenRVTensorExtLitePlugin TianChenRVTensorExtLiteTarget tianchenrv-tensorext-lite-extension-plugin-test tianchenrv-construction-protocol-common-test tianchenrv-target-artifact-export-test tianchenrv-rvv-extension-plugin-test tcrv-opt tcrv-translate -j2`
- `./build/bin/tianchenrv-construction-protocol-common-test`
- `./build/bin/tianchenrv-template-extension-plugin-test`
- `./build/bin/tianchenrv-toy-extension-plugin-test`
- `./build/bin/tianchenrv-tensorext-lite-extension-plugin-test`
- `./build/bin/tianchenrv-target-artifact-export-test`
- `./build/bin/tianchenrv-rvv-extension-plugin-test`
- `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter='ConstructionProtocol|Template|template|Toy|toy|TensorExtLite|tensorext-lite|tensorext_lite'` from `build/test`
- Core neutrality and duplicate-route ref scans.
- `git diff --check`
- `git diff --cached --check`
- Trellis validation before finish/archive and after archive.

No `ssh rvv` evidence was run because this task changes non-executable
construction metadata validation and generated artifact emission only.

### Status

[OK] Finished, archived, and ready to commit.


## Session 59: TensorExtLite construction-protocol extension instantiation

**Date**: 2026-05-13
**Task**: TensorExtLite construction-protocol extension instantiation
**Branch**: `main`

### Summary

Instantiated `TensorExtLite` as a richer non-RVV construction-protocol
extension family that reaches deterministic generated metadata artifact output
from a typed ODS role-op boundary.

### Main Changes

- Added `tcrv_tensorext_lite.lowering_boundary` and
  `tcrv_tensorext_lite.tile_mma_skeleton` under a new TensorExtLite dialect.
- Added TensorExtLite construction manifest, fragment-MMA-like archetype,
  semantic role graph `configure -> load_frag -> tile_mma -> store_frag`,
  common-interface realization, typed role realization, generated EmitC route
  mapping, and evidence profile validation.
- Added plugin-local capability proposal, lowering-boundary validation,
  selected planning metadata, and emission-plan metadata for
  `tensorext_lite.tile_mma`.
- Added TensorExtLite target artifact export for
  `none-executable-tensorext-lite-fragment-mma-metadata`, including generated
  source-like role-to-call evidence and no runtime/hardware/correctness/perf
  claims.
- Registered TensorExtLite through builtin plugin and target bundle frontdoors
  without adding TensorExtLite semantic branches to `tcrv.exec`,
  `lib/Transforms`, or common orchestration passes.
- Added C++ and lit/FileCheck coverage for the valid path and fail-closed
  stale, missing, wrong-route, wrong-role, wrong-interface, malformed
  selected-plan, and stale role-realization cases.

### Checks

- `cmake --build build --target TianChenRVTensorExtLiteDialect TianChenRVTensorExtLitePlugin TianChenRVTensorExtLiteTarget tianchenrv-tensorext-lite-extension-plugin-test TianChenRVTemplateDialect TianChenRVTemplatePlugin TianChenRVTemplateTarget tianchenrv-template-extension-plugin-test TianChenRVToyDialect TianChenRVToyPlugin TianChenRVToyTarget tianchenrv-toy-extension-plugin-test tianchenrv-target-artifact-export-test tianchenrv-rvv-extension-plugin-test tcrv-opt tcrv-translate -j2`
- `./build/bin/tianchenrv-tensorext-lite-extension-plugin-test`
- `./build/bin/tianchenrv-template-extension-plugin-test`
- `./build/bin/tianchenrv-toy-extension-plugin-test`
- `./build/bin/tianchenrv-rvv-extension-plugin-test`
- `./build/bin/tianchenrv-target-artifact-export-test`
- `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter='TensorExtLite|tensorext-lite|tensorext_lite|Toy|toy|Template|template'` from `build/test`
- Core neutrality scan:
  `rg -n "tensorext|TensorExtLite|tcrv_tensorext_lite" lib/Transforms include/TianChenRV/Dialect/Exec lib/Dialect/Exec`
- `git diff --check`
- `git diff --cached --check`
- Trellis validation before finish/archive and after archive.

No `ssh rvv` evidence was run because this task changes TensorExtLite
construction/plugin/target metadata evidence only and makes no RVV runtime,
correctness, or performance claim.

### Status

[OK] Implementation checks passed; finish/archive and commit pending.


## Session 59: First concrete extension-family template instantiation

**Date**: 2026-05-13
**Task**: First concrete extension-family template instantiation
**Branch**: `main`

### Summary

Upgraded the existing Toy family from metadata-only integration into the first
minimal non-RVV consumer of the Extension-Family Plugin Construction Protocol.
Toy now has a protocol-backed manifest, role graph, ODS compute role op,
plugin-local planning metadata, target artifact preflight, and generated
role-graph-to-EmitC source-like output.

### Main Changes

- Added `ToyConstructionProtocol` with manifest, archetype, semantic role
  graph, family declaration, common-interface realization, typed role
  realization, EmitC route mapping, generated output route, and validators.
- Added `tcrv_toy.compute_skeleton` implementing
  `TCRVEmitCLowerableOpInterface`.
- Rewired Toy proposal, legality, selected-boundary materialization,
  emission-plan selected metadata, target route metadata, and artifact
  preflight to consume protocol-derived fields.
- Extended Toy target output with construction protocol, semantic roles,
  validated role-op provenance, typed role graph, EmitC route, evidence
  profile, `generated_emitc_step[...]`, and `generated_source`.
- Added focused C++ and lit/FileCheck coverage for positive and fail-closed
  Toy construction paths.
- Confirmed no Toy-specific semantic branch was added to core `tcrv.exec` or
  `lib/Transforms`.

### Checks

- `cmake --build build --target TianChenRVToyDialect TianChenRVToyPlugin TianChenRVToyTarget tianchenrv-toy-extension-plugin-test tcrv-opt tcrv-translate -j2`
- `./build/bin/tianchenrv-toy-extension-plugin-test`
- `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter='Toy|toy'` from `build/test`
- `cmake --build build --target TianChenRVTemplateDialect TianChenRVTemplatePlugin TianChenRVTemplateTarget tianchenrv-template-extension-plugin-test tianchenrv-target-artifact-export-test -j2`
- `./build/bin/tianchenrv-template-extension-plugin-test`
- `./build/bin/tianchenrv-target-artifact-export-test`
- `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter='Template|template'` from `build/test`
- `cmake --build build --target tianchenrv-rvv-extension-plugin-test -j2`
- `./build/bin/tianchenrv-rvv-extension-plugin-test`
- `rg -n "tcrv_toy|toy-plugin|Toy" lib/Transforms include/TianChenRV/Dialect/Exec lib/Dialect/Exec || true`
- `git diff --check`

`clang-format` was not available as `clang-format` or
`/usr/lib/llvm-20/bin/clang-format`; formatting was kept manually consistent.

No `ssh rvv` evidence was run because Toy construction output is
non-executable generated artifact evidence only and no RVV runtime artifact was
changed.

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


## Session 58: Template ODS role-op boundary materialization

**Date**: 2026-05-13
**Task**: Template ODS role-op boundary materialization
**Branch**: `main`

### Summary

Materialized the first real Template TableGen/ODS role-op boundary and wired it
into the existing Template construction protocol, selected-boundary
materialization, target artifact validation, and generated role-to-EmitC
output.

### Main Changes

- Added `tcrv_template.compute_skeleton` as a minimal ODS-backed Template
  compute role op implementing `TCRVEmitCLowerableOpInterface`.
- Added Template dialect verifier coverage for role-op identity, selected
  variant/capability mirror, source role, typed role id, role-specific
  interface, EmitC call, and forbidden generic-compute/unknown attributes.
- Added construction-protocol validation that reads the generated op interface
  source op/source role and cross-checks it against the manifest, typed role
  graph, EmitC mapping, and evidence profile.
- Updated Template selected-boundary materialization to create both
  `tcrv_template.lowering_boundary` and `tcrv_template.compute_skeleton`.
- Updated Template target artifact preflight to require exactly one matching
  compute role op before exporting generated output.
- Added artifact output fields for validated role-op/interface provenance while
  preserving deterministic `generated_emitc_step[...]` and `generated_source`.
- Added focused C++ and lit/FileCheck coverage for positive ODS/interface
  validation and missing/stale/wrong-role/wrong-interface failure modes.
- Confirmed no Template semantic branches were added to core `tcrv.exec` or
  `lib/Transforms`.

### Checks

- `cmake --build build --target TianChenRVTemplateDialect TianChenRVTemplatePlugin TianChenRVTemplateTarget tianchenrv-template-extension-plugin-test tcrv-opt tcrv-translate -j2`
- `./build/bin/tianchenrv-template-extension-plugin-test`
- `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter='Template|template'` from `build/test`
- `cmake --build build --target tianchenrv-target-artifact-export-test -j2`
- `./build/bin/tianchenrv-target-artifact-export-test`
- `cmake --build build --target tianchenrv-rvv-extension-plugin-test -j2`
- `./build/bin/tianchenrv-rvv-extension-plugin-test`
- `git diff --check`
- Core neutrality scan:
  `rg -n "tcrv_template|template-plugin|Template" lib/Transforms include/TianChenRV/Dialect/Exec lib/Dialect/Exec`

`clang-format` was not available as `clang-format` or
`/usr/lib/llvm-20/bin/clang-format`; formatting was kept manually consistent
with nearby code and whitespace was checked with `git diff --check`.

No `ssh rvv` evidence was run because this task changes Template construction
and artifact validation only and makes no RVV runtime/correctness/performance
claim.

### Status

[OK] Implementation checks passed; finish/archive and commit pending.

## Session 61: RVV runtime AVL/VL ABI boundary

**Date**: 2026-05-13
**Task**: RVV runtime AVL/VL ABI boundary
**Branch**: `main`

### Summary

Materialized the RVV runtime-length boundary as a target-owned contract
consumed by selected config, planning metadata, and target artifact validation
for the existing i32-vadd RVV microkernel route.

### Main Changes

- Added `RVVRuntimeLengthContract` for runtime element-count C name, runtime
  AVL source/role, runtime VL source/scope, and descriptor-local
  element-count metadata.
- Rewired `RVVBinarySelectedConfigContract` to delegate runtime length state to
  that contract.
- Rewired RVV selected emission planning to append descriptor-local
  `tcrv_rvv.descriptor_element_count` via the runtime-length contract.
- Updated C++ target artifact export test helpers to consume runtime metadata
  from the selected-config contract, matching production planning/export.
- Added C++ coverage for runtime-length metadata emission and descriptor-only
  length authority rejection.
- Added lit fail-closed coverage for stale
  `tcrv_rvv.runtime_avl_source = "descriptor-element-count"` before source
  export.
- Confirmed no changes under `lib/Transforms`,
  `include/TianChenRV/Dialect/Exec`, or `lib/Dialect/Exec`.

### Checks

- `cmake --build build --target TianChenRVRVVTarget tianchenrv-target-artifact-export-test tcrv-translate -j2`
- `./build/bin/tianchenrv-target-artifact-export-test`
- `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv Target/RVVMicrokernel/rvv-microkernel-runtime-abi-role-binding.mlir` from `build/test`
- `cmake --build build --target TianChenRVRVVDialect TianChenRVRVVPlugin TianChenRVRVVTarget TianChenRVScalarTarget tianchenrv-rvv-extension-plugin-test tianchenrv-rvv-selected-lowering-boundary-test tianchenrv-rvv-binary-planning-test tianchenrv-rvv-binary-variant-legality-test tianchenrv-target-artifact-export-test tcrv-opt tcrv-translate -j2`
- `./build/bin/tianchenrv-rvv-extension-plugin-test`
- `./build/bin/tianchenrv-rvv-selected-lowering-boundary-test`
- `./build/bin/tianchenrv-rvv-binary-planning-test`
- `./build/bin/tianchenrv-rvv-binary-variant-legality-test`
- `./build/bin/tianchenrv-target-artifact-export-test`
- Focused 10-test lit set covering RVV runtime ABI role binding, auto
  materialization, source/header/object, descriptor-only quarantine, stale
  boundary/config, missing policy, selected shape metadata, and RVV+scalar
  runtime ABI binding.
- Broader RVV lit filter
  `rvv-microkernel|RVVScalarDispatch|RVVSmokeProbe|LoweringBoundary|rvv-`:
  104 selected tests passed.
- `git diff --check`
- Trellis validation before finish and after archive passed.
- `git diff --cached --check`

No `ssh rvv` evidence was run because generated RVV intrinsic source/body
semantics, object bytes, runtime behavior, correctness, and performance claims
were not changed.

### Status

[OK] Implementation checks and Trellis archive passed; commit pending.


## Session 58: RVV selected variant materialization route

**Date**: 2026-05-14
**Task**: RVV selected variant materialization route
**Branch**: `main`

### Summary

Extended default i32-vadd upstream linalg route coverage through compiler-produced tcrv_rvv lowering boundary, typed microkernel, source/header/object artifact output, and fail-closed selected boundary/config/runtime metadata checks; repaired stale vmul selected-config FileCheck.

### Main Changes

- Confirmed the production path already materializes the bounded
  compiler-produced route from marked `linalg.generic` i32-vadd through
  `tcrv.exec.kernel`, RVV plugin planning, `tcrv_rvv.lowering_boundary`, and
  `tcrv_rvv.i32_vadd_microkernel`.
- Extended `linalg-i32-vadd-to-exec.mlir` to prove that the same upstream route
  reaches generated RVV source, header, and RISC-V relocatable object artifacts.
- Added FileCheck assertions for frontend-lowering source identity, selected
  config metadata, runtime AVL/VL metadata, runtime element-count ABI metadata,
  typed `setvl/with_vl/load/add/store` body, selected-config emission authority,
  and IR-backed callable ABI output.
- Added fail-closed coverage for missing typed RVV microkernel body, missing
  selected-boundary source identity, stale selected vector suffix, and stale
  runtime AVL role metadata.
- Repaired a stale `linalg-i32-vmul-to-rvv-artifact.mlir` FileCheck line to
  match the selected-config contract authority wording introduced by the
  current target artifact route.
- Confirmed no changes under `lib/Transforms`,
  `include/TianChenRV/Dialect/Exec`, or `lib/Dialect/Exec`.

### Git Commits

| Hash | Message |
|------|---------|
| `this commit` | (see git log) |

### Testing

- `cmake --build build --target TianChenRVRVVDialect TianChenRVRVVPlugin TianChenRVRVVTarget TianChenRVScalarTarget tianchenrv-rvv-extension-plugin-test tianchenrv-rvv-selected-lowering-boundary-test tianchenrv-rvv-binary-planning-test tianchenrv-rvv-binary-variant-legality-test tianchenrv-target-artifact-export-test tcrv-opt tcrv-translate -j2`
- `./build/bin/tianchenrv-rvv-extension-plugin-test`
- `./build/bin/tianchenrv-rvv-selected-lowering-boundary-test`
- `./build/bin/tianchenrv-rvv-binary-planning-test`
- `./build/bin/tianchenrv-rvv-binary-variant-legality-test`
- `./build/bin/tianchenrv-target-artifact-export-test`
- `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv Transforms/LinalgToExec/linalg-i32-vadd-to-exec.mlir` from `build/test`
- `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv Transforms/LinalgToExec` from `build/test`
- `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv Target/RVVMicrokernel` from `build/test`
- `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv Target/RVVScalarDispatch` from `build/test`
- `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv Transforms/LoweringBoundary` from `build/test`
- `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-14-rvv-selected-variant-materialization-route`
- `git diff --check`

The first full `Transforms/LinalgToExec` run failed on stale selected-config
authority text in `linalg-i32-vmul-to-rvv-artifact.mlir`; the FileCheck line was
updated and the directory passed on rerun.

No `ssh rvv` evidence was run because this round changed lit coverage only and
did not change generated RVV source/object bytes, runtime behavior,
correctness, or performance claims.

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 59: RVV selected variant materialization implementation

**Date**: 2026-05-14
**Task**: RVV selected variant materialization implementation
**Branch**: `main`

### Summary

Implemented op-owned RVV selected source identity on materialized microkernel ops, added target artifact preflight validation, updated focused lit fixtures, and archived the Trellis task.

### Main Changes

(Add details)

### Git Commits

(No commits - planning session)

### Testing

- [OK] (Add test results)

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 60: RVV selected variant bundle full-state closure

**Date**: 2026-05-14
**Task**: RVV selected variant bundle full-state closure
**Branch**: `main`

### Summary

Closed the compiler-produced `vector-dynamic-i32-vadd` selected bundle route
around full selected RVV state: source identity, selected vector config, and
runtime AVL/VL authority now flow into RVV+scalar dispatch bundle metadata
through `RVVBinarySelectedConfigContract`, and the bundle test now fails closed
on stale or missing selected config/runtime-length metadata.

### Main Changes

- Reconciled the current HEAD mismatch: HEAD is supervisor-resume commit
  `29a5903`, while the relevant prior RVV owner commit is `4df5aaa`.
- Added dispatch-bundle metadata formatting helpers to
  `RVVBinarySelectedConfigContract` for selected vector config, runtime VL
  boundary, and selected source identity.
- Rewired RVV+scalar dispatch bundle metadata construction to use those
  contract helpers instead of local string assembly.
- Extended the dynamic vector i32-vadd plan-and-export bundle lit test with
  fail-closed stale/missing selected LMUL and stale/missing runtime AVL/VL
  metadata mutations.
- Confirmed no changes under core `tcrv.exec` dialect sources and no new
  generic RVV semantic branch.

### Checks

- `cmake --build build --target TianChenRVRVVTarget TianChenRVScalarTarget tcrv-translate tcrv-opt -j2`
- `cmake --build build --target tianchenrv-target-artifact-export-test tianchenrv-rvv-binary-planning-test tianchenrv-rvv-selected-lowering-boundary-test tianchenrv-rvv-extension-plugin-test -j2`
- `./build/bin/tianchenrv-target-artifact-export-test`
- `./build/bin/tianchenrv-rvv-binary-planning-test`
- `./build/bin/tianchenrv-rvv-selected-lowering-boundary-test`
- `./build/bin/tianchenrv-rvv-extension-plugin-test`
- `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv Target/TargetArtifactBundleExport/plan-vector-dynamic-i32-vadd-and-export-target-artifact-bundle.mlir`
- `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv Target/TargetArtifactBundleExport`
- `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv Target/RVVMicrokernel Target/RVVScalarDispatch`
- `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv Target/ArtifactExport Transforms/VectorToExec Transforms/LinalgToExec`
- `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv Target`
- `git diff --check`
- `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-14-rvv-selected-variant-bundle-full-state-closure`

No `ssh rvv` evidence was run because generated RVV source/object semantics,
runtime behavior, correctness, and performance claims were not changed.

### Status

[OK] Implementation checks passed; finish/archive and commit pending.


## Session 61: RVV compiler-produced artifact e2e closure

**Date**: 2026-05-14
**Task**: RVV compiler-produced artifact e2e closure
**Branch**: `main`

### Summary

Closed the dynamic vector i32-vadd compiler-produced artifact evidence round:
the existing production route already reached generated RVV source/header/object
artifacts through the plan-and-export bundle front door, and this session added
focused lit coverage for the exact `--lower-vector-i32-vadd-frontend` route.
The generated bundle was then compiled, linked, and run on `ssh rvv` through the
external ABI caller with runtime counts `7,16,23`.

### Main Changes

- Added `test/Scripts/rvv-microkernel-bundle-e2e.test` coverage for
  `rvv_microkernel_e2e.py --lower-vector-i32-vadd-frontend
  --use-target-artifact-bundle --use-plan-and-export-bundle-front-door`.
- Created and archived Trellis task
  `05-14-rvv-compiler-produced-artifact-e2e-closure`.
- No production C++ change was needed; current HEAD already consumed the
  selected RVV config/runtime-length contracts and emitted source/header/object
  artifacts for this migrated vadd slice.

### Testing

- `cmake --build build --target TianChenRVRVVDialect TianChenRVRVVPlugin TianChenRVRVVTarget TianChenRVScalarTarget tianchenrv-rvv-extension-plugin-test tianchenrv-rvv-selected-lowering-boundary-test tianchenrv-rvv-binary-planning-test tianchenrv-target-artifact-export-test tcrv-opt tcrv-translate -j2`
- `./build/bin/tianchenrv-target-artifact-export-test`
- `./build/bin/tianchenrv-rvv-binary-planning-test`
- `./build/bin/tianchenrv-rvv-selected-lowering-boundary-test`
- `./build/bin/tianchenrv-rvv-extension-plugin-test`
- `python3 scripts/rvv_microkernel_e2e.py --self-test`
- `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv Target/TargetArtifactBundleExport/plan-vector-dynamic-i32-vadd-and-export-target-artifact-bundle.mlir`
- `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv Target/TargetArtifactBundleExport Target/RVVMicrokernel Target/RVVScalarDispatch Target/ArtifactExport Transforms/VectorToExec Transforms/LinalgToExec`
- `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv Scripts/rvv-microkernel-bundle-e2e.test`
- `python3 scripts/rvv_microkernel_e2e.py --arithmetic-family i32-vadd --lower-vector-i32-vadd-frontend --use-target-artifact-bundle --use-plan-and-export-bundle-front-door --expect-selected-kernel frontend_vector_dynamic_i32_vadd --runtime-count 7 --runtime-count 16 --runtime-count 23 --run-id codex-current-ssh --overwrite --timeout 120 --connect-timeout 10`
- `git diff --check`
- `python3 ./.trellis/scripts/task.py validate .trellis/tasks/archive/2026-05/05-14-05-14-rvv-compiler-produced-artifact-e2e-closure`

### Evidence

`ssh rvv` succeeded on `riscv64` with clang `Ubuntu clang version 18.1.3
(1ubuntu1)`. Both the source-built object path and generated bundle object path
printed `tcrv_rvv_microkernel_external_abi_ok counts=7,16,23`.

### Status

[OK] Completed and archived; commit pending.


## Session 68: RVV source-frontdoor op-family artifact parity

**Date**: 2026-05-14
**Task**: RVV source-frontdoor op-family artifact parity
**Branch**: `main`

### Summary

Replaced the dynamic MLIR vector source-frontdoor transform's vadd/vsub mode
split with one registry-backed op-family policy surface. The default
`--tcrv-lower-source-rvv-binary-to-exec` path and explicit vadd/vsub
compatibility aliases now filter through the same policy, while the dynamic
body recognizer still derives computation identity from source vector/arith
operations and the RVV binary family registry.

### Main Changes

- Added `VectorFrontendFamilyPolicy` in
  `lib/Transforms/LowerSourceRVVBinaryToExec.cpp` for default source-frontdoor
  and explicit vadd/vsub compatibility alias restrictions.
- Removed the transform-local `VectorFrontendAdapterMode` /
  `VAddOnly` / `VSubOnly` / `ArithmeticFamily` split.
- Made dynamic vector accepted arithmetic diagnostics derive from the registry
  backed dynamic vector family set.
- Added fail-closed coverage rejecting an `i32-vmul` dynamic vector marker/body
  on the bounded add/sub vector source-frontdoor route.
- Updated `Passes.td` to document the registry-policy-backed construction
  surface.

### Testing

- `cmake --build build --target TianChenRVTransforms tcrv-opt -j2`
- `cmake --build build --target TianChenRVRVVPlugin TianChenRVRVVTarget tcrv-translate tianchenrv-rvv-extension-plugin-test tianchenrv-target-artifact-export-test -j2`
- `./build/bin/tianchenrv-rvv-extension-plugin-test`
- `./build/bin/tianchenrv-target-artifact-export-test`
- Focused lit for dynamic vector vadd/vsub positive, explicit alias invalid,
  shared binary invalid, artifact bundle export, and RVV script bundle tests.
- `python3 scripts/rvv_microkernel_e2e.py --self-test`
- `python3 scripts/rvv_scalar_dispatch_e2e.py --self-test`
- Local dry-runs:
  `codex-family-policy-vector-vadd-dry`,
  `codex-family-policy-vector-vsub-dry`,
  `codex-family-policy-vector-vsub-dispatch-dry`.
- Focused `ssh rvv` evidence:
  `codex-family-policy-vector-vadd-ssh`, `ssh_evidence=true`,
  `tcrv_rvv_microkernel_external_abi_ok counts=7,16,23`; and
  `codex-family-policy-vector-vsub-ssh`, `ssh_evidence=true`,
  `tcrv_rvv_i32_vsub_microkernel_external_abi_ok counts=7,16,23`.
- Ref-scans for removed transform mode branches, descriptor-only authority,
  and generic core family branches.
- `git diff --check`
- Trellis validation before finish and after archive.

### Status

[OK] Completed and archived; commit pending.


## Session 63: Bounded MLIR vector to RVV selected route

**Date**: 2026-05-14
**Task**: Bounded MLIR vector to RVV selected route
**Branch**: `main`

### Summary

Closed the explicit VectorToExec i32-vsub front-door gap on top of the
existing generic source RVV binary owner. The new pass
`--tcrv-lower-vector-rvv-i32-vsub-to-exec` reuses the same source-frontdoor
lowering path, accepts only the bounded dynamic vector/SCF vsub route, and now
fails closed when a vsub marker is paired with an add body.

### Main Changes

- Added the explicit vector i32-vsub pass in `Passes.td`, `Passes.h`,
  `LowerSourceRVVBinaryToExec.cpp`, and `tcrv-opt`.
- Added `VSubOnly` validation so explicit vsub rejects fixed vadd and
  marker/body mismatch before materializing `tcrv.exec`.
- Updated `rvv_microkernel_e2e.py` so `--lower-vector-i32-vsub-frontend` uses
  the explicit pass and reports the explicit pipeline label.
- Extended focused VectorToExec, bundle export, and e2e lit coverage.
- Updated the lowering-runtime spec to include the explicit i32-vsub vector
  source-tail authority surface.

### Evidence

- Direct explicit vector vsub bundle:
  `artifacts/tmp/rvv_microkernel_bundle_e2e/codex-vector-vsub-explicit-ssh`.
  Remote `ssh rvv` source-linked and bundle-object-linked runs printed
  `tcrv_rvv_i32_vsub_microkernel_external_abi_ok counts=7,16,23`.
- RVV+scalar dispatch vector vsub bundle:
  `artifacts/tmp/tianchenrv-rvv-dispatch-bundle-e2e/codex-vector-vsub-dispatch-ssh`.
  Remote `ssh rvv` source-linked and bundle-object-linked runs printed
  `tcrv_rvv_scalar_i32_vsub_bundle_external_abi_ok runtime_counts=7,16 branches=scalar_and_rvv`.

### Checks

- `cmake --build build --target TianChenRVTransforms tcrv-opt tcrv-translate tianchenrv-rvv-extension-plugin-test tianchenrv-target-artifact-export-test -j2`
- `cmake --build build --target TianChenRVTransforms tcrv-opt tcrv-translate -j2`
- `build/bin/tianchenrv-rvv-extension-plugin-test`
- `build/bin/tianchenrv-target-artifact-export-test`
- `python3 scripts/rvv_microkernel_e2e.py --self-test`
- Focused lit from `build/test` for explicit vector vsub, invalid vector
  source, vector vsub bundle export, and RVV microkernel bundle e2e; 5 passed.
- Local explicit vector vsub dry-run bundle evidence with runtime counts
  `7,16,23`.
- `ssh rvv` direct explicit vector vsub bundle evidence.
- `ssh rvv` RVV+scalar dispatch vector vsub bundle evidence.
- Ref-scan for descriptor authority and generic core target-family branches.
- `git diff --check`
- Trellis validation for the active task.

### Self-Repair

The first focused lit run caught that vsub-only mode did not cross-check the
dynamic body family after marker validation. A vsub marker paired with an
`arith.addi` body could incorrectly lower as vadd. The fix extends
marker/body cross-checking to `VSubOnly`, and the new invalid test locks this
case.

### Status

[OK] Implementation and validation complete; archive and commit pending.


## Session 63: RVV generated artifact runtime ABI closure

**Date**: 2026-05-14
**Task**: RVV generated artifact runtime ABI closure
**Branch**: `main`

### Summary

Closed the direct RVV generated artifact ABI proof path for the selected
`i32-vadd` and `i32-vsub` microkernel artifacts. Direct helper dry-runs now
construct a generated external C caller from the exported generated header and
compiler-emitted runtime ABI signature, while non-dry-run evidence uses that
same caller to compile, link, and run against both the generated source-built
object and the generated object artifact on `ssh rvv`.

### Main Changes

- Updated `scripts/rvv_microkernel_e2e.py` so direct helper dry-run mode emits
  `rvv_microkernel_external_caller.c` whenever the generated direct header is
  available.
- Recorded the generated caller path, hash, ABI signature, runtime counts,
  success marker, and `source_only` dry-run flag in evidence JSON.
- Extended `test/Scripts/rvv-microkernel-e2e.test` to check direct vadd/vsub
  generated caller files, arithmetic calls, runtime counts, success markers,
  caller hashes, and the updated claim scope.
- Kept descriptor-only compute/config/runtime authority unchanged and
  quarantined; no core `tcrv.exec` or generic target-export RVV branch was
  added.

### Testing

- `cmake --build build --target tcrv-opt tcrv-translate -j2`
- `python3 scripts/rvv_microkernel_e2e.py --self-test`
- `python3 scripts/rvv_microkernel_e2e.py --dry-run --run-id codex-abi-vadd-dry --overwrite --input test/Target/EmissionManifest/emission-manifest-rvv-microkernel.mlir`
- `python3 scripts/rvv_microkernel_e2e.py --dry-run --arithmetic-family=i32-vsub --run-id codex-abi-vsub-dry --overwrite`
- `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'rvv-microkernel-e2e\\.test'` from `build/test`
- `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'rvv-microkernel-bundle-e2e\\.test'` from `build/test`
- `python3 scripts/rvv_microkernel_e2e.py --run-id codex-abi-vadd-ssh --overwrite --input test/Target/EmissionManifest/emission-manifest-rvv-microkernel.mlir --timeout 120`
- `python3 scripts/rvv_microkernel_e2e.py --arithmetic-family=i32-vsub --run-id codex-abi-vsub-ssh --overwrite --timeout 120`

Both `ssh rvv` runs returned `status: success` and `ssh_evidence: true`.
The vadd run used
`tcrv_rvv_i32_vadd_microkernel_rvv_microkernel_manifest_rvv_first_slice` and
observed `tcrv_rvv_microkernel_external_abi_ok`; the vsub run used
`tcrv_rvv_i32_vsub_microkernel_export_i32_vsub_rvv_sub_slice` and observed
`tcrv_rvv_i32_vsub_microkernel_external_abi_ok`. Both used runtime counts
`7,16` and compile flags `-O2 -march=rv64gcv -mabi=lp64d`.

### Evidence

- `artifacts/tmp/rvv_microkernel_e2e/codex-abi-vadd-ssh/evidence.json`
- `artifacts/tmp/rvv_microkernel_e2e/codex-abi-vsub-ssh/evidence.json`

### Status

[OK] Implemented; final diff checks, archive, and commit pending.


## Session 65: RVV EmitC body mapping production route

**Date**: 2026-05-14
**Task**: RVV EmitC body emission production route
**Branch**: `main`

### Summary

Promoted selected RVV EmitC metadata for migrated dynamic vector i32 vadd/vsub
from artifact annotation into the production microkernel body emission path.
The RVV microkernel source exporter now builds an explicit EmitC body mapping
from selected plan metadata and passes it into the common EmitC lowerable route
for header selection and arithmetic `emitc.call_opaque` callee selection.

### Main Changes

- Added `RVVBinaryEmitCBodyMapping` to the selected config contract surface.
- Collected selected emission-plan metadata while resolving the selected RVV
  runtime ABI path and made frontend-selected records fail closed when the
  body mapping metadata is missing or stale.
- Rewired `RVVBinaryEmitCLowerable` so route kind, required header, and
  arithmetic intrinsic come from the selected body mapping rather than
  hard-coded target source strings.
- Emitted source/object evidence for `emitc_body_mapping_source` and the
  concrete mapping consumed by the body route.
- Added vadd/vsub VectorToExec checks proving `selected_plan_metadata` body
  mapping consumption and fail-closed self-check export mutations.

### Testing

- `cmake --build build --target TianChenRVRVVTarget TianChenRVRVVPlugin tcrv-opt tcrv-translate tianchenrv-target-artifact-export-test tianchenrv-rvv-extension-plugin-test -j2`
- `./build/bin/tianchenrv-target-artifact-export-test`
- `./build/bin/tianchenrv-rvv-extension-plugin-test`
- `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv Transforms/VectorToExec/vector-dynamic-i32-vadd-to-exec.mlir Transforms/VectorToExec/vector-dynamic-i32-vsub-to-exec.mlir` from `build/test`
- `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv Target/TargetArtifactBundleExport/plan-vector-dynamic-i32-vadd-and-export-target-artifact-bundle.mlir Target/TargetArtifactBundleExport/plan-vector-dynamic-i32-vsub-and-export-target-artifact-bundle.mlir Scripts/rvv-microkernel-bundle-e2e.test` from `build/test`
- `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv Target/RVVMicrokernel/rvv-microkernel-pipeline.mlir Target/RVVMicrokernel/rvv-microkernel-family-sub.mlir Scripts/rvv-microkernel-e2e.test` from `build/test`
- `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv Target/RVVScalarDispatch/rvv-scalar-i32-vadd-dispatch-c.mlir Target/RVVScalarDispatch/rvv-scalar-i32-vsub-dispatch-generic-route.mlir Scripts/rvv-scalar-dispatch-bundle-e2e.test` from `build/test`
- Bounded reference scan over touched RVV plugin/target/export files for
  descriptor/direct-string authority terms.
- `git diff --check`

No `ssh rvv` evidence was run because valid generated RVV intrinsic behavior
stayed the same; this round changed production authority consumption, emitted
comments/object evidence, and fail-closed metadata checks, not runtime
correctness or performance behavior.

### Status

[OK] Completed locally; archive and commit pending.


## Session 66: RVV op-family runtime ABI dispatch parity

**Date**: 2026-05-14
**Task**: RVV op-family runtime ABI dispatch parity
**Branch**: `main`

### Summary

Closed the next vadd/vsub runtime ABI dispatch parity slice. The support-layer
i32 finite binary runtime ABI helper surface now has selected-family overloads
for callable parameters, role requirements, mem-window specs, runtime length
specs, dispatch runtime specs, dispatch ABI parameters, and callable role
binding. The historical no-argument i32 helpers remain documented as
compatibility defaults for `i32-vadd` only.

### Main Changes

- Added selected-family helper overloads in
  `include/TianChenRV/Support/RuntimeABI.h`,
  `include/TianChenRV/Support/RuntimeABIParam.h`,
  `include/TianChenRV/Support/RuntimeABIMemWindow.h`, and
  `lib/Support/RuntimeABIContract.cpp`.
- Extended `test/Support/RuntimeABICallablePlanTest.cpp` so `i32-vadd`,
  `i32-vsub`, and `i32-vmul` all prove selected-family helper parity against
  `I32BinaryRuntimeABIContract`.
- Updated vsub/vmul target-artifact C++ fixtures to build runtime ABI mirrors
  from the selected family id rather than the no-argument vadd default.
- Added
  `test/Target/RVVScalarDispatch/rvv-scalar-i32-vsub-dispatch-runtime-abi-role-binding.mlir`
  covering positive vsub dispatch ordered ABI roles and fail-closed mutations
  for missing runtime length role, stale runtime-count metadata, duplicate
  role, wrong type, wrong ownership, unknown role, guard type mismatch, and
  detached dispatch ABI metadata.
- Updated
  `.trellis/spec/lowering-runtime/emission-runtime-contract.md` with the
  executable selected-family helper contract and compatibility-wrapper rule.

### Testing

- `cmake --build build --target tianchenrv-runtime-abi-callable-plan-test tianchenrv-target-artifact-export-test tcrv-opt tcrv-translate -j2`
- `build/bin/tianchenrv-runtime-abi-callable-plan-test`
- `build/bin/tianchenrv-target-artifact-export-test`
- `cmake --build build --target tianchenrv-rvv-extension-plugin-test -j2`
- `build/bin/tianchenrv-rvv-extension-plugin-test`
- `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv Target/RVVScalarDispatch/rvv-scalar-i32-vsub-dispatch-runtime-abi-role-binding.mlir Target/RVVScalarDispatch/rvv-scalar-i32-vadd-dispatch-runtime-abi-role-binding.mlir Target/RVVScalarDispatch/rvv-scalar-i32-vsub-dispatch-generic-route.mlir Target/RVVScalarDispatch/rvv-scalar-i32-vadd-dispatch-generic-route.mlir Target/TargetArtifactBundleExport/plan-vector-dynamic-i32-vadd-and-export-target-artifact-bundle.mlir Target/TargetArtifactBundleExport/plan-vector-dynamic-i32-vsub-and-export-target-artifact-bundle.mlir Scripts/rvv-scalar-dispatch-e2e.test Scripts/rvv-scalar-dispatch-bundle-e2e.test` from `build/test`
- `python3 scripts/rvv_scalar_dispatch_e2e.py --self-test`
- `python3 scripts/rvv_scalar_dispatch_e2e.py --dry-run --use-target-artifact-bundle --use-plan-and-export-bundle-front-door --arithmetic-family=i32-vsub --input test/Target/TargetArtifactBundleExport/plan-vector-dynamic-i32-vsub-and-export-target-artifact-bundle.mlir --expect-selected-kernel frontend_vector_dynamic_bundle_i32_vsub --run-id codex-parity-vsub-dynamic-dry --overwrite --timeout 120`
- `python3 scripts/rvv_scalar_dispatch_e2e.py --use-target-artifact-bundle --use-plan-and-export-bundle-front-door --arithmetic-family=i32-vsub --input test/Target/TargetArtifactBundleExport/plan-vector-dynamic-i32-vsub-and-export-target-artifact-bundle.mlir --expect-selected-kernel frontend_vector_dynamic_bundle_i32_vsub --run-id codex-parity-vsub-dynamic-ssh --overwrite --timeout 120 --connect-timeout 10`
- Bounded support/RVV ref-scan for remaining vadd defaults, descriptor
  authority, and generic-core RVV branch terms.
- `git diff --check`
- `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-14-rvv-op-family-runtime-abi-dispatch-parity`
- `python3 ./.trellis/scripts/task.py finish`
- `python3 ./.trellis/scripts/task.py archive --no-commit 05-14-rvv-op-family-runtime-abi-dispatch-parity`
- `python3 ./.trellis/scripts/task.py validate .trellis/tasks/archive/2026-05/05-14-rvv-op-family-runtime-abi-dispatch-parity`

The `ssh rvv` dynamic vsub dispatch bundle run returned `status: success`,
`ssh_evidence: true`, `ssh_evidence_verified: true`, and selected kernel
`frontend_vector_dynamic_bundle_i32_vsub`. Artifact root:
`artifacts/tmp/tianchenrv-rvv-dispatch-bundle-e2e/codex-parity-vsub-dynamic-ssh`.

### Status

[OK] Completed and archived; commit pending.


## Session 63: RVV selected route default quarantine

**Date**: 2026-05-14
**Task**: RVV selected route default quarantine
**Branch**: `main`

### Summary

Quarantined descriptor-only authority on the migrated dynamic vector
`i32-vadd`/`i32-vsub` selected route. The default RVV+scalar dispatch bundle
route now records explicit descriptor compute/config/runtime quarantine claims,
and frontend-lowering selected variants must keep matching selected
lowering-boundary source identity even when reached through explicit RVV
self-check helper export rather than the generic artifact front door.

### Main Changes

- Added `selectedVariantRequiresBoundarySourceIdentity` in
  `lib/Target/RVV/RVVMicrokernel.cpp`, making `frontend-lowering` selected
  variants require selected lowering-boundary source identity inside
  `buildModuleRecord()`.
- Added descriptor compute/config/runtime quarantine route claims to RVV
  microkernel artifact routes and RVV+scalar dispatch artifact routes.
- Extended dynamic vector vadd/vsub bundle tests to assert quarantine claims in
  generated bundle indexes.
- Added fail-closed vadd/vsub self-check-helper coverage for modules where
  selected source identity is stripped from both the RVV lowering boundary and
  RVV microkernel op. The vsub helper path previously emitted C for this
  stripped frontend-lowering module.
- Confirmed changed production files are limited to RVV target/export code; no
  generic core orchestration or `tcrv.exec` sources changed.

### Testing

- `cmake --build build --target TianChenRVRVVTarget tcrv-opt tcrv-translate tianchenrv-target-artifact-export-test tianchenrv-rvv-binary-planning-test tianchenrv-rvv-selected-lowering-boundary-test tianchenrv-rvv-extension-plugin-test -j2`
- `./build/bin/tianchenrv-target-artifact-export-test`
- `./build/bin/tianchenrv-rvv-binary-planning-test`
- `./build/bin/tianchenrv-rvv-selected-lowering-boundary-test`
- `./build/bin/tianchenrv-rvv-extension-plugin-test`
- `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv Target/TargetArtifactBundleExport/plan-vector-dynamic-i32-vadd-and-export-target-artifact-bundle.mlir Target/TargetArtifactBundleExport/plan-vector-dynamic-i32-vsub-and-export-target-artifact-bundle.mlir` from `build/test`
- `python3 scripts/rvv_microkernel_e2e.py --self-test`
- `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv Scripts/rvv-microkernel-bundle-e2e.test` from `build/test`
- `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv Target/TargetArtifactBundleExport/target-artifact-bundle-positive.mlir Target/RVVScalarDispatch/rvv-scalar-i32-vadd-dispatch-c.mlir Target/RVVScalarDispatch/rvv-scalar-i32-vsub-dispatch-generic-route.mlir Target/RVVMicrokernel/rvv-microkernel-pipeline.mlir` from `build/test`
- Manual stripped-vsub self-check helper probe failed closed with `requires
  selected RVV binary source identity before target artifact export`.
- `git diff --check`

No `ssh rvv` evidence was run because this round changed fail-closed
preflight/quarantine metadata only. It did not change generated RVV
source/object semantics, runtime behavior, correctness claims, or performance
claims.

### Status

[OK] Implementation checks passed; finish/archive and commit pending.


## Session 64: RVV extension-family EmitC emission route

**Date**: 2026-05-14
**Task**: RVV extension-family EmitC emission route
**Branch**: `main`

### Summary

Added selected-plan authority for the RVV extension-family EmitC route on the
migrated dynamic vector `i32-vadd`/`i32-vsub` slice. The RVV plugin planner now
publishes route kind, EmitC source authority, required RVV header, and selected
arithmetic intrinsic metadata, and the RVV target/export path validates those
fields before source/header/object/bundle emission.

### Main Changes

- Added shared RVV selected EmitC route metadata helpers in
  `include/TianChenRV/Target/RVV/RVVSelectedConfigContract.h`.
- Appended the route metadata from
  `lib/Plugin/RVV/RVVBinarySelectedEmissionPlanning.cpp` after selected source
  identity, preserving selected vector-shape-driven intrinsic suffixes.
- Made `lib/Target/RVV/RVVMicrokernel.cpp` and
  `lib/Target/RVV/RVVScalarDispatch.cpp` require the same route metadata during
  production selected-plan validation and route preflight.
- Extended target artifact bundle lit coverage for vadd/vsub with positive
  index/source checks and fail-closed missing route-kind / stale-intrinsic
  cases.
- Updated `scripts/rvv_microkernel_e2e.py` so direct/bundle evidence requires
  `tcrv_rvv.emitc_route_kind`, `emitc_source_authority`,
  `emitc_required_header`, and `emitc_arithmetic_intrinsic`.
- Kept changes out of `lib/Transforms`, `include/TianChenRV/Dialect/Exec`, and
  `lib/Dialect/Exec`; core orchestration remains target-neutral.

### Testing

- `cmake --build build --target TianChenRVRVVTarget TianChenRVRVVPlugin tcrv-opt tcrv-translate tianchenrv-target-artifact-export-test tianchenrv-rvv-extension-plugin-test -j2`
- `cmake --build build --target tianchenrv-target-artifact-export-test -j2`
- `./build/bin/tianchenrv-target-artifact-export-test`
- `./build/bin/tianchenrv-rvv-extension-plugin-test`
- `./build/bin/tianchenrv-rvv-binary-planning-test`
- `./build/bin/tianchenrv-rvv-selected-lowering-boundary-test`
- `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv Target/TargetArtifactBundleExport/plan-vector-dynamic-i32-vadd-and-export-target-artifact-bundle.mlir Target/TargetArtifactBundleExport/plan-vector-dynamic-i32-vsub-and-export-target-artifact-bundle.mlir` from `build/test`
- `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv Scripts/rvv-microkernel-bundle-e2e.test` from `build/test`
- `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv Transforms/VectorToExec/vector-dynamic-i32-vadd-to-exec.mlir Transforms/VectorToExec/vector-dynamic-i32-vsub-to-exec.mlir Transforms/LinalgToExec/linalg-i32-vadd-to-exec.mlir Transforms/LinalgToExec/linalg-i32-vsub-to-rvv-artifact.mlir Target/RVVMicrokernel/rvv-microkernel-auto-materialization.mlir Target/RVVMicrokernel/rvv-microkernel-family-sub.mlir Target/RVVMicrokernel/rvv-microkernel-i32m2-family-sub.mlir Target/RVVMicrokernel/rvv-microkernel-descriptor-only-production-rejects.mlir Target/RVVScalarDispatch/rvv-scalar-i32-vadd-dispatch-generic-route.mlir Target/RVVScalarDispatch/rvv-scalar-i32-vsub-dispatch-generic-route.mlir Target/RVVScalarDispatch/rvv-scalar-i32-vsub-dispatch-i32m2-generic-route.mlir` from `build/test`
- `python3 scripts/rvv_microkernel_e2e.py --self-test`
- `python3 scripts/rvv_microkernel_e2e.py --arithmetic-family i32-vadd --lower-vector-i32-vadd-frontend --use-target-artifact-bundle --use-plan-and-export-bundle-front-door --expect-selected-kernel frontend_vector_dynamic_i32_vadd --runtime-count 7 --runtime-count 16 --runtime-count 23 --run-id codex-emitc-route-vadd --overwrite --timeout 120 --connect-timeout 10`
- `python3 scripts/rvv_microkernel_e2e.py --arithmetic-family i32-vsub --lower-vector-i32-vsub-frontend --use-target-artifact-bundle --use-plan-and-export-bundle-front-door --expect-selected-kernel frontend_vector_dynamic_i32_vsub --runtime-count 7 --runtime-count 16 --runtime-count 23 --run-id codex-emitc-route-vsub --overwrite --timeout 120 --connect-timeout 10`
- `git diff --check`
- `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-14-rvv-extension-family-emitc-emission-route`
- `python3 ./.trellis/scripts/task.py finish`
- `python3 ./.trellis/scripts/task.py archive --no-commit 05-14-rvv-extension-family-emitc-emission-route`
- `python3 ./.trellis/scripts/task.py validate .trellis/tasks/archive/2026-05/05-14-rvv-extension-family-emitc-emission-route`
- `git diff --cached --check`

Both `ssh rvv` e2e runs returned `status: success`, `ssh_evidence: true`, and
runtime counts `7,16,23`.

### Status

[OK] Completed and archived; commit pending.


## Session 62: RVV selected op-family production route

**Date**: 2026-05-14
**Task**: RVV selected op-family production route
**Branch**: `main`

### Summary

Made the selected RVV finite binary artifact preflight reusable for
`vector-dynamic-i32-vsub` instead of relying on an i32-vadd-only source-identity
guard. The vsub compiler-produced route now has plan-and-export bundle evidence
through the same selected source identity, selected config, runtime length,
source/header/object, and external caller path as the prior vadd route.

### Main Changes

- Changed RVV target-artifact candidate validation in
  `lib/Target/RVV/RVVMicrokernel.cpp` so every finite typed RVV binary route
  requires selected lowering-boundary source identity before export.
- Added `--lower-vector-i32-vsub-frontend` to
  `scripts/rvv_microkernel_e2e.py`, using the selected binary lowering plus
  execution-planning route and preserving one-frontend-only guardrails.
- Extended `test/Scripts/rvv-microkernel-bundle-e2e.test` with dry-run
  plan-and-export bundle coverage for the dynamic vector i32-vsub frontend,
  including selected kernel, runtime counts, source/header/object routes,
  `__riscv_vsub_vv_i32m1`, and caller ABI subtraction checks.
- Added a fail-closed target-artifact export mutation test that strips vsub
  op-owned selected source identity from the lowering boundary and microkernel
  and requires bundle export to fail before writing an index.
- Confirmed no changes under `lib/Transforms`,
  `include/TianChenRV/Dialect/Exec`, or `lib/Dialect/Exec`.

### Testing

- `cmake --build build --target TianChenRVRVVTarget tcrv-translate tcrv-opt -j2`
- `cmake --build build --target tianchenrv-target-artifact-export-test tianchenrv-rvv-binary-planning-test tianchenrv-rvv-selected-lowering-boundary-test tianchenrv-rvv-extension-plugin-test -j2`
- `cmake --build build --target TianChenRVRVVDialect TianChenRVRVVPlugin TianChenRVRVVTarget TianChenRVScalarTarget tianchenrv-rvv-extension-plugin-test tianchenrv-rvv-selected-lowering-boundary-test tianchenrv-rvv-binary-planning-test tianchenrv-target-artifact-export-test tcrv-opt tcrv-translate -j2`
- `./build/bin/tianchenrv-target-artifact-export-test`
- `./build/bin/tianchenrv-rvv-binary-planning-test`
- `./build/bin/tianchenrv-rvv-selected-lowering-boundary-test`
- `./build/bin/tianchenrv-rvv-extension-plugin-test`
- `python3 scripts/rvv_microkernel_e2e.py --self-test`
- `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter=plan-vector-dynamic-i32-vsub-and-export-target-artifact-bundle` from `build/test`
- `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter=plan-vector-dynamic-i32-vadd-and-export-target-artifact-bundle` from `build/test`
- `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter=vector-dynamic-i32-vsub-to-exec` from `build/test`
- `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter=rvv-microkernel-bundle-e2e` from `build/test`
- Manual vsub dry-run bundle probes for plan-and-export and two-step selected
  planning/export routes with runtime counts `7,16,23`.
- Manual fail-closed probe that stripped vsub selected source identity and
  observed `requires selected RVV binary source identity before target artifact
  export`.
- `git diff --check`

No `ssh rvv` evidence was run because this round changed production preflight,
runner plumbing, and lit evidence only; it did not change generated RVV
source/object semantics or make runtime correctness/performance claims.

### Status

[OK] Completed and archived; commit pending.


## Session 67: RVV vector frontend default artifact route

**Date**: 2026-05-14
**Task**: RVV vector frontend default artifact route
**Branch**: `main`

### Summary

Promoted the bounded dynamic MLIR vector i32 vadd/vsub artifact evidence from
explicit vector adapter surfaces to the production source RVV binary frontdoor.
The normal route now carries dynamic vector vadd and vsub through the shared
source-derived finite family construction, selected RVV materialization,
selected config/runtime length metadata, EmitC bundle export, and existing
RuntimeABICallablePlan/RVVScalarDispatch consumers.

### Main Changes

- Changed `lib/Transforms/LowerSourceRVVBinaryToExec.cpp` so dynamic vector
  vadd and vsub both construct the selected frontend lowering contract from the
  source-derived finite family contract, removing the vadd-only construction
  branch from the dynamic vector path.
- Updated `Passes.td` descriptions to make
  `--tcrv-lower-source-rvv-binary-to-exec` the documented default source
  frontdoor and keep explicit vector vadd/vsub passes as compatibility
  adapters.
- Updated `scripts/rvv_microkernel_e2e.py` vector frontend modes to run and
  report `tcrv-lower-source-rvv-binary-to-exec + tcrv-execution-planning-pipeline`.
- Updated `scripts/rvv_scalar_dispatch_e2e.py` plan-and-export bundle evidence
  to use the shared source RVV binary frontend path name.
- Updated focused artifact/script tests so dynamic vector vadd/vsub generated
  artifact evidence goes through the default source frontdoor, with explicit
  vsub retained only as compatibility coverage.

### Testing

- `cmake --build build --target TianChenRVTransforms TianChenRVRVVPlugin TianChenRVRVVTarget tcrv-opt tcrv-translate tianchenrv-rvv-extension-plugin-test tianchenrv-target-artifact-export-test -j2`
- `./build/bin/tianchenrv-rvv-extension-plugin-test`
- `./build/bin/tianchenrv-target-artifact-export-test`
- `python3 scripts/rvv_microkernel_e2e.py --self-test`
- `python3 scripts/rvv_scalar_dispatch_e2e.py --self-test`
- `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv Transforms/VectorToExec/vector-dynamic-i32-vadd-to-exec.mlir Transforms/VectorToExec/vector-dynamic-i32-vsub-to-exec.mlir Transforms/VectorToExec/vector-dynamic-i32-vsub-explicit-invalid.mlir Transforms/VectorToExec/vector-dynamic-i32-binary-invalid.mlir Target/TargetArtifactBundleExport/plan-vector-dynamic-i32-vadd-and-export-target-artifact-bundle.mlir Target/TargetArtifactBundleExport/plan-vector-dynamic-i32-vsub-and-export-target-artifact-bundle.mlir Scripts/rvv-microkernel-bundle-e2e.test Scripts/rvv-scalar-dispatch-bundle-e2e.test`
- Default vector vadd/vsub dry-run probes for direct microkernel bundle and
  plan-and-export bundle paths.
- RVVScalarDispatch vsub dry-run through the plan-and-export bundle front door.
- `ssh rvv` direct microkernel bundle evidence:
  `codex-default-vector-vsub-ssh` and `codex-default-vector-vadd-ssh`.
- `ssh rvv` RVVScalarDispatch bundle evidence:
  `codex-default-vector-vsub-dispatch-ssh`.
- Ref-scans for explicit-only vector routes, descriptor-only authority, and
  generic core family branches.
- `git diff --check`

All three `ssh rvv` runs returned `status: success` and `ssh_evidence: true`.
The claim scope is limited to the bounded generated artifact or dispatch route
actually run.

### Spec Update Judgment

No code-spec update was needed. This round did not add a new command signature,
payload field, dialect contract, or cross-layer schema; it rewired production
evidence to the already-specified source frontdoor, selected boundary,
EmitC-route, descriptor-quarantine, and runtime ABI contracts.

### Status

[OK] Completed and archived; commit pending.


## Session 60: RVV source-frontdoor artifact/runtime closure

**Date**: 2026-05-14
**Task**: RVV source-frontdoor artifact/runtime closure
**Branch**: `main`

### Summary

Extended RVV dispatch bundle selected source identity with dtype/operator/EmitC provenance for source-frontdoor i32 vadd/vsub, verified focused bundle tests plus local and ssh rvv artifact evidence.

### Main Changes

- Extended
  `RVVSelectedConfig::formatDispatchContractSelectedSourceIdentityMetadataValue`
  so RVV+scalar dispatch bundle route claims now include source kind, dtype,
  finite family, arithmetic operator, selected RVV microkernel op, selected
  EmitC source op, and `TCRVEmitCLowerableOpInterface`.
- Updated dynamic vector i32 vadd/vsub TargetArtifactBundleExport checks so
  generated bundle indexes must publish the complete typed source identity.
- Kept descriptor-only authority quarantined and did not add core `tcrv.exec`
  compute semantics, explicit-only frontend flags, or generic core family
  branches.

### Git Commits

Commit pending in this round.

### Testing

- `cmake --build build --target TianChenRVRVVTarget TianChenRVRVVPlugin tcrv-opt tcrv-translate tianchenrv-rvv-extension-plugin-test tianchenrv-target-artifact-export-test -j2`
- `./build/bin/tianchenrv-target-artifact-export-test`
- `./build/bin/tianchenrv-rvv-extension-plugin-test`
- `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv Target/TargetArtifactBundleExport/plan-vector-dynamic-i32-vadd-and-export-target-artifact-bundle.mlir Target/TargetArtifactBundleExport/plan-vector-dynamic-i32-vsub-and-export-target-artifact-bundle.mlir` from `build/test`
- `python3 scripts/rvv_microkernel_e2e.py --self-test`
- `python3 scripts/rvv_scalar_dispatch_e2e.py --self-test`
- Local `rvv_microkernel_e2e.py` plan-and-export dynamic vector source-frontdoor dry-runs for i32-vadd and i32-vsub, runtime counts `7,16,23`.
- Local `rvv_scalar_dispatch_e2e.py` plan-and-export dispatch bundle dry-runs for i32-vadd and i32-vsub, with generated bundle indexes carrying the new `tcrv_rvv.dispatch_contract_selected_source_identity` values.
- `ssh rvv` `rvv_microkernel_e2e.py` source-frontdoor bundle runs `codex-source-identity-vsub-ssh` and `codex-source-identity-vadd-ssh`, both `status: success` and `ssh_evidence: true`.
- Ref-scans for explicit-only route drift, descriptor quarantine, and generic `tcrv.exec` core neutrality.
- `git diff --check`
- Trellis task validation before finish.

### Self-Repair

The first scalar-dispatch dry-run used linalg bundle fixtures but asserted
dynamic-vector selected kernel names. The runner failed closed as expected; the
commands were rerun with `frontend_bundle_i32_vadd` and
`frontend_bundle_i32_vsub` and passed.

### Spec Update Judgment

No `.trellis/spec/` update was needed. The round did not add a new dialect op,
schema field, command signature, plugin protocol, or architecture rule; it
completed an existing selected-source identity contract at the generated
artifact/dispatch bundle surface.

### Status

[OK] **Completed**

### Next Steps

- None - task complete
