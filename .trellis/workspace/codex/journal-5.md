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
