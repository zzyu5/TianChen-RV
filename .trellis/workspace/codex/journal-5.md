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
