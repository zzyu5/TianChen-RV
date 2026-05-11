# Journal - codex (Part 4)

> Continuation from `journal-3.md` (archived at ~2000 lines)
> Started: 2026-05-12

---



## Session 35: RVV selected emission descriptor exit

**Date**: 2026-05-12
**Task**: RVV selected emission descriptor exit
**Branch**: `main`

### Summary

Removed descriptor authority from RVV typed-source selected emission planning and kept typed body plus exec ABI as production authority.

### Main Changes

### Main Changes

- Rewired RVV selected emission/readiness so typed-source i32/i64 add/sub/mul paths no longer call descriptor-selected plan reconstruction.
- Added typed i64 microkernel control/dataflow validation in selected emission planning, matching the existing typed i32 selected body authority.
- Added post-typed-plan mirror checks for stale tcrv_rvv.lowering_descriptor and tcrv_rvv.element_count metadata.
- Added C++ coverage for descriptor-only i32/i64 selected emission/readiness fail-closed behavior and stale descriptor mirror rejection.
- Updated one lit diagnostic to the new typed-plan mirror mismatch wording.

### Testing

- artifacts/tmp/tianchenrv-build/bin/tianchenrv-rvv-extension-plugin-test
- artifacts/tmp/tianchenrv-build/bin/tianchenrv-rvv-selected-lowering-boundary-test
- artifacts/tmp/tianchenrv-build/bin/tianchenrv-runtime-abi-callable-plan-test
- artifacts/tmp/tianchenrv-build/bin/tianchenrv-target-artifact-export-test
- artifacts/tmp/tianchenrv-build/bin/tianchenrv-rvv-binary-planning-test
- focused lit filter: rvv-microkernel auto-materialization/family-mul/i32 descriptor mismatch/i64 vadd-vsub-vmul, 9/9 passed
- focused lit filter: rvv-microkernel-descriptor-element-mismatch-fails, 1/1 passed
- git diff --check
- cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2, 207/207 passed

### Status

[OK] Completed; no ssh rvv runtime, correctness, or performance claim was made.


### Git Commits

(No commits - planning session)

### Testing

- [OK] (Add test results)

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 36: Scalar selected emission descriptor exit

**Date**: 2026-05-12
**Task**: scalar-selected-emission-descriptor-exit
**Branch**: `main`

### Summary

Removed descriptor authority from scalar fallback selected emission/readiness
for the bounded i32/i64 add/sub/mul typed microkernel families.

### Main Changes

- Rewired `ScalarExtensionPlugin` selected emission/readiness to build
  supported scalar fallback plans from typed `tcrv_scalar` microkernel ops,
  common EmitC route metadata, and exec-IR callable ABI metadata.
- Quarantined `tcrv_scalar.lowering_descriptor` and
  `tcrv_scalar.element_count` as optional legacy mirror metadata after typed
  plan construction.
- Added C++ coverage for descriptorless typed-source success across bounded
  scalar fallback families, descriptor-only metadata-only fail-closed behavior,
  and stale descriptor mirror rejection.
- Updated RVV+scalar dispatch/export fixtures so scalar fallback authority
  comes from typed scalar ops and common EmitC route metadata.
- Updated affected lit fixtures for descriptorless scalar target source
  artifact routes and new mirror metadata diagnostics.

### Testing

- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-scalar-extension-plugin-test`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-target-artifact-export-test`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-runtime-abi-callable-plan-test`
- Focused lit: scalar lowering-boundary diagnostic, scalar target source
  artifact routes, RVV scalar dispatch generic route, and
  `rvv-scalar-dispatch-e2e.test`, 4/4 passed.
- Focused lit: `rvv-scalar-dispatch-bundle-e2e.test` and i64 target artifact
  bundle export routes, 4/4 passed.
- `git diff --check`
- `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-12-scalar-selected-emission-descriptor-exit`
- `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`,
  207/207 passed.

### Status

[OK] Completed and ready to archive. No `ssh rvv` runtime, correctness, or
performance claim was made.


## Session 38: I32 runtime ABI compatibility wrapper removal

**Date**: 2026-05-12
**Task**: remove-i32-vadd-runtime-abi-compat-wrappers
**Branch**: `main`

### Summary

Removed the obsolete add-only `I32VAdd*` runtime ABI compatibility wrapper API
surface after production owners had moved to the family-aware
`I32BinaryRuntimeABIContract`.

### Main Changes

- Deleted temporary `I32VAdd*` runtime ABI wrapper declarations from support
  headers and their definitions from support sources.
- Kept the real i32-vadd family registration, dialect, route, and microkernel
  support intact; only the runtime ABI compatibility API was removed.
- Added support test coverage proving add/sub/mul use the same family-aware
  callable ABI shape while preserving per-family RVV/scalar/dispatch ABI
  identities.
- Updated the lowering-runtime spec so it states the temporary add-only ABI
  wrappers are retired and active owners must use `I32Binary*` APIs directly.

### Testing

- Wrapper-name `rg` check over `include`, `lib`, `test`, and relevant spec
  files found no deleted compatibility wrapper names.
- `git diff --check`
- `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-12-remove-i32-vadd-runtime-abi-compat-wrappers`
- `cmake --build artifacts/tmp/tianchenrv-build --target tianchenrv-runtime-abi-callable-plan-test -j2`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-runtime-abi-callable-plan-test`
- `cmake --build artifacts/tmp/tianchenrv-build --target tianchenrv-target-artifact-export-test -j2`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-target-artifact-export-test`
- `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`,
  207/207 passed.

### Status

[OK] Completed and ready to archive. No `ssh rvv` runtime, correctness, or
performance claim was made.


## Session 38: finite descriptor registry quarantine

**Date**: 2026-05-12
**Task**: rvv-scalar-descriptor-registry-quarantine
**Branch**: `main`

### Summary

Quarantined finite RVV/scalar/i32/RVV+scalar descriptor registries as
registration, compatibility naming, and legacy mirror validation surfaces after
the selected EmitC route migration.

### Main Changes

- Renamed public registry lookup/getter surfaces from descriptor-shaped names
  to `RegistrationRecord`, `RegistrationBy...`, and
  `Legacy...Mirror` names across RVV, scalar, i32, and RVV+scalar dispatch
  boundaries.
- Retagged selected descriptor mirror metadata as
  `legacy-rvv-binary-descriptor-mirror` and
  `legacy-scalar-binary-descriptor-mirror`; typed RVV/scalar source metadata
  remains the production authority for default artifact export.
- Kept direct RVV descriptor-only planning and selected lowering-boundary paths
  fail-closed, with diagnostics now saying legacy-registration-only rather
  than descriptor fallback.
- Added target artifact export regression coverage that rejects legacy mirror
  roles when route preflight requires typed selected-plan metadata.
- Updated registry/planning/plugin/artifact-export/lit tests to assert
  registration/mirror scope and stale descriptor quarantine.

### Testing

- Focused C++ build and runs:
  `tianchenrv-target-artifact-export-test`,
  `tianchenrv-i32-binary-family-registry-test`,
  `tianchenrv-rvv-extension-plugin-test`,
  `tianchenrv-scalar-extension-plugin-test`,
  `tianchenrv-rvv-binary-planning-test`,
  `tianchenrv-runtime-abi-callable-plan-test`.
- Focused lit over direct RVV/scalar export, RVV+scalar dispatch, bundle,
  dry-run, and lowering-boundary quarantine routes, 62/62 passed after
  rebuilding `tcrv-opt`.
- `git diff --check`
- `git diff --cached --check`
- `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-12-rvv-scalar-descriptor-registry-quarantine`
- `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`,
  207/207 passed after updating one stale registry FileCheck expectation.

### Status

[OK] Completed and ready to archive. Python stayed tooling-only, `tcrv.exec`
stayed compute-free, and no `ssh rvv` runtime, correctness, or performance
claim was made.


## Session 37: RVV+scalar dispatch descriptor exit

**Date**: 2026-05-12
**Task**: rvv-scalar-dispatch-descriptor-exit
**Branch**: `main`

### Summary

Removed descriptor authority from the default RVV+scalar dispatch composite
bundle identity path for the bounded selected-component route.

### Main Changes

- Added a route-local composite bundle metadata callback to
  `TargetArtifactCompositeExporter` and let bundle records take dispatch ABI,
  component group, and external ABI identity from selected component groups
  before static route fallback metadata.
- Rewired RVV+scalar dispatch composite identity so dispatch function stem,
  header guard stem, self-check marker, runtime ABI name, component group, and
  external ABI name are derived from selected RVV/scalar
  `selected_binary_family` plan metadata.
- Kept finite descriptors as route registration and mismatch validation
  metadata only; stale selected component metadata now fails closed before
  bundle metadata export can proceed.
- Updated target/export and i32 registry tests to require route-local runtime
  ABI, bundle metadata, and candidate preflight callbacks.

### Testing

- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-target-artifact-export-test`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-rvv-extension-plugin-test`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-scalar-extension-plugin-test`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-i32-binary-family-registry-test`
- Focused lit filter:
  `Target/RVVScalarDispatch|rvv-scalar-dispatch-e2e|rvv-scalar-dispatch-bundle-e2e`,
  15/15 passed.
- `git diff --check`
- `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`,
  207/207 passed.

### Status

[OK] Completed and ready to archive. No `ssh rvv` runtime, correctness, or
performance claim was made.
