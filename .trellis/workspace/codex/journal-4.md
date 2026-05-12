# Journal - codex (Part 4)

> Continuation from `journal-3.md` (archived at ~2000 lines)
> Started: 2026-05-12

---

## Session 40: RVV selected-boundary descriptor mirror quarantine

**Date**: 2026-05-12
**Task**: rvv-selected-boundary-descriptor-mirror-quarantine
**Branch**: `main`

### Summary

Quarantined the remaining RVV selected-boundary proposal descriptor mirror
surface so default typed RVV family/body proposal planning no longer exposes or
uses a lowering-descriptor attach switch.

### Main Changes

- Removed `attachLoweringDescriptorAttr`, `shouldAttachLoweringDescriptorAttr`,
  and RVV plugin proposal/selected-plan `getLoweringDescriptor()` APIs.
- Removed the RVV extension plugin branch that could attach
  `tcrv_rvv.lowering_descriptor` to default proposal metadata.
- Renamed RVV target/selected-config descriptor access consumed by RVV plugin
  code to `getLegacyLoweringDescriptorMirror()`.
- Updated RVV planning/registry tests to assert typed family, source-kind,
  selected shape, and capability authority.
- Updated the RVV plugin spec to state default typed RVV proposals must not
  emit `tcrv_rvv.lowering_descriptor`.

### Testing

- Focused C++ builds:
  `tianchenrv-rvv-extension-plugin-test`,
  `tianchenrv-rvv-binary-planning-test`,
  `tianchenrv-target-artifact-export-test`,
  `tianchenrv-i32-binary-family-registry-test`,
  `tianchenrv-rvv-selected-lowering-boundary-test`,
  `tianchenrv-rvv-binary-variant-legality-test`, `tcrv-opt`,
  `tcrv-translate`.
- Focused C++ test binaries above all passed.
- Focused lit filter over Plugin/RVV, LoweringBoundary, LinalgToExec,
  RVVMicrokernel, RVVScalarDispatch, and TargetArtifactBundleExport passed
  87/87 selected tests.
- Ref-scan found no remaining RVV proposal attach/getter API; remaining
  descriptor references are target registration, explicit legacy mirror
  validation, stale-mirror negative tests, or quarantine tests.
- `git diff --check` passed.
- `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`,
  209/209 passed.

### Status

[OK] Completed. Default RVV selected-boundary proposal/materialization is
typed-family/body plus selected-config plus EmitC/source-authority driven;
remaining RVV descriptor references are legacy mirror/quarantine only. No
`ssh rvv` runtime, correctness, or performance claim was made.



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

| Hash | Message |
|------|---------|
| this commit | feat(rvv): require typed variant legality authority |

### Testing

- `cmake --build artifacts/tmp/tianchenrv-build --target tianchenrv-rvv-binary-variant-legality-test tianchenrv-rvv-binary-planning-test tianchenrv-rvv-selected-lowering-boundary-test tianchenrv-rvv-extension-plugin-test -j2`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-rvv-binary-variant-legality-test`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-rvv-binary-planning-test`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-rvv-selected-lowering-boundary-test`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-rvv-extension-plugin-test`
- `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'linalg-i32-vadd-to-exec|rvv.*artifact'`
- Targeted rerun of the 6 lit tests that first exposed stale descriptor/quarantine diagnostics.
- `git diff --check`
- `git diff --cached --check`
- `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-12-rvv-variant-legality-descriptor-exit`
- `python3 ./.trellis/scripts/task.py validate .trellis/tasks/archive/2026-05/05-12-rvv-variant-legality-descriptor-exit`
- `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`,
  209/209 passed.

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 47: RVV typed-family EmitC artifact authority

**Date**: 2026-05-13
**Task**: RVV typed-family EmitC artifact authority
**Branch**: `main`

### Summary

Created the Trellis task/PRD from the Hermes brief, verified that the current
RVV production source path already uses typed family-body authority plus
generated `TCRVEmitCLowerableOpInterface` provenance and the common EmitC
source-authority boundary, then pinned the remaining descriptor-only production
rejection with focused lit coverage.

### Main Changes

- Added a descriptor-only RVV i32 production-input regression under
  `test/Target/RVVMicrokernel/`.
- The regression exercises the marked direct route
  `tcrv-export-rvv-i32-vmul-microkernel-c` and checks that descriptor-only
  metadata fails during artifact-backed planning before exact target artifact
  export and before any RVV C source output.
- No spec update was needed because the existing specs already describe the
  typed-family authority, interface-backed EmitC provenance, and descriptor
  mirror boundary.

### Testing

- `cmake --build build --target TianChenRVTarget TianChenRVTransforms tcrv-translate tianchenrv-target-artifact-export-test -j2`
- `build/bin/tianchenrv-target-artifact-export-test`
- From `build/test`:
  `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'RVVMicrokernel|RVVScalarDispatch|target-source-artifact-routes|target-artifact-export-registry|emitc-lowerable|rvv-microkernel-descriptor-only-production-rejects'`
  with 46/46 selected tests passed.
- `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-13-rvv-typed-family-emitc-artifact-authority`

### Status

[OK] Completed. No `ssh rvv` runtime, correctness, or performance claim was
made.

### Next Steps

- None - task complete


## Session 43: RVV common EmitC source boundary production owner

**Date**: 2026-05-12
**Task**: RVV Common EmitC Source Boundary Production Owner
**Branch**: `main`

### Summary

Rewired RVV direct runtime-callable microkernel source emission so RVV target
code still builds the typed family EmitC lowerable route locally, while the
common conversion/EmitC boundary owns source-authority emission through
`lowerTCRVEmitCLowerableToEmitCSource`.

### Main Changes

- `RVVMicrokernel.cpp` now constructs `RVVBinaryEmitCLowerable` locally and
  calls the common lower-to-EmitC source boundary for both library source and
  self-check harness body emission.
- Removed direct RVV imports/usages of the lower-level source-authority emitter
  APIs.
- Added RVV generated-source comments and FileCheck coverage for
  `emitc_common_lower_to_emitc_boundary: TCRVLowerToEmitCSourceAuthority`.
- Preserved RVV-local selected vector shape, intrinsic names, dataflow plan,
  callable ABI mapping, route metadata, and harness wrapper ownership.
- Spec update judgment: no spec files changed because the existing EmitC route,
  RVV plugin, emission-runtime, and testing specs already describe this
  boundary.

### Testing

- `cmake --build build --target TianChenRVConversionEmitC TianChenRVRVVTarget tcrv-translate tianchenrv-target-artifact-export-test -j2`
- Focused lit filter for RVV microkernel, RVV artifact source export,
  `rvv-microkernel-e2e`, linalg-to-RVV artifact, module target profile, and
  RVV probe replay checks: 40/40 passed.
- `python3 scripts/rvv_microkernel_e2e.py --dry-run --arithmetic-family=i32-vmul --run-id codex-rvv-common-emitc-boundary-dry --overwrite`
- `python3 scripts/rvv_microkernel_e2e.py --self-check-harness --arithmetic-family=i32-vadd --run-id codex-rvv-common-emitc-boundary-ssh --overwrite`
- `git diff --check`
- `./build/bin/tianchenrv-target-artifact-export-test`
- `cmake --build build --target check-tianchenrv -j2`, 210/210 passed.
- `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-12-rvv-common-emitc-source-boundary-production-owner`

### Status

[OK] Completed. The real RVV claim is bounded to generated i32-vadd
self-check executable correctness only.

### Next Steps

- None - task complete


## Session 45: Direct target translate route artifact coherence

**Date**: 2026-05-12
**Task**: Direct target translate route artifact coherence
**Branch**: `main`

### Summary

Rewired emission-plan-backed direct `tcrv-translate` RVV and RVV+scalar
source/header/object routes onto the same generic target artifact exporter and
execution-plan coherence surface used by `--tcrv-export-target-*`.

### Main Changes

- Added an exact-route generic target artifact export API for registered
  standalone and composite route ids.
- Extended target translate route registration with optional target artifact
  route metadata and validation.
- Marked direct RVV microkernel and RVV+scalar dispatch source/header/object
  routes with their target artifact route ids while leaving self-check helper
  routes target-local.
- Updated `tcrv-translate` so marked direct routes with emission-plan
  diagnostics populate built-in registries, run execution-plan coherence, and
  invoke the generic exact-route exporter.
- Updated C++ and lit coverage for route metadata, exact-route diagnostics,
  coherence preflight, and target-local helper preservation.

### Testing

- `cmake --build build --target TianChenRVTarget tcrv-translate tianchenrv-target-artifact-export-test -j2`
- `build/bin/tianchenrv-target-artifact-export-test`
- From `build/test`:
  `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'RVVScalarDispatch|RVVMicrokernel|target-source-artifact-routes|target-artifact-export-registry'`
  with 45/45 selected tests passed.
- `git diff --check`
- `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-12-05-12-direct-target-translate-route-artifact-coherence`
- `python3 ./.trellis/scripts/task.py validate .trellis/tasks/archive/2026-05/05-12-05-12-direct-target-translate-route-artifact-coherence`

### Status

[OK] Completed. No `ssh rvv` runtime, correctness, or performance claim was
made.

### Next Steps

- None - task complete


## Session 44: RVV+scalar dispatch common EmitC source boundary

**Date**: 2026-05-12
**Task**: RVV Scalar Dispatch Common EmitC Source Boundary
**Branch**: `main`

### Summary

Rewired the RVV+scalar host dispatcher body source emission so the
target-owned dispatch-control route now crosses the common
`lowerTCRVEmitCLowerableToEmitCSource` boundary instead of calling the
lower-level EmitC source-authority materializer directly from
`RVVScalarDispatch.cpp`.

### Main Changes

- Added a target-local `DispatchControlEmitCLowerable` adapter over the
  existing dispatch route construction.
- Removed direct dispatch target imports/usages of
  `emitTCRVEmitCLowerableRouteAsCppSource` and
  `TCRVEmitCSourceAuthorityOptions`.
- Preserved dispatch-local authority for selected RVV/scalar callable symbols,
  ABI parameter bindings, runtime guard linkage, fallback target linkage,
  route ids, and the `tcrv.exec.case` / `tcrv.exec.fallback` call steps.
- Added generated-source metadata for
  `dispatch_emitc_common_lower_to_emitc_boundary:
  TCRVLowerToEmitCSourceAuthority`.
- Updated RVV+scalar dispatch FileCheck coverage and the dry-run evidence
  helper validator to require the dispatch common-boundary marker.

### Testing

- `cmake --build build --target TianChenRVConversionEmitC TianChenRVBuiltinTargetArtifactExporters tcrv-translate tcrv-opt -j2`
- Focused lit filter for RVV+scalar dispatch direct/generic routes, dispatch
  script tests, and bundle/front-door tests: 12/12 passed after one FileCheck
  ordering repair.
- `python3 scripts/rvv_scalar_dispatch_e2e.py --dry-run --arithmetic-family=i32-vadd --run-id codex-dispatch-common-emitc-boundary-dry --overwrite`
- `python3 scripts/rvv_scalar_dispatch_e2e.py --dry-run --use-target-artifact-bundle --use-plan-and-export-bundle-front-door --arithmetic-family=i32-vmul --run-id codex-dispatch-common-emitc-boundary-bundle-dry --overwrite`
- `git diff --check`
- `./build/bin/tianchenrv-target-artifact-export-test`
- `cmake --build build --target check-tianchenrv -j2`, 210/210 passed.
- `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-12-05-12-rvv-scalar-dispatch-common-emitc-source-boundary`

### Status

[OK] Completed. No `ssh rvv` runtime, correctness, or performance claim was
made; dry-run evidence is local compiler artifact validation only.

### Next Steps

- None - task complete


## Session 39: Descriptor mirror retirement from production source-authority surfaces

**Date**: 2026-05-12
**Task**: descriptor-mirror-retirement-from-production-surfaces
**Branch**: `main`

### Summary

Removed descriptor/registration-record shaped authority from the Support
runtime ABI callable planning API. Production RVV, scalar fallback,
RVV+scalar dispatch, and offload call sites now pass selected-family runtime ABI
contracts into Support planning while descriptor mirrors remain target-local
legacy validation data.

### Main Changes

- Added Support-owned `FiniteBinaryRuntimeABIContract` and made the i32 shared
  runtime ABI contract selected-family-id keyed rather than descriptor-record
  keyed.
- Removed target descriptor and i32 family registry includes plus
  descriptor-shaped callable-plan overloads from Support callable planning.
- Reworked target-owned RVV runtime ABI contracts to derive from the generic
  Support finite-binary contract.
- Updated RVV, scalar, dispatch, and offload production call sites to adapt
  selected typed family registration data before invoking Support planning.
- Updated Support and target artifact C++ tests plus the lowering/runtime spec
  to enforce descriptorless Support callable planning.

### Testing

- Focused builds for Support, RVV target, scalar target, offload target,
  built-in target artifact exporters, `tcrv-opt`, and `tcrv-translate`.
- C++ tests:
  `tianchenrv-runtime-abi-callable-plan-test`,
  `tianchenrv-target-artifact-export-test`,
  `tianchenrv-rvv-extension-plugin-test`,
  `tianchenrv-scalar-extension-plugin-test`,
  `tianchenrv-rvv-binary-planning-test`.
- Focused lit filter:
  `Target/ArtifactExport|Target/TargetArtifactBundleExport|Target/RVVScalarDispatch|Transforms/LinalgToExec`,
  50/50 passed.
- Dry-run bundle validations for direct `i32-vmul` RVV and RVV+scalar dispatch
  through `--use-target-artifact-bundle --use-plan-and-export-bundle-front-door`.
- `git diff --check`.
- Trellis task validation.
- `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`,
  209/209 passed.

### Status

[OK] Completed and ready to archive. Production/default paths are descriptorless
at the Support callable-planning API, common EmitC remains the source route for
RVV/scalar/dispatch, `tcrv.exec` stayed compute-free, Python stayed
tooling-only, and no `ssh rvv` runtime, correctness, or performance claim was
made.


## Session 38: Scalar fallback EmitC source authority production route

**Date**: 2026-05-12
**Task**: Scalar fallback EmitC source authority production route
**Branch**: `main`

### Summary

Scalar fallback runtime-callable source now uses the MLIR EmitC / MLIR Cpp
emitter source authority for its production callable body, and RVV+scalar
dispatch consumes that migrated scalar component.

### Main Changes

- Extended the common `TCRVEmitCLowerableRoute` source-authority materializer
  with a runtime-element-count loop shape for scalar routes, using EmitC
  subscript/load/apply/call_opaque/call operations rather than the legacy
  route-to-C renderer.
- Rewired `lib/Target/Scalar/ScalarMicrokernel.cpp` so scalar production source
  calls `emitTCRVEmitCLowerableRouteAsCppSource`.
- Kept the legacy renderer as an explicitly named diagnostic compatibility
  helper; scalar and RVV production target code no longer call it.
- Updated scalar, RVV+scalar dispatch, target artifact bundle, LinalgToExec,
  ExecutionPlanning, and e2e script fixtures to assert MLIR Cpp emitter source
  authority and typed scalar source-op provenance instead of old scalar loop
  spelling.
- No fresh `ssh rvv` run was performed, so no new RVV runtime, correctness, or
  performance claim was made.

### Git Commits

| Hash | Message |
|------|---------|
| this commit | feat(scalar): use emitc source authority for fallback source |

### Testing

- `cmake --build artifacts/tmp/tianchenrv-build --target TianChenRVConversionEmitC TianChenRVScalarTarget -j2`
- `cmake --build artifacts/tmp/tianchenrv-build --target tianchenrv-emitc-lowerable-interface-test -j2`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-emitc-lowerable-interface-test`
- `cmake --build artifacts/tmp/tianchenrv-build --target TianChenRVConversionEmitC TianChenRVScalarTarget TianChenRVBuiltinTargetArtifactExporters tianchenrv-target-artifact-export-test tcrv-translate -j2`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-target-artifact-export-test`
- Focused lit for scalar source, RVV+scalar dispatch, target artifact bundles,
  LinalgToExec, ExecutionPlanning, and RVV+scalar e2e scripts.
- `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`,
  209/209 passed.
- `git diff --check`
- `git diff --cached --check`
- `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-12-scalar-emitc-source-authority-production-route`
- `python3 ./.trellis/scripts/task.py validate .trellis/tasks/archive/2026-05/05-12-scalar-emitc-source-authority-production-route`

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 41: EmitC source authority production route

**Date**: 2026-05-12
**Task**: EmitC source authority production route
**Branch**: `main`

### Summary

The bounded direct RVV runtime-callable source export now uses a verified MLIR
EmitC module translated by `mlir::emitc::translateToCpp` as the production
source authority instead of the custom route-to-C renderer.

### Main Changes

- Added a conversion / EmitC source authority adapter that materializes the
  runtime-AVL-to-VL route as EmitC functions, EmitC control flow, route
  `call_opaque` steps, and source-op provenance comments.
- Rewired the direct RVV microkernel source export to call the MLIR-backed
  source authority for generated source bytes while preserving clang as the
  default native compiler path.
- Renamed the custom route renderer as a legacy diagnostic compatibility path;
  scalar compatibility can still use it, but the bounded RVV default route no
  longer does.
- Updated C++ and lit coverage across RVV source, artifact export, dispatch,
  bundle, script, and transform fixtures to assert MLIR EmitC / MLIR Cpp
  emitter authority rather than old handwritten loop spelling.

### Testing

- Focused changed-owner builds for `TianChenRVConversionEmitC`,
  `TianChenRVRVVTarget`, `TianChenRVScalarTarget`,
  `tianchenrv-target-artifact-export-test`, and `tcrv-translate`.
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-emitc-lowerable-interface-test`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-target-artifact-export-test`
- Focused and broader lit filters for RVV source/export, dispatch, bundle,
  script, and transform fixtures.
- `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`,
  209/209 passed.

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 39: RVV+scalar dispatch selected-plan route authority

**Date**: 2026-05-12
**Task**: rvv-scalar-dispatch-selected-plan-route-authority
**Branch**: `main`

### Summary

Rewired RVV+scalar dispatch composite candidate family authority so selected
RVV/scalar component plans choose the finite family before descriptor-derived
route registration records are consulted.

### Main Changes

- Changed RVV+scalar dispatch candidate collection and composite matching to
  read `tcrv_rvv.selected_binary_family` and
  `tcrv_scalar.selected_binary_family` selected-plan metadata before checking
  finite route ids, emission kinds, runtime ABI names, runtime glue roles, and
  artifact kinds.
- Kept finite `RVVScalarBinaryFamilyDescriptor` /
  `DispatchBinaryFamilyDescriptor` data as route-registration and stale-mirror
  validation metadata only.
- Preserved non-RVV/scalar composite route matching by using route ids only as
  a relevance filter before selected-plan authority checks.
- Added C++ target artifact export coverage for missing selected family
  metadata and stale scalar route data after selected-plan family authority is
  established.
- Updated the i32-vsub i32m2 RVV+scalar dispatch lit stale-family diagnostic
  to the new selected-plan-first route mismatch.

### Testing

- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-target-artifact-export-test`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-i32-binary-family-registry-test`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-scalar-extension-plugin-test`
- Focused lit filter:
  `RVVScalarDispatch|rvv-scalar-dispatch-e2e|rvv-scalar-dispatch-bundle-e2e`,
  15/15 passed.
- Focused lit filter: `target-artifact-bundle-guards`, 1/1 passed after
  matcher relevance filtering was restored for non-RVV/scalar routes.
- `python3 scripts/rvv_scalar_dispatch_e2e.py --dry-run --arithmetic-family=i32-vmul --lower-linalg-frontend --run-id codex-selected-plan-route-authority-vmul --overwrite --input test/Target/RVVScalarDispatch/rvv-scalar-i32-vmul-dispatch-generic-route.mlir`
- `git diff --check`
- `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-12-rvv-scalar-dispatch-selected-plan-route-authority`
- `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`,
  209/209 passed.

### Status

[OK] Completed and ready to archive. Python stayed tooling-only,
`tcrv.exec` stayed compute-free, extension details stayed plugin/target-local,
and no `ssh rvv` runtime, correctness, or performance claim was made.


## Session 40: RVV selected variant descriptor attribute exit

**Date**: 2026-05-12
**Task**: rvv-selected-variant-descriptor-attr-exit
**Branch**: `main`

### Summary

Removed the remaining implicit default selected-family legality path that let a
descriptorless RVV selected variant with only `tcrv_rvv.element_count` stand in
for typed i32-vadd authority. Default selected RVV binary identity now needs
typed selected-source metadata, kernel frontend authority, or an actual typed
RVV microkernel body; descriptor metadata remains legacy mirror/quarantine only.

### Main Changes

- Removed the implicit element-count-only i32-vadd fallback from RVV binary
  variant legality.
- Added a negative C++ legality case proving descriptorless
  `element_count`-only metadata no longer establishes selected-family
  authority.
- Updated the default selected-lowering-boundary fixture to carry typed
  selected-source metadata.
- Updated RVV direct and RVV+scalar e2e tooling samples to validate typed
  selected-family/EmitC authority instead of selected descriptor role names.
- Recorded PRD completion notes and inventory classification in the Trellis
  task.

### Testing

- `cmake --build artifacts/tmp/tianchenrv-build --target tianchenrv-rvv-binary-variant-legality-test tianchenrv-rvv-selected-lowering-boundary-test -j2`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-rvv-binary-variant-legality-test`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-rvv-selected-lowering-boundary-test`
- `python3 scripts/rvv_microkernel_e2e.py --self-test`
- `python3 scripts/rvv_scalar_dispatch_e2e.py --self-test`
- `cmake --build artifacts/tmp/tianchenrv-build --target TianChenRVRVVPlugin TianChenRVRVVTarget TianChenRVBuiltinTargetArtifactExporters tcrv-opt tcrv-translate tianchenrv-rvv-binary-planning-test tianchenrv-rvv-binary-variant-legality-test tianchenrv-rvv-selected-lowering-boundary-test tianchenrv-rvv-extension-plugin-test tianchenrv-target-artifact-export-test -j2`
- Focused C++ tests for RVV planning, legality, selected lowering boundary,
  extension plugin, and target artifact export.
- Focused lit filters for RVV microkernel, RVV+scalar dispatch, and
  frontend-to-artifact descriptor absence/quarantine paths, 15/15 passed.
- `git diff --check`
- `git diff --cached --check`
- `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`,
  209/209 passed.

### Status

[OK] Completed. No `ssh rvv` runtime, correctness, or performance claim was
made.


## Session 38: RVV self-check typed authority descriptor exit

**Date**: 2026-05-12
**Task**: rvv-self-check-typed-authority
**Branch**: `main`

### Summary

Removed descriptor compute authority from the touched direct RVV and
RVV+scalar dispatch self-check default artifact paths.

### Main Changes

- Added a shared typed self-check expectation boundary that derives the scalar
  element C type from IR-backed runtime ABI pointer roles and formats expected
  arithmetic from typed family/dataflow authority.
- Rewired direct RVV microkernel self-check generation to use verified typed
  dataflow/body provenance plus generated EmitC route and ABI authority instead
  of descriptor `getScalarCType()` or `getCArithmeticCheckExpression()`.
- Rewired RVV+scalar dispatch self-check generation to use validated selected
  RVV/scalar component family authority plus shared dispatch ABI instead of
  descriptor compute helpers.
- Updated direct RVV and dispatch lit tests, including a stale descriptor mirror
  negative case for direct RVV self-check export.

### Testing

- `git diff --check`
- Focused build:
  `cmake --build artifacts/tmp/tianchenrv-build --target TianChenRVRVVTarget TianChenRVBuiltinTargetArtifactExporters TianChenRVScalarTarget tcrv-translate tcrv-opt tianchenrv-target-artifact-export-test -j2`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-target-artifact-export-test`
- Focused lit filter:
  `rvv-microkernel-(family-sub|pipeline)|rvv-scalar-i32-vadd-dispatch-c|rvv-scalar-i32-vsub-dispatch-generic-route|rvv-scalar-i64-vmul-dispatch-generic-route`,
  6/6 passed.
- Focused frontend-to-artifact lit filter: `linalg-i32-vadd-to-exec`, 2/2
  passed.
- `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-12-rvv-self-check-typed-authority`
- `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`,
  209/209 passed.

### Status

[OK] Completed and ready to archive. Python stayed tooling-only, `tcrv.exec`
stayed compute-free, and no `ssh rvv` runtime, correctness, or performance
claim was made.


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


## Session 39: RVV descriptor compute API exit

**Date**: 2026-05-12
**Task**: rvv-descriptor-compute-api-exit
**Branch**: `main`

### Summary

Quarantined descriptor-owned arithmetic/type helper authority from the live RVV,
scalar fallback, and RVV+scalar default artifact/export path. Generated
compute/self-check/provenance now uses typed family-op authority, the common
EmitC route, and IR-backed runtime ABI roles instead of descriptor compute
helpers.

### Main Changes

- Removed public descriptor compute helpers and mirror fields:
  `getCArithmeticCheckExpression`, descriptor `getCOperator`,
  descriptor `getScalarCType`, `cOperator`, and `scalarCType`.
- Removed default generated `arithmetic_c_operator` comments and replaced them
  with typed `arithmetic_source` provenance.
- Removed default `selected_binary_config` `lowering_descriptor` comments from
  generated dispatch summaries.
- Rewired scalar fallback helper C element type derivation to runtime ABI
  pointer roles instead of descriptor scalar type metadata.
- Rewired RVV+scalar dispatch self-check arithmetic to selected typed component
  family authority instead of suffix/descriptor-derived compute metadata.
- Updated direct e2e evidence vocabulary from descriptor `c_operator` to typed
  `arithmetic_token`.

### Testing

- `git diff --check`
- `git diff --cached --check`
- Focused source search over `include`, `lib`, `test`, and
  `scripts/rvv_microkernel_e2e.py`: no live hits for descriptor compute helper
  names or default descriptor arithmetic/provenance vocabulary.
- Focused build targets:
  `TianChenRVRVVTarget`, `TianChenRVScalarTarget`,
  `TianChenRVBuiltinTargetArtifactExporters`, `tcrv-translate`, `tcrv-opt`,
  `tianchenrv-target-artifact-export-test`, and
  `tianchenrv-i32-binary-family-registry-test`.
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-target-artifact-export-test`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-i32-binary-family-registry-test`
- `python3 scripts/rvv_microkernel_e2e.py --self-test`
- Focused lit filter for direct RVV, scalar fallback, RVV+scalar dispatch, and
  frontend-to-artifact paths, 13/13 passed.
- `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-12-rvv-descriptor-compute-api-exit`
- `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`,
  209/209 passed.

### Status

[OK] Completed and ready to archive. No `ssh rvv` runtime, correctness, or
performance claim was made.


## Session 38: RVV plugin proposal planning descriptor exit

**Date**: 2026-05-12
**Task**: RVV plugin proposal planning descriptor exit
**Branch**: `main`

### Summary

Removed legacy `tcrv_rvv.lowering_descriptor` authority from direct RVV binary
proposal planning. Direct proposal identity now comes from typed RVV frontend
family/body authority; descriptor text is accepted only as post-authority mirror
metadata.

### Main Changes

- Replaced the direct legacy descriptor candidate route in
  `resolveRVVBinaryFamilyForProposal` with typed-body-first resolution plus
  descriptor mirror validation.
- Descriptor-only direct RVV planning now fails closed as a recoverable proposal
  decline and cannot select a supported RVV binary proposal plan.
- Stale descriptor mirrors beside typed bodies now fail with diagnostics naming
  typed RVV microkernel body authority.
- Updated RVV plugin tests to cover descriptor-only rejection, matching mirror
  acceptance after typed authority, and stale mirror rejection.
- Updated the RVV plugin spec to record descriptor text as mirror metadata after
  typed authority, not plugin planning authority.

### Testing

- `cmake --build artifacts/tmp/tianchenrv-build --target tianchenrv-rvv-binary-planning-test -j2`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-rvv-binary-planning-test`
- `cmake --build artifacts/tmp/tianchenrv-build --target tianchenrv-rvv-extension-plugin-test -j2`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-rvv-extension-plugin-test`
- `cmake --build artifacts/tmp/tianchenrv-build --target tianchenrv-rvv-selected-lowering-boundary-test tianchenrv-rvv-binary-variant-legality-test -j2`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-rvv-selected-lowering-boundary-test`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-rvv-binary-variant-legality-test`
- Focused lit filters for RVV linalg artifact and RVV microkernel paths, 15/15
  and 8/8 passed.
- `git diff --check`
- `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`,
  209/209 passed.

### Status

[OK] Completed and ready to archive. No `ssh rvv` runtime, correctness, or
performance claim was made.


## Session 39: RVV i64 binary descriptor-exit closure

**Date**: 2026-05-12
**Task**: rvv-i64-binary-descriptor-exit-closure
**Branch**: `main`

### Summary

Closed a real RVV descriptor-driven materialization escape hatch for the
bounded i64-vsub/i64-vmul default route. Descriptor-only finite RVV binary
variants can no longer build selected microkernel materialization plans; the
selected lowering boundary now requires typed RVV microkernel authority or the
descriptorless typed-family frontend/default path.

### Main Changes

- Removed the public descriptor-to-selected-plan/materialization helper chain:
  `buildRVVBinarySelectedPlanFromVariant`,
  `buildRVVBinaryMicrokernelMaterializationPlanFromVariant`, and
  `buildRVVBinarySelectedMicrokernelMaterializationPlan`.
- Rewired RVV selected lowering-boundary materialization so
  `tcrv_rvv.lowering_descriptor` without a typed microkernel attachment fails
  closed as legacy-registration-only before any source/export path can use it.
- Replaced descriptor-derived materialization validation in RVV variant
  legality with bounded legacy descriptor mirror validation.
- Added i64-vsub and i64-vmul lit coverage for descriptor-only selected
  lowering-boundary quarantine.

### Testing

- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-rvv-binary-planning-test`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-rvv-selected-lowering-boundary-test`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-rvv-binary-variant-legality-test`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-rvv-extension-plugin-test`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-scalar-extension-plugin-test`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-target-artifact-export-test`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-runtime-abi-callable-plan-test`
- Focused lit filter for i64-vsub/i64-vmul lowering-boundary quarantine,
  frontend/direct RVV export, microkernel export, RVV+scalar dispatch export,
  and bundle export: 10/10 passed.
- `git diff --check`
- `git diff --cached --check`
- `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`,
  209/209 passed.

### Status

[OK] Completed and ready to archive. Descriptor authority remains absent from
the default i64-vsub/i64-vmul selected RVV/scalar production route; no `ssh rvv`
runtime, correctness, or performance claim was made.


## Session 38: RVV i64-vadd default EmitC route

**Date**: 2026-05-12
**Task**: rvv-i64-vadd-default-emitc-route
**Branch**: `main`

### Summary

Validated the bounded i64-vadd production/default route as typed frontend/body
identity -> RVV/scalar selected family ops -> common EmitC/export route, with
descriptors limited to optional legacy mirror/cross-check metadata.

### Main Changes

- Created the Trellis task and PRD for the i64-vadd default EmitC route.
- Confirmed current HEAD already carries i64-vadd through typed frontend
  inference, descriptorless RVV/scalar selected plans, typed
  `tcrv_rvv.i64_vadd_microkernel` / `tcrv_scalar.i64_vadd_microkernel`
  materialization, and selected-plan metadata consumed by direct and dispatch
  export routes.
- Added explicit i64-vadd frontend negative coverage for stale
  `tcrv_rvv.lowering_descriptor`, `tcrv_scalar.lowering_descriptor`, and
  `selected_lowering_descriptor` input metadata.

### Testing

- `artifacts/tmp/tianchenrv-build/bin/tcrv-opt test/Transforms/LinalgToExec/linalg-finite-binary-frontend-invalid.mlir --split-input-file --verify-diagnostics --tcrv-lower-linalg-rvv-binary-to-exec`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-rvv-binary-planning-test`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-rvv-selected-lowering-boundary-test`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-scalar-extension-plugin-test`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-target-artifact-export-test`
- `git diff --check`
- `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-12-rvv-i64-vadd-default-emitc-route`
- `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`,
  207/207 passed.

### Status

[OK] Completed and ready to archive. Descriptor authority is absent from the
default i64-vadd selected RVV/scalar production route; no `ssh rvv` runtime,
correctness, or performance claim was made.


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


## Session 36: RVV i64 target export body authority

**Date**: 2026-05-12
**Task**: RVV i64 target export body authority
**Branch**: `main`

### Summary

Made RVV i64 target artifact export resolve typed microkernel bodies before
legacy descriptor mirrors, with focused C++/lit coverage and the Trellis task
archived.

### Main Changes

- Rewired `resolveSelectedI64FamilyForPath` so the selected
  `tcrv_rvv.i64_vadd_microkernel`, `tcrv_rvv.i64_vsub_microkernel`, or
  `tcrv_rvv.i64_vmul_microkernel` body is discovered before any legacy
  descriptor mirror.
- Added fail-closed target/export diagnostics for stale
  `tcrv_rvv.lowering_descriptor` mirrors and descriptor-only i64 export.
- Added focused target artifact export C++ coverage for matching mirror
  acceptance, stale mirror rejection, and descriptor-only rejection.
- Added i64 vsub/vmul lit coverage for matching and stale descriptor mirrors
  through `tcrv-opt | tcrv-translate`, and updated the existing i64 vadd
  stale-descriptor diagnostic expectation.
- Recorded the i64 source/header/object body-authority rule in
  `.trellis/spec/lowering-runtime/emission-runtime-contract.md`.

### Git Commits

| Hash | Message |
|------|---------|
| this commit | feat(rvv): make i64 target export body authoritative |

### Testing

- `cmake --build artifacts/tmp/tianchenrv-build --target tianchenrv-target-artifact-export-test -j2`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-target-artifact-export-test`
- Focused lit for RVV i64 vadd/vsub/vmul microkernel export, i64 vsub/vmul
  RVV+scalar dispatch, i64 vsub/vmul target artifact bundle, and i64 vadd
  linalg/export stale-descriptor diagnostics.
- `git diff --check`
- `git diff --cached --check`
- `python3 ./.trellis/scripts/task.py validate .trellis/tasks/archive/2026-05/05-12-rvv-i64-target-export-body-authority`
- `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`,
  209/209 passed.

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 37: RVV variant legality descriptor exit

**Date**: 2026-05-12
**Task**: RVV variant legality descriptor exit
**Branch**: `main`

### Summary

RVV finite binary legality now requires typed RVV body or selected-source authority before descriptor mirrors; descriptor-only direct variants fail closed.

### Main Changes

- Rewired RVV variant legality to resolve typed body, selected-source metadata, frontend lowering authority, or the existing descriptorless default i32 route before accepting finite RVV binary metadata.
- Added proposal selected binary dtype, family, operator, and source-kind metadata, with planning preserving existing source kind across pipeline reruns.
- Updated RVV legality/plugin/lit tests for descriptor-only i32 and i64 rejection, stale descriptor mirror failures, typed i32/i64 legality, and smoke-probe separation.
- Checks passed: focused RVV C++ tests, focused RVV artifact lit filter, targeted failed lit files, git diff checks, Trellis validate before and after archive, and full check-tianchenrv.


### Git Commits

(No commits - planning session)

### Testing

- [OK] (Add test results)

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 38: RVV+scalar dispatch EmitC source authority production route

**Date**: 2026-05-12
**Task**: RVV+scalar dispatch EmitC source authority production route
**Branch**: `main`

### Summary

Rewired the production RVV+scalar dispatch wrapper to build a verified MLIR EmitC dispatch-control route from tcrv.exec dispatch/case/fallback, selected component callable plans, and ordered runtime ABI parameters. Preserved RVV/scalar component source authority in their plugin/target routes, updated fail-closed source/artifact/script coverage, verified dry-run and bounded ssh rvv i32-vmul dispatch slice, archived the Trellis task.

### Main Changes

- Added a dispatch-control shape to the common EmitC lowerable materializer: runtime guard compare, `emitc.if`, selected case call, early return, and fallback call.
- Rewired `RVVScalarDispatch` source export so the production dispatcher function is emitted by `mlir::emitc::translateToCpp` from selected `tcrv.exec.dispatch`/`case`/`fallback`, validated component callable plans, and ABI-ordered parameters.
- Preserved RVV and scalar component bodies in their existing plugin/target-owned EmitC source-authority routes; the dispatch route emits only runtime control glue.
- Added fail-closed coverage for malformed guard/runtime ABI facts, stale RVV/scalar component routes, route-family mismatch, and script/bundle stale selected-family checks.
- Updated lit/FileCheck expectations across direct dispatch, target artifact bundle, generic artifact export, and linalg front-door tests for the new EmitC guard compare and dispatch provenance.

### Git Commits

| Hash | Message |
|------|---------|
| `this commit` | (see git log) |

### Testing

- `cmake --build artifacts/tmp/tianchenrv-build --target TianChenRVConversionEmitC TianChenRVBuiltinTargetArtifactExporters tcrv-translate tianchenrv-emitc-lowerable-interface-test tianchenrv-target-artifact-export-test -j2`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-emitc-lowerable-interface-test`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-target-artifact-export-test`
- `python3 scripts/rvv_scalar_dispatch_e2e.py --self-test`
- Focused lit: `Target/RVVScalarDispatch`, `Target/TargetArtifactBundleExport`, `Scripts/rvv-scalar-dispatch-e2e.test`, `Scripts/rvv-scalar-dispatch-bundle-e2e.test`, plus the two full-check fallout tests.
- Dry-run bundle validation: `i32-vmul` through `--use-target-artifact-bundle --use-plan-and-export-bundle-front-door`.
- Bounded `ssh rvv` validation: `i32-vmul` target-artifact bundle external caller correctness only.
- `git diff --check`, `git diff --cached --check`, Trellis task validation before and after archive, and `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2` with 209/209 passed.

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 39: Scalar selected-boundary descriptor-only quarantine

**Date**: 2026-05-12
**Task**: Scalar selected-boundary descriptor-only quarantine
**Branch**: `main`

### Summary

Quarantined scalar fallback descriptor-only selected-boundary materialization while preserving descriptorless typed default and explicit typed scalar microkernel source-export paths.

### Main Changes

- Rewired `ScalarExtensionPlugin::materializeSelectedLoweringBoundary` so a selected scalar fallback variant carrying only legacy `tcrv_scalar.lowering_descriptor` and/or `tcrv_scalar.element_count` metadata fails before `tcrv_scalar.lowering_boundary` creation.
- Preserved descriptorless default typed scalar materialization for finite i32/i64 add/sub/mul families and explicit typed scalar microkernel paths.
- Kept legacy scalar descriptor metadata as optional mirror data only after typed scalar microkernel authority exists; stale mirrors still fail as mirror mismatches.
- Added C++ regression coverage for descriptor-only boundary rejection across finite scalar families and lit coverage for a valid scalar i32-vadd descriptor-only selected-boundary quarantine fixture.
- Updated scalar fallback and emission-runtime specs so descriptor-only scalar selected-boundary materialization is quarantined and typed scalar family ops remain the production authority.
- Checks passed: `tianchenrv-scalar-extension-plugin-test`, focused lit filter for scalar lowering-boundary/emission/artifact fixtures, `git diff --check`, Trellis task validation, and full `check-tianchenrv` with 210/210 tests passing.


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


## Session 40: EmitC source authority legacy renderer removal

**Date**: 2026-05-12
**Task**: EmitC Source Authority Owner
**Branch**: `main`

### Summary

Removed the remaining legacy route-to-C renderer from the conversion/EmitC
public API and implementation. The conversion library now exposes route
materialization to MLIR EmitC and MLIR Cpp source-authority emission, but no
longer offers a direct custom C renderer for `TCRVEmitCLowerableRoute`.

### Main Changes

- Removed `TCRVEmitCLegacyDiagnosticSourceRenderOptions` and
  `renderTCRVEmitCLowerableRouteAsLegacyDiagnosticCFunction` from
  `TCRVEmitCLowerableMaterializer.h`.
- Deleted the internal `LegacyDiagnosticRouteCSourceRenderer` implementation
  from `TCRVEmitCLowerableMaterializer.cpp`.
- Updated `tianchenrv-emitc-lowerable-interface-test` to cover only MLIR EmitC
  materialization and `mlir::emitc::translateToCpp` source-authority behavior.
- Kept RVV, scalar fallback, and RVV+scalar dispatch production source routes
  on the existing MLIR EmitC source-authority path; no runtime evidence claim
  was made.

### Testing

- `cmake --build artifacts/tmp/tianchenrv-build --target TianChenRVConversionEmitC tianchenrv-emitc-lowerable-interface-test -j2`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-emitc-lowerable-interface-test`
- From `artifacts/tmp/tianchenrv-build/test`:
  `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'target-source-artifact-routes|scalar-target-source-artifact-routes|scalar-target-vmul-source-artifact-routes|rvv-microkernel-family-(sub|mul)|rvv-scalar-i32-vmul-dispatch-generic-route'`
  with 8/8 selected tests passed.
- `rg -n "renderTCRVEmitCLowerableRouteAsLegacyDiagnosticCFunction|TCRVEmitCLegacyDiagnosticSourceRenderOptions|LegacyDiagnosticRouteCSourceRenderer|legacy diagnostic EmitC C source renderer" include lib test`
  returned no hits.
- `git diff --check`
- `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-12-emitc-source-authority-owner`

### Status

[OK] Completed. No `ssh rvv` runtime, correctness, or performance claim was
made.

### Next Steps

- None - task complete


## Session 42: Common extension lower-to-EmitC source boundary

**Date**: 2026-05-12
**Task**: Common Extension Interface + Lower-To-EmitC Pass Owner
**Branch**: `main`

### Summary

Added a common conversion/EmitC source-authority API and rewired scalar
fallback production source export through it. Scalar target code still owns the
family route mapping, callee names, ABI mapping, and validation, but the
route-to-EmitC-to-C++ source authority step now goes through the shared
conversion boundary.

### Main Changes

- Added `TCRVLowerToEmitCSourceResult`,
  `TCRVLowerToEmitCSourceOptions`, and
  `lowerTCRVEmitCLowerableToEmitCSource` in the conversion/EmitC library.
- The new API builds a verified `TCRVEmitCLowerableRoute` from
  `TCRVEmitCLowerableInterface`, emits through MLIR EmitC /
  `mlir::emitc::translateToCpp`, and fails if the emitted source lacks
  `tcrv_emitc.source_authority=mlir_emitc_cpp_emitter`.
- Rewired scalar runtime-callable source export to call the common boundary
  for production/default scalar source routes instead of calling the
  source-authority materializer directly from target-local code.
- Added C++ coverage for positive common-boundary lowering and fail-closed
  missing op-interface provenance / bad route verification.
- Updated scalar artifact and frontend lit expectations to assert the common
  lower-to-EmitC boundary marker.

### Testing

- `cmake --build build --target TianChenRVConversionEmitC tianchenrv-emitc-lowerable-interface-test TianChenRVScalarTarget tcrv-opt tcrv-translate tianchenrv-target-artifact-export-test -j2`
- `./build/bin/tianchenrv-emitc-lowerable-interface-test`
- `./build/bin/tianchenrv-target-artifact-export-test`
- From `build/test`:
  `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'linalg-i32-vmul-to-scalar-artifact|linalg-i32-vsub-to-scalar-artifact|scalar-target-source-artifact-routes|scalar-target-vmul-source-artifact-routes|target-source-artifact-routes'`
  with 5/5 selected tests passed.
- `git diff --check`
- `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-12-common-extension-lower-to-emitc-pass-owner`
- `python3 ./.trellis/scripts/task.py validate .trellis/tasks/archive/2026-05/05-12-common-extension-lower-to-emitc-pass-owner`
- `cmake --build build --target check-tianchenrv -j2`, 210/210 passed.

### Status

[OK] Completed. No `ssh rvv` runtime, correctness, or performance claim was
made.

### Next Steps

- None - task complete


## Session 46: Direct artifact route planning authority

**Date**: 2026-05-12
**Task**: Direct artifact route planning authority
**Branch**: `main`

### Summary

Migrated marked tcrv-translate direct artifact routes to common planning/coherence/exact-route authority, updated RVV and RVV+scalar FileCheck coverage, and archived the Trellis task.

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
