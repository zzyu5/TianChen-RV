# RVV selected-body target artifact export gate

## Goal

Make RVV target/export validation consume provider-derived selected-body route descriptions before accepting an RVV target artifact candidate. Target/export may mirror route facts in candidate metadata, but it must not authorize RVV route support from manifest route IDs, retained i32m1 artifact names, descriptor residue, or stale construction metadata by itself.

This task keeps the active RVV Stage 1 line focused on:

```text
selected tcrv.exec RVV variant
  -> explicit typed tcrv_rvv body
  -> RVV provider-derived route description
  -> target artifact candidate validation/export
  -> neutral common EmitC/materialization mechanics
```

## Background

- Previous commit `f07928c` made the RVV provider derive supported routes from selected-body config/intrinsic descriptors.
- Stage 1 remains open downstream if `lib/Target/RVV/RVVTargetSupportBundle.cpp` still authorizes artifacts through manifest route IDs, `getRVVConstructionManifest().emitcRoute.routeID`, `getRVVI32M1ArithmeticArtifactMetadata`, bounded i32m1 artifact descriptions, descriptor/direct-C metadata, or exact route-name residue.
- Retained i32m1 names are allowed only as derived specialization labels behind typed selected-body validation.

## Requirements

- Target candidate validation must call or consume the RVV provider-selected route description before accepting route ID, artifact kind, runtime ABI, and RVV metadata.
- Candidate metadata may mirror selected-body route facts, but metadata cannot be the authority for route support.
- Manifest route IDs and retained i32m1 artifact names must be treated only as derived specialization names for the current supported selected body.
- Export must reject candidates when the selected variant/body is absent, unsupported, mismatched, or only described by stale route/artifact metadata.
- Common EmitC/materialization must remain neutral and must not choose RVV semantics.
- The production/default target/export path must be rewired, not just supplemented by an unused helper.

## Acceptance Criteria

- [x] A positive focused test proves an existing explicit typed i32m1 selected body reaches accepted artifact validation through the provider-derived route description.
- [x] Negative focused tests prove stale route ID, stale artifact metadata, missing selected body, and mismatched selected-body operation/config fail closed.
- [x] Target/export code no longer treats `getRVVConstructionManifest().emitcRoute.routeID`, `getRVVI32M1ArithmeticArtifactMetadata`, descriptor/direct-C metadata, artifact names, or finite i32 route cases as independent RVV route authority.
- [x] Directly relevant RVV construction/provider code remains the source of selected-body route description and target/export only verifies or consumes that provenance.
- [x] Focused lit/C++ checks for the changed target/export behavior pass.
- [x] The task is finished/archived if the bounded gate is complete; otherwise task notes identify the exact remaining continuation point.

## Non-Goals

- Do not add new RVV route coverage: no broadcast, compare/select, reduction, conversion, new dtype, new LMUL, source-shape, or intrinsic case growth.
- Do not generalize Linalg, Vector, StableHLO, or frontend lowering.
- Do not move RVV semantic choice into common EmitC/export.
- Do not preserve old i32 route authority through compatibility wrappers.
- Do not rename every i32m1 artifact string for aesthetics; retained names are acceptable only as derived labels after selected-body validation.
- Do not work on Scalar, IME, Offload, TensorExt, autotuning, dashboards, broad smoke matrices, report/status machinery, or artifact-index-only evidence.
- Do not claim runtime, correctness, or performance without real `ssh rvv` evidence.

## Relevant Specs And Context

- `.trellis/spec/index.md`
- `.trellis/spec/extension-plugins/rvv-plugin.md`
- `.trellis/spec/lowering-runtime/emitc-route.md`
- `.trellis/spec/lowering-runtime/emission-runtime-contract.md`
- `.trellis/tasks/archive/2026-05/05-19-rvv-selected-body-config-intrinsic-route-surface/prd.md`

## Initial Code Surface

- `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`
- `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
- `lib/Plugin/RVV/Construction/RVVConstructionProtocol.cpp`
- `lib/Plugin/RVV/RVVExtensionPlugin.cpp`
- `lib/Target/RVV/RVVTargetSupportBundle.cpp`
- Focused tests under `test/Target` and `test/Plugin` that cover target artifact validation, export rejection, and provider-derived selected-body route metadata.

## Validation Plan

- Run a bounded reference scan over `lib/Target/RVV/RVVTargetSupportBundle.cpp` and directly relevant RVV provider/construction files for route-id authority, i32m1 artifact metadata, descriptor/direct-C/source-export residue, and stale metadata authority.
- Run focused lit tests that exercise RVV target artifact candidate validation/export and provider route derivation.
- Avoid broad unrelated pytest/lit matrices.

## Implementation Notes

- `validateRVVRuntimeAVLVLArtifactMetadata` now receives the provider-derived `RVVSelectedBodyEmitCRouteDescription` and validates `tcrv_rvv.*` metadata against `getRVVSelectedBodyConfigArtifactMetadata(description)`.
- RVV target/export no longer calls the old fixed `verifyRVVI32M1ArithmeticArtifactMetadata` or `getRVVI32M1ArithmeticArtifactMetadata` gate.
- The construction-template adapter no longer pre-authorizes LMUL through a static target-side `lmul = m1` boundary expectation; unsupported selected-body config reaches the RVV provider route description and fails closed there.
- Target C++ coverage now includes unsupported LMUL selected-body rejection through the provider route description, plus existing stale route, stale operation metadata, missing body, missing route metadata, stale runtime ABI, descriptor/direct-C residue, and header composite negatives.
- Conversion negative lit expectations were updated to the current selected-body descriptor/config diagnostics so the focused route-provider evidence matches the active provider behavior.
- No spec update was needed: the relevant RVV selected-body route description and target/export mirror rules already exist in `.trellis/spec/extension-plugins/rvv-plugin.md` and `.trellis/spec/lowering-runtime/emitc-route.md`.

## Checks Run

- `cmake --build build --target tianchenrv-target-artifact-export-test -j2`
- `./build/bin/tianchenrv-target-artifact-export-test`
- `cmake --build build --target tcrv-opt tcrv-translate -j2`
- `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'Target/RVV/(vector-materialized-target-artifact-exporters|emitc-to-cpp-selected-boundary-negative|emitc-to-cpp-selected-boundary-attrs-negative|emitc-to-cpp-non-materialized)'` from `build/test`
- `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'Conversion/EmitC/rvv-first-slice-(materialization|materialization-negative|materialization-missing-abi|config-vl-contract-negative|vl-contract-negative)'` from `build/test`
- `cmake --build build --target tianchenrv-rvv-extension-plugin-test -j2`
- `./build/bin/tianchenrv-rvv-extension-plugin-test`
- `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-19-05-19-rvv-selected-body-target-artifact-export-gate`
- Bounded ref-scans over RVV target/provider/construction files and focused tests for old i32m1 artifact metadata authority, manifest route-id authority, descriptor/direct-C/source-export residue, and selected-body descriptor diagnostics.
- `git diff --check`
