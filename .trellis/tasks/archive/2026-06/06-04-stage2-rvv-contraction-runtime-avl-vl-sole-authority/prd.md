# Stage2 RVV contraction runtime AVL/VL sole authority cleanup

## Goal

Make the RVV contraction-family target-artifact runtime-boundary consumers treat `RVVRuntimeAVLVLSelectedBoundaryContract` as the sole acceptance authority for runtime `n` / AVL / VL facts. MAcc and widening-dot / widening-dot-reduce validation may retain route-local runtime/control fields and candidate metadata only as explicit mirrors that are checked after the shared selected-boundary contract.

## What I already know

* The previous completed commit is `ee87e05a rvv: close compare select runtime mirror metadata labels`.
* The next bounded owner is the contraction family, specifically MAcc and widening-dot / widening-dot-reduce route validation and candidate metadata.
* The intended production chain is selected `tcrv.exec` runtime parameter `n` plus typed `tcrv_rvv` contraction body, then RVV plugin-owned runtime AVL/VL selected-boundary contract, then provider-built contraction `TCRVEmitCLowerableRoute`, then common EmitC materialization and target artifact validation.
* Route-local fields such as `runtimeControlPlanID`, `runtimeABIOrder`, setvl/VL/loop names, and widening-dot candidate metadata are mirrors only.
* Stale or missing route-local runtime mirrors must fail closed with diagnostics that identify `route-local runtime AVL/VL mirror` instead of accepting route-local fields as authority.

## Assumptions

* This round changes only MAcc and widening-dot / widening-dot-reduce runtime-boundary consumers unless repository evidence shows a shared helper is the minimal safe place.
* No runtime `ssh rvv` evidence is needed if emitted C, runtime ABI order/counts, statement ordering, contraction correctness, and runtime behavior are not changed.
* Existing compare/select runtime mirror cleanup provides the preferred local pattern.

## Requirements

* MAcc and widening-dot target validation must call the shared runtime AVL/VL selected-boundary validator before route-local runtime mirror checks.
* Route-local runtime fields must be named, documented, or diagnosed as mirrors only.
* Stale or missing route-local runtime mirrors must fail closed with targeted diagnostics.
* Widening-dot candidate metadata runtime labels must be mirror-only labels, not selected typed authority labels.
* Common EmitC/export must remain neutral and must not infer RVV runtime facts from route ids, artifact names, test names, manifests, descriptors, C strings, or metadata mirrors.
* No new MAcc, widening-dot, reduction, conversion, compare/select, elementwise, scalar splat-store, memory, source-front-door, high-level frontend, intrinsic, dtype, or LMUL coverage may be added.

## Acceptance Criteria

* [x] Focused production diff aligns MAcc and widening-dot route contracts/builders or target validation with the shared runtime AVL/VL selected-boundary contract.
* [x] Positive focused target-artifact checks still pass for touched MAcc and widening-dot paths.
* [x] Negative focused target-artifact checks prove stale route-local runtime mirrors cannot override the embedded selected-boundary contract.
* [x] Widening-dot candidate metadata runtime labels are mirror-only.
* [x] Diagnostics for stale/missing route-local fields say `route-local runtime AVL/VL mirror`.
* [x] `tianchenrv-target-artifact-export-test` and focused contraction lit/export checks pass.
* [x] If provider/planning code changes, `tianchenrv-rvv-extension-plugin-test` passes.
* [x] Bounded grep and added-line old-authority scan do not show newly introduced old route authority.
* [x] `git diff --check` passes and git status is clean after commit.

## Out of Scope

* Adding new operation coverage, dtype coverage, LMUL coverage, frontend lowering, source-front-door routes, descriptors, dashboards, or runtime behavior changes.
* Moving RVV semantics into common EmitC/export.
* Treating route-local fields, candidate metadata, artifact metadata, or status/result labels as acceptance authority.

## Technical Notes

* Read first: `.trellis/spec/index.md`, `.trellis/spec/extension-plugins/rvv-plugin.md`, `.trellis/spec/lowering-runtime/emitc-route.md`.
* Inspect: `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`, `lib/Plugin/RVV/EmitC/`, `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`, focused target/export tests, recent archived runtime sole-authority tasks under `.trellis/tasks/archive/2026-06/`.
* Existing compare/select cleanup should be used as the comparison pattern.

## Implementation Summary

* `RVVMAccRouteValidationContract` and `RVVWideningDotReduceRouteValidationContract` now document `runtimeControlPlanID`, `runtimeABIOrder`, setvl/VL names, and related route-local runtime copies as target-side consistency mirrors whose acceptance authority is `runtimeAVLVLContract`.
* MAcc and widening-dot / widening-dot-reduce target validation now calls `validateRVVRuntimeAVLVLSelectedBoundaryContract` first and then validates retained route-local runtime fields through `validateRVVRouteLocalRuntimeAVLVLMirrors`.
* Direct route-local acceptance checks for `runtimeControlPlanID` and `runtimeABIOrder` were removed from the touched MAcc and widening-dot validation paths.
* MAcc, widening MAcc, and widening-dot candidate metadata runtime labels now use `route-local runtime AVL/VL control plan mirror` and `route-local runtime AVL/VL ABI order mirror`.
* Added stale route-local runtime control and ABI mirror negative coverage for plain MAcc, widening MAcc, and widening-dot candidate validation.

## Evidence

* Built `tianchenrv-target-artifact-export-test`, `tianchenrv-rvv-extension-plugin-test`, `tcrv-opt`, and `tcrv-translate`.
* Ran `build/bin/tianchenrv-target-artifact-export-test`: passed.
* Ran `build/bin/tianchenrv-rvv-extension-plugin-test`: passed.
* Ran focused lit from `build/test` with filter `explicit-selected-body-artifact-(macc-add|.*macc-add|.*widening-dot.*)`: 9 passed, 468 excluded.
* Ran bounded grep for stale MAcc/widening-dot runtime authority labels: none found.
* Ran added-line old-authority scan over touched files: no old-authority additions.
* Ran `git diff --check`: passed.
* Did not run `ssh rvv` because this round changes target-side validation contracts, metadata mirror labels, and stale-mirror tests only; it does not change emitted C, runtime ABI order/counts, statement ordering, contraction computation, or runtime behavior.
