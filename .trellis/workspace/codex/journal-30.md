> Continuation from `journal-29.md` (archived at more than 2000 lines)

## Session 590: Stage2 RVV Gearbox Gate 2 candidate facts

**Date**: 2026-06-11
**Task**: Stage2 RVV production-kernel Gearbox resource-aware low-precision contraction campaign
**Branch**: `main`

### Summary

Continued the active macro task and completed the Gate 2
widening-product/reduction resource-candidate fact slice. The previous slice
made packed-i4 load/unpack resource facts explicit. This round made the
selected widening-product and widening-reduction candidate facts explicit in
the RVV-owned low-precision resource selection, consumed them during
selected-body realization and route planning, mirrored them into target
artifacts, and rejected stale provider-side and target-side facts.

### Main Changes

- Added explicit widening-product and widening-reduction candidate fact fields
  to RVV low-precision resource candidate/selection payloads.
- Derived signed i8 and unsigned u8 candidate facts from provider-owned
  primitive chain facts instead of from route ids, artifact names, helper names,
  or emitted C spelling.
- Required selected-body realization to consume pre-realized candidate facts and
  emit them on `tcrv_rvv.gearbox_cross_region_handoff`.
- Required selected-body route planning and provider family validation to
  reject stale Gearbox handoff candidate facts before constructing a
  `TCRVEmitCLowerableRoute`.
- Mirrored the candidate facts through route metadata, target support bundle
  keys, and target artifact validation with stale target-mirror rejection.
- Updated the RVV plugin code-spec to record the new resource candidate facts,
  handoff fields, target stale-mirror behavior, and focused test requirements.

### Testing

- [OK] `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-rvv-extension-plugin-test tianchenrv-target-artifact-export-test -j2`
- [OK] `cmake --build build --target tianchenrv-rvv-extension-plugin-test tianchenrv-target-artifact-export-test -j2`
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `build/bin/tianchenrv-target-artifact-export-test`
- [OK] `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'rvv-gearbox-widening-product-reduce-dequantize-f32|explicit-selected-body-artifact-widening-product-reduce-dequantize-f32|explicit-selected-body-artifact-widening-product-reduce-add-unsigned-u8'`
- [OK] `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'pre-realized-selected-body-artifact-widening-product-reduce-dequantize-f32|pre-realized-selected-body-artifact-widening-product-reduce-dequant-clamp-f32|pre-realized-selected-body-artifact-widening-product-reduce-dequantize-f32-packed-i4|pre-realized-selected-body-artifact-widening-product-reduce-dequant-clamp-f32-packed-i4|explicit-selected-body-artifact-widening-product-reduce-add\\.mlir|explicit-selected-body-artifact-widening-product-reduce-add-unsigned-u8'`
- [OK] Manual FileCheck stale-provider/stale-target cases for
  `STALE-WIDENING-CANDIDATE`, `STALE-REDUCTION-CANDIDATE`,
  `STALE-HANDOFF-CANDIDATE`, `STALE-TARGET-CANDIDATE`, and
  `STALE-RESOURCE-REDUCTION-CANDIDATE`.

### Self-Repair

- Seeded C++ selected-body and target artifact fixtures with the new explicit
  candidate facts after stricter selected-body realization and target
  validation exposed missing fixture attrs.
- Updated explicit selected-body dequant and pre-realized dequant expectations
  to match the current fail-closed diagnostics and stable metadata/header order.
- Added the code-spec update after recognizing that the new candidate fields are
  a cross-layer compiler contract, not only an implementation detail.

### Status

[OPEN MACRO TASK] Gate 2 is complete for the current low-precision resource
fact spine: packed-i4 load/unpack facts plus widening-product/reduction
candidate facts are explicit, produced by RVV-owned resource paths, consumed by
selected-body realization and route planning, mirrored by target artifacts, and
stale-rejected in focused signed/unsigned and packed-i4-adjacent coverage.

Gates 3 and 4 remain open. The next continuation point is Gate 3 selected-body
realization consumption using this completed fact spine to build
resource-aware executable bodies without changing computation semantics, ABI
roles, dispatch, fallback, or runtime AVL. Do not move to Gate 4
performance-preferred dispatch without fresh source-backed same-target measured
win evidence.

### Git Commits

Final coherent commit is created after this journal entry.
