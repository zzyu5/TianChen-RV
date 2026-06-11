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

## Session 591: Stage2 RVV Gearbox Gate 3A realization fact consumption

**Date**: 2026-06-11
**Task**: Stage2 RVV production-kernel Gearbox resource-aware low-precision contraction campaign
**Branch**: `main`

### Summary

Continued the active macro task and completed the Gate 3A selected-body
realization-consumption slice for the current low-precision
product-reduction/dequantize-f32 spine. This round made the route-planning path
consume realization-produced product/dequant region and phase facts from
realized `with_vl` scopes rather than rebuilding those accepted facts locally
from candidate names or route metadata.

### Main Changes

- Added explicit `product_region_index`, `dequant_region_index`,
  `product_phase`, and `dequant_phase` low-precision resource attr constants.
- Required RVV selected-body realization to write product/dequant region and
  phase facts on realized producer and consumer `with_vl` scopes.
- Required the Gearbox schedule materialization path to materialize or
  fail-closed-validate the same realization producer/decision, realized
  unroll/region/peak-live, and product/dequant region/phase facts for explicit
  selected-body fixtures.
- Changed low-precision route-family planning to read realization-produced
  resource facts before route construction and to reject stale realized-body
  product/dequant facts.
- Added focused stale `product_phase` and `dequant_region_index` FileCheck
  coverage for the pre-realized product-reduction/dequantize-f32 artifact path.
- Updated the RVV plugin code-spec to preserve the Gate 3A realization fact
  consumption contract.

### Testing

- [OK] `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-rvv-extension-plugin-test tianchenrv-target-artifact-export-test -j2`
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `build/bin/tianchenrv-target-artifact-export-test`
- [OK] `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'explicit-selected-body-artifact-widening-product-reduce-dequantize-f32|explicit-selected-body-artifact-widening-product-reduce-add-unsigned-u8|explicit-selected-body-artifact-widening-product-reduce-add\\.mlir|rvv-gearbox-widening-product-reduce-dequantize-f32'`
- [OK] `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'pre-realized-selected-body-artifact-widening-product-reduce-dequantize-f32|pre-realized-selected-body-artifact-widening-product-reduce-dequant-clamp-f32|pre-realized-selected-body-artifact-widening-product-reduce-dequantize-f32-packed-i4|pre-realized-selected-body-artifact-widening-product-reduce-dequant-clamp-f32-packed-i4|rvv-gearbox-widening-product-reduce-dequantize-f32'`

### Self-Repair

- The first explicit selected-body lit run exposed that the Gearbox schedule
  materialization path wrote provider resource facts but not the new
  realization region/phase facts. The fix moved the fact materialization into
  `materializeLowPrecisionResourceAttrs`, so both explicit and pre-realized
  paths share the same producer-owned fact surface.
- The first pre-realized FileCheck update used an over-broad unordered match
  and could bind to the consumer `with_vl`. The check was tightened to an
  ordered selected-variant anchor and `SAME` region/phase assertions.

### Status

[OPEN MACRO TASK] Gate 3A is complete for the current
product-reduction/dequantize-f32 realization-consumption slice. Gate 3 remains
open for extending the same resource-aware realized-body consumption pattern to
the next low-precision body-family representative. Gate 4 remains later work
and still requires fresh source-backed same-target measured-win evidence before
performance-preferred dispatch admission.

### Git Commits

Final coherent commit is created after this journal entry.
