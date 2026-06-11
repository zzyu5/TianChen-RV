# Stage2 RVV Production-Kernel Gearbox Resource-Aware Low-Precision Contraction Campaign

## Direction

This is a macro production-capability campaign for RVV low-precision
production kernels. The owner is the spine that lets typed `tcrv_rvv` selected
bodies, Gearbox/resource schedules, low-precision contraction primitive facts,
route planning, target mirrors, and later same-target measurements agree
without q8/q4 names, artifact names, route IDs, helper names, or common EmitC
semantics becoming authority.

## Campaign Gates

1. Production resource/primitive fact spine exists with explicit producers,
   consumers, target mirrors, and stale-fact rejection.
2. Low-precision contraction primitive/resource surface covers typed
   packed-i4/i8/u8 load/unpack, widening product, and reduction facts without
   q8/q4 route authority.
3. Gearbox selected-body realization consumes those facts to build
   resource-aware executable bodies without changing semantics, ABI roles,
   dispatch, fallback behavior, or runtime AVL.
4. Same-target measurement admits only source-backed measured wins into
   performance-preferred dispatch while preserving correctness fallback for
   no-win records.

## Repository Findings

Gate 1 is already present in production source from the prior archived
resource/primitive and selected-dispatch campaigns:

- Dequant/dequant-clamp resource candidates carry planning contract, selected
  candidate, source/product/accumulator/result facts, primitive chain facts,
  realization facts, target capability mirrors, and stale mirror rejection.
- Packed-i4 resource selections carry remediation/performance admission facts
  through provider policy and target validation.
- Target artifact validation already verifies resource/primitive mirrors when
  a low-precision resource selection is present.

The missing bounded Gate 2 slice is the plain product-reduction resource spine.
`WideningProductReduceAdd` already supports signed and unsigned low-precision
primitive facts, including unsigned u8/u16/u32 route and target artifact
validation, but it did not participate in low-precision resource selection.
The non-dequant resource-selection path also carried signed-only assumptions
for source signedness, product roles, extension policy, and accumulator/result
type expectations.

## Current Slice

Implement the first Gate 2 production slice:

- Add provider-owned plain product-reduction low-precision resource candidates
  for signed i8 and unsigned u8 widening-product reduction.
- Admit `WideningProductReduceAdd` into low-precision resource selection.
- Populate resource selection from provider primitive facts, including source
  signedness, source/product/accumulator/result dtype, widening-product roles,
  extension policy, primitive chain contract/kind, widening/reduction
  intrinsics, and target mirrors.
- Keep existing dequant/dequant-clamp and strided low-precision representatives
  working with their current resource contracts.
- Prove unsigned u8 stale resource facts fail closed at target export.

## Acceptance Criteria For This Slice

- The unsigned u8 product-reduction route emits `tcrv_rvv.low_precision_resource`
  candidate, selected candidate, source signedness, u32 accumulator/result,
  primitive chain, intrinsic, and target mirror metadata.
- Target header artifact mirrors the new resource facts.
- Stale `tcrv_rvv.low_precision_resource.source_signedness` is rejected before
  target export.
- Existing signed product-reduction, dequantize-f32, and dequant-clamp-f32
  focused plan/header paths still pass.
- `tianchenrv-rvv-extension-plugin-test` and
  `tianchenrv-target-artifact-export-test` pass.

## Non-Goals

- No q8/q4-named route IDs, artifact authority, or helper-op authority.
- No new high-level Linalg/Vector/StableHLO frontend work.
- No selected-dispatch performance-preferred admission without fresh
  source-backed same-target measured-win evidence.
- No runtime `ssh rvv` claim in this slice; this is a provider/target resource
  fact-spine slice.

## Status After Current Slice

Gate 2 is advanced but not complete. The macro task remains active.

Continuation point: extend the same production resource/primitive spine toward
the next low-precision contraction primitive gap, especially packed-i4
load/unpack and widening-product/reduction resource candidates that Gearbox
selected-body realization can consume without relying on q8/q4 labels or
artifact names.
