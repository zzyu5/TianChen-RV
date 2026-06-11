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

The first Gate 2 slice added the plain product-reduction resource spine.
`WideningProductReduceAdd` now supports signed and unsigned low-precision
primitive facts, including unsigned u8/u16/u32 route planning, resource
selection, and target artifact validation.

The second Gate 2 slice completed the packed-i4 load/unpack resource-fact
spine. Packed-i4 candidates now carry an explicit load/unpack contract,
storage-load fact, unpack-plan fact, and unpacked-source fact. Gearbox
materializes those facts, selected-body realization consumes and mirrors them,
route planning rejects stale handoff facts before route construction, and target
artifact validation rejects stale provider or metadata mirrors. The authority
remains typed `tcrv_rvv` body/config/runtime facts plus RVV-owned resource
candidates, not q8/q4 labels, artifact names, route IDs, helper names, or common
EmitC semantics.

## Completed Slice

The current round implemented the Gate 2 packed-i4 load/unpack slice:

- Add explicit packed-i4 load/unpack resource facts to the low-precision
  contraction resource candidate/selection surface, including a contract,
  storage-load fact, unpack-plan fact, and unpacked-source fact.
- Produce those facts from the RVV-owned Gearbox/resource candidate path when
  the selected candidate is packed-i4.
- Consume the same facts in selected-body realization and route-planning
  validation before `TCRVEmitCLowerableRoute` construction, including the
  realized `with_vl` and Gearbox cross-region handoff path.
- Mirror the facts through route metadata/target artifact validation and reject
  stale provider facts or stale target metadata before artifact acceptance.
- Keep existing packed-i4 dequant/dequant-clamp, signed/unsigned
  product-reduction, and dequant/dequant-clamp resource paths passing.

## Acceptance Criteria For This Slice

- Packed-i4 load/unpack facts are explicit fields in the production
  low-precision resource candidate/selection surface, not only text inside
  remediation strings.
- Gearbox/resource candidate materialization produces the packed-i4
  load/unpack facts and selected-body realization consumes them before route
  construction.
- Route planning and target validation compare packed-i4 load/unpack provider
  facts and target mirrors exactly.
- A stale provider-side packed-i4 load/unpack fact fails in the production
  provider/target validation path.
- A stale target artifact mirror for a packed-i4 load/unpack fact fails before
  target export.
- Existing packed-i4 dequant/dequant-clamp, signed product-reduction, unsigned
  product-reduction, dequantize-f32, and dequant-clamp-f32 focused plan/header
  paths still pass.
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

Gate 2 is advanced but not complete. The packed-i4 load/unpack fact slice is
complete; widening-product/reduction resource candidate facts remain open. The
macro task remains active.

Continuation point: extend the same production resource/primitive spine toward
the next low-precision contraction primitive gap that Gearbox selected-body
realization consumes directly, especially widening-product/reduction resource
candidates without relying on q8/q4 labels or artifact names.
