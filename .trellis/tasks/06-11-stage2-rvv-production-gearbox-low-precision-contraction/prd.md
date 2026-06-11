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

The current bounded Gate 2 slice consumes those facts as route-planning and
target-validation authority. The plain product-reduction consumer path must not
let the resource selection pick its own signed/unsigned primitive fact family
before it is checked against the provider-derived route description/plan.
Missing, stale, or inconsistent product-reduction resource facts must fail
closed before target artifact acceptance.

## Current Slice

Implement the current Gate 2 consumption slice:

- Make plain product-reduction route-planning validation choose signed/unsigned
  widening-reduction primitive facts from provider-derived plan/description
  signedness, not from the untrusted resource selection being validated.
- Keep resource selection comparison against provider primitive/source/product,
  accumulator/result, widening-product role, extension-policy, route-family,
  producer/consumer, target capability, and mirror facts fail closed.
- Add focused negative coverage for a stale plain product-reduction resource
  signedness selection and a stale product-reduction resource chain mirror.
- Keep existing signed/unsigned product-reduction and product-reduction
  dequant/dequant-clamp paths passing.

## Acceptance Criteria For This Slice

- Plain product-reduction resource validation consumes provider-derived
  primitive signedness when selecting the expected signed/unsigned
  widening-reduction fact family.
- A stale plain product-reduction resource signedness selection fails in the
  production target-validation/provider-fact consumer path.
- A stale target artifact mirror for the product-reduction resource chain
  relation fails before target export.
- Existing signed product-reduction, unsigned product-reduction,
  dequantize-f32, and dequant-clamp-f32 focused plan/header paths still pass.
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

Gate 2 will be advanced but not complete. The macro task remains active.

Continuation point: extend the same production resource/primitive spine toward
the next low-precision contraction primitive gap that Gearbox selected-body
realization consumes directly, especially packed-i4 load/unpack and
widening-product/reduction resource candidates without relying on q8/q4 labels
or artifact names.
