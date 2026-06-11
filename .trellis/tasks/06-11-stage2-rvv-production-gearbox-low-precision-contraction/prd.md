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

## Completed Gate 2 Slices

The prior rounds completed the Gate 2 widening-product/reduction
resource-candidate fact spine:

- Add explicit widening-product and widening-reduction resource-candidate facts
  to the production low-precision resource candidate/selection surface, separate
  from but derived from the provider-owned primitive chain facts.
- Produce those candidate facts in the RVV-owned Gearbox/resource candidate
  path for product-reduction, dequantize, dequant-clamp, and packed-i4-adjacent
  low-precision candidates.
- Consume the same candidate facts in selected-body realization and Gearbox
  cross-region handoff validation before route construction.
- Carry the facts through `TCRVEmitCLowerableRoute` planning metadata and target
  artifact mirrors, with stale provider and stale target rejection.
- Keep existing packed-i4 load/unpack, signed/unsigned product-reduction, and
  dequant/dequant-clamp resource paths passing.

## Current Gate 3A Slice

The current round owns selected-body realization consumption for the completed
low-precision resource fact spine. The bounded target is the
`widening_product_reduce_dequantize_f32` / product-reduce-add-adjacent spine,
including packed-i4 load/unpack and widening-product/reduction candidate facts.

Selected-body realization must consume the selected low-precision resource
candidate before route construction and materialize resource-aware realized
body structure. The realized producer/consumer `with_vl` scopes,
`tcrv_rvv.vsetvl_region_marker` ops, and
`tcrv_rvv.gearbox_cross_region_handoff` must carry provider-verifiable resource
facts derived from the selected candidate: realization decision, unroll,
region count, peak live-vector pressure, product/dequant region indices,
product/dequant phases, packed-i4 load/unpack facts when selected, and
widening-product/reduction primitive/candidate facts. Route planning must read
and compare those realized-body facts before constructing a
`TCRVEmitCLowerableRoute`; it must not reconstruct acceptance from route IDs,
artifact names, helper names, primitive strings alone, or common EmitC logic.

## Acceptance Criteria For Gate 3A

- Production source movement happens in RVV selected-body realization or its
  direct route-planning consumers, not in common EmitC semantic inference.
- Realized product/reduction/dequant producer and consumer scopes carry
  resource-aware schedule facts from the selected candidate, including
  product/dequant region index and phase facts in addition to the existing
  realized unroll, region count, peak-live, packed-i4, and primitive/candidate
  fields.
- Route planning consumes the realized-body schedule facts and rejects missing
  or stale producer/consumer `with_vl`, marker, or Gearbox handoff facts before
  `TCRVEmitCLowerableRoute` construction.
- Packed-i4 load/unpack facts and widening-product/reduction candidate facts
  remain required when selected and continue to fail closed when stale at the
  provider/handoff/target boundary.
- Focused lit/FileCheck fixtures prove realized-body fact consumption and stale
  realized-body/handoff/target fact rejection for the dequantize-f32
  product-reduction representative, with adjacent packed-i4 and dequant-clamp
  coverage kept passing.
- `tianchenrv-rvv-extension-plugin-test` and
  `tianchenrv-target-artifact-export-test` pass.

## Non-Goals

- No q8/q4-named route IDs, artifact authority, or helper-op authority.
- No new high-level Linalg/Vector/StableHLO frontend work.
- No selected-dispatch performance-preferred admission without fresh
  source-backed same-target measured-win evidence.
- No runtime `ssh rvv` claim in this slice; this is a provider/target resource
  fact-spine slice.

## Status Before Current Slice

Gate 2 is complete for the current low-precision resource fact spine:
packed-i4 load/unpack facts and widening-product/reduction candidate facts are
explicit, provider-produced, selected-body consumed, route-planning mirrored,
target-artifact mirrored, and stale-rejected in focused signed/unsigned and
packed-i4-adjacent coverage. The macro task remains active because Gate 3 and
Gate 4 campaign gates are not complete.

Continuation point: move to Gate 3 selected-body realization consumption using
the completed resource fact spine to build resource-aware executable bodies
without changing computation semantics, ABI roles, dispatch, fallback, or
runtime AVL. Do not move to Gate 4 performance-preferred dispatch without fresh
source-backed same-target measured-win evidence.

## Status After Gate 3A Slice

Gate 3A is complete for the current product-reduction/dequantize-f32
realization-consumption slice. RVV selected-body realization and Gearbox
schedule materialization now write provider-verifiable realization
producer/decision, realized unroll, realized `vsetvl` region count, realized
peak-live vector pressure, product/dequant region indexes, and product/dequant
phases onto realized producer/consumer `with_vl` scopes. Route planning reads
those realization-produced facts before constructing a
`TCRVEmitCLowerableRoute` and rejects stale product/dequant fact values before
route acceptance.

Gate 3 is not fully complete for the macro campaign. Remaining Gate 3 work is
to extend the same resource-aware realized-body consumption pattern across the
next low-precision body-family representative, keeping packed-i4 load/unpack
facts and widening-product/reduction candidate facts as producer-owned inputs.
Gate 4 remains later work and still requires fresh source-backed same-target
measured-win evidence before any performance-preferred dispatch admission.

## Current Gate 3B Slice

This round owns the next low-precision body-family representative:
`widening_product_reduce_dequant_clamp_f32`. The goal is to extend the Gate 3A
realized-body fact consumption contract from product/reduction/dequant facts to
the dequant-clamp epilogue without changing computation semantics, ABI roles,
dispatch/fallback behavior, or runtime AVL.

The dequant-clamp realized producer/consumer scopes and Gearbox handoff must
carry clamp/compare-select region and phase facts derived from the selected
low-precision resource candidate. Route planning and target artifact
validation must consume those facts before route or artifact acceptance and
must reject stale dequant-clamp realized-body, handoff, provider, or target
mirrors.

## Acceptance Criteria For Gate 3B

- Production source movement happens in the RVV Gearbox/resource schedule,
  selected-body realization, route-planning consumer, dialect handoff
  structure, or target validation spine; Common EmitC does not infer
  dequant-clamp semantics.
- Dequant-clamp producer/consumer `with_vl` scopes carry the completed Gate 3A
  realization facts plus clamp/compare-select phase and region facts derived
  from the selected resource candidate.
- `tcrv_rvv.gearbox_cross_region_handoff` carries dequant-clamp clamp region,
  clamp phase, compare/select phase, and select-layout facts when the selected
  resource candidate is a dequant-clamp candidate, and rejects those facts for
  non-clamp candidates.
- Route planning reads the dequant-clamp realization facts from the realized
  body before constructing `TCRVEmitCLowerableRoute` and rejects stale
  producer/consumer or handoff clamp facts.
- Target artifact validation mirrors the provider-selected dequant-clamp
  realization facts exactly and rejects stale target mirrors before artifact
  acceptance.
- Focused lit/FileCheck coverage proves positive dequant-clamp fact
  materialization/consumption and at least one stale realized-body or handoff
  dequant-clamp fact failure before route acceptance, while completed Gate 2
  and Gate 3A tests remain passing.
