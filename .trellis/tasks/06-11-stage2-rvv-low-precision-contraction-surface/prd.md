# Stage2 RVV Low-Precision Contraction Primitive-Surface Campaign

## Direction

This is a macro production-capability campaign for the typed low-precision RVV
contraction primitive surface. The owner is the compiler path from selected
typed `tcrv_rvv` body/config/runtime facts to RVV plugin-owned primitive
legality and route construction, then through neutral EmitC and target artifact
consumers.

q8/q4 examples are production pressure tests only. They must not become route
ids, artifact names, helper wrappers, source-front-door authority, or common
EmitC semantics.

## Campaign Outcomes

1. Foundation outcome: the typed low-precision primitive surface contract
   exists and is consumed by RVV plugin validation for i8/u8 element/config,
   low-precision loads, signed/unsigned extension, widening product, widening
   accumulation or reduction facts, policy, runtime AVL/VL, and fail-closed
   unsupported combinations.
2. Selected-body realization outcome: the RVV plugin can materialize
   representative signed and unsigned low-precision product-reduction bodies
   from that primitive surface without q8/q4 route authority.
3. Provider/artifact carry-through outcome: the route provider and
   artifact/export consumers carry primitive facts through
   `TCRVEmitCLowerableRoute` and target validation without Common EmitC
   semantic inference.
4. Measurement-disposition outcome: measured same-target comparison or
   dispatch preference, if later claimed, uses only fresh source-backed
   evidence and the conservative no-win boundary from the previous campaign.

## Repository Findings

- The previous Gearbox/resource-aware campaign is archived and complete for its
  scope. It recorded no-win measurement disposition and dispatch/fallback
  consumption boundaries, but those evidence outcomes must not act as route,
  dtype/config, schedule, or artifact-name authority.
- Existing production code already has route-supported signed i8 and unsigned
  u8 widening product/reduction representatives, dequant/dequant-clamp
  resource paths, MAcc paths, low-precision resource selections, and target
  stale-mirror rejection.
- Commit `65f67e38` concentrated the signed/unsigned product-reduction dtype,
  signedness, SEW/LMUL, policy, runtime control, product/reduction relation,
  intrinsic, seed splat, layout, and store-VL facts into a provider-owned
  primitive route payload consumed by target artifact validation.
- The current gap is adjacent primitive ownership drift: stable low-precision
  primitive facts are still repeated across selected-body descriptions,
  primitive fact structs, route payload metadata, emission-plan mirrors, and
  target validation checks. The next cleanup must keep typed body/config/runtime
  and provider-built primitive facts as compiler authority, while treating
  emitted metadata and target artifact fields as mirrors of the provider route
  payload.

## Completed Slice: Gearbox Low-Precision Resource-Schedule Canonicalization

The previous bounded slice followed the packed-i4 measurement-disposition
quarantine by canonicalizing the stable Gearbox/resource schedule facts that
survive as compiler authority. Compiler authority remains:

```text
selected tcrv.exec RVV variant
  -> typed packed-i4 low-precision pre-realized body/config/runtime facts
  -> RVV Gearbox resource candidate and stable resource-schedule facts
  -> RVV contraction selected-body realization owner
  -> realized tcrv_rvv unpack/product/reduction/dequant or dequant-clamp body
  -> RVV provider-owned primitive/resource route payload
  -> TCRVEmitCLowerableRoute / neutral EmitC metadata mirrors
  -> support-bundle export / target artifact validation of compiler facts
```

That cleanup inspected the directly related packed-i4 dequantize and
dequant-clamp production route, Gearbox handoff, selected-body realization,
provider route planning, and target-validation surfaces. Stable schedule and
resource-cost facts now have one plugin-local compiler contract sourced from
typed body/config/resource facts. Measurement results, no-win conclusions,
source evidence IDs, remediation plans, performance admission, and dispatch
preference remain only as measurement-disposition or policy evidence.

The concrete blocker completed in that slice was that
`isRVVLowPrecisionResourceAcceptedPackedI4StableScheduleDecision` depended on
packed-i4 remediation plan fields while claiming to decide a stable schedule.
The fix concentrated the stable packed-i4 schedule facts into a canonical
helper/API consumed by the Gearbox candidate selector, selected-body
realization/provider route validation, and target artifact resource
validation. Measurement-disposition helpers remain separate and continue to
fail closed for stale evidence/admission mirrors.

## Completed Slice: Selected-Body Realization Admission/Evidence Boundary Cleanup

The previous bounded slice cleaned the handoff between selected-body
realization compiler facts and packed-i4 measurement-disposition policy/evidence
facts. Provider-side realization compiler-fact gates now validate
typed/resource/realization, packed load/unpack, Gearbox handoff, stable
schedule, and resource-cost facts without consuming packed-i4
admission/remediation/performance/measurement/no-win/dispatch fields as route,
schedule, or resource acceptance facts.

Packed-i4 measurement-disposition fields remain preserved and are read through
explicit policy/evidence helpers. Missing or stale policy/evidence attrs fail
with named measurement-disposition policy/evidence diagnostics, while stale
target artifact mirrors remain handled by separate evidence/admission mirror
validators.

## Completed Slice: Provider Primitive Route-Payload Canonicalization

The previous bounded slice canonicalized stable low-precision primitive facts at
the RVV provider route-payload boundary. The provider already creates
`RVVLowPrecisionWideningReductionPrimitiveFacts` and
`RVVLowPrecisionPrimitiveRoutePayload`, but related fields still appear in the
route-family plan, route description, emission metadata, support-bundle export,
and target artifact validation. That slice made provider validation the
canonical primitive route-payload gate: typed `tcrv_rvv` body/config/runtime
facts and provider primitive facts feed one validated primitive route payload;
Common EmitC and target artifacts only carry or compare mirrors of that payload.

The source-backed field classification remains:

- Compiler authority: selected typed `tcrv_rvv` body/config/runtime facts,
  source dtype/signedness, source load/extension, product dtype/SEW/LMUL,
  accumulator and primitive reduction-result dtype/SEW/LMUL,
  product-reduction relation, widening product intrinsic, reduction intrinsic,
  scalar seed splat, tail/mask policy, runtime AVL/VL facts, accumulator/result
  layout, reduction store VL, packed-i4 load/unpack facts, and provider-owned
  `RVVLowPrecisionPrimitiveRoutePayload`.
- Mirror/test facts: route-description scalar copies, emission-plan
  `tcrv_rvv.low_precision_primitive.*` metadata, target support-bundle fields,
  candidate metadata, object/header mirrors, lit `PLAN`/`HEADER` checks, and
  C++ fixture mutation records. These may prove exact carry-through but must
  not choose primitive semantics.
- Policy/evidence facts: realization admission proof, remediation plans,
  performance feedback/admission, same-target measurement evidence IDs,
  maturity/no-win outcomes, and dispatch preference. They are out of the
  primitive compiler-fact boundary and must not satisfy primitive route-payload
  validation.

Production behavior now fails closed when source dtype or signedness,
load/extension, product/accumulator/result dtype, SEW/LMUL,
product-reduction relation, intrinsic, policy, runtime AVL/VL,
accumulator/result layout, or store-VL facts disagree with the provider-owned
primitive payload. Target artifact validation consumes provider-built mirrors
only after the provider payload contract is valid; it must not reconstruct a
second primitive authority from artifact metadata, candidate IDs, route IDs,
admission state, or exact intrinsic spellings.

## Completed Slice: Support-Bundle/Export Primitive Mirror-Boundary Cleanup

The previous bounded slice cleaned the adjacent support-bundle/export boundary
for the same low-precision product-reduction and packed-i4
dequant/dequant-clamp representatives. The target support bundle exposes
`tcrv_rvv.low_precision_primitive.*` fields to generated headers and bundle
metadata only as `low_precision_primitive.payload_mirror.*` header labels.
Candidate metadata keys remain `tcrv_rvv.low_precision_primitive.*` for target
provider-payload validation, but generated header evidence no longer makes
support-bundle metadata look like a second source of dtype, signedness,
SEW/LMUL, policy, runtime AVL/VL, product-reduction relation, intrinsics,
seed/layout/store-VL, packed-i4 resource, admission, or measurement authority.

## Completed Slice: Route-Description/Emission-Plan Candidate Mirror-Boundary Cleanup

This bounded slice cleaned the adjacent route-description, emission-plan, and
candidate-metadata boundary for the same low-precision primitive payload. The
legacy route-description scalar primitive fields now mirror the provider-built
`RVVLowPrecisionPrimitiveRoutePayload` after payload construction instead of
being populated directly from the route-family plan. Emission-plan primitive
metadata is emitted through a single payload-mirror helper and now carries
`tcrv_rvv.low_precision_primitive.payload_mirror_source =
provider-built-low-precision-primitive-route-payload.v1`. Product-reduction
metadata mirrors for source/product/accumulator/result SEW/LMUL, relation,
intrinsics, seed/layout, and store-VL are sourced from the payload. Target
candidate validation requires the payload-mirror source marker and still
rejects stale or missing primitive field mirrors against the rebuilt provider
payload.

The cleanup owner is the path:

```text
selected typed low-precision body/config/runtime facts
  -> RVV provider primitive facts
  -> RVVLowPrecisionPrimitiveRoutePayload
  -> route-description scalar mirrors
  -> emission metadata / candidate metadata payload mirrors
  -> support-bundle/header mirror transport labels
  -> target artifact validation against the provider payload
```

Route-description scalar fields, emission-plan metadata, candidate metadata,
support-bundle/header evidence, and C++/lit fixtures may copy primitive fields
only as exact mirrors of the provider payload. Missing, stale, or marker-less
primitive mirrors must fail at the provider/target mirror boundary, not be
repaired from route ids, candidate IDs, artifact names, support-bundle metadata,
intrinsic spellings, admission/remediation/measurement/no-win, or dispatch
fields. Common EmitC remains neutral and does not infer or choose any
low-precision primitive semantics.

This round must not add q8/q4 route authority, artifact-name authority,
helper-only wrappers, source-front-door positive routes, Common EmitC semantic
inference, or measured-win/admission claims.

## Acceptance Criteria For This Slice

- Active task metadata, PRD, implementation/check context, and RVV
  low-precision spec sections identify route-description/emission-plan/candidate
  primitive mirror-boundary cleanup under the same macro task.
- Source-backed field classification is recorded for compiler authority,
  mirror/test facts, and policy/evidence facts at the route-description,
  emission-plan, and candidate primitive mirror boundary.
- Route-description scalar primitive mirrors are populated from the
  provider-built `RVVLowPrecisionPrimitiveRoutePayload` and validated against
  that payload before provider materialization.
- Emission metadata serializes low-precision primitive fields through a single
  payload-mirror helper and carries
  `tcrv_rvv.low_precision_primitive.payload_mirror_source =
  provider-built-low-precision-primitive-route-payload.v1`.
- Product-reduction emission metadata for source/product/accumulator/result
  SEW/LMUL, relation, intrinsics, seed/layout, and store-VL is sourced from the
  provider payload rather than stale route-description scalar mirrors.
- Target artifact validation continues to validate the provider payload first,
  then consumes candidate primitive mirrors and the payload-mirror source marker
  only as exact mirrors of that payload.
- Missing, stale, or marker-less candidate primitive mirrors are rejected
  without inventing primitive semantics from route ids, artifact names, intrinsic
  spellings, admission/remediation/measurement/no-win, or dispatch fields.
- Packed-i4 dequantize/dequant-clamp resource and policy/evidence boundaries
  remain intact; packed-i4 policy/evidence fields stay outside primitive
  compiler-fact validation.
- Focused negative coverage proves stale or missing primitive candidate mirrors
  and stale or missing payload-mirror source markers are rejected at the target
  mirror boundary, alongside existing signed/unsigned product-reduction and
  packed-i4 dequant/dequant-clamp positive fixtures.
- Common EmitC remains neutral and only carries provider-built payloads and
  mirrors.
- A bounded scan over touched production files, active task text, directly
  affected fixtures, and RVV low-precision spec sections finds no
  admission/remediation/performance/measurement/same-target/no-win/dispatch
  wording used as primitive route-description/emission-plan/candidate compiler
  authority.
- `tcrv-opt`, `tcrv-translate`, `tianchenrv-rvv-extension-plugin-test`, and
  `tianchenrv-target-artifact-export-test` build; the two C++ test binaries run.
- Focused lit coverage for signed/unsigned product-reduction and packed-i4
  dequantize/dequant-clamp fixtures passes.
- `python3 ./.trellis/scripts/task.py validate
  .trellis/tasks/06-11-stage2-rvv-low-precision-contraction-surface` passes.
- `git diff --check` and `git diff --cached --check` pass.
- One coherent commit is created for the cleanup slice. The macro task remains
  active unless all campaign outcomes are genuinely complete.

## Non-Goals

- No q8/q4-named route ids, artifact names, helper wrappers, or source-front-door
  positive routes.
- No high-level Linalg/Vector/StableHLO frontend work.
- No broad dtype/LMUL clone batch.
- No measured-win or performance-preferred admission without fresh
  source-backed same-target evidence.
- No dashboard, report-only, metadata-only, IME, Offload, TensorExt, Scalar, or
  future-plugin work.
- No runtime/correctness/performance claim and therefore no required `ssh rvv`
  evidence in this slice.
- No broad repository rename. Historical archived text and immutable evidence
  path identifiers may remain outside the bounded active cleanup surface.

## Status Before This Round

The foundation outcome is complete for the current product-reduction primitive
contract hardening slice. RVV provider route metadata mirrors
source/product/accumulator/result SEW/LMUL plus tail policy, mask policy,
runtime control plan, and runtime AVL source under
`tcrv_rvv.low_precision_primitive.*` whenever a low-precision primitive
contract is present.

The selected-body realization outcome is complete for the bounded signed i8 and
unsigned u8 product-reduction representatives. The shared pre-realized typed
surface carries explicit `source_signedness`, the verifier accepts only
signedness-consistent signed or unsigned product-reduction contracts, and the
RVV contraction selected-body realization owner materializes explicit
`tcrv_rvv` structure before route construction.

The provider/artifact carry-through outcome is complete for the plain
signed/unsigned product-reduction representative slice. Product-reduction route
descriptions carry `RVVLowPrecisionPrimitiveRoutePayload`; emission metadata
serializes `tcrv_rvv.low_precision_primitive.*` only from that provider payload;
target support-bundle export maps those mirrors; and target artifact validation
rejects missing or stale primitive route payload mirrors before artifact
acceptance.

The measurement-disposition outcome remains later work unless fresh
source-backed same-target measurement evidence is explicitly introduced. The
macro task remains open after this cleanup slice for any adjacent
low-precision primitive-surface gaps or a future measurement-disposition slice
that has real RVV target evidence.

The previous packed-i4 measurement-disposition policy-boundary cleanup is
complete. Gearbox handoff diagnostics, provider route planning, and target
artifact validation distinguish stable resource facts from
measurement-disposition evidence/admission mirrors.

The Gearbox low-precision resource-schedule canonicalization slice is also
complete. The accepted packed-i4 stable schedule decision uses a single RVV
plugin-local `RVVLowPrecisionPackedI4StableResourceScheduleFacts` helper for
schedule-decision, unroll/accumulator/region/live-vector budget, and
resource-cost facts. That stable helper is consumed by Gearbox candidate
selection, Gearbox handoff/resource schedule verification, provider route
planning validation, and target artifact resource validation.

The support-bundle/export primitive mirror cleanup is complete. The remaining
open boundary for this round was the adjacent route-description/emission-plan
candidate metadata surface: signed/unsigned product-reduction and packed-i4
dequant/dequant-clamp paths carried low-precision primitive facts through the
provider payload, route-description scalar mirrors, emission metadata,
candidate metadata, support-bundle export, and target artifact validation. This
slice made route-description scalar mirrors and emission-plan candidate
primitive metadata visibly payload-mirror-only and kept target validation
anchored on provider-built payload mirrors.

## Expected Status After This Round

The route-description/emission-plan/candidate primitive mirror-boundary cleanup
slice is complete for signed/unsigned product-reduction, standalone
widening-product metadata affected by the shared payload helper, and packed-i4
dequant/dequant-clamp representatives. The macro campaign remains in progress
for any adjacent low-precision primitive-surface cleanup and for future
measurement-disposition work only when fresh source-backed same-target RVV
evidence exists. The next continuation point is to inspect whether any
remaining low-precision primitive/resource facts still pass through route
descriptions, emission-plan metadata, support bundles, or target artifacts
without an explicit provider-payload/resource-owner mirror marker.
