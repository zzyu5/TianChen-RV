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

## Completed Slice: Resource-Owner Mirror-Source Cleanup

The previous bounded slice cleaned the adjacent low-precision resource mirror
boundary. Emission metadata now carries
`tcrv_rvv.low_precision_resource.resource_owner_mirror_source =
provider-owned-low-precision-contraction-resource-selection.v1` whenever the
provider route description contains `RVVLowPrecisionContractionResourceSelection`.
Target candidate validation rejects missing or stale marker values before
accepting stable `tcrv_rvv.low_precision_resource.*` mirrors, and support-bundle
header evidence exposes the marker as
`low_precision_resource.resource_owner_mirror.source`.

The completed cleanup owner was:

```text
selected typed low-precision body/config/runtime facts
  -> RVV Gearbox/resource candidate and stable schedule facts
  -> RVVLowPrecisionContractionResourceSelection
  -> emission metadata / candidate metadata resource-owner mirrors
  -> support-bundle/header mirror-source transport evidence
  -> target artifact validation against the provider resource selection
```

## Completed Slice: Low-Precision Mirror Transport Contract Consolidation

The previous bounded slice consolidated the primitive/resource mirror transport
boundary after the adjacent payload and resource-owner marker slices. Primitive
payload and resource-owner marker key/source/header/diagnostic labels are now
provided by one shared RVV transport contract and consumed by route planning,
target support-bundle/header evidence, target artifact validation, and focused
C++ tests.

The completed cleanup owner was:

```text
provider-built low-precision primitive payload / resource selection
  -> shared RVV mirror transport contract for marker key, source value,
     header label, and authority label
  -> emission metadata / candidate metadata marker production
  -> support-bundle/header marker transport evidence
  -> target artifact validation using the same contract before mirror checks
```

## Current Slice: Resource-Selection Policy/Evidence Quarantine

This bounded slice cleans the boundary inside and around
`RVVLowPrecisionContractionResourceSelection`. That aggregate intentionally
carries a broad resource-selection record today, but production route planning
and target resource mirror validation should consume only stable compiler facts:
selected typed body/config/runtime facts, Gearbox resource candidate facts,
stable schedule/resource-cost facts, selected-body realization facts,
provider-owned primitive facts, target capability mirrors, and legality facts.
Realization admission, remediation planning, performance feedback/admission,
same-target/no-win measurement evidence, selected-dispatch policy output, and
dispatch preference remain explicit policy/evidence records.

The cleanup owner for this round is:

```text
selected typed low-precision tcrv_rvv body/config/runtime facts
  -> Gearbox stable resource and schedule facts
  -> RVVLowPrecisionStableResourceCompilerFacts view
  -> route-planning stable resource metadata mirrors
  -> target artifact stable resource mirror validation
  -> separate measurement-disposition and selected-dispatch policy/evidence helpers
```

Source-backed field classification for this slice:

- Compiler authority: selected typed `tcrv_rvv` body/config/runtime facts,
  Gearbox/resource candidate selection, stable packed-i4 schedule/resource-cost
  facts, selected-body realization facts, provider-owned primitive payloads,
  target capability provider/legality mirrors, route-family provider support,
  and the `RVVLowPrecisionStableResourceCompilerFacts` view derived from the
  provider selection before route planning or target stable resource validation.
- Mirror/test facts: `tcrv_rvv.low_precision_resource.*` emission metadata,
  candidate metadata, support-bundle/header comments, lit `PLAN`/`HEADER`
  checks, and C++ fixture mutations. These may prove exact carry-through only
  after the stable compiler-fact view exists and the shared resource-owner
  mirror-source marker is present.
- Policy/evidence facts: realization admission proof, remediation plans,
  performance feedback/admission, same-target measurement evidence IDs,
  maturity/no-win outcomes, selected-dispatch policy output, and dispatch
  preference. These remain in measurement-disposition or selected-dispatch
  helper boundaries and must not satisfy primitive payload, stable resource,
  schedule, route, support-bundle, or artifact acceptance.

This round must not add q8/q4 route authority, artifact-name authority,
helper-only wrappers, source-front-door positive routes, Common EmitC semantic
inference, or measured-win/admission claims.

## Acceptance Criteria For This Slice

- Active task metadata, PRD, implementation/check context, and RVV low-precision
  spec sections identify resource-selection policy/evidence quarantine under the
  same macro task.
- `RVVLowPrecisionStableResourceCompilerFacts` or an equivalent stable view
  contains only compiler/resource/primitive/realization/target facts needed for
  stable route-planning and target stable resource mirror validation; it excludes
  admission/remediation/performance/measurement/no-win/dispatch fields.
- Route planning emits stable `tcrv_rvv.low_precision_resource.*` metadata from
  the stable compiler-fact view, while policy/evidence metadata remains in a
  separately named admission/remediation/performance/dispatch block.
- Target artifact stable resource mirror validation consumes the stable
  compiler-fact view, not the policy-bearing aggregate. Measurement-disposition
  evidence mirrors and selected-dispatch policy-output mirrors remain in their
  existing explicit policy/evidence validators.
- Missing or stale stable resource mirrors still fail closed against provider
  stable facts, including signed/unsigned product-reduction and packed-i4
  dequantize/dequant-clamp representatives where directly affected.
- Missing, stale, or metadata-only policy/evidence fields do not satisfy
  primitive payload, stable resource, schedule, route, support-bundle, or
  artifact acceptance. If policy/evidence fields are present, their own helpers
  continue to fail closed on stale mirrors.
- Common EmitC remains neutral and only carries provider-built payloads,
  stable resource mirrors, shared mirror-source markers, and explicit
  policy/evidence mirrors.
- Focused C++ coverage proves the stable resource compiler-fact view ignores
  stale policy/evidence admission, remediation, and dispatch fields while
  preserving stable resource/schedule/primitive/target facts.
- A bounded scan over touched production files, active task text, directly
  affected fixtures, and RVV low-precision spec sections finds no
  admission/remediation/performance/measurement/same-target/no-win/dispatch
  wording used as primitive payload or stable resource compiler authority.
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

The support-bundle/export primitive mirror cleanup is complete. The
route-description/emission-plan/candidate primitive mirror cleanup is complete:
route-description scalar primitive fields and emission-plan candidate primitive
metadata are visibly payload-mirror-only and target validation is anchored on
provider-built payload mirrors.

The resource-owner mirror-source cleanup is complete. Emission metadata,
candidate metadata, support-bundle/header evidence, and target validation now
carry and require the provider-owned low-precision resource-selection marker.
The low-precision mirror transport contract consolidation slice is complete.
Primitive payload and resource-owner marker key/source/header/diagnostic labels
are provided by one shared RVV transport contract and consumed by route planning,
target support-bundle/header evidence, target artifact validation, and focused
C++ tests. The remaining open boundary for this round is that stable
resource/primitive compiler facts still sit beside policy/evidence/admission
and result records in `RVVLowPrecisionContractionResourceSelection`; production
route planning and stable target resource validation need a policy-free
compiler-fact view.

## Expected Status After This Round

The resource-selection policy/evidence quarantine slice is complete. The RVV
provider exposes a stable low-precision resource compiler-fact view derived from
provider-owned resource selection but excluding realization-admission,
remediation, performance, measurement, no-win, and dispatch-preference policy
fields. Route planning emits stable `tcrv_rvv.low_precision_resource.*`
metadata through that view, while admission/remediation/performance/dispatch
metadata remains in a separately guarded policy/evidence block. Target artifact
stable resource mirror validation consumes the stable compiler-fact view instead
of the policy-bearing aggregate, while measurement-disposition and selected
dispatch policy validators remain explicit policy/evidence boundaries.

The macro campaign remains in progress for adjacent low-precision
primitive/resource cleanup and for future measurement-disposition work only when
fresh source-backed same-target RVV evidence exists. The next continuation point
is to inspect whether any remaining low-precision primitive/resource support
bundle or artifact acceptance paths still consume the broad provider aggregate
instead of stable compiler-fact views or explicitly named policy/evidence
validators.

## Completed Slice: Support-Bundle/Artifact Acceptance Stable-Resource Cleanup

This bounded follow-up slice completed the consumer-side cleanup after the
stable compiler-fact view extraction. Stable low-precision artifact acceptance
now uses the same policy-free compiler-fact boundary through the target
provider-fact checks, target candidate mirror checks, and support-bundle/header
evidence labels:

```text
selected typed low-precision body/config/runtime facts
  -> provider-owned RVVLowPrecisionContractionResourceSelection
  -> RVVLowPrecisionStableResourceCompilerFacts
  -> route-planning stable resource metadata
  -> target artifact stable provider/candidate validation
  -> support-bundle/header stable-resource mirrors plus explicit
     measurement-disposition and selected-dispatch policy mirror labels
```

The target artifact validators for stable primitive/resource/realization facts
take `RVVLowPrecisionStableResourceCompilerFacts` when checking stable
acceptance prerequisites. The wide
`RVVLowPrecisionContractionResourceSelection` remains only for explicitly named
measurement-disposition evidence/admission validation and selected-dispatch
policy-output validation.

Support-bundle/header comments no longer present performance, remediation,
admission, measurement, maturity/no-win, selected-dispatch policy output, or
dispatch-preference fields as bare `low_precision_resource.*` header facts.
They now use
`low_precision_resource.measurement_disposition_evidence_mirror.*` or
`low_precision_resource.selected_dispatch_policy_output_mirror.*` labels while
leaving candidate metadata keys as `tcrv_rvv.low_precision_resource.*` for
target artifact validators.

The macro campaign remains open. Remaining milestones are adjacent
low-precision primitive/resource cleanup only where live consumers still blur
stable compiler facts with mirror/policy state, and future
measurement-disposition work only if fresh source-backed same-target RVV
evidence is introduced. The next continuation point is to inspect route
planning/provider-side low-precision resource consumers outside the target
support-bundle/artifact acceptance path, then return to selected-body
realization or primitive-surface coverage only after no remaining acceptance
consumer promotes policy/evidence fields.

## Completed Slice: Provider/Statement-Plan Stable-Resource Consumer Cleanup

This bounded follow-up slice completed the provider/route-planning-side
resource consumer cleanup named by the previous continuation point. The
remaining broad consumers were not target support-bundle or artifact validators;
they were provider-side preflight checks:

```text
RVVLowPrecisionContractionResourceSelection
  -> RVVLowPrecisionStableResourceCompilerFacts
  -> route-family provider-plan stable mirror check
  -> direct-contraction statement-plan stable preflight
  -> explicit measurement-disposition and selected-dispatch helpers only for
     admission/remediation/performance/no-win/dispatch policy state
```

The stable compiler-fact view now has an explicit equality helper. Route-family
provider-plan validation uses that helper when comparing route-description
low-precision resource mirrors against the validated family plan, so
description-only policy/evidence drift no longer blocks route construction as
if it were resource authority. Stable route/provider facts still fail closed
when selected candidate, realization, resource-cost/schedule, primitive, target
capability, legality, or rejection facts disagree.

The direct-contraction statement-plan owner now builds its packed-i4
statement-plan preflight from `RVVLowPrecisionStableResourceCompilerFacts`.
It still requires stable packed-i4 operand form, signedness, packing/unpack
facts, realization decision, region/phase facts, resource-cost facts, and
schedule facts before materialization. It no longer directly gates statement
construction on measurement-disposition remediation, performance feedback,
admission, maturity/no-win, selected-dispatch policy output, or dispatch
preference fields. Those policy/evidence fields remain validated by the
existing measurement-disposition and selected-dispatch policy helpers.

Focused C++ coverage now proves both boundaries: provider route-family mirror
validation ignores description-only policy/evidence drift while preserving
stable compiler-fact checks, and the direct-contraction statement-plan owner
still builds the packed-i4 low-shift/product/rescale/high-nibble-vwmacc
statement plan when only provider policy/evidence fields are stale.

The macro campaign remains open. The next continuation point is to run a final
targeted scan for any remaining low-precision primitive/resource consumer that
uses the broad resource aggregate outside explicit policy/evidence helpers. If
that scan is clean, return to adjacent low-precision primitive/resource surface
coverage or future measurement-disposition work only when fresh source-backed
same-target RVV evidence is introduced.

## Completed Slice: Remaining-Consumer Stable-Fact Guard Cleanup

This bounded follow-up slice completed the remaining-consumer cleanup named by
the previous continuation point. The targeted inventory found the remaining
production consumers of `RVVLowPrecisionContractionResourceSelection` in these
categories:

- Stable compiler authority: provider plan/description mirror equality, direct
  contraction route-construction preflight, direct statement-plan preflight,
  and target stable resource candidate mirror validation. These now consume
  `RVVLowPrecisionStableResourceCompilerFacts` for selected candidate,
  legality, live-vector budget, target capability mirrors, resource-cost/
  schedule, primitive, and realization facts.
- Derivation/verification: construction of the broad provider aggregate from
  typed body/config/runtime/pass facts, provider-owned resource verification,
  and stable view construction. These may still touch the broad aggregate
  because they create or verify the source record before stable view extraction.
- Policy/evidence helpers: measurement-disposition, selected-dispatch
  policy-output validation, no-win/maturity, admission, remediation, and
  metadata-only performance-claim rejection. These remain explicitly named
  policy/evidence boundaries and do not decide route, statement, stable
  schedule, support-bundle, or artifact acceptance.
- Mirror transport: route-planning metadata emission and support-bundle/header
  export still serialize stable resource mirrors from the stable view and
  serialize admission/remediation/performance/dispatch fields only in named
  policy/evidence mirror blocks.

The production cleanup removed the last direct stable-route consumers of the
policy-bearing aggregate in the focused surface:

```text
RVVLowPrecisionContractionResourceSelection
  -> RVVLowPrecisionStableResourceCompilerFacts
  -> provider-plan stable mirror equality
  -> direct route-construction selected-candidate/legal/budget preflight
  -> direct statement-plan packed/grouped candidate preflight
  -> target stable resource candidate mirror validation
```

The target stable resource candidate mirror validator now has a deleted broad
aggregate overload, so a future call that tries to validate stable
`tcrv_rvv.low_precision_resource.*` candidate mirrors directly from
`RVVLowPrecisionContractionResourceSelection` fails at compile time. Broad
aggregate usage remains valid only for derivation, explicit
measurement-disposition or selected-dispatch policy/evidence validators, and
policy/evidence mirror transport.

The macro campaign remains open. The remaining macro milestones are adjacent
low-precision primitive/resource surface coverage and future
measurement-disposition work only if fresh source-backed same-target RVV
evidence is introduced. The next continuation point is to return to adjacent
primitive/resource coverage or selected-body realization/resource-aware coverage
only after preserving this stable-fact guard in focused scans.
