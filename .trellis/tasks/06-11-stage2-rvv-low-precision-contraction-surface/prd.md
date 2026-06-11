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
- The current gap is language and ownership drift: stale numbered-milestone terminology still
  appears in the active task/spec and in production low-precision helper names
  or diagnostics, where it can imply route, schedule, support, or admission
  authority rather than a project milestone or measurement-disposition record.

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

## Current Round Slice: Selected-Body Realization Admission/Evidence Boundary Cleanup

This bounded slice cleans the handoff between selected-body realization
compiler facts and packed-i4 measurement-disposition policy/evidence facts.
The current code already carries explicit stable schedule/resource facts, but
the selected-body realization/provider handoff still has dense
admission/remediation/performance fields near the route-realization fact gate.
That can make policy evidence look like resource or route support authority
unless the boundary is explicit in helper names, diagnostics, and validation
call sites.

The source-backed field classification for this slice is:

- Stable compiler authority: typed `tcrv_rvv` body/config/runtime facts,
  selected resource candidate facts, packed load/unpack facts, primitive-chain
  facts, realization producer/decision, realized region/phase facts, Gearbox
  cross-region handoff structure, stable packed-i4 schedule/resource-cost
  facts, provider route payloads, route-family plan mirrors, target capability
  mirrors, and target resource validation.
- Policy/evidence facts: realization admission proof, remediation handoff,
  remediation plans, performance feedback, same-target measurement evidence
  IDs, performance admission closure/reopen facts, beyond-local admission,
  maturity/no-win outcome, dispatch preference, and selected-dispatch
  performance policy mirrors.

Production behavior for this slice must enforce that stable compiler facts are
validated by resource/realization helpers, while policy/evidence facts are
validated only by explicitly named measurement-disposition policy/evidence
helpers. If policy/evidence attrs remain serialized for target validation, they
must stay grouped as policy/evidence mirrors and fail closed as stale mirrors;
they must not satisfy route, schedule, resource, or artifact acceptance.

This round must not add q8/q4 route authority, artifact-name authority,
helper-only wrappers, source-front-door positive routes, Common EmitC semantic
inference, or measured-win/admission claims.

## Acceptance Criteria For This Slice

- Active task metadata, PRD, implementation/check context, and RVV
  low-precision spec sections identify the selected-body realization
  admission/evidence boundary cleanup under the same macro task.
- Source-backed field classification is recorded for stable compiler authority
  versus policy/evidence facts at the selected-body realization handoff.
- Provider-side route-realization helpers validate typed/resource/realization
  compiler facts, packed load/unpack facts, stable schedule/resource-cost
  facts, and handoff structure without consuming performance feedback,
  remediation, same-target evidence, no-win, admission, or dispatch preference
  fields as route/resource/schedule acceptance facts.
- Packed-i4 measurement-disposition fields remain preserved and validated
  through explicitly named policy/evidence helpers, including realization
  admission proof, remediation planning, performance feedback/admission,
  maturity/no-win, same-target evidence ID, and dispatch preference.
- Target artifact validation keeps stable resource mirrors and
  measurement-disposition evidence/admission mirrors in separate helpers; stale
  policy/evidence mirrors continue to fail closed as policy/evidence mirrors,
  not as resource/schedule facts.
- Focused negative coverage proves stale or missing policy/evidence cannot
  satisfy route, schedule, resource, or artifact acceptance and is diagnosed at
  the named measurement-disposition boundary.
- Common EmitC remains neutral and only carries provider-built payloads and
  mirrors.
- A bounded scan over touched production files, active task text, directly
  affected packed-i4 fixtures, and the RVV low-precision spec sections finds no
  admission/remediation/performance/measurement/same-target/no-win wording that
  implies compiler authority outside explicit measurement-disposition or
  policy/evidence contexts. Immutable historical evidence path strings may
  remain only as evidence IDs.
- Focused packed-i4 dequantize/dequant-clamp lit coverage remains passing,
  alongside the signed/unsigned product-reduction payload coverage needed to
  guard the existing provider payload exactness.
- `tcrv-opt`, `tcrv-translate`, `tianchenrv-rvv-extension-plugin-test`, and
  `tianchenrv-target-artifact-export-test` build; the two C++ test binaries run.
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

The remaining open boundary for this round is the selected-body realization
handoff itself: packed-i4 admission, remediation, performance, measurement,
same-target, no-win, and dispatch-preference fields remain serialized near
realization/provider handoff checks. This round makes those fields explicit
measurement-disposition policy/evidence facts while keeping resource and
schedule acceptance anchored in stable compiler facts.

## Status After This Round

The selected-body realization admission/evidence boundary cleanup slice is
complete. Provider-side realization compiler-fact gates now validate
typed/resource/realization, packed load/unpack, Gearbox handoff, stable
schedule, and resource-cost facts without consuming packed-i4
admission/remediation/performance/measurement/no-win/dispatch fields as route,
schedule, or resource acceptance facts.

Packed-i4 measurement-disposition fields remain preserved and are read through
explicit policy/evidence helpers. Missing policy/evidence attrs now fail with a
named measurement-disposition policy/evidence diagnostic before route-plan
construction can treat them as compiler facts, and stale target artifact mirrors
remain handled by the existing separate evidence/admission mirror validators.

The macro campaign remains in progress. The next continuation point is adjacent
low-precision primitive-surface cleanup or a future measurement-disposition
slice only when fresh source-backed same-target RVV evidence is introduced.
