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

## Current Round Slice: Gearbox Low-Precision Resource-Schedule Canonicalization

The current bounded slice follows the packed-i4 measurement-disposition
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

The cleanup must inspect the directly related packed-i4 dequantize and
dequant-clamp production route, Gearbox handoff, selected-body realization,
provider route planning, and target-validation surfaces. Stable schedule and
resource-cost facts must have one plugin-local compiler contract sourced from
typed body/config/resource facts. Measurement results, no-win conclusions,
source evidence IDs, remediation plans, performance admission, and dispatch
preference may remain only as measurement-disposition or policy evidence. They
must not participate in the stable schedule decision or target artifact
resource acceptance.

The concrete blocker found for this round is that
`isRVVLowPrecisionResourceAcceptedPackedI4StableScheduleDecision` still depends
on packed-i4 remediation plan fields while claiming to decide a stable schedule.
That can make measurement-disposition remediation facts look like permanent
schedule authority. This slice must concentrate the stable packed-i4 schedule
facts into a canonical helper/API consumed by the Gearbox candidate selector,
selected-body realization/provider route validation, and target artifact
resource validation. Measurement-disposition helpers must remain separate and
continue to fail closed for stale evidence/admission mirrors.

This round must not add q8/q4 route authority, artifact-name authority,
helper-only wrappers, source-front-door positive routes, Common EmitC semantic
inference, or measured-win/admission claims.

## Acceptance Criteria For This Slice

- Active task metadata, PRD, implementation/check context, and RVV
  low-precision spec sections identify the Gearbox low-precision
  resource-schedule canonicalization milestone under the same macro task.
- The packed-i4 stable schedule decision no longer depends on remediation,
  performance admission, beyond-local admission, dispatch preference, same-target
  evidence, or no-win/campaign outcome fields.
- A plugin-local compiler helper/API defines the accepted packed-i4 stable
  resource-schedule facts and is consumed by Gearbox candidate selection,
  provider route planning validation, and target artifact resource validation.
- Selected-body realization and provider payload exactness remains intact for
  primitive/config/resource facts: signed packed-i4 dequantize and
  dequant-clamp route descriptions still carry provider-owned resource facts,
  primitive payloads, realization facts, runtime ABI facts, stable schedule
  facts, and target mirrors before artifact acceptance.
- Measurement evidence and no-win/admission/remediation conclusions remain
  preserved as measurement-disposition or policy evidence only; no useful
  evidence is deleted.
- Common EmitC remains neutral and only carries provider-built payloads and
  mirrors.
- A bounded scan over touched production files, active task text, directly
  affected packed-i4 fixtures, and the RVV low-precision spec sections finds no
  Gate/admission/result/support/remediation wording that implies stable
  schedule/resource authority outside explicit measurement-disposition or
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
artifact validation now distinguish stable resource facts from
measurement-disposition evidence/admission mirrors. The remaining open boundary
for this round is the stable schedule decision itself: the current accepted
packed-i4 schedule helper still requires remediation planning facts, which
keeps policy/evidence fields too close to schedule authority. This round
completes a bounded canonicalization of that stable schedule/resource contract
without claiming new runtime, correctness, or performance evidence.

## Status After This Round

The Gearbox low-precision resource-schedule canonicalization slice is complete.
The accepted packed-i4 stable schedule decision now uses a single RVV
plugin-local `RVVLowPrecisionPackedI4StableResourceScheduleFacts` helper for
schedule-decision, unroll/accumulator/region/live-vector budget, and
resource-cost facts. That stable helper is consumed by Gearbox candidate
selection, Gearbox handoff/resource schedule verification, provider route
planning validation, and target artifact resource validation.

The stable schedule decision no longer depends on remediation plan,
performance admission, beyond-local admission, dispatch preference, no-win
classification, same-target evidence IDs, or campaign outcome strings.
Measurement-disposition admission/remediation evidence remains preserved and
validated through explicitly named policy/evidence helpers, including the
provider handoff structure checks and target artifact evidence/admission mirror
checks.

Focused C++ and lit coverage confirms packed-i4 dequantize/dequant-clamp,
signed/unsigned product-reduction payload exactness, and the generated-bundle
dry-run evidence boundary still pass. The macro campaign remains in progress.
The next continuation point is any adjacent low-precision primitive-surface gap
or a future measurement-disposition slice with fresh source-backed same-target
RVV evidence before any performance-preferred or measured-win claim.
