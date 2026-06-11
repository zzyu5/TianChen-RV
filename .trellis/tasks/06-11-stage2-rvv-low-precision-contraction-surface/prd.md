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

## Current Round Slice: Outcome-Language And Measurement-Disposition Boundary Cleanup

The current bounded slice normalizes milestone language and authority
boundaries after the first provider/artifact carry-through payload slice.
Compiler authority remains:

```text
selected tcrv.exec RVV variant
  -> typed pre-realized low-precision product-reduction body
     carrying source_signedness plus source/product/accumulator/result config
  -> RVV contraction selected-body realization owner
  -> realized tcrv_rvv setvl/with_vl/load/widening_product/standalone_reduce/store body
  -> RVV provider-owned primitive route payload
  -> TCRVEmitCLowerableRoute / neutral EmitC metadata mirrors
  -> support-bundle export / target artifact validation exact mirror checks
```

The cleanup must remove or quarantine stale numbered-milestone authority from the
active task/spec and the directly related low-precision production
provider/artifact surfaces. Measurement results, no-win conclusions, source
evidence IDs, and admission conclusions may remain only as
measurement-disposition or policy evidence. They must not become route support,
schedule authority, dtype/config authority, artifact-name authority, or
selected-body authority.

This round must not add q8/q4 route authority, artifact-name authority,
helper-only wrappers, source-front-door positive routes, Common EmitC semantic
inference, or measured-win/admission claims.

## Acceptance Criteria For This Slice

- Active task metadata, PRD, implementation/check context, and RVV
  low-precision spec sections use outcome/milestone language:
  foundation outcome, selected-body realization outcome,
  provider/artifact carry-through outcome, and measurement-disposition outcome.
- Production low-precision helper names, diagnostics, and target validation
  labels touched in this slice no longer use numbered-milestone terminology
  where that name implies route, schedule, support, artifact, or admission
  authority.
- Provider payload and target validation exactness from `65f67e38` remains
  intact: signed and unsigned product-reduction route descriptions still carry
  a provider-owned low-precision primitive route payload, and target artifact
  validation still compares candidate mirrors against that payload before
  artifact acceptance.
- Measurement evidence and no-win/admission conclusions remain preserved as
  measurement-disposition or policy evidence only; no useful evidence is
  deleted.
- Common EmitC remains neutral and only carries provider-built payloads and
  mirrors.
- A bounded scan over the active task, RVV low-precision spec sections, touched
  production files, and directly affected tests finds no uppercase
  numbered-milestone terminology except immutable historical evidence path
  strings or archived text outside this active slice.
- Focused signed and unsigned product-reduction artifact lit coverage remains
  passing.
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
