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

## Campaign Gates

1. Typed low-precision primitive surface contract exists and is consumed by RVV
   plugin validation for i8/u8 element/config, low-precision loads,
   signed/unsigned extension, widening product, widening accumulation or
   reduction facts, policy, runtime AVL/VL, and fail-closed unsupported
   combinations.
2. Plugin-local selected-body realization can materialize one representative
   low-precision contraction body from that primitive surface without q8/q4
   route authority.
3. Route provider and artifact/export consumers carry the primitive facts
   through `TCRVEmitCLowerableRoute` and target validation without common EmitC
   semantic inference.
4. Measured same-target comparison or dispatch admission, if later claimed, uses
   only fresh source-backed evidence and the conservative no-win boundary from
   the previous campaign.

## Repository Findings

- The previous Gearbox/resource-aware campaign is archived and complete for its
  scope. It closed Gate 4 no-win admission and dispatch/fallback consumption
  boundaries.
- Existing production code already has route-supported signed i8 and unsigned
  u8 widening product/reduction representatives, dequant/dequant-clamp
  resource paths, MAcc paths, low-precision resource selections, and target
  stale-mirror rejection.
- The current Gate 1 gap is not a missing q8 wrapper. It is that low-precision
  primitive facts are validated partly as ordinary `tcrv_rvv.*` route fields
  and partly as `tcrv_rvv.low_precision_primitive.*` mirrors. The primitive
  namespace does not yet carry its own source/product/accumulator/result
  SEW/LMUL plus policy/runtime AVL/VL mirror set, even though those facts are
  required by the primitive contract and already exist in provider-derived route
  facts.

## Current Round Slice: Gate 1 Foundation

Harden the existing production-owned low-precision primitive contract for
product-reduction representatives. The bounded change is to carry and validate
the primitive's typed config/runtime facts in the
`tcrv_rvv.low_precision_primitive.*` mirror namespace, derived from the same
selected typed body/config/runtime facts already used by the RVV provider.

This round does not add a new op family, q8/q4 route, artifact-name authority,
or common EmitC inference. It reuses existing signed i8 and unsigned u8
product-reduction representatives as positive/fail-closed Gate 1 evidence.

## Acceptance Criteria For This Slice

- Production source movement happens in RVV provider route planning and target
  validation/support-bundle consumers, not in common EmitC semantic inference.
- Route metadata for low-precision widening-product/reduction primitives
  includes provider-derived primitive source/product/accumulator/result
  SEW/LMUL plus tail policy, mask policy, runtime control plan, and runtime AVL
  source under `tcrv_rvv.low_precision_primitive.*`.
- Target artifact validation compares those primitive mirrors against the
  provider-owned contract before accepting the artifact.
- Focused FileCheck coverage proves positive signed i8 and unsigned u8 mirrors
  and at least one fail-closed stale runtime AVL/VL or policy/typed config
  primitive mirror.
- Existing stale source signedness, extension, product/reduction, accumulator,
  resource, and Gate 4 no-win boundaries remain untouched unless directly
  affected.

## Non-Goals

- No q8/q4-named route ids, artifact names, helper wrappers, or source-front-door
  positive routes.
- No high-level Linalg/Vector/StableHLO frontend work.
- No broad dtype/LMUL clone batch.
- No measured-win or performance-preferred admission without fresh source-backed
  same-target evidence.
- No dashboard, report-only, metadata-only, IME, Offload, TensorExt, Scalar, or
  future-plugin work.
- No runtime/correctness/performance claim and therefore no required `ssh rvv`
  evidence in this slice.

## Definition Of Done

- PRD and task context reflect the macro campaign and this Gate 1 slice.
- Compiler source diff is in RVV provider/target path.
- Focused target tests pass or any missing toolchain is reported explicitly.
- `git diff --check` and `git diff --cached --check` pass.
- Macro task remains active with Gate 1 slice completion and Gate 2-4
  continuation state.
- One coherent commit is created for the slice.

## Status After Current Gate 1 Slice

Gate 1 foundation is complete for the current product-reduction primitive
contract hardening slice. RVV provider route metadata now mirrors
source/product/accumulator/result SEW/LMUL plus tail policy, mask policy,
runtime control plan, and runtime AVL source under
`tcrv_rvv.low_precision_primitive.*` whenever a low-precision primitive
contract is present.

Target support-bundle export maps those primitive mirrors, and target artifact
validation rejects stale primitive typed-config/runtime mirrors before artifact
acceptance. Focused signed i8 and unsigned u8 product-reduction lit coverage
checks positive primitive mirrors and stale runtime AVL/policy rejection.

The macro task remains open. Gate 2 should next move from this primitive mirror
foundation to plugin-local selected-body realization of one representative
low-precision contraction body from the primitive surface. Gate 3 remains route
provider/artifact/export carry-through beyond this foundation, and Gate 4
remains later measured same-target evidence/admission work only if fresh
source-backed measurements exist.
