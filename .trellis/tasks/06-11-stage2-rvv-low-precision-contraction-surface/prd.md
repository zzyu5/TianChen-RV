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

## Current Round Slice: Gate 2 Selected-Body Realization

Gate 1 is complete for the typed low-precision primitive mirror/validation
foundation. The current bounded Gate 2 slice is to make one representative
low-precision product-reduction contraction body enter production through RVV
plugin-local selected-body realization instead of arriving only as an already
realized explicit `setvl` / `with_vl` body.

The representative is the signed i8 widening-product plus i16-to-i32 widening
reduction chain:

```text
selected tcrv.exec RVV variant
  -> typed pre-realized low-precision product-reduction body
  -> RVV contraction selected-body realization owner
  -> realized tcrv_rvv setvl/with_vl/load/widening_product/standalone_reduce/store body
  -> existing route provider / primitive facts / target artifact validation
```

This round must not add q8/q4 route authority, artifact-name authority,
helper-only wrappers, source-front-door positive routes, or common EmitC
semantic inference. If the full signed/unsigned realization surface is too
large, this slice lands the signed i8 representative and leaves unsigned u8
realization as the precise Gate 2 continuation point.

## Acceptance Criteria For This Slice

- Production source movement happens in the RVV dialect and RVV plugin-local
  selected-body realization owner, with downstream consumption by the existing
  route provider/target validation surfaces. Common EmitC remains neutral.
- A selected typed low-precision product-reduction body can be materialized by
  the contraction realization owner into explicit `tcrv_rvv` structure before
  route-family facts, route-control plans, statement plans, or
  `TCRVEmitCLowerableRoute` construction.
- The realization validates the representative body against provider-owned
  low-precision widening-reduction primitive facts for source signedness,
  source load/extension, source/product/accumulator/result SEW/LMUL,
  product-reduction relation, accumulator/result layout, policy, and runtime
  AVL/VL facts, failing closed with targeted diagnostics for stale or missing
  primitive facts.
- Focused lit/FileCheck coverage proves the pre-realized representative body is
  removed, realized `setvl` / `with_vl` / load / `widening_product` /
  `standalone_reduce` / store structure is produced, and existing provider
  primitive/resource mirrors reach emission-plan and target artifact checks.
- Focused negative coverage proves stale selected-body primitive facts are
  rejected before provider route construction or target artifact acceptance.
- Existing explicit signed i8 and unsigned u8 product-reduction target artifact
  coverage remains passing; unsigned u8 selected-body realization may remain a
  follow-up if not included in this slice.

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

- PRD and task context reflect the macro campaign, completed Gate 1 foundation,
  and current Gate 2 selected-body realization slice.
- Compiler source diff is in RVV dialect/plugin selected-body realization and
  any directly necessary provider/target consumers.
- Focused target tests pass or any missing toolchain is reported explicitly.
- `git diff --check` and `git diff --cached --check` pass.
- Macro task remains active with Gate 1 complete, this Gate 2 slice recorded,
  and remaining Gate 2/Gate 3/Gate 4
  continuation state.
- One coherent commit is created for the slice.

## Status After Current Gate 2 Signed Slice

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

The current Gate 2 signed representative slice is complete. The RVV dialect now
has `tcrv_rvv.typed_widening_product_reduce_pre_realized_body` for a signed i8
widening-product plus i16-to-i32 widening-reduction selected body. The RVV
contraction selected-body realization owner validates that pre-realized body
against provider-owned signed low-precision widening-reduction primitive facts,
materializes explicit `setvl` / `with_vl` / load / `widening_product` /
`standalone_reduce` / store structure, and then lets the existing route
provider, emission-plan mirrors, and target artifact validation consume the
realized body. Focused target coverage proves the pre-realized body is removed,
the realized structure is emitted, primitive mirrors reach emission-plan/header
export, stale source SEW is rejected before route construction, and existing
explicit signed i8 plus unsigned u8 product-reduction fixtures remain passing.

The macro task remains open. The precise Gate 2 continuation point is unsigned
u8 selected-body realization for the same product-reduction primitive surface,
or a bounded decision to leave unsigned u8 as explicit-body-only until the next
campaign gate. Gate 3 remains route provider/artifact/export carry-through
beyond this realization foundation, and Gate 4 remains later measured
same-target evidence/admission work only if fresh source-backed measurements
exist.
