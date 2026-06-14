# Research: N1 RVV coverage + capability-profile divergence state

- **Query**: Current N1 RVV feature coverage + the capability-profile-driven DIVERGENCE state, vs the goal "覆盖增量 toward full coverage" + the spec's N1 evidence bar ("同一 kernel 在多个真实 profile 上，被 capability 查询导向不同的合法性/选择/dispatch 结果").
- **Scope**: internal (read-only)
- **Date**: 2026-06-14

## Verdict (one line)

The capability-driven legality DIVERGENCE *mechanism* is real (4 query axes: `supported_sew`/`supported_lmul`/`required_tail_policy`/`required_mask_policy`) and is exercised one-directionally in negative lit tests — but it is (a) **wired only to hand-authored test-fixture attributes, never to the real probe→capability path**, and (b) **never demonstrated as the SAME kernel diverging across two profiles in a single artifact**; so against the spec's "真实 profile" N1 bar it is currently synthetic and unproven.

## Findings

### 1. Coverage today — what converts + what is lit-covered

**Conversion architecture** (`lib/Conversion/RVV/RVVToEmitC.cpp`): one giant `VariantToEmitCFunc` `OpConversionPattern` (registered `patterns.add<VariantToEmitCFunc>` at `RVVToEmitC.cpp:5521`) that walks the typed `tcrv_rvv` body and dispatches per op-kind through `emit*` helpers. There is **one** pattern, not per-op patterns; "coverage" = the set of `emit*` op-kinds it legalizes.

**RVV dialect op-kinds defined** (`include/TianChenRV/Dialect/RVV/IR/RVVOps.td`) and converted (each has an `emit*` in `RVVToEmitC.cpp`):

| Family | Ops | Converter helper |
|---|---|---|
| Envelope | `SetVLOp`, `WithVLOp`, `RuntimeABIValueOp`, `VSetVLRegionMarkerOp` | loop/config scaffolding |
| Load/Store | `LoadOp`, `StoreOp`, `BroadcastLoadOp` | `emitLoad`/`emitStore`/`emitBroadcastLoad` |
| Strided | `StridedLoadOp`/`StridedStoreOp` (+`Masked*`) | `emitStridedLoad`/`emitStridedStore` |
| Indexed/gather-scatter | `IndexLoadOp`, `IndexedLoadOp`/`IndexedStoreOp` (+`Masked*`) | `emitIndexLoad`/`emitIndexedLoad`/`emitIndexedStore`/`emitIndexByteScale` |
| Segment | `Segment2LoadOp`/`Segment2StoreOp` (+`Masked*`) | `emitSegment2Store` etc. (**segment2 only**) |
| Mask memory | `MaskLoadOp`, `MaskedLoadOp`/`MaskedStoreOp`, `MaskAndOp` | `emitMaskLoad`/`emitMaskedLoad`/`emitMaskedStore`/`emitMaskAnd` |
| Elementwise | `BinaryOp`/`MaskedBinaryOp`, `SplatOp`, `MoveOp`/`MaskedMoveOp` | `emitBinary`/`emitMaskedBinary`/`emitSplat`/`emitMove` |
| Compare/select | `CompareOp`, `SelectOp` | `emitCompare`/`emitSelect` |
| Reduce | `ReduceOp`, `StandaloneReduceOp`/`MaskedStandaloneReduceOp` | `emitReduce`/`emitStandaloneReduce` |
| MAcc | `MAccOp`/`MaskedMAccOp` | `emitMAcc`/`emitMaskedMAcc` |
| Widening | `WideningProductOp`, `WideningMAccOp`, `WideningDotReduceOp`/`MaskedWideningDotReduceOp`, `WideningConvertOp` | `emitWideningProduct`/`emitWideningMAcc`/`emitWideningDotReduce`/`emitWideningConvert` |
| Low-precision / Gearbox | `DequantizeOp` (+chain), gearbox handoff/region markers | `emitDequantize`/`emitDequantizeChain` |
| Legacy fixed-i32 | `I32*Op` (Add/Sub/Mul/CmpEq/Select/Load/Store/BroadcastLoad) | early fixed-i32 path |

Roughly matches the PRD's "~16 RVV families." Lit fixture density (`test/Conversion/EmitC/` = 56 `.mlir` files; counts by family): mask 93, select 81, compare 28, binary 21, broadcast 20, splat/move 16, macc 15, strided 14, reduce 12, widening 11, segment2 6, indexed 6, gather 5, scatter 4, elementwise 3, dequant 2.

**dtype / SEW / LMUL support limits** (the real "full coverage" gap):

- **dtype** (`vectorDType`, `RVVToEmitC.cpp:468-495`): signed `i8/i16/i32/i64`, unsigned `u8/u16/u32/u64`, float **`f32` only**. **`f64` returns ""** (unsupported) despite spec profiles listing fp64 (`RVVToEmitC.cpp:492` is the only float branch; `isF32()` is the only float predicate used at :531/:1697/:5406/:5494). **No fp16 / bf16 / zvfh.** Those strings appear in the repo ONLY as `hasRVVVectorHint`/`isaVectorHint` text checks (`RVVCapabilityProfile.cpp:110-111`, `RVVEmitCRoutePlanning.cpp:106`) and as a `verify.mlir` "unknown capability @zvfh" negative — **no fp16/bf16 conversion path exists.**
- **LMUL**: most families are pinned to **`m1`** (e.g. `RVVToEmitC.cpp:1492`, :2481, :2548, :2619, :2942 all `!= "m1"` → match-fail). `maskWidthForConfig` (`:440-450`) supports only `{sew32,sew64} × {m1,m2}`; everything else returns 0 → unsupported. So effective config space ≈ **SEW∈{32,64}, LMUL∈{m1, m2 in a few elementwise paths}.** No `mf2/mf4/mf8`, no `m4/m8`.
- **Segment**: segment2 only (no segment3/4; no `vlseg3/4`).

**Features NOT covered (the gap toward "full RVV")**: f64, fp16/bf16 + zvfh, LMUL widths beyond m1/m2 (fractional mf*, m4, m8), segment3/4, `vrgather`/permute/slide, fixed-point/saturating (`vsadd`/`vssub`/`vssra`/`vsmul`), fault-only-first (`vleff`), vector-vector divide/min/max beyond the binary mnemonics wired, narrowing/clipping shifts. (Confirmed absent: grep for `vrgather|vfslide|vsadd|vssub|vssra|fault.only.first|vleff|segment3|segment4|vlseg3|vlseg4` over lib/include/test returned no implementation hits.)

### 2. Profile divergence — the N1 evidence state

**Is there a divergence mechanism?** Yes, and it is genuine capability *querying* (not metadata-only):

- **Converter gate** `checkCapabilityConfigGate` (`RVVToEmitC.cpp:2032-2085`): resolves each `requires` symbol to a provider op in the kernel, reads its `supported_sew` / `supported_lmul` comma-allow-lists, and `notifyMatchFailure`s the body if the typed SEW/LMUL is excluded ("capability gates this body out").
- **Route-planning gate** `verifyRVVSelectedTargetCapabilityForTypedConfig` (`RVVEmitCRoutePlanning.cpp:253-303`): four capability-vs-typed-body axes that hard-error on mismatch — `supported_sew`, `supported_lmul`, `required_tail_policy`, `required_mask_policy`. Facts come from `collectRVVSelectedTargetCapabilityFacts` (`:158-247`) reading provider `getProperty(...)`.

**Are there MULTIPLE real profiles?** Spec defines four target profiles in `.trellis/spec/capability-model/profiles.md`: `rvv-main` (the real `ssh rvv` hw, RVV 1.0, zvl128b), `k3-ime` (later, N2), `riscv-sophgo-offload` (offload). Only `rvv-main` is a real present RVV profile. The C++ probe→capability builder `buildRVVTargetCapabilitiesFromProbeFacts` (`RVVCapabilityProfile.cpp:276-341`) produces ONE capability set from probe facts.

**Is the SAME kernel routed to DIFFERENT outcomes on two profiles?** **No single artifact demonstrates it.** Evidence:
- Negative: `test/Conversion/EmitC/rvv-target-capability-selected-route-gating-negative.mlir` — a `supported_sew = "64"` provider rejects a SEW=32 macc body (`// CHECK: no registered backend ... fully legalizes`). One profile, reject only.
- Negative legality set: `test/Transforms/PluginVariantLegality/rvv-target-capability-selected-route-gating-invalid.mlir` — empty/missing/unavailable/wrong-kind/ambiguous-provider rejections. All negative.
- Positive fixture: `test/Target/RVV/explicit-selected-body-artifact-scalar-broadcast-macc-add.mlir` — a `--tcrv-materialize-emission-plans` PLAN/HEADER test, single `@rvv_profile` provider with `supported_sew="32" supported_lmul="m1"` admitting a SEW32/m1 body. Different kernel, different file, different pass — **not** the paired "accept on A, reject on B" demo. (It also carries a scalar-fallback dispatch case, a separate mechanism.)

So: divergence is exercised **one-directionally (negative only)**, across **separate kernels in separate files**, never as one kernel × two profiles.

### 3. The legality/selection/dispatch path — does it actually query capability to diverge?

- **Legality**: YES diverges. The two gates above are real capability queries; SEW/LMUL/tail/mask mismatch is a hard reject. Legality alone satisfies N1's "合法性/选择/dispatch" wording, so it is a *legitimate* N1 vehicle — IF on real profiles.
- **HEADLINE GAP — the divergence axes are not wired to the real probe path.** `grep -rn "supported_sew|supported_lmul|required_tail_policy|required_mask_policy" lib/ include/` returns ONLY read sites (the two gates) + the four name-constants; **zero writers** in `lib/`. `RVVCapabilityProfile.cpp` (the authoritative probe→`TargetCapabilitySet` conversion) sets `architecture`, `isa_vector_hints`, `count`, `bytes`, toolchain versions, `selected_march/mabi` — and **none** of the four divergence axes (grep on that file = empty). The only producers of `supported_sew`/`supported_lmul` are **hand-authored `tcrv.exec.capability`/`tcrv.exec.target` attributes in test fixtures**. ⇒ A real probed `rvv-main` capability set **cannot exercise the gate**; the divergence is reachable only from synthetic IR. This is the precise miss against the spec's "真实 profile" bar.
- **Selection among variants**: ABSENT (state as fact, not flaw). `RVVExtensionPlugin::proposeVariants` returns empty (`RVVExtensionPlugin.cpp:446-452`); `collectVariantProposals` only `addRecoverableDecline` (`:454-465`). No capability-driven *picking among candidate variants*; the typed body is pre-materialized upstream and the plugin validates/legalizes/costs the single given body (`estimateVariantCost` returns a flat 1.0, `:495-512`).
- **Dispatch**: `tcrv.exec.dispatch` exists (case + fallback, e.g. the positive fixture) but the case/fallback split is authored, not capability-derived per profile.

### 4. Proposed bounded N1 coverage-increment target

A) **The missing N1 evidence artifact (highest leverage, smallest lift)**: one kernel × two capability profiles in a single lit file (or split-input), showing **divergent legality outcome** driven by capability query — e.g. profile A `supported_sew="32"` ACCEPTS, profile B `supported_sew="64"` REJECTS the identical SEW32 body (and/or an LMUL/tail/mask axis). This directly instantiates the spec's "同一 kernel … 不同的合法性/选择/dispatch."

B) **Close the probe→gating-facts gap (makes A real, not synthetic)**: have the real RVV probe→capability path (`buildRVVTargetCapabilitiesFromProbeFacts` / a derived profile capability) actually carry `supported_sew`/`supported_lmul` (and optionally tail/mask) facts from `rvv-main` probe data, so a real profile exercises the gate. Without B, A remains test-fixture-only and fails the "真实 profile" bar. (Honor I5/I6: probe stays sanitized facts; the plugin-local C++ profile validator is the authority that fills `TargetCapabilitySet`.)

C) **Coverage features toward "full RVV"** (bounded, pick the divergence-bearing ones first): f64 conversion (`vectorDType` f64 rung + intrinsics); then a second config axis to make divergence non-trivial — e.g. LMUL m4/m8 or fractional `mf2` (extend `maskWidthForConfig` + drop the `!= "m1"` pins). Larger later: fp16/bf16 + zvfh (a real capability-gated extension → strong N1 divergence story: zvfh-present profile legalizes the f16 body, zvfh-absent profile rejects it), segment3/4, `vrgather`/permute, fixed-point/saturating, fault-only-first.

**Recommended minimal "N1 lamp"**: A + B on the SEW axis with the existing macc/elementwise kernel (no new conversion needed) → first real two-profile divergence; then C's zvfh-or-f64 axis to make divergence carry a *coverage* increment, not just a config reject.

## Caveats / Not Found

- I did not build/run; all claims are static (file:line). The gates' runtime behavior on `ssh rvv` is not verified here (and the N1 bar is about compile-time capability divergence, which is what the gates implement).
- "Selection among variants is absent" is a description of the current plugin (`proposeVariants` empty), not a judgment — the typed-body-first construction protocol is by design (see `collectVariantProposals` recoverable-decline contract).
- Whether any *third-party* pass writes `supported_sew` outside `lib/`/`include/` (e.g. a Python tool) was not exhaustively traced; the grep covered `lib/` + `include/` (the authoritative C++ capability path) and `test/`.
