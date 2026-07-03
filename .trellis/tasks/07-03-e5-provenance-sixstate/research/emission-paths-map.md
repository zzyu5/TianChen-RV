# Research: E5 — strong vs weak emission paths + the mechanical strong/weak discriminator

- **Query**: Map the STRONG (typed-primitive) and WEAK (descriptor-selected hand-written) emission paths with exact HEAD anchors; identify the mechanical discriminator that E6 currently cannot compute; state what a provenance manifest would serialize for each.
- **Scope**: internal
- **Date**: 2026-07-03
- **HEAD**: `c6ebe0b1401c7e218ee34b73510e91d47a550aa3` (branch `refactor/full-refactor-m1`)

> All line numbers below re-grepped at this HEAD. Prior-audit anchors (execution-doc §2 pins `7185a62b`) are stale — the file:line here supersede them.

---

## 0. The headline: what the discriminator actually is (and is NOT)

The load-bearing output of E5 is the mechanical strong-vs-weak rule. It is **NOT** "VerbatimOp
absent". Both paths emit `emitc.verbatim` and `emitc.call_opaque`, and BOTH ultimately lower to
pure `emitc` (where they are indistinguishable). The real discriminator lives **one stage earlier**,
at the *pre-realized / construction* stage:

- **STRONG** ⟺ the kernel body is realized as **typed `tcrv_rvv.*` dialect primitive ops**
  (`tcrv_rvv.widening_product`, `tcrv_rvv.standalone_reduce`, `tcrv_rvv.dequantize`,
  `tcrv_rvv.load`, `tcrv_rvv.store`, …), and that body carries a **complete role-step sequence**
  (a `semanticRoleGraph` of primitive-op names) with **no opaque hand-written helper**.
- **WEAK** ⟺ the body is constructed by a **descriptor** (`FlatBlockDotDescriptor` →
  `FlatDecodePrimitive` / `FlatFoldModel`) that selects among **hand-written C fragments**, emitted
  directly as `emitc.verbatim` / `emitc.call_opaque` (`__riscv_*` intrinsic strings) — there are
  **zero typed `tcrv_rvv` primitive ops** in the body.

The mechanical test already half-exists in code as a *rejecter*
(`rejectMixedPreRealizedContractionBody<allowlist>`) and a *sequence inspector*
(`collectSelectedExecutableRoleSequence`). E5 turns these from validators into a **classifier that
serializes the verdict**.

---

## 1. The STRONG (typed-primitive) construction path

### 1a. Files found

| File Path | Role |
|---|---|
| `lib/Plugin/RVV/RVVReductionSourceFrontDoor.cpp` | Front door for the K=32 signed i8 dot-reduce; **stamps `rvv_construction_protocol = "extension-family-construction-protocol.v1"`** on the emitted op (`:91-96`, `:470-471`) |
| `lib/Plugin/RVV/RVVDequantDotSourceFrontDoor.cpp` | Front door for the product-reduce-dequantize f32 shape; stamps `rvv_construction_protocol` (`:94`) |
| `lib/Plugin/RVV/RVVPackedI4DotSourceFrontDoor.cpp` | Front door for the offset-binary N=3 shape; stamps `rvv_construction_protocol` (`:106`); header role-graph comment `… -> tcrv_rvv.standalone_reduce (signed widening reduce, i16 -> i32)` (`:12`) |
| `lib/Plugin/RVV/RVVCodebookDotSourceFrontDoor.cpp` | Front door for the codebook N=3 (LUT) shape; stamps `rvv_construction_protocol` (`:109`) |
| `lib/Plugin/RVV/Construction/RVVContractionRouteIdentity.cpp` | **The static route registry** (routes 1–7, `:33-374`); each `ContractionRouteIdentity` = ordered `ContractionSourceSpec[]` with `bodyStepPosition` / `headOperandIndex` |
| `lib/Plugin/RVV/Construction/RVVConstructionProtocol.cpp` | **`kRetainedSelectedBodySpecializations[]`** (`:357`): op_kind → `semanticRoleGraph` (the `+`-joined typed-primitive chain) + emitc-route IDs; `getRVVSelectedBodyExecutableRoleSteps()` (`:4946`, `:4976`, `:4991`) returns the ordered `ExecutableRoleStep[]` |
| `lib/Plugin/RVV/EmitC/RVVEmitCContractionRouteFamilyPreRealizedValidators.cpp` | Pre-realized body validators; the **typed-op allowlist gate** `rejectMixedPreRealizedContractionBody<…>` (`:927-932`, `:1074-1083`, …) |
| `lib/Plugin/RVV/EmitC/RVVEmitCContractionRouteFamilyInternal.h` | `rejectMixedPreRealizedContractionBody<…>` template def (`:527-541`) |
| `include/TianChenRV/Plugin/ConstructionProtocol.h` | Shared `Manifest`, `SemanticRole`, `ExecutableRoleStep`, `SelectedExecutableRoleStep`, `collectSelectedExecutableRoleSequence` (`:51-223`) |
| `include/TianChenRV/Plugin/RVV/RVVConstructionProtocol.h` | RVV aliases; `getRVVSelectedBodyExecutableRoleSteps(...)` decls (`:113-119`) |
| `include/TianChenRV/Dialect/RVV/IR/RVVOps.td` | The typed primitive ODS ops: `standalone_reduce` (`:3394`), `widening_dot_reduce` (`:3591`), `widening_product` (`:3632`), `dequantize` (`:9185`), `TypedWideningProductReduceDequantizePreRealizedBodyOp` (`:1571`) |

### 1b. The three STRONG shapes and their primitive-ID sequences (already data)

From `kRetainedSelectedBodySpecializations[]` (`RVVConstructionProtocol.cpp:538-564`), the
`semanticRoleGraph` field IS the ordered pattern-primitive-ID list:

| six-state row | op_kind (route key) | `semanticRoleGraph` = ordered primitive IDs |
|---|---|---|
| `product_reduce / q4_0_nibble` | `widening_product_reduce_dequantize_f32` | `tcrv_rvv.widening_product` + `tcrv_rvv.standalone_reduce` + `tcrv_rvv.gearbox_cross_region_handoff` + `tcrv_rvv.dequantize` (`:550-552`) |
| `product_reduce / offset_binary_n3` | packed-i4 offset-binary N=3 route (front door `bounded_packed_i4_offset_binary_dot_source`) | same base chain, **spliced** with the N=3 extra product factor (qhi) via the **generic N-operand product-reduction role-step spec** (`:2149-2165`) + registry Route 4 (`RVVContractionRouteIdentity.cpp:164-187`) |
| `product_reduce / codebook_n3` | codebook-gather N=3 route (front door `bounded_codebook_gather_dot_source`) | same base chain, spliced with the extra product factor + the `ConstantTableLoad` aux **filtered out** (`:2163`); registry Route 5 (`RVVContractionRouteIdentity.cpp:229-258`) |

The nibble/offset-binary/codebook product *head* swaps to `tcrv_rvv.packed_i4_nibble_unpack_product`
in the packed-i4 flip (`RVVConstructionProtocol.cpp:1964`, `:3865`), but the reduce+dequant tail is
identical. The **generic N-operand spec** (`:2149-2165`) explicitly retired the former
per-route `isOffsetBinaryProductRoute`/`isCodebookProductRoute` gates: "A future N-operand
product-reduction route needs only a registry entry."

### 1c. The mechanical "no opaque helper" gate (already in code)

`rejectMixedPreRealizedContractionBody<RealizedOps...>`
(`RVVEmitCContractionRouteFamilyInternal.h:527-541`) walks `variant.getBody().front()` and errors if
any op OTHER than the allowlisted typed ops is present. For the product-reduce-dequantize strong
shape the allowlist is (`RVVEmitCContractionRouteFamilyPreRealizedValidators.cpp:1075-1083`):

```
SetVLOp, WithVLOp, LoadOp, WideningProductOp, StandaloneReduceOp,
GearboxCrossRegionHandoffOp, DequantizeOp, StoreOp
```

A body that passes this gate provably contains **only** typed `tcrv_rvv` primitives — i.e. **no
opaque hand-written helper**. This is exactly the "无不透明手写 helper" half of the strong test in
实验总纲 line 27. (Note: this is currently a *rejecter* invoked during validation; it is NOT emitted
as a verdict anywhere — that is the E5 gap.)

### 1d. The mechanical "manifest exists" proof (already in code)

`collectSelectedExecutableRoleSequence(spec)` (`ConstructionProtocol.h:221-223`, impl in
`lib/Plugin/Construction/ConstructionProtocol.cpp`) pairs each declared `ExecutableRoleStep`
(from `getRVVSelectedBodyExecutableRoleSteps`) with the actual `mlir::Operation*` in the emitted
role block, returning a `SelectedExecutableRoleSequenceInspection` whose `complete()` ⟺
`matchedRoleOps == steps.size()`. `complete() == true` ⟺ every declared primitive is a real typed
op in the body. **That is the machine-checkable "manifest exists ∧ maps onto typed ops" fact E6 needs.**

---

## 2. The WEAK (descriptor-selected hand-written) path

### 2a. Files found

| File Path | Role |
|---|---|
| `lib/Conversion/RVV/RVVToEmitCBlockQuantLinear.cpp` | The shared `emitFlatBlockDot` body (`:5334`) + `deriveFlatBlockDotDescriptor` (`:5222-5333`). Emits pure `emitc`: 65× `emitc::VerbatimOp`, 10× `emitc::CallOpaqueOp` (`__riscv_*` intrinsics), plus `emitc::Add/Mul/Cast/For/Expression` |
| `lib/Conversion/RVV/RVVToEmitCCodebookFp4.cpp` | Thin shims for `iq4_nl` (`:33`) and `mxfp4` (`:658`) that build a `FlatBlockDotDescriptor` with `hasCodebook`/`CodebookGatherNibble` and call the SAME `emitFlatBlockDot`. (`nvfp4` is a bespoke `emitNvfp4Q8_0BlockDot` — own layout — and is **dispatch-wired, not weak**.) |
| `lib/Conversion/RVV/RVVToEmitCInternal.h` | `enum class FlatDecodePrimitive` (`:73-79`), `enum class FlatFoldModel` (`:103-108`), `struct FlatBlockDotDescriptor` (`:117-154`), `deriveFlatBlockDotDescriptor` decl (`:162-163`) |

### 2b. The 7 constructed-weak rows all route through ONE emit site

`deriveFlatBlockDotDescriptor(op)` (`RVVToEmitCBlockQuantLinear.cpp:5222`) sets
`d.decodePrimitive` from the op `kind`:

| six-state weak row | `FlatDecodePrimitive` | fold model |
|---|---|---|
| `vec_dot / q8_0` | `PlainI8` (`:5280`) | `SumiTimesScales` |
| `vec_dot / q4_0` | `OffsetBinaryNibble` (`:5285`) | `LeftAssoc` |
| `vec_dot / q4_1` | `UnsignedNibble` (`:5290`) | `ScalePlusMin` |
| `vec_dot / q5_0` | `FiveBitOffsetBinary` (`:5295`) | `ScalesTimesSumi` |
| `vec_dot / q5_1` | `FiveBitOffsetBinary` (`:5301`) | `ScalePlusMin` |
| `vec_dot / iq4_nl` | `CodebookGatherNibble` + `hasCodebook` (`:5311`; via CodebookFp4 shim) | `SumiTimesScales` |
| `vec_dot / mxfp4` | `CodebookGatherNibble` + `hasCodebook` + `E8M0` scale (`:5322`) | `SumiTimesScales` |

The decode switch (`:5531-5628`) emits the arithmetic as `emitOpaqueCall` → an `emitc.verbatim`
step-comment + an `emitc.call_opaque` with a raw `__riscv_*` intrinsic callee (e.g. `vwmul`,
`vrgather_vv_i8`, `vand_vx`). **These `FlatDecodePrimitive` cases ARE the "被描述符选择的手写片段"
(descriptor-selected hand-written fragments) of 实验总纲 line 27 / [L-8].**

### 2c. What marks the weak body opaque

- **Zero `tcrv_rvv.*` typed primitive ops.** The body never constructs `widening_product` /
  `standalone_reduce` / `dequantize`; it emits `emitc.call_opaque("__riscv_vwmul…")` directly.
- **No `rvv_construction_protocol` attr, no `semanticRoleGraph`, no role-step sequence.**
  `getRVVSelectedBodyExecutableRoleSteps` has nothing to return for these ops (they are not
  selected-body construction routes).
- The decode/fold identity lives in a **`FlatBlockDotDescriptor` enum field**, i.e. a *selection
  among hand-written templates*, not a *composition of typed primitives*.

---

## 3. The mechanical discriminator, stated precisely

Applied at the **pre-realized / construction stage** (before final EmitC lowering), per kernel:

```
strong (== six-state "constructed")      ⟺  role-step sequence is COMPLETE
                                             (collectSelectedExecutableRoleSequence.complete())
                                             ∧ rejectMixedPreRealizedContractionBody passes
                                               (body has zero non-allowlisted / opaque op)

weak   (== six-state "constructed-weak")  ⟺  body constructed from a FlatBlockDotDescriptor
                                             (FlatDecodePrimitive / FlatFoldModel selected)
                                             ∧ body is pure emitc (verbatim/call_opaque),
                                               zero typed tcrv_rvv primitives
```

Equivalently, the construction path itself knows which it is: the four typed-body **front doors**
(`RVV{Reduction,DequantDot,PackedI4Dot,CodebookDot}SourceFrontDoor.cpp`) produce STRONG; the
descriptor-driven **`emitFlatBlockDot`** (`RVVToEmitCBlockQuantLinear.cpp`) produces WEAK. The
cleanest, undebatable capture is at those two emit sites (see `provenance-manifest-design.md`).

### Why E6 cannot compute this today

`coverage_metrics.py` is stdlib-only and, by the implementation-stack red line, **"does NOT read or
touch any C++/ODS, and does NOT derive strong-vs-weak from code"** (`coverage_metrics.py:8-9`). The
strong/weak fact lives in C++ (typed-op body vs descriptor) and in the transient pre-realized IR —
neither is visible to the Python tool. E6 therefore hand-labels every strong/weak row with
`auto_readout:"pending-E5"`. E5's job is to **serialize the C++-side verdict into an artifact the
Python tool can read**.

---

## 4. What a provenance manifest serializes for each path

| | STRONG (constructed) | WEAK (constructed-weak) |
|---|---|---|
| primitive-ID list | ordered `semanticRoleGraph` typed-op names (e.g. `tcrv_rvv.widening_product` → `standalone_reduce` → `gearbox_cross_region_handoff` → `dequantize`) — **already exists** as `kRetainedSelectedBodySpecializations` data + `getRVVSelectedBodyExecutableRoleSteps` | the descriptor fragment IDs (`FlatDecodePrimitive::OffsetBinaryNibble`, `FlatFoldModel::LeftAssoc`) — **needs a tiny new emit** |
| opaque-helper flag | `false` (rejectMixed passes) | `true` (emits `emitc.verbatim`/`call_opaque`) |
| construction-protocol | `rvv_construction_protocol = "extension-family-construction-protocol.v1"` already stamped | none |
| derived six-state | `constructed` | `constructed-weak` |

## Caveats / Not found

- **CONFIRM AT IMPLEMENT (advisor item 1):** I confirmed the machinery covers the N=3 offset-binary
  and codebook shapes *generically* (`RVVConstructionProtocol.cpp:2149-2165`, `:6344-6373`), and that
  `widening_product_reduce_dequantize_f32` has a full explicit `semanticRoleGraph` (`:550-552`). I did
  **not** execute `getRVVSelectedBodyExecutableRoleSteps` for the two N=3 op names to byte-confirm a
  complete non-empty sequence, nor confirm whether they resolve via `findRouteByTypedComputeOpNameRaw`
  directly or via the op_kind route. Implement MUST verify all three strong shapes return a complete
  role sequence; if the offset-binary/codebook op names are not directly reachable, that is a small
  net-new lookup, not net-new construction.
- The strong export-e2e tests take a high-level *source* op as input
  (`test/Target/RVV/{non-deferred-wide-product-reduce-dequantize-f32,packed-i4-offset-binary-dot-product-reduce,codebook-gather-dot-product-reduce}-front-door-export-e2e.mlir`;
  a 4th `q4-k-q8-k-super-block-…` exists but is the Fork-C super-block shape = deprioritized,
  not one of the 3 strong rows). op_kind is derived inside the front door, not visible in the test input.
- `rejectMixedPreRealizedContractionBody` is a template rejecter keyed per body-op-kind; there is no
  single generic "is this body all-typed?" entry point yet — E5 either reuses the per-kind allowlists
  or adds one classifier.
