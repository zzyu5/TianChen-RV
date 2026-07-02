# Research: Track-B emitter-body auto-construction — scope & plan

- **Query**: Where do the block-dot op EMITTER BODIES live and how are they produced today? How far does body auto-construction already go? What capability-facts + descriptor fields would drive a generated body? Sketch q8_0. Feasibility verdict + first step + design forks.
- **Scope**: internal (code trace + plan)
- **Date**: 2026-07-02
- **Task frame**: this is the design scoping for pillar **P2** (generic construction full-coverage) in `prd.md` — "a generic, capability-driven construction mechanism that auto-generates all kernel bodies, not per-kernel hand-written monoliths." The PRD's body-shape taxonomy C1–C8 is the coverage target; this doc is about the C2–C7 emitter bodies.
- **READ-ONLY**: no code was modified.

---

## TL;DR — the crux, stated at paper grade

There are **two distinct notions of "body"** in this codebase, and conflating them is the trap:

1. **The RVV-dialect body** inside `tcrv_rvv.with_vl` (what the front door constructs). For the **decomposed** routes (dequant / C1) this is a straight-line of *generic composable micro-ops* (`load → widening_product → standalone_reduce → dequantize → store`). For the **monolithic** block-dots (C2–C7) this is **ONE typed op** (`tcrv_rvv.q8_0_q8_0_block_dot`, `…q4_0_q8_0…`, …) carrying block-format facts as attributes.
2. **The EmitC body** — the actual C arithmetic (load / nibble-unpack / widening-product / reduce / per-block fp16 scale). For decomposed routes this is produced by *walking the generic slices* through composable slice-emitters. For monolithic block-dots this is produced by **ONE hand-written `emitQxxx` method per op** in `RVVToEmitCBlockQuantLinear.cpp` — this is what "mirrors ggml `_generic` C" and is what is STILL hand-written.

**What the 24 front doors already auto-construct is #1 (the scaffold + typed op).** What Track-B DEEPENING targets is #2 (the emitter body) — replacing the N bespoke `emitQxxx` methods with a mechanism that constructs the emitter body from capability-facts + a block-format descriptor.

The honest frontier is **not** the vague "facts drive the *how* (schedule) not the *what* (arithmetic)." What the code actually shows is finer and stronger:

- **Facts/attrs ALREADY drive a real chunk of "the what"**: `qk`, `weight_block_stride`, `activation_block_stride`, `quant_byte_offset`, `activation_high_byte_offset` drive block geometry (`blockBaseValue`, `nb = n/qk`, `blockLen = qk` vs `qk/2`); and `deriveBlockDotFacts` bakes gearbox-selected LMUL / unroll-factor / strip-elision into the emitted vector types and loop shape. This is descriptor-parameterized today.
- **What is STILL selected by OP IDENTITY** (via the `kBlockDotKernels` recognizer→method table): (a) the **decode primitive** (offset-binary vs unsigned-nibble vs plain-`vwmul` vs 5-bit-`qh`-merge vs gather vs bit-dance), and (b) the **scale/fold model** (`d_x*d_y`-first vs left-assoc `((sumi*d_x)*d_y)`).

So the deepening is precisely: **lift decode-primitive + scale-model selection from `op-identity → bespoke emitQxxx` to `descriptor-field → parameterized primitive dispatch`.** That is the paper-grade statement of the frontier and it is what the feasibility verdict below turns on.

---

## §1 — WHERE the emitter bodies live and how they are produced today (trace)

### 1a. Front door → scaffold + ONE attr-carrying op (DONE)
`lib/Plugin/RVV/RVVQ80BlockDotSourceFrontDoor.cpp` (representative). From a marked generic source func carrying only the ggml `vec_dot` **operator identity** (signature `func(%s: memref<?xf32>, %n: index, %vx: memref<?xi8>, %vy: memref<?xi8>)`, body may be a bare `return`), `materializeKernel` (:413-496) builds the whole `tcrv.exec.kernel + capability + variant + setvl + with_vl + dispatch/fallback` scaffold, and `createBlockDot` (:269-288) stamps ONE op:

```cpp
mlir::OperationState state(loc, tcrvrvv::GgmlBlockDotQ80Q80Op::getOperationName());
state.addOperands({weight, activation, out, n, vl});
state.addAttribute("kind", builder.getStringAttr("ggml_q8_0_q8_0_block_dot"));
state.addAttribute("scale_model", builder.getStringAttr("dual-fp16-per-block-d_x.d_y"));
state.addAttribute("qk", builder.getI64IntegerAttr(kQK));                       // 32
state.addAttribute("weight_block_stride", builder.getI64IntegerAttr(kWeightBlockStride));   // 34
state.addAttribute("activation_block_stride", builder.getI64IntegerAttr(kActivationBlockStride)); // 34
state.addAttribute("quant_byte_offset", builder.getI64IntegerAttr(kQuantByteOffset));       // 2
```

Note the shape knobs (`integer_core_lmul` / `multi_block_factor` / `strip_elision`) are deliberately **left OFF** — the attr-less op lowers at the emitter's default LMUL, and the gearbox/schedule pass refines them later (see §3). This scaffold-construction is DONE for all 24 ops.

### 1b. The family table + recognition (DONE)
`include/TianChenRV/Plugin/RVV/RVVMonolithicBlockDotFamily.h`. The per-op family table `monolithicBlockDotOpTable()` (:180-299) has 24 rows: `{opName, routeFamily (SuperBlock|Flat), kind, abiRoles}`. `findSelectedMonolithicBlockDotBody(withVL)` (:329-343) recognizes "the entire `with_vl` body is exactly one supported block-dot op." This is the "trait at the C++ level" — generic wiring, per-op data.

### 1c. Emission plan = DIAGNOSTIC MIRROR ONLY (not the arithmetic)
`lib/Plugin/RVV/RVVExtensionPlugin.cpp`. `buildVariantEmissionPlan` (:802-886) calls `findSelectedMonolithicBlockDotBody` and, on a hit, dispatches to `buildMonolithicBlockDotEmissionPlan` (:490-538). Read the load-bearing comment at :445-448:

> "The real emitted bytes come from the existing block-dot emitter unchanged; materialize-emission-plans only APPENDS the diagnostic mirror…"

So the emission-plan stage attaches route-id / runtime-ABI / artifact metadata (keyed by route family + the op's `kind`/`scale_model` attrs) — it does **not** produce the arithmetic. The arithmetic is elsewhere.

### 1d. The arithmetic emitter body — ONE hand-written method per op (STILL hand-written)
`lib/Conversion/RVV/RVVToEmitC.cpp:333` — the dispatch table `kBlockDotKernels` (**43 entries**, first-match-wins, a recognizer→method pair per kernel):

```cpp
static constexpr BlockDotKernel kBlockDotKernels[] = {
    {&isQ4_0Q8_0BlockDotBody,  &VariantToEmitCFunc::emitQ4_0Q8_0BlockDot},
    ...
    {&isQ8_0Q8_0BlockDotBody,  &VariantToEmitCFunc::emitQ8_0Q8_0BlockDot},
    {&isQ4_1Q8_1BlockDotBody,  &VariantToEmitCFunc::emitQ4_1Q8_1BlockDot},
    {&isQ5_0Q8_0BlockDotBody,  &VariantToEmitCFunc::emitQ5_0Q8_0BlockDot},
    ... };
```

The methods live in `lib/Conversion/RVV/RVVToEmitCBlockQuantLinear.cpp` (7466 LOC; flat-plain + Q4_0-GEMM/repack), `RVVToEmitCKQuant.cpp` (4736 LOC; K-quant super-blocks), `RVVToEmitCGridCodebook.cpp` / `RVVToEmitCCodebookFp4.cpp` / `RVVToEmitCTernaryBinary.cpp` (codebook / ternary / binary).

**Form of the body: typed-MLIR construction, NOT a C string template.** Each `emitQxxx` builds `emitc` dialect ops via `rewriter.create<emitc::…>` — `emitQ8_0Q8_0BlockDot` alone is a ~375-line sequence; the file has **1149 `emitc::` builder calls vs 135 verbatim/opaque strings**. The only opaque (string) pieces are the RVV intrinsic call names (`__riscv_vwmul_…`, `vwredsum`, `vle8`, the `(float)*(const _Float16*)` scale read) emitted as `emitc.call_opaque` / `emitc.verbatim` — i.e. the intrinsics themselves are strings, but the *control/data structure* (loops, variables, expressions, casts, address arithmetic) is first-class typed `emitc`. See `emitQ8_0Q8_0BlockDot` at `RVVToEmitCBlockQuantLinear.cpp:6769-7144`.

### 1e. Target side — the export path exists for BOTH monolithic route families (machinery, not e2e proof)
`lib/Target/RVV/RVVTargetSupportBundle.cpp` (explicitly in the trace list). The target-artifact-export layer (:1792-1955+) registers a purpose-built candidate validator `validateRVVMonolithicBlockDotTargetArtifactCandidate` (:1840) plus bare-object exporters keyed by BOTH monolithic route ids — `monolithicBlockDotFamilyForRouteID` (:1824-1837) maps the route id to `SuperBlock` **or** `Flat`, and the validator checks the honest monolithic metadata mirror (route id, archetype, op-kind→ABI-roles via `findMonolithicBlockDotOpEntryByKind`). Header comment :1804-1812: "the bare object exporters are keyed by the monolithic route ids, so [every decomposed route stays on its own ids and never reaches this validator]. Each monolithic plan produces [an object]."

**Read this precisely**: the export *machinery* (validator + exporter) is present for both families. That is NOT the same as "every one of the 24 ops is production-export e2e sealed." What I verified is the presence of the emission-plan `getSupported` path (§1c) + this target-export validator/exporter. I did NOT run the full source→object chain per op, and the PRD marks super-block C5 as byte-exact-witnessed but **未接 production** (not production-connected). See §7 and Caveats for the per-bucket export-status honesty.

**Verdict on question 1**: the emitter body is a **hand-written typed-MLIR (`emitc` dialect) builder construction, one method per op**, that mirrors ggml `_generic` C, with RVV intrinsics as opaque call strings. It is NOT a C string template and NOT a set of composable generic micro-ops.

---

## §2 — How far body auto-construction ALREADY goes (the honest frontier)

### 2a. The dequant / C1 rung = the model for a genuinely auto-constructed body
`lib/Plugin/RVV/RVVDequantDotSourceFrontDoor.cpp`. This front door does **more** than the block-dot front doors: it auto-constructs the *whole RVV-dialect body out of generic composable micro-ops* (the body builder at :410-…):

```
load i8<core>  →  widening_product i16<wide>  →  standalone_reduce i32m1  →  dequantize (i32→f32 runtime-scale)  →  store f32
```

Two facts make it a real auto-construction and the model to imitate:

1. **The integer-core LMUL is the RETURN VALUE of a capability-fact-driven function** — `selectIntegerCoreLMUL` (:366-399), keyed off `march`/VLEN, not a hand-picked constant. (Header comment :7-12: "The integer-core LMUL anchor is the RETURN VALUE of … front door's selectIntegerCoreLMUL … VLEN256, NOW with the i32→f32 dequant fused in.")
2. **The EmitC side is a slice-walker, not a bespoke monolith.** `lib/Conversion/RVV/RVVToEmitCDeferredDequant.cpp:104-105` literally: "Walk the scope into ordered product/reduce slices + the dequant epilogue"; `emitDequantProductReduceSlice` (:30) + `emitLoad` compose generic per-slice emitters. Dispatch is a *separate* table `kDequantKernels` (`RVVToEmitC.cpp:548-564`) — the dequant body is **NOT** one of the 43 `kBlockDotKernels` bespoke methods.

So C1 already achieves: *facts → LMUL width* + *body assembled from generic composable slices* + *production-export end-to-end* (VLEN128 m2/m4 + VLEN256 m1/m2, per `prd.md` C1 = ✅ and the P1 finding — and C1 is the **single proven e2e Track-B body** per the P1 finding + memory).

### 2b. What the block-dot ops borrow from this vs where they stop
- The **LMUL/unroll/elision baking** already generalizes to block-dots via `deriveBlockDotFacts` (see §3) — the *schedule* dimension is descriptor-driven for all 24 ops.
- The **arithmetic** does NOT. The block-dot decode/scale is a monolithic `emitQxxx` method, not composable slices. The dequant body's arithmetic is trivial (plain `vwmul`); block-dot arithmetic (nibble-unpack, offset-binary, `qh` 5th-bit merge, gather, super-block bit-dance, per-block fp16 scale loop) is richer and is exactly the part still written by hand.

**Honest current frontier**: *schedule facts* (LMUL/unroll/elision) + *block geometry* (strides/offsets/qk) are already descriptor-driven for the whole zoo; the *decode primitive* + *fold/scale model* are still selected by op identity and hand-written per op. The dequant rung proves a fully fact-driven, composable-slice body is achievable **for the trivial-arithmetic case** — extending that to the rich decode primitives is the open work.

---

## §3 — Capability facts + format-descriptor fields that would drive a generated body

Two groups. Group A already exists in the code (proof they are the right axes); Group B is what a generated flat-plain body would additionally need to parameterize.

### Group A — already threaded (attrs + gearbox facts)
Block-format descriptor (typed I4 op attrs, read straight off the op — see `emitQ8_0Q8_0BlockDot:6818-6824` and `emitQ4_0Q8_0BlockDot:94-99`):
- `qk` — elements per block (32 for flat, 256/`QK_K` for super-block)
- `weight_block_stride`, `activation_block_stride` — AoS block byte strides (e.g. 34 for `block_q8_0`, 18 for `block_q4_0`)
- `quant_byte_offset` — where quants start within a block (past the fp16 scale)
- `activation_high_byte_offset` — present only when the weight packs two nibbles that pair with two activation halves (q4_0); absent for q8_0
- `activation_quant_byte_offset` — second offset for formats whose activation quant sits elsewhere (q5_0)
- `scale_model` — string tag (`dual-fp16-per-block-d_x.d_y`, …) that today only names the fold; would become the descriptor field selecting the fold primitive

Capability / schedule facts (baked by the gearbox schedule pass, read via `deriveBlockDotFacts`, `include/TianChenRV/Conversion/RVV/RVVToEmitCSupport.h:377-395`):
- `integer_core_lmul` — i8 source LMUL anchor (`mf4|m1|m2`); drives the widened i16 product LMUL (`m1→m2`, `m2→m4`, `mf4→mf2`) and the `vle`/`vsetvl` spelling
- `multi_block_factor` — outer-loop unroll factor (`1|2|4`)
- `strip_elision` — `robust` (inner strip loop, VLEN-universal, sumi-carry seed) vs `elided` (single `vsetvl`, VLEN≥128 only)

The derivation of these facts from raw capability is `getRVVStripVLMAXElements` (single truth source, cited in `emitQ4_0Q8_0BlockDot:66`) — i.e. VLEN → VLMAX-in-elements → (elided? factor? anchor?).

### Group B — the additional descriptor fields a generated flat-plain body needs
- **`decode_primitive`** enum, selecting the already-factored helper:
  - `plain_i8` → `vwmul` (q8_0)
  - `offset_binary_nibble` → `emitOffsetBinaryDecodeProductValue` (q4_0: `xor 0x88` + `sll/sra` sign-extend + `vwmul`+`vwmacc`)
  - `unsigned_nibble` → `emitUnsignedNibbleDecodeProductValue` (q4_1: `vand 0x0F` + `vsrl` + reinterpret)
  - `five_bit_offset_binary` → `emitFiveBitOffsetBinaryDecodeProductValue` (q5_0: nibble + `qh` 5th-bit merge, width-parametrized on `coreLmul`/`wideLmul`)
- **`block_len`** = `qk` (q8_0, whole block) vs `qk/2` (q4_0/q4_1/q5_0, nibble half-block) — derivable from `decode_primitive`, not an independent field
- **`num_activation_loads`** = 1 (q8_0) vs 2 (`y0`,`y1` halves for offset-binary) — derivable from `decode_primitive` + `activation_high_byte_offset` presence
- **`fold_model`** enum: `dx_dy_first` (q8_0: `sumf += (float)sumi * (d_x*d_y)`) vs `left_assoc` (q4_0: `sumf += ((float)sumi*d_x)*d_y`) vs `scale_plus_min` (q4_1/q5_1 Family-B, add the MIN term). Byte-exactness of fp non-associativity is pinned by this field, so it MUST be explicit (not inferred).

These are the exact quantities the existing hand-written methods read/branch on; enumerating them as a descriptor is the mechanization.

---

## §4 — q8_0 representative: what auto-constructing its body requires; the flat-plain common shape; where divergence enters

### 4a. Why q8_0 is the right representative
q8_0 is **2-operand, flat-plain, no codebook, no super-block, already in `kBlockDotKernels`, and the CORE emit is byte-exact today**. It therefore **isolates Seam A (emitter-body auto-construction) from Seam B (the route-arity wall, §7)**: a generated q8_0 body needs nothing from the blocked C3 route work, and — because a Fork-B refactor changes no op / no ABI arity / no route identity — it preserves whatever production-export status q8_0 has today by construction (§6a). (Note: "byte-exact CORE emit" ≠ "production-export e2e sealed" — do not read this as an e2e claim; see §7 and Caveats.)

### 4b. The common flat-plain body shape (q8_0 skeleton, `emitQ8_0Q8_0BlockDot:6769-7144`)
Every flat-plain emitter is structurally this skeleton (line cites from the q8_0 method):
1. `float sumf = 0.0f;` (:6835)
2. `size_t nb = n / qk;` (:6844)
3. block loop `for (ib=0; ib<nb; …)` — factor-1 (:7076) or multi-block unroll main+tail (:7091-7123), folds kept in strict ascending order for fp byte-exactness
4. per block: `blockBaseValue` address arithmetic `xb = vx + ib*stride` (:6853-6864); two `fp16Read` scale reads (`(float)*(const _Float16*)`, :6867-6871)
5. **integer core** `emitIntegerCore` (:6933): `int32_t sumi=0`; then either the *elided* core (one `vsetvl`, one strip reduce, :6950-6968) or the *robust* core (inner strip loop over the block bytes with VLMAX step + sumi-carry seed, :6970-7009)
6. **strip reduce** `emitStripReduce` (:6878): load chunks → decode/product → `vmv_v_x` seed (lane0 = carried sumi or 0) → `vwredsum` → `vmv_x_s` extract
7. **fold** `emitFold` (:7019): grouped into one `emitc.expression` so `mlir-translate` renders a single C statement and the compiler fuses the SAME FMA ggml does under `-ffp-contract`
8. `*s = sumf;` (:7126-7140)

Steps 1,2,3,4(addr),5,6(seed/reduce/extract),7(expression grouping),8 are **byte-for-byte the same shape** across q4_0/q8_0/q4_1/q5_0/q5_1. I verified q4_0 (`:29-455`) and q5_0 (`:455-…`) share this skeleton verbatim (same lambda names, same loop structure).

### 4c. Where per-op divergence enters (exactly three points)
1. **The decode+product inside `emitStripReduce`** (step 6). This is the ONLY arithmetic divergence, and it is **already factored into named helper functions**:
   - q8_0: inline `vwmul` i8×i8 (`:6899-6902`) — no decode
   - q4_0: `emitOffsetBinaryDecodeProductValue` (`RVVToEmitC.cpp:2570`) = `emitOffsetBinaryDecodeValue` (`:2592`: `vxor 0x88` → `vsll 4`/`vsra 4` low, `vsra 4` high) + `emitOffsetBinaryProductFromDecodedValue` (`:2629`: `vwmul` + `vwmacc`), loads `y0`,`y1` (`:181-183`)
   - q4_1: `emitUnsignedNibbleDecodeProductValue` (`:2649`: `vand 0x0F` + `vsrl 4` + reinterpret u8→i8)
   - q5_0 ("high-bit"): `emitFiveBitOffsetBinaryDecodeProductValue` — nibble decode PLUS the `qh` 5th-bit merge (a `vid`+shift vector indexing the per-element `qh` bit, width-parametrized on `coreLmul`/`wideLmul` so wide/narrow are byte-identical; `emitQ5_0Q8_0BlockDot:490-501`). q5_0 also reads `activation_quant_byte_offset` (`:524`).
2. **`block_len`** — `qk` (q8_0) vs `qk/2` (nibble formats), one line (`:6824` vs q4_0 `:99`).
3. **`fold_model`** in `emitFold` — `d_x*d_y`-first (q8_0, `:7035-7042`) vs left-assoc (q4_0) vs +MIN (q4_1/q5_1). One `emitc.expression` builder shape per model.
   - **⚠ CORRECTION (found during Fork-B impl, commit-side truth):** the fold count is **4, not 3**, and it MUST key off `kind`, NOT `scale_model`. q8_0 (`(float)sumi*(d_x*d_y)`, cast emitted first — `SumiTimesScales`) and q5_0 (`(d_x*d_y)*(float)sumi`, scales-mul emitted first — `ScalesTimesSumi`) are a **distinct emitc-op sequence** (different operand AND emission order) yet BOTH carry `scale_model="dual-fp16-per-block-d_x.d_y"`. Grouping them (as this doc originally did) breaks byte-exact. The 4 folds: `SumiTimesScales`(q8_0) / `LeftAssoc`(q4_0) / `ScalesTimesSumi`(q5_0) / `ScalePlusMin`(q4_1==q5_1). Fork B (commit-pending) implemented this correctly.

**So auto-constructing q8_0's body from facts = ONE `emitFlatBlockDot(facts, descriptor)` method that emits the shared skeleton (4b) and dispatches steps 6/2/7 on `decode_primitive` / `block_len` / `fold_model`.** The `decode_primitive` dispatch targets ALREADY exist as factored helpers. q8_0 is the `plain_i8` / `block_len=qk` / `fold=dx_dy_first` instance.

---

## §5 — FEASIBILITY verdict (brutally honest, per bucket)

**There IS a parameterizable common body-shape template for the flat-plain bucket. It does NOT generalize cleanly to super-block / codebook / IQ-gather.** Estimates below are of "fraction of the emitted body that is the shared parameterizable skeleton vs op-specific bespoke residue," read from the actual methods.

| Bucket (PRD class) | Ops | Mechanical (shared skeleton) | Bespoke residue (named) | Verdict |
|---|---|---|---|---|
| **Flat-plain (C2/C3)** | q4_0, q8_0, q4_1, q5_0, q5_1 | **~85%** — steps 1-5,6-frame,7-frame,8 identical | decode primitive (4, all already factored helpers) + fold model (3) + `block_len` flag | **TRACTABLE.** One `emitFlatBlockDot(descriptor)` + a 4-entry decode-primitive dispatch + 3-entry fold dispatch. Byte-exact achievable because the emitted `emitc` ops are unchanged. |
| **Flat-codebook (C4)** | iq4_nl, mxfp4, nvfp4 | **~60%** — same block loop / fold / store; decode replaced by a gather | 16-entry codebook table broadcast + `vluxei` gather + fp4 scale reconstruction (E8M0 / UE4M3 sub-block). A bigger "decode step" but still a swap on the same skeleton. | **PLAUSIBLE as a 2nd primitive class**, not free. Gather + scale-recon is one factored primitive per sub-family, not a one-liner. |
| **IQ-gather (C6)** | iq1_s, iq1_m, iq2_*, iq3_* | **~40%** | `vluxei16` grid gather + sign codebook (`signs64` — currently blocked as an op-attr per memory) + ternary grid unpack. Not a primitive swap: the gather+sign decode is structurally distinct per format. | **HARD.** Each format is close to bespoke; shared skeleton is thin. |
| **Super-block K-quant (C5)** | q4_K, q5_K, q6_K, q2_K, q3_K | **~30%** | NOT a flat skeleton. `QK_K=256` super-block loop, 6-bit scale/min **bit-dance**, `aux32` accumulation, deferred fp32 fold/min. q4_K alone is **8 sub-methods** in the table (`emitQ4_KNibbleUnpack` / `…ScaleMinBitDance` / `…ScaledDot` / `…MinTerm` / `…SumsFoldScaleD` / `…HorizontalFold` / `…Aux32Partial` / `…BlockDot`, `RVVToEmitC.cpp:394-409`). | **BESPOKE-DOMINATED.** The bit-dance is genuinely irreducible to a primitive swap; a generator here is a *different, larger* mechanism (a super-block sub-op vocabulary), not the flat template. |
| **Ternary/binary** | tq1_0, tq2_0, q1_0 | ~35% | base-3 trit unpack / 2-bit-field+bias / binary sign mask (`vlm` → i8 sign) | Each a special decode; treat as its own primitive. |

**Bottom line for the verdict**: the flat-plain bucket (C2/C3, the q8_0 family) is a real, high-mechanical common template and is the correct first target. The claim "one generic mechanism generates the whole zoo" is **honest for flat-plain, aspirational for codebook, and a materially larger separate effort for super-block/IQ-gather** — the bit-dance and grid-gather are not decode-primitive swaps on a shared skeleton. Do not let the paper claim "the whole zoo" from a flat-plain demonstrator; the defensible claim is "capability-facts + a block-format descriptor generate the flat-plain contraction bucket byte-exact, and the decode-primitive/fold-model become descriptor fields rather than op identity."

---

## §6 — First implementable step + the key design forks

### 6a. The PRIMARY fork (this is the architectural decision, not the C-string one)
- **Fork A — decompose to composable micro-ops** (the dequant/C1 model): lower block-dot arithmetic into generic first-class RVV micro-ops (a `nibble_unpack`, an `offset_binary_decode`, a `widening_product`, a `qh_merge`, …) that the existing *slice-walker* emits, and have a body-builder assemble them from the descriptor. This is the **purest "capability-driven"** framing and the one the paper's strongest claim leans toward (body = composed generic ops, not a monolith).
  - **Cost/risk**: requires a large NEW first-class micro-op vocabulary + verifiers, and — critically — for offset-binary it collides with **Seam B** (§7): a 3-input product-reduction needs the N-operand route redesign that pillar P1 is about. Byte-exact recomposition against 429 lit is the risk.
- **Fork B — parameterize the monolith** (recommended first step): KEEP the typed `GgmlBlockDot*` op, replace the N per-op `emitQxxx` methods with ONE `emitFlatBlockDot(rewriter, scope, facts, descriptor)` that emits the shared skeleton and dispatches decode/fold on descriptor fields (the factored helpers already exist).
  - **Cost/risk**: LOW and byte-exact-tractable — same `emitc` ops, skeleton already shared, primitives already factored. **It is independent of Seam B by CONSTRUCTION**: refactoring the internals of `emitQxxx` changes no op, no ABI arity, and no route identity, so it touches neither the route-planning layer nor the target-export validator — it *preserves whatever production-export status each op has today, sealed or not*. (This is a stronger and safer argument than "the ops already export," which this doc does not claim — see §7.)

**Recommendation**: Fork B for flat-plain as the first landed step (it is where q8_0 sits and it demonstrates "descriptor → body" byte-exact without the route wall). Note in the paper that Fork A is the north-star spirit and that the two forks meet only when the N-operand route (P1) lands.

### 6b. The SECONDARY fork (orthogonal) — recommend, don't leave open
Body-as-typed-MLIR-construction (`emitc` builder) **vs** body-as-emitted-C-string-template. **Recommend keeping typed-MLIR.** Byte-exact migration against the 429-test lit is far easier emitting the *same `emitc` ops* the current methods emit (the harness already renders them identically via `mlir-translate`) than reproducing `mlir-translate`'s textual output byte-for-byte from a string template. A string template would also lose the `emitc.expression` FMA-grouping guarantee that pins fp byte-exactness (`emitFold`). Tradeoff: a string template would be more legible/ggml-like, but the byte-exact-during-migration constraint dominates.

### 6c. First implementable step (concrete, byte-exact-gated)
1. Define a `FlatBlockDotDescriptor` (the Group-B fields of §3) and a `deriveFlatBlockDotDescriptor(op)` that reads the op's attrs (mirrors `deriveBlockDotFacts`, adds `decode_primitive`/`fold_model` from `kind`/`scale_model` + attr presence).
2. Extract the shared skeleton of `emitQ8_0Q8_0BlockDot` into `emitFlatBlockDot(…, descriptor)`, with the decode step calling a `switch(descriptor.decode_primitive)` over the existing helpers and the fold step over a `switch(descriptor.fold_model)`.
3. Re-point q8_0 (the `plain_i8` instance) at it FIRST; prove byte-exact under a forced clean rebuild + BEFORE==AFTER on the q8_0 lit (per `build-incremental-unreliable` memory — absolute fingerprints are stale, use equality).
4. Migrate q4_0, then q4_1/q5_0/q5_1 one at a time, each byte-exact-gated. Each migration deletes one `emitQxxx`, leaving the `kBlockDotKernels` entry pointing at `emitFlatBlockDot` with a descriptor (or a thin shim).

### 6d. How it plugs into the in-flight collapsed table-driven front-door (P1)
- The 24 front doors are being collapsed to a table-driven front door (pillar P1 / the `RVVMonolithicBlockDotFamily.h` table is the seed — per-op DATA + generic wiring). The emitter-side `FlatBlockDotDescriptor` is the **mirror** of that front-door table on the EmitC side: the front-door table row (`opName, routeFamily, kind, abiRoles`) already carries most of what the descriptor needs; the natural end state is ONE descriptor definition consumed by BOTH the front door (to stamp the op) and the emitter (to generate the body).
- **Sequencing**: Fork B for flat-plain can land BEFORE P1 completes (it is emitter-local). But Fork A / any *decomposed* offset-binary path, and any attempt to make the descriptor the single source across front-door + route + emitter, **is gated on P1** closing the N-operand route (see Seam B).

---

## §7 — The two hard seams (name both; the second is already located)

### Seam A — emitter body is a monolith, not composable (THIS doc)
Block-dot arithmetic is a bespoke `emitQxxx` per op, not composable micro-ops. Feasibility = can the flat-plain skeleton + pluggable decode/fold primitive be parameterized byte-exact. **Verdict: yes for flat-plain (§5).** This seam is *emitter-local* and does not require route/export changes for the monolithic ops.

### Seam B — the N-operand route-arity multi-validator wall (upstream, IN-FLIGHT = P1)
This blocks the **decomposed** offset-binary route (C3, `PackedI4OffsetBinaryXI8ProductOp`), i.e. Fork A for q4_0. It does **not** block Fork B (§6a): Fork B is byte-exact-by-construction and never changes route arity. From `research/P1-root-cause-multivalidator-FINDING.md` (self-contained, precise cites):

- The offset-binary product op is **3-input** (weight + `activation_low` + `activation_high`; `RVVOps.td:3814-3871`; front door `RVVPackedI4DotSourceFrontDoor.cpp:575,578,606-623`). The production product-reduction route is a **2-input (lhs×rhs)** ABI model, baked into **multiple independent mirror-validators**:
  1. **Config-binding** — `RVVEmitCRouteConfigBinding.cpp:1582-1592` (hardcodes `source.lmul=MF4` / `product.lmul=MF2`); `assignRVVGenericLoadBinding` (`:2763-2765`) rejects a 2nd `rhs-input-buffer`.
  2. **Route-facts** — `RVVEmitCContractionRouteFamilyPlanOwners.cpp` (`buildRVVWideningDotReduceRouteFacts:474-516`; `getRVVWideningDotReduceRouteFacts:1060-1066` drops actual source/product LMUL).
  3. **Construction-protocol conformance** — `lib/Plugin/RVV/Construction/RVVConstructionProtocol.cpp`: `appendWideningProductReduceAddRoleSteps` (**:2142**) emits a 2-load/5-runtime_abi (12-step) spec; the offset-binary body has 3 loads/6 runtime_abi → role-step 12 rejected; `verifyRVVSelectedBodySelectedRoleSequence` (**:6367**); `typedComputeOpName` canonicalizes to the wrong op.
- **The finding**: q4_0 decomposed-route e2e is a **deliberate multi-validator redesign**, not one bounded arc. A 10-file route-family layer was built ADDITIVELY (verified dormant, 429/429 lit green) and then hit the construction-protocol wall; it was DISCARDED as a diagnostic, not committed. There is probably a 3rd subsystem (emit/role) downstream.

**Why this matters for the plan**: Fork B (parameterize the monolith) is *independent of Seam B by construction* — it is the low-risk first step and can land now. Only the *decomposed/composable* offset-binary path (Fork A, the purest paper claim) requires closing Seam B, and closing it is the multi-quarter N-operand route redesign (pillar P1). If the paper wants "body composed of generic ops" for offset-binary specifically, that is gated on P1; if it wants "descriptor-parameterized emitter, byte-exact, capability-driven schedule + geometry," Fork B delivers that for flat-plain now.

---

## Caveats / Not found

- **Production-export status is per-bucket, NOT uniform — do not over-claim.** What I verified: the emission-plan `getSupported` path (§1c) and the target-export validator/exporter registered for both monolithic route families (§1e, `RVVTargetSupportBundle.cpp:1792-1955`). What I did NOT verify: a per-op source→object e2e seal. The honest per-bucket status is: **dequant/C1 = proven e2e** (single proven Track-B body); **decomposed offset-binary/C3 = BLOCKED (Seam B)**; **super-block K-quant/C5 = byte-exact-witnessed but 未接 production** (per PRD, even though the target-export machinery accepts the super-block route id); **flat-plain monolithic = emission-plan-supported + target-export machinery present, but per-op e2e seal UNVERIFIED here**. Fork B's safety does not depend on resolving this (it preserves current status by construction), but the paper must not assert flat-plain e2e without a run.
- **`RVVTargetSupportBundle.cpp`** was in the explicit trace list; I characterized it by grep + reading the export-layer comment block (:1792-1955), not a full read of the exporter body. If e2e-seal status per op becomes load-bearing, deep-read the exporter + run the source→object chain for the target op.
- **Timescale honesty (per PRD "Scale 诚实")**: Fork B flat-plain is a bounded, weeks-scale, byte-exact-gated refactor of 5 emitter methods. Fork A for offset-binary + the "single descriptor across front-door/route/emitter" end state is gated on P1's multi-validator redesign = multi-quarter. Super-block/IQ-gather generation (C5/C6) is a *separate, larger* mechanism, not the flat template. Do not conflate these in a schedule.
- **Byte-exact gate**: absolute fingerprints in memory (`f810ce6b`/`cb04b219`) are STALE; use forced clean rebuild + BEFORE==AFTER equality (memory `build-incremental-unreliable`). ODS `RVVOps.cpp.inc` regenerates each build and `tcrv-opt` sometimes fails to relink.
- **Not exhaustively read**: I read the q8_0, q4_0, q5_0 flat emitters in full/part and the offset-binary/unsigned-nibble decode helpers; I did NOT deep-read the K-quant (`RVVToEmitCKQuant.cpp`) or codebook (`RVVToEmitCGridCodebook.cpp`) methods line-by-line — the §5 super-block/codebook fractions are estimated from the dispatch-table sub-method count (`RVVToEmitC.cpp:368-419`) and the family-header structural notes, not a full read of those emitters. If C5/C6 becomes the target, deep-read those before committing fractions.
- **`selectIntegerCoreLMUL` internals** (dequant) read only via the front-door header comment + P1 finding; the exact capability→LMUL mapping function body was not opened (not needed for this scope — the point is it is fact-driven, not constant).

---

## Key file map (absolute paths)

| Path | Role |
|---|---|
| `/home/kingdom/phdworks/TianchenRV/lib/Plugin/RVV/RVVQ80BlockDotSourceFrontDoor.cpp` | Representative front door: scaffold + attr-carrying op construction (`createBlockDot:269`, `materializeKernel:413`) |
| `/home/kingdom/phdworks/TianchenRV/include/TianChenRV/Plugin/RVV/RVVMonolithicBlockDotFamily.h` | 24-op family table + recognition; the "per-op data / generic wiring" seed for P1 |
| `/home/kingdom/phdworks/TianchenRV/lib/Plugin/RVV/RVVExtensionPlugin.cpp` | Emission plan = diagnostic mirror only (`buildMonolithicBlockDotEmissionPlan:490`, comment :445-448) |
| `/home/kingdom/phdworks/TianchenRV/lib/Target/RVV/RVVTargetSupportBundle.cpp` | Target-artifact export: monolithic validator + bare-object exporters for both route families (`validateRVVMonolithicBlockDotTargetArtifactCandidate:1840`, `monolithicBlockDotFamilyForRouteID:1824`) |
| `/home/kingdom/phdworks/TianchenRV/lib/Conversion/RVV/RVVToEmitC.cpp` | `kBlockDotKernels` dispatch table (:333, 43 entries); decode helpers (`emitOffsetBinaryDecodeValue:2592`, `emitUnsignedNibbleDecodeProductValue:2649`) |
| `/home/kingdom/phdworks/TianchenRV/lib/Conversion/RVV/RVVToEmitCBlockQuantLinear.cpp` | The hand-written flat emitter bodies (`emitQ8_0Q8_0BlockDot:6769`, `emitQ4_0Q8_0BlockDot:29`, `emitQ5_0Q8_0BlockDot:455`, `emitQ4_1Q8_1BlockDot:6348`) |
| `/home/kingdom/phdworks/TianchenRV/include/TianChenRV/Conversion/RVV/RVVToEmitCSupport.h` | `BlockDotFacts` + `deriveBlockDotFacts` (:377-395) — the schedule-fact read handle |
| `/home/kingdom/phdworks/TianchenRV/lib/Plugin/RVV/RVVDequantDotSourceFrontDoor.cpp` | The C1 auto-constructed decomposed body (model rung); `selectIntegerCoreLMUL:366` |
| `/home/kingdom/phdworks/TianchenRV/lib/Conversion/RVV/RVVToEmitCDeferredDequant.cpp` | Dequant slice-walker emitter (`emitDequantProductReduceSlice:30`, "Walk the scope into ordered … slices" :104) |
| `/home/kingdom/phdworks/TianchenRV/lib/Plugin/RVV/Construction/RVVConstructionProtocol.cpp` | Seam B 2nd validator (`appendWideningProductReduceAddRoleSteps:2142`, `verifyRVVSelectedBodySelectedRoleSequence:6367`) |
| `/home/kingdom/phdworks/TianchenRV/.trellis/tasks/07-01-arch-refactor-noperand-core/research/P1-root-cause-multivalidator-FINDING.md` | Full Seam B multi-validator finding (per-file cites) |
