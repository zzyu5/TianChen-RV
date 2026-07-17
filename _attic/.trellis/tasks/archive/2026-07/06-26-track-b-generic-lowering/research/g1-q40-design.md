# Research: Track B G1 — auto-constructing the q4_0 nibble-decode block-dot body generically

- **Query**: Investigate + DESIGN Track B G1 (auto-CONSTRUCT the q4_0 nibble-decode block-dot body from a generic op, capability-driven, replacing recognize-then-delegate to the hand-written emitter). DESIGN + FEASIBILITY pass — not full implementation.
- **Scope**: internal (code-verified)
- **Date**: 2026-06-29
- **Task**: `.trellis/tasks/06-26-track-b-generic-lowering` (parent `06-26-compiler-maturity-retest`)

---

## TL;DR verdict

- **q4_0 today = recognize + DELEGATE** (confirmed, quoted below). The front door auto-CONSTRUCTS only the kernel/variant/dispatch/fallback *scaffold* around **one monolithic** `tcrv_rvv.q4_0_q8_0_block_dot` op; the **body** (block loop + nibble decode + per-block fp16 scale + fp32 fold) is carried inside that op and emitted by the 415-line hand-written `emitQ4_0Q8_0BlockDot`. This is the q4_K front door's "does NOT hand-roll any of it" pattern.
- **The nibble-decode integer core is ALREADY a first-class generic op** with a byte-exact lit: `tcrv_rvv.packed_i4_offset_binary_x_i8_product` lowers via the SAME `emitOffsetBinaryDecodeProductValue` the hand-written block-dot calls internally. So the *hardest-looking* piece (nibble unpack) is the *easiest* to auto-construct — it already exists and is byte-exact gated.
- **The real gap** for a fully-generic q4_0 is NOT the nibble unpack; it is the **block loop + per-block fp16 scale read + left-associative fp32 fold**, none of which are expressible as generic structural ops today (rung-1/2 bodies are single-strip, no outer loop, scale is a runtime by-value f32 not a per-block memory fp16 read).
- **Bounded first step** = a NEW rung-3 front door that auto-constructs the **nibble integer-core body** (load×3 + the existing `PackedI4OffsetBinaryXI8ProductOp` + `StandaloneReduceOp` + store) from a marked generic source, byte-exact against the existing nibble-core lit. Low risk, ~1 focused step. This proves "auto-construction reaches nibble-decode" — it does NOT meet the full G1 DoD.
- **Full G1** (retire the monolithic op + hand-written emitter) = multi-day, needs NEW ODS vocab for block-loop / fp16-scale-read / fp32-fold. fp32-fold byte-exactness is the hairiest seam.
- **Beat (row-②) is NOT reachable in G1.** G1 = coverage/parity. Beat is deferred to G5 (cm wider shapes through the generic mechanism). Stated flatly per PRD line 24 and memory `kernel-wins-dont-transplant-to-e2e`.

---

## Findings

### 1. Current state — q4_0 Track-B is recognize + DELEGATE (NOT auto-constructed body)

**The front door builds ONE monolithic op, not a generic-op body.**
`lib/Plugin/RVV/RVVQ40BlockDotSourceFrontDoor.cpp:486-489`:

```cpp
// The auto-constructed attr-less block dot-product op (the nibble unpack +
// per-block fp16 scale + reduce are first-class STRUCTURE inside this op).
(void)createBlockDot(builder, loc, vx, vy, s, n, setvl.getVl());
```

`createBlockDot` (`:274-295`) emits a single `tcrvrvv::GgmlBlockDotQ40Q80Op` carrying only the bounded format facts (qk, strides, offsets, scale_model) and NO shape knobs. The front-door header self-describes the delegation (`RVVQ40BlockDotSourceFrontDoor.cpp:16-21`):

> "the front door does NOT hand-roll the nibble unpack as fragile straight-line vector ops; the harder seam the dequant rung explicitly deferred (the per-block fp16 scale loop + the nibble decode) is carried by the op the front door constructs."

**The body comes from the hand-written emitter.** The monolithic op is recognized by op-identity and routed to the 415-line hand emitter:
- Dispatch table `lib/Conversion/RVV/RVVToEmitC.cpp:333-335`: `{&isQ4_0Q8_0BlockDotBody, &VariantToEmitCFunc::emitQ4_0Q8_0BlockDot}`.
- Recognizer `RVVToEmitC.cpp:1038-1050`: body must be exactly one `GgmlBlockDotQ40Q80Op`.
- Emitter `lib/Conversion/RVV/RVVToEmitCBlockQuantLinear.cpp:29-444`: hand-writes the whole kernel — `sumf=0`, `nb=n/QK`, the `emitc.for` block loop, per-block `xb=vx+ib*18` / `yb=vy+ib*34`, the two `(float)*(const _Float16 *)` scale reads, the inner strip core, and the fp32 fold.

This is identical in shape to the q4_K front door, whose comment is the canonical statement of the delegate pattern (`lib/Plugin/RVV/RVVQ4KBlockDotSourceFrontDoor.cpp:19-22`):

> "ALL of that ... is FIRST-CLASS STRUCTURE inside that op + its existing q4_K emitter (RVVToEmitCKQuant.cpp). So the front door does NOT hand-roll any of it; it supplies the q4_K super-block-format CONSTANTS as the typed integer attrs the verifier pins."

**Verdict: q4_0 is recognize + delegate. The body is NOT auto-constructed.** What IS auto-constructed is only the exec scaffold (capabilities, variant, the fallback variant via the plugin registry, dispatch/fallback) — the same scaffold the dequant rung builds, but wrapped around one opaque monolithic op instead of a generic-op body.

### 2. The rung-1/2 auto-construction precedent (the reusable scaffold)

Two front doors TRULY auto-construct the RVV body from a generic `vector` source, and are two-board byte-exact sealed:

| Front door | File | Generic source consumed | Body it auto-constructs |
|---|---|---|---|
| Reduction (MVP / rung-1) | `lib/Plugin/RVV/RVVReductionSourceFrontDoor.cpp` | `func(lhs,rhs:memref<?xi8>, out,acc:memref<?xi32>, n)` with `transfer_read×2 → extsi×2 → muli → multi_reduction<add>` | `load(i8)×2 → widening_product(i8→i16) → standalone_reduce(i16→i32) → store` |
| Dequant (rung-2) | `lib/Plugin/RVV/RVVDequantDotSourceFrontDoor.cpp` | rung-1 body + `sitofp → mulf %scale → f32 store` | rung-1 body + `dequantize(i32→f32 scale)` before the store |

**The auto-construction mechanism (3 parts), all reusable for G1:**

1. **Matcher** (`matchBoundedWideningDotReduceSourceFunc`, `RVVReductionSourceFrontDoor.cpp:165-323`) — recognizes the generic `vector.multi_reduction` dataflow by structure (exact op counts + operand wiring + static `vector<32xi8>`/`vector<32xi32>` shapes), returning the SSA roles (lhs/rhs/out/acc/n).

2. **Capability-fact-driven LMUL anchor** (`selectIntegerCoreLMUL`, `RVVReductionSourceFrontDoor.cpp:343-379`). This is the genuine capability flip and the part G1 must extend:
   - Runs the SHARED block-dot schedule authority `enumerateBlockDotShapeCandidates(descriptor, deriveMinimumVLEN(march), budget)` + `selectGenericSchedule(...)` — NOT a hand `vlen<256 ? "m2":"m1"` switch.
   - Descriptor built inline: `coreLMULs={m1,m2}`, `quantFormat="plain-int8"`, `blockLen=32`, `factorCap=1`.
   - Returns the `lmul` knob: **m2 @ VLEN128, m1 @ VLEN256** → the byte-flip. Fail-closed (`std::nullopt`) if every candidate pruned (I7).

3. **Body builder** (`materializeKernel`, `RVVReductionSourceFrontDoor.cpp:649-766`) threads the selected anchor through fine-grained generic ops:
   - `loadLMUL = selectedIntegerCoreLMUL`, `productLMUL = getRVVNextWiderLMUL(loadLMUL)`, `anchorSEW = 8`.
   - Builds: `createSetVL` / `createWithVL` (carry SEW+LMUL anchor) → `createRVVLoad`×2 (`tcrv_rvv.load`, i8/anchor) → `createWideningProduct` (`tcrv_rvv.widening_product`, i16/wider) → `createStandaloneReduce` (`tcrv_rvv.standalone_reduce`, i32m1) → `createRVVStore`.
   - The emitted bytes FLIP with the capability fact because each generic op's type carries the anchor (`e8m2 + vwmul_i16m4 + vwredsum_i16m4` vs `e8m1 + vwmul_i16m2 + vwredsum_i16m2`). The unchanged `RVVToEmitC` emitter lowers each generic op individually.

**The exec scaffold around the body is IDENTICAL across rung-1/2/3** (createCapability, createConservativeFallbackCapability, createVariant, materializeConservativeFallbackVariantViaPlugin, createDispatch). The q4_0 front door already has all of this verbatim — only its body-building call differs (`createBlockDot` vs the rung-1/2 generic-op chain).

**Key structural difference rung-1/2 vs the q4_0 KERNEL:** rung-1/2 bodies are a **single bounded contraction** — ONE strip, NO outer block loop, the scale (rung-2) is a runtime by-value f32 scalar (`tcrv_rvv.dequantize`), NOT a per-block fp16 read from memory. The q4_0 *kernel* is genuinely a multi-block loop (`nb=n/QK`) with per-block dual fp16 scales read from `xb[0]`/`yb[0]`.

### 3. The gap for nibble-decode — what is reusable, what is missing

**Already exists and is byte-exact (the easy part):** the nibble-decode integer core is a first-class generic op.

- ODS: `tcrv_rvv.packed_i4_offset_binary_x_i8_product` (`include/TianChenRV/Dialect/RVV/IR/RVVOps.td:3814-3871`). Operands `(weight, activation_low, activation_high, vl)` → one i16 product; carries the offset-binary one-sided unpack as typed STRUCTURE.
- Emitter: `emitPackedI4OffsetBinaryXI8Product` (`RVVToEmitC.cpp:2729-2763`) calls **the SAME `emitOffsetBinaryDecodeProductValue`** (`RVVToEmitC.cpp:2463-2483`) that the hand-written block-dot's inner strip calls at `RVVToEmitCBlockQuantLinear.cpp:176`. The decode chain (`vxor.vx 0x88 → vsll/vsra sign-extend → vwmul/vwmacc`) is identical bytes whether reached via the monolithic op or the generic op.
- Byte-exact lit ALREADY committed: `test/Conversion/RVV/rvv-to-emitc-packed-i4-offset-binary-x-i8-product-reduce.mlir`. Its body is exactly a generic-op composition: `tcrv_rvv.load`×3 (i8/mf4) → `packed_i4_offset_binary_x_i8_product` (→ i16/mf2) → `standalone_reduce` (→ i32/m1) → `store`. CHECK lines pin every `__riscv_*` call.

So auto-constructing the nibble integer core = building that exact generic-op chain from a matcher, capability-anchoring the LMUL, reusing the unchanged emitter. The nibble unpack itself needs **no new emitter vocabulary**.

**Reusable L0–L3 emitter vocabulary (the refactor just built):**
- `emitOffsetBinaryDecodeProductValue` / `emitOffsetBinaryDecodeValue` / `emitOffsetBinaryProductFromDecodedValue` (`RVVToEmitC.cpp:2463-2540`) — the factored nibble decode + asymmetric product, reached through the generic op already.
- `emitVCall` / `emitVCallBuilt` / `emitVCallVoid(Built)` / `emitOpaqueCall(Built)` (`RVVToEmitCSupport.cpp:864-952`) — the verbatim-step-comment + call_opaque idiom; byte-identical to hand-spliced.
- `deriveWideningChain(base)` (`RVVToEmitCSupport.cpp:1001-1010`) — single-source i8→i16→i32 LMUL ladder; **L1 fixed the latent m2-base bug** (m2 no longer collapses into the m1 branch), so a G-series body that anchors the nibble core at m1→m2 is now safe. This is the "L1 修的 m2 bug 是 G1 的硬前置" the PRD calls out.
- `deriveBlockDotFacts` (`RVVToEmitCSupport.h:384-395`) — read-handle for the stamped schedule facts (coreLmul / multiBlockFactor / stripElision); pure read, used by the hand emitter today.

**Missing for the FULL q4_0 kernel (the hard part) — no generic vocabulary exists:**
1. **The outer block loop** `for (ib=0; ib<nb; ib++)`, `nb=n/QK`. rung-1/2 bodies have NO outer loop. Needs either a generic structural op carrying the AoS block loop, or an scf/emitc-for representation in the auto-constructed body.
2. **Per-block address arithmetic** `xb=vx+ib*18`, `yb=vy+ib*34` (`RVVToEmitCBlockQuantLinear.cpp:123-134`) — emitc.mul + emitc.add nodes inside the loop.
3. **Per-block dual fp16 scale read** `(float)*(const _Float16 *)(xb)` via the verbatim opaque callee `"(float)*(const _Float16 *)"` (`RVVToEmitCBlockQuantLinear.cpp:115, 138-142`). The rung-2 `dequantize` op is a runtime by-value f32 scale — NOT a per-block memory fp16 cast-deref. Needs a new generic op or sanctioned opaque "scalar fp16 cast-deref load."
4. **The left-associative fp32 fold** `sumf = sumf + ((float)sumi * d_x) * d_y` wrapped in ONE `emitc.ExpressionOp` (`RVVToEmitCBlockQuantLinear.cpp:303-330`) so mlir-translate prints a SINGLE C statement and `-ffp-contract=on` fuses the SAME FMA ggml does. fp non-associativity makes this byte-exactness-sensitive in a way the integer core is not.

---

## G1 DESIGN

### What q4_0 would be auto-constructed from

A marked generic source carrying the ggml `ggml_vec_dot_q4_0_q8_0` operator identity (the q4_0 front door's signature already does this: `func(s:memref<?xf32>, n:index, vx:memref<?xi8>, vy:memref<?xi8>)`). Unlike rung-1/2, the block-dot has no compact generic vector dataflow form — recognition is by operator-identity (the eight ggml vec_dot ABI roles), which the current q4_0 matcher already does (`RVVQ40BlockDotSourceFrontDoor.cpp:151-174`).

### Body-construction steps (full G1 target)

Per block `ib` over `nb = n/QK`:
1. address: `xb = vx + ib*18`, `yb = vy + ib*34`  (NEW: block-loop + address ops)
2. scales: `d_x = fp16read(xb)`, `d_y = fp16read(yb)`  (NEW: fp16 cast-deref op)
3. strip (REUSABLE generic ops): `load(w,i8/anchor)`, `load(y0)`, `load(y1)` → `packed_i4_offset_binary_x_i8_product` (→ i16/wider) → `standalone_reduce` (→ i32 sumi)
4. fold: `sumf += (sumi * d_x) * d_y` in one `emitc.ExpressionOp`  (NEW: fp32-fold op)
5. after loop: `*s = sumf`  (scalar store)

### Where the capability-driven LMUL flip enters

Reuse `selectIntegerCoreLMUL` (the rung-1/2 gearbox call) to anchor the integer-core load/product/reduce LMULs in step 3, threaded into the generic ops' types exactly as rung-1/2 does. This connects wa1/wa2 wide-LMUL but emits through the generic path (PRD line 20).

**HONEST flip framing — do NOT smuggle the q4_0 kernel's (non-)flip.** The existing q4_0 KERNEL front door explicitly disclaims a VLEN byte-flip (`RVVQ40BlockDotSourceFrontDoor.cpp:22-33`): q4_0's integer-core anchor is m1 at every Zvl128b tier; the real divergence is `(m1, factor=4, elided)` vs `(m1, factor=2, robust)`, ridden by the unmodified `--tcrv-rvv-materialize-q4-0-schedule` autotuner. If a NEW rung-3 nibble-core body anchors at the gearbox-selected mf4/m1 (or m1/m2) via `selectIntegerCoreLMUL`, that is a NEW flip on a NEW body — it is NOT the q4_0 kernel's flip and must not be claimed as such.

---

## BOUNDED FIRST byte-exact step

**Build a NEW rung-3 front door that auto-constructs the q4_0 nibble INTEGER-CORE body** (the single-strip generic-op composition), copy-then-adapt from `RVVDequantDotSourceFrontDoor.cpp`:

- **New distinct marker** (e.g. `bounded_packed_i4_offset_binary_dot_source`), mutually exclusive with the reduction/dequant markers and with the existing q4_0 KERNEL marker `ggml_q4_0_q8_0_block_dot_source` (each pass early-returns on marker mismatch).
- **Matcher**: a generic source whose body is the nibble integer-core intent (weight + two plain-i8 activation halves + acc + out + n). Can mirror the rung-1/2 structural matcher or the q4_0 operator-identity shell.
- **Capability anchor**: reuse `selectIntegerCoreLMUL` (mf4/m1 — or m1/m2 — per the lit's chosen anchor).
- **Body builder**: emit the EXACT generic-op chain already proven byte-exact — `createRVVLoad`×3 (i8/anchor) → a new `createPackedI4OffsetBinaryProduct` helper emitting `tcrv_rvv.packed_i4_offset_binary_x_i8_product` (kind `signed_packed_i4_offset_binary_x_i8_product`, product_relation `offset-binary-i4mf4-x-i8mf4x2-to-i16mf2`) → `createStandaloneReduce` → `createRVVStore`. Reuse the unchanged exec scaffold (capabilities/variant/fallback/dispatch).

**The gate (two bars — keep them separate):**
- **Bar reached by the bounded step (the only one it claims):** the new front door, run on the marked generic source, materializes a body that the unchanged `--tcrv-rvv-lower-to-emitc` lowers to C **byte-identical to the existing `test/Conversion/RVV/rvv-to-emitc-packed-i4-offset-binary-x-i8-product-reduce.mlir` emit** (same `__riscv_vxor_vx_i8mf4 / vsll / vsra / vwmul / vwmacc / vwredsum` chain). New lit: source-front-door MLIR → CHECK the emitted intrinsics match the nibble-core lit. Two-board: re-run under VLEN128/VLEN256 march if the chosen anchor flips, OR pin the no-flip mf4 form like the existing lit. This proves **auto-construction reaches nibble-decode**.
- **Bar NOT reached (the full G1 DoD, do NOT claim it):** PRD G1 DoD is "auto-constructed body ≡ 对应手写 emitter (vs scalar oracle + vs ggml `_vlN`), 两板封印" — i.e. byte-exact against the **full** `emitQ4_0Q8_0BlockDot` kernel emit (block loop + fp16 scales + fp32 fold + store). The bounded step does NOT produce that; it produces only the integer core. The follow-up trellis-implement must NOT ship the bounded step claiming G1 done.

**What the bounded step does NOT touch:** the existing q4_0 KERNEL front door, the monolithic `GgmlBlockDotQ40Q80Op`, and `emitQ4_0Q8_0BlockDot` all STAY (it adds a separate rung-3 front door, like dequant-vs-reduction). The block loop, the per-block fp16 scale read, and the fp32 `ExpressionOp` fold are NOT auto-constructed in the bounded step.

---

## Honest risks / difficulty verdict

- **Bounded first step: LOW risk, ~1 focused step.** The nibble op + reduce emit is already byte-exact and lit-pinned; the work is the matcher + a one-op body-builder helper, copy-then-adapt from the dequant front door. The scaffold is verbatim-reusable. Main care: the new marker must be mutually exclusive with the three existing markers; reuse `selectIntegerCoreLMUL` exactly (don't re-implement selection).
- **Full G1: MULTI-DAY, HIGH risk. Needs NEW ODS structure** for the three missing axes:
  - **Block loop + address arithmetic** — new generic representation (a block-loop structural op, or an scf/emitc-for in the auto-constructed body). New vocabulary axis (rung-1/2 are single-strip, no outer loop).
  - **Per-block fp16 scale read** — new generic op or sanctioned opaque "scalar fp16 cast-deref load" (`(float)*(const _Float16 *)`); rung-2's `dequantize` is a by-value f32 scale, structurally different.
  - **fp32 fold — the hairiest seam.** Must reproduce the single `emitc.ExpressionOp` carrying `sumf + ((float)sumi*d_x)*d_y` so `-ffp-contract` fuses the same FMA; fp non-associativity means a different node grouping breaks byte-exactness even when the math is "equal." Needs a new ODS op that emits exactly that expression node, or a generic "left-associative fp32 block-fold" op.
- **Where byte-exactness could break:** (1) the fp32 fold expression grouping / FMA contraction; (2) emitc node ORDER (the hand emitter is meticulous about verbatim-comment-then-call ordering — `emitVCallBuilt`'s positional contract — and seed-load position; a generic builder must reproduce it); (3) the fp16 cast-deref callee string verbatim; (4) address-arithmetic node order (mul-then-add, blockOffset==0 path).
- **Beat (row-②) is NOT reachable in G1.** G1 is true auto-construction = mechanism/coverage/parity, by PRD design (lines 17, 20, 24). Beat is deferred to G5 (cm wider shapes synthesized through the generic mechanism, measured faster). Memory `kernel-wins-dont-transplant-to-e2e` + `winc-structural-null`: compute-side kernel parity does not transplant to e2e decode. So G1's claim ceiling is byte-exact parity + a capability flip lit, NOT a speed win.

---

## Files cited (absolute paths)

- `/home/kingdom/phdworks/TianchenRV/lib/Plugin/RVV/RVVQ40BlockDotSourceFrontDoor.cpp` — q4_0 front door (recognize+delegate; `createBlockDot` :274, delegation comment :16-33, marker :81-82)
- `/home/kingdom/phdworks/TianchenRV/lib/Plugin/RVV/RVVReductionSourceFrontDoor.cpp` — rung-1 auto-construct (matcher :165, `selectIntegerCoreLMUL` :343, body builder :649)
- `/home/kingdom/phdworks/TianchenRV/lib/Plugin/RVV/RVVDequantDotSourceFrontDoor.cpp` — rung-2 auto-construct (`createDequantize` :545, body builder :687) — **copy-then-adapt base for the bounded step**
- `/home/kingdom/phdworks/TianchenRV/lib/Plugin/RVV/RVVQ4KBlockDotSourceFrontDoor.cpp` — q4_K front door, canonical "does NOT hand-roll any of it" delegate statement (:19-41)
- `/home/kingdom/phdworks/TianchenRV/lib/Conversion/RVV/RVVToEmitCBlockQuantLinear.cpp` — hand-written `emitQ4_0Q8_0BlockDot` (:29-444; nibble strip :154-208, fp32 fold ExpressionOp :303-330)
- `/home/kingdom/phdworks/TianchenRV/lib/Conversion/RVV/RVVToEmitC.cpp` — dispatch table (:333), `isQ4_0Q8_0BlockDotBody` (:1038), `emitOffsetBinaryDecodeProductValue` (:2463), `emitPackedI4OffsetBinaryXI8Product` (:2729)
- `/home/kingdom/phdworks/TianchenRV/include/TianChenRV/Dialect/RVV/IR/RVVOps.td` — ODS: `WideningProductOp` (:3632), `PackedI4OffsetBinaryXI8ProductOp` (:3814), `GgmlBlockDotQ40Q80Op` (:3873), `DequantizeOp` (:8601)
- `/home/kingdom/phdworks/TianchenRV/include/TianChenRV/Conversion/RVV/RVVToEmitCSupport.h` — emitter vocab (`emitVCall*` :244-325, `deriveWideningChain` :345-353, `deriveBlockDotFacts` :377-395)
- `/home/kingdom/phdworks/TianchenRV/lib/Conversion/RVV/RVVToEmitCSupport.cpp` — `deriveWideningChain` impl + L1 m2-fix (:1001), `emitVCall*` impls (:864)
- `/home/kingdom/phdworks/TianchenRV/test/Conversion/RVV/rvv-to-emitc-packed-i4-offset-binary-x-i8-product-reduce.mlir` — **the byte-exact gate anchor** for the bounded step (generic-op nibble core, already lit-pinned)

## Caveats / Not Found

- Did not build/run lits (design pass only). The "byte-identical" gate is asserted from the shared `emitOffsetBinaryDecodeProductValue` call path + the existing lit's CHECK lines, not from a fresh diff. The follow-up trellis-implement should diff the new front-door emit against the existing nibble lit's emit to confirm.
- Did not verify whether anchoring the nibble core at m1/m2 (vs the lit's mf4) yields a clean two-board byte-flip for the half-block (16-byte) strip; the bounded step can pin the no-flip mf4 form (matching the existing lit) and defer the flip claim. The reduction/dequant rungs DO flip m2↔m1 for the K=32 full block — the q4_0 half-block reduction-count may differ; confirm before claiming a q4_0-core flip.
- The build is unreliable for byte-exact gates (memory `build-incremental-unreliable`): use forced/clean rebuild + BEFORE/AFTER-EQUALITY, not absolute fingerprints.
