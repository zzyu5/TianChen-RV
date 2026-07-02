# Research: Fork C — can the SUPER-BLOCK block-dot bodies be descriptor-parameterized?

- **Query**: After Fork B consolidated the flat-plain + flat-codebook block-dot bodies into ONE descriptor-driven `emitFlatBlockDot`, nvfp4 was kept as a documented exception because it is "structurally a super-block, not a flat block-dot." Can the super-block bodies (q4_K/q5_K/q6_K/q2_K/q3_K + nvfp4) be descriptor-parameterized the same way — a `emitSuperBlockBlockDot(descriptor)` "Fork C" — or is the super-block bit-dance a principled HARD BOUNDARY that resists mechanization?
- **Scope**: internal (deep code trace of the K-quant + nvfp4 emitters, contrast against the landed flat skeleton) + feasibility verdict
- **Date**: 2026-07-02
- **Task frame**: this validates/corrects the earlier scope's §5 super-block row (rated ~30% shared, "a different, larger mechanism"). The prior doc's Caveat explicitly stated it did NOT deep-read `RVVToEmitCKQuant.cpp`; this doc closes that gap with a full read of all five K-quant emitters + nvfp4.
- **READ-ONLY**: no code was modified.

---

## TL;DR — the verdict, stated up front

**Fork C — a SEPARATE `emitSuperBlockBlockDot(descriptor)` super-block template, emitter-local and byte-exact-by-construction the way Fork B was — is TRACTABLE, not a hard boundary. But it is LOW-ROI, and it will NOT reuse the flat template.** Precise reading: an emitter refactor that emits the *same* emitc ops per format (dispatching per-region on a format descriptor) is byte-exact by construction — the identical argument that made Fork B safe (`emitFlatBlockDot` emits the same ops the five bespoke flat methods did). It touches no route/ABI/op identity, so — like Fork B — it is *independent of the P1 route-arity wall by construction*. What makes it low-value (not impossible) is that the shareable scaffold is *already factored and small* relative to the irreducible per-format bulk. See §4.

**Do NOT mislabel this a "hard boundary."** The genuinely-bespoke piece is the per-format DECODE (unpack + scale/min bit-dance) — but that was ALSO true of the flat template (each decode primitive is a bespoke helper the descriptor dispatches to), and Fork B shipped anyway. "Decode is bespoke" establishes "the super-block descriptor, like the flat one, dispatches to bespoke decode helpers," NOT "Fork C resists mechanization." The one thing that genuinely resists *clean* parameterization is **q2_K's scalar accumulator** (a different loop nest, §1c/§3b) — but that is ONE format being a structural outlier that stays special-cased, not the family being a boundary.

**Crucial refinement the prior estimate missed: there are TWO different "super-block" families, and they must not be conflated.**
1. **K-quant super-blocks** (q4_K/q5_K/q6_K/q2_K/q3_K): `aux8[256]` element-reordered unpack → 8-lane `aux32` vector accumulator → scale/min bit-dance → deferred fp32 fold. A `emitSuperBlockBlockDot` would be a NEW template for these (different from the flat one, no flat-helper reuse), tractable but low-ROI; q2_K is the outlier. See §4.
2. **Codebook super-blocks** (nvfp4; and the iq4_xs sibling): codebook **gather** decode (the SAME primitive already in the flat descriptor) + a per-sub-block micro-exponent scale loop + `sumi`→`sumf` fold. Structurally = *flat codebook + a sub-block scale loop*. **This is the one case that folds into the EXISTING flat template** — nvfp4 was kept out of Fork B not because it needs the K-quant machinery, but because its outer loop has a sub-block scale phase the flat skeleton has no slot for. See §3c and §6.

**The single genuinely-shared, already-mechanized cross-format piece is `deriveWideningChain` (the LMUL/widening schedule knob) — the super-block analog of `BlockDotFacts`. That is the mechanizable ~25–35%, and it is ALREADY factored and shared** (q4_K/q5_K/q6_K/q3_K/nvfp4 all consume it). So the marginal dedup a K-quant Fork C would add on top is small — which is the honest reason to deprioritize it, not "it can't be built."

---

## §1 — The ACTUAL shared structure across the 5 K-quant emitters (+ nvfp4)

### 1a. What IS shared (the skeleton)

Reading `lib/Conversion/RVV/RVVToEmitCKQuant.cpp` (4736 LOC) in full, the genuinely cross-format-shared structure is four things:

1. **The super-block outer loop.** `size_t nb = n / QK_K;` (QK_K = 256), `for (size_t ib = 0; ib < nb; ib += 1)`. Cites: q4_K `:2845-2846`, q5_K `:3186-3187`, q2_K `:3623-3624`, q6_K `:398-406`. This is a `for` loop — shared, but trivially so.

2. **The `aux8[256]` element-ordered scratch buffer** (for the four K-quants that use it): a function-scoped `int8_t aux8[256]` that the unpack writes and the per-sub-block dot reads contiguously. Cites: q4_K `:2850`, q5_K `:3191`, q2_K `:3628`, q6_K `aux8Array` param. **nvfp4 has NO aux8** (it gathers, it does not materialize).

3. **The per-sub-block widening MAC into `aux32`**: `vwmul` i8→i16 then `vwmacc` (scale·i16)→i32, accumulating an 8-lane `vint32m2_t`. This is shared in SHAPE for q4_K/q5_K/q6_K/q3_K (q2_K and nvfp4 diverge, see §1b). The MAC helper is factored: `emitQ4_KScaledDotIntoAux32` (`:1102`), `emitQ6_KSuperBlockAux32Core` region B (`:147-269`).

4. **`deriveWideningChain(coreLmul)`** — THE shared schedule mechanism. Header `include/TianChenRV/Conversion/RVV/RVVToEmitCSupport.h:345-353`; impl `lib/Conversion/RVV/RVVToEmitCSupport.cpp:1001`. It maps the `integer_core_lmul` gearbox knob → `{l8, l16, l32, stripWidth, foldGroups}`. Consumed identically by q4_K `:2795`, q6_K `:646`, q3_K `:4092`, **and nvfp4 `:750`**. This is the super-block analog of the flat path's `BlockDotFacts`/`deriveBlockDotFacts`: the "how" (LMUL width, strip width, fold-group count) IS already descriptor-driven and cross-format-shared. **It is the honest ~25–35% and it is already done — which is exactly why a further Fork C consolidation adds little.**

### 1b. Intra-format consolidation already achieved (proof the mechanizable part is mechanized)

- **q4_K ↔ q5_K genuinely share the integer core.** q5_K builds a `Q4_KIntegerCoreContext cx` (`:3273`) and calls `emitQ4_KSuperBlockAux32Core(...)` (`:3315`) — the SAME method q4_K uses. So the 6-bit utmp/kmask bit-dance + the aux8→aux32 MAC + the fold-back are ONE shared code path for these two formats. q5_K adds only its own qh 5th-bit merge in the unpack. **This is the existence proof that a shared super-block core is buildable — it already exists for 2 of 5.**
- **q6_K** shares `emitQ6_KSuperBlockAux32Core` (`:29`) between its two consumers (`emitQ6_KQ8_KAux32Partial:338`, `emitQ6_KQ8_KBlockDot:506`).
- **q4_K's "8 sub-methods"** (`RVVToEmitC.cpp:394-409`: NibbleUnpack / ScaleMinBitDance / ScaledDot / MinTerm / SumsFoldScaleD / HorizontalFold / Aux32Partial / BlockDot) are NOT eight steps of one pipeline — they are **first-class-op witnesses**: standalone `tcrv_rvv.q4_k_*` ops each with its own recognizer in `kBlockDotKernels`, so each is independently reachable from an input IR that carries that op. The REAL monolithic block-dot path is `emitQ4_KQ8_KBlockDot` (`:2743`) → `emitQ4_KSuperBlockAux32Core` (`:1617`) → three brick helpers (`emitQ4_KPlainNibbleUnpack:847`, `emitQ4_KScaleMinBitDanceCore:973`, `emitQ4_KScaledDotIntoAux32:1102`). (This 8-op decomposition is a *separate, larger* Fork-A-flavored option — see §5c — distinct from the emitter-local Fork C.)

### 1c. What is NOT shared (the "what") — four axes, all divergent

| Axis | q4_K | q5_K | q6_K | q2_K | q3_K |
|---|---|---|---|---|---|
| **subBlock / #sub-blocks** | 32 / 8 | 32 / 8 | 16 / 16 | 16 / 16 | 16 / 16 |
| **quant bits + unpack** | 4-bit plain nibble (`vand 0x0F`/`vsrl 0x04`), no bias | 4-bit nibble **+ qh 5th-bit** merge | 6-bit = ql 4-bit **+ qh 2-bit** merge, **bias −32** | 2-bit unpack | 3-bit = 2-bit **+ hmask** subtractive high bit |
| **scale/min decode** | **6-bit utmp/kmask bit-dance** | **SAME utmp/kmask** (reuses q4_K) | **plain int8 scales** (NO dance) | **4-bit packed scales+mins** | **6-bit scales SPLIT across two arrays** |
| **accumulator model** | 8-lane `aux32` vec + `sums8[8]` deferred fp32 fold + min term | 8-lane `aux32` + `sums8[8]` | 8-lane `aux32` + partial variant | **SCALAR `sumf` in-loop, NO 8-lane vector** | 8-lane `aux32` + `sums8`-style |
| **min/bias correction** | `dmin·mins·bsums` (bsums min term) | `dmin·mins·bsums` | −32 bias in unpack | packed scale+min | subtractive hmask |

Cites for the divergences: q6_K unpack `:75-145` (the exact `quants.c:828-831` permutation, bias −32 at `:116-123`); q2_K `:3574` subBlock=16, `:3646-3656` **"q2_K has NO 8-lane sums vector — the positive term is the scalar isum"**; q3_K `:4052-4063` subBlock=16 + `hmaskOffset`/`scalesOffset=96` split-array scales; q4_K/q5_K subBlock=32 (`:2774`, `:3136`).

**Validation of the prior ~30% estimate: CONFIRMED (~25–35%), with a sharper reading.** The shared part is the *schedule* (loop frame + `deriveWideningChain` + the aux8→aux32 MAC shape). The un-shared 65–75% is bespoke *decode* (unpack + scale/min), and 4/5 formats also share the *accumulator model* (8-lane `aux32` vector) — **q2_K is the sole accumulator outlier** (scalar `sumf`, `:3646`). So the residue is "bespoke decode helpers + one structural outlier," not "irreducibly monolithic." That is a *degree* statement (low dedup ceiling), not a *kind* statement (resistant).

---

## §2 — The bit-dance specifically: how the 6-bit scale/min unpacks differ

The bit-dance is the most bespoke region, and the answer is decisive: **the "6-bit scale/min bit-dance" is not even the same *kind* of operation across the five formats — it exists as such in exactly TWO of them.** But (see §2c) "bespoke decode" is a *dispatched-to helper*, exactly as in the flat template — not a mechanization defeater.

### 2a. q4_K / q5_K — the genuine utmp/kmask 6-bit dance
`emitQ4_KScaleMinBitDanceCore` (`:973-1081`). Loads THREE `uint32_t` words `w0/w1/w2` from the 12 packed scale bytes (`:1020-1022`), then a fixed permutation with three masks (`:998-1000`):
```
kmask1 = 0x3f3f3f3f;  kmask2 = 0x0f0f0f0f;  kmask3 = 0x03030303;
u3 = ((w2 >> 4) & kmask2) | (((w1 >> 6) & kmask3) << 4);   // :1039-1041
u2 =  w1 & kmask1;                                          // :1042
u1 = ( w2       & kmask2) | (((w0 >> 6) & kmask3) << 4);    // :1043-1045
u0 =  w0 & kmask1;                                          // :1046
```
producing `utmp[4]` whose 16 bytes are exactly `[scales[0..7], mins[0..7]]` (all 8 six-bit scales AND 8 six-bit mins). Mirrors ggml `quants.c:685-690`. Emitted as STRUCTURED `emitc.bitwise_*` ops (`:1023-1038`), not raw strings. q5_K reuses this verbatim (`:3315`).

### 2b. The other three do NOT have this operation at all
- **q6_K**: scales are **plain `int8_t`**, read directly per sub-block — NO bit-dance. Its "6-bit" is the *quant values* (ql 4-bit + qh 2-bit merge, `:102-144`), a different thing entirely.
- **q2_K**: scales AND mins are **4-bit packed** in a single byte per sub-block (`scalesOffset=0`, `:3577`); the decode is a 4-bit split, not the utmp 32-bit-word dance.
- **q3_K**: scales are **6-bit but SPLIT across two arrays** (a low-4-bit array + a high-2-bit array at `scalesOffset=96`, `:4057`), reassembled with a different mask sequence than q4_K's.

### 2c. Could a `scale_min_layout` descriptor field parameterize this?
A `scale_min_layout` enum with values `{utmp_kmask_6bit, plain_i8, packed_4bit, split_6bit}` would **select among four different code blocks**, each a bespoke helper, sharing only the output contract ("produce a per-sub-block scale source"). **This is exactly the shape of the flat path's `decode_primitive` switch** (`emitFlatBlockDot:5531-5628`) — a dispatch to bespoke helpers, and Fork B shipped it byte-exact. So a `scale_min_layout` field IS a legitimate descriptor field (a dispatch selector), NOT a parameterization defeater. The honest caveat is (a) unlike the flat body where decode was the ONLY divergence on an 85%-shared skeleton, here the scale decode is one of ~four divergence axes, and (b) `scale_min_layout` co-varies with `accumulator_model` (q2_K's packed-4-bit scales come *with* its scalar accumulator) — so the reachable combinations are just the five formats. That means the descriptor's "primitive" fields degenerate toward "which of the five," which is why the marginal dedup is low — but it is still constructible and byte-exact, mirroring how the flat `kind` string selects the primitive bundle.

---

## §3 — Contrast against the landed `emitFlatBlockDot` (what "parameterized" looks like)

### 3a. Why the flat template works
`emitFlatBlockDot` (`RVVToEmitCBlockQuantLinear.cpp:5334-5949`, ~616 LOC) is ONE method with a shared skeleton and exactly **three pluggable points**:
- `decode_primitive` switch — 5 cases (PlainI8 / OffsetBinaryNibble / UnsignedNibble / FiveBitOffsetBinary / CodebookGatherNibble), `:5531-5628`, each calling a pre-existing factored helper.
- `fold_model` switch — 4 cases (SumiTimesScales / LeftAssoc / ScalesTimesSumi / ScalePlusMin), `:5765-5824`.
- `block_len` (qk vs qk/2), a scalar.
Descriptor: `FlatBlockDotDescriptor` (`RVVToEmitCInternal.h:117-154`) — ~20 scalar/enum fields. `deriveFlatBlockDotDescriptor` (`:5222-5332`) maps `kind` → the three Group-B primitive fields, reads geometry off the I4 attrs.

The template works because the flat formats share THREE invariants:
1. **ONE scalar integer accumulator `int32_t sumi` per block, folded ONCE** (`emitIntegerCore:5663`, `emitFold:5750`).
2. **The decode is strip-local**: one vector load decoded in place inside `emitStripReduce` (`:5505`); no materialized scratch.
3. **The scale is ONE fp16 pair `(d_x, d_y)` per block** (`emitBlockCore:5854-5858`), a scalar pair.

### 3b. The K-quants share NONE of these three invariants — so Fork C is a DIFFERENT template, not an extension of the flat one
1. Accumulator is an **8-lane `vint32m2_t` vector** whose lanes ARE the sub-blocks (or, for q2_K, a scalar accumulated in-loop with a DIFFERENT reduction). Not `int32_t sumi`.
2. Decode is **two-phase**: unpack ALL 256 elements into `aux8[256]` in a format-specific *element permutation* FIRST (q4_K `:1643`, q6_K `:75-145`), THEN dot per sub-block. The `aux8` materialization phase has **no analog** in the flat skeleton.
3. Scale is **8 per-sub-block six-bit scales + 8 mins** extracted by the bit-dance (or int8, or packed) — a per-sub-block scale VECTOR, plus a separate deferred min/bsums term. Not a scalar `(d_x, d_y)`.

The consequence for Fork C is about *reuse*, not *feasibility*: **the flat helpers are NOT reusable** — `emitStripReduce`, `emitIntegerCore`, `emitFold`, `emitBlockCore`, `fp16ReadAt` all assume the scalar-sumi / one-scale-pair / strip-local model. A super-block `emitSuperBlockBlockDot` would build its OWN skeleton (super-block loop + aux8 unpack phase + per-sub-block MAC into `aux32` + deferred fp32/min fold) with its OWN pluggable points. Only `deriveWideningChain` (which is NOT part of `FlatBlockDotDescriptor` — it lives separately in `BlockDotFacts`/support.h) crosses over, and it is already shared. So Fork C is a *parallel* template; it does not sit on top of Fork B.

### 3c. nvfp4 — the ONE case that folds into the EXISTING flat template
`emitNVFP4Q8_0BlockDot` (`RVVToEmitCCodebookFp4.cpp:713`). QK=64, QK_SUB=16, numSubBlocks=4 (`:764-771`). Its decode is the **16-entry FP4 codebook GATHER** (`:820-829`, `vrgather_vv_i8`) — the SAME `CodebookGatherNibble` primitive already in `emitFlatBlockDot` (`:5589-5627`). It has **no aux8, no bit-dance, no aux32 vector**; it computes a per-sub-block `sumi` and folds into a scalar `sumf`. The ONLY thing separating it from the flat codebook path is (a) a per-sub-block **UE4M3 micro-exponent scale** reconstruction (`emitUE4M3Scale:848`, a structured `ldexpf`-based decode) and (b) the outer loop iterating `numSubBlocks` sub-blocks per super-block before the fold. Structurally: **flat codebook + a sub-block scale loop.** This is why it was a *documented Fork-B exception* and not a K-quant — it is much closer to the flat family than to q4_K, and it is extended into the flat template, not the K-quant one (see §6).

---

## §4 — FEASIBILITY verdict (brutally honest)

### K-quant super-blocks (q4_K/q5_K/q6_K/q2_K/q3_K): TRACTABLE + byte-exact-achievable as a SEPARATE super-block template — but LOW-ROI.

**Achievability (the "tractable" claim):** an `emitSuperBlockBlockDot(descriptor)` that emits the *same* emitc ops each format emits today — a shared super-block-loop + aux8-unpack-phase + aux32-MAC + deferred-fold skeleton, with per-region switches on the format descriptor (`quant_unpack`, `scale_min_layout`, `accumulator_model`, `min_bias_model`) — is byte-exact **by construction**, the same guarantee Fork B relied on. The existence proof is already in-tree: q4_K and q5_K share `emitQ4_KSuperBlockAux32Core` byte-exact (`:3315`); q6_K shares its core across two consumers (`:29`); all consume `deriveWideningChain`. It is emitter-local: no op, no ABI arity, no route identity changes, so — exactly like Fork B — it is **independent of the P1 construction-protocol / route-arity wall by construction** (that wall binds only the decomposed, N-operand Fork-A path — see §5c).

**Why it is LOW-ROI (the honest blocker):**
- The shareable scaffold (`deriveWideningChain` + the aux32 MAC + the loop frame) is **already factored and small** relative to the irreducible per-format unpack + scale-decode bulk. The template would mostly re-house the five decode bodies behind a shared signature; the marginal dedup is the loop frame + `nb=n/256` + aux8 decl.
- The descriptor's "primitive" fields degenerate toward "which of the five formats" (§2c), because `scale_min_layout` / `accumulator_model` / `subBlock` / `bias_model` co-vary — only five combinations exist. So "add a K-quant format" still means writing a new unpack + scale decode + picking an accumulator model; the template does not turn it into "fill a descriptor row." (This asymmetry with the flat path is real but is *degree*: adding a flat format also required a new decode helper — the flat template just had a thinner residue to write.)
- **q2_K is a genuine structural outlier**: its scalar-`sumf` in-loop accumulator (`:3646`) is a different loop nest from the vector-`aux32` path, so it stays a special-case branch inside the template, not a descriptor row. It is the one place a "clean" parameterization genuinely does not reach.

**Byte-exact testing burden (not an achievability question):** five formats, each gated by `check-tianchenrv` (783/780/3) under a forced clean relink + BEFORE==AFTER equality (memory `build-incremental-unreliable`). This is the cost, not a boundary — Fork B carried the same nature and shipped.

**Net:** constructible and byte-exact, but the win is code-org (some dedup + one uniform super-block emitter signature), NOT perf and NOT novelty, and the dedup ceiling is low because the scaffold is already shared. **Recommendation: do not prioritize building it** (see §7) — but this is "not worth it," not "can't be done." Do not record it as a boundary.

### Codebook super-blocks (nvfp4): TRACTABLE as a bounded extension of the EXISTING flat template.
nvfp4's decode is already `emitFlatBlockDot`'s `CodebookGatherNibble`. It is tractable to fold it into an *extended* flat descriptor — see §6.

---

## §5 — Next moves (three distinct options; pick by whether a paper section needs it)

### 5a. K-quant `emitSuperBlockBlockDot(descriptor)` — first byte-exact-gated step + descriptor fields (IF pursued)
1. Define a `SuperBlockDotDescriptor` with: `qk`(256) / `subBlock`(16|32) / `numSubBlocks`; `quantUnpack` enum `{nibble4, nibble4_qh5, ql4_qh2_bias32, bits2, bits2_hmask3}`; `scaleMinLayout` enum `{utmp_kmask_6bit, plain_i8, packed_4bit, split_6bit}`; `accumulatorModel` enum `{aux32_vec_sums8, scalar_sumf}`; `minBiasModel` enum `{bsums_min, unpack_bias, hmask_sub, packed}`; plus the geometry byte offsets already on the typed ops. `deriveWideningChain` stays separate (schedule, already shared).
2. Extract the shared super-block skeleton from q4_K's `emitQ4_KSuperBlockAux32Core` + `emitQ4_KQ8_KBlockDot` into `emitSuperBlockBlockDot`, with `quantUnpack`/`scaleMinLayout`/`minBiasModel` dispatching to the existing per-format helpers and the fold/accumulator on `accumulatorModel`.
3. Re-point **q4_K FIRST** (it owns the richest core), prove byte-exact under a forced clean relink + BEFORE==AFTER on the q4_K lit; then q5_K (already shares the core — cheapest second); then q6_K; then q3_K. **q2_K last and as an explicit `scalar_sumf` special-case branch** (its loop nest does not fit the vector path).
Each migration is byte-exact-gated (783/780/3, clean relink). This is bounded and emitter-local, but see the ROI verdict in §4 — the recommendation is to NOT do this unless §7's trigger fires.

### 5b. nvfp4 → extended flat descriptor (bounded tidy)
Extend `FlatBlockDotDescriptor` with a small super-block-codebook extension:
- `int64_t numSubBlocks` (1 for true flat; 4 for nvfp4) — drives an inner sub-block loop around the existing gather+dot.
- `subBlockScaleSource` enum `{None, UE4M3}` — selects the per-sub-block scale read (`emitUE4M3Scale`).
- The existing `hasCodebook` / `codebook` / gather path is reused unchanged.
First byte-exact-gated step: add `numSubBlocks=1` as the default (no behavior change for all current flat formats — prove BEFORE==AFTER on the full flat lit under forced clean relink), THEN re-point nvfp4 at the extended body with `numSubBlocks=4` + `UE4M3`, gated byte-exact on the nvfp4 lit alone. **Caveat:** nvfp4 is a *single* format, so the dedup win is one method (~200 LOC) folded into the flat body at the cost of an inner-loop + scale-source branch that every flat format then carries (guarded, zero-node at `numSubBlocks=1`). Marginal. Worth it only as tidying.

### 5c. The SEPARATE, LARGER option — a composable super-block sub-op vocabulary (Fork-A-flavored)
Distinct from Fork C (§5a is an emitter-local monolith refactor; this is a decomposition into first-class ops composed by a body-builder): `unpack_to_aux8(layout)` → `decode_scale_min(layout)` → `scaled_dot_into_aux32` → `min_term(bsums)` → `deferred_fp32_fold`. **Already partially sketched** by q4_K's 8 first-class ops (`RVVToEmitC.cpp:394-409`; emitters at `RVVToEmitCKQuant.cpp:1871/1959/2126/2306/2452/2624/1695`). Generalizing = making the ops format-generic (a `layout` attr) + a composing body-builder. **This is the ONLY super-block option that carries the P1 wall**: it introduces/changes first-class ops and their ABI, so it hits the construction-protocol / route-arity redesign documented in `P1-root-cause-multivalidator-FINDING.md` — multi-quarter, not emitter-local. It buys a "composable super-block vocabulary" *novelty claim*, not perf. Pursue ONLY if the paper explicitly wants that demonstrator; even then, q4_K/q5_K only.

---

## §6 — Effort/risk + relationship to `emitFlatBlockDot`

| Move | Effort | Byte-exact | Win | Recommend |
|---|---|---|---|---|
| **Fork C**: K-quant `emitSuperBlockBlockDot(descriptor)` (emitter-local, à la Fork B) | Medium (extract shared super-block skeleton; 5 formats; q2_K special-cased) | **Achievable by construction**; testing burden = 5 formats × clean relink × 783/780/3 | Marginal dedup + uniform signature; NO perf, NO novelty (scaffold already shared) | **No — deprioritize (low-ROI), but constructible; do NOT log as a boundary** |
| nvfp4 → extended flat descriptor (§5b) | Low–medium (bounded) | Achievable; guarded default + nvfp4 lit | Tidies 1 exception, ~200 LOC | **Optional tidy** |
| Composable super-block sub-op vocabulary (§5c, Fork-A) | Multi-quarter; **hits P1 route-arity wall** | High risk | Composability *claim* only | **No, unless a paper section needs the demonstrator (then q4_K/q5_K only)** |

**Relationship to the landed `emitFlatBlockDot`:**
- **Flat skeleton helpers are NOT reusable** for a K-quant Fork C (§3b): `emitStripReduce` / `emitIntegerCore` / `emitFold` / `emitBlockCore` assume scalar-sumi + one-scale-pair + strip-local decode. Fork C would be a **PARALLEL super-block template with its own skeleton**, not an extension of the flat body. This is the key sequencing fact: Fork C does not build on Fork B; they are siblings.
- **The ONE thing that crosses over** is `deriveWideningChain` (support.h:345 / support.cpp:1001) — already shared by the flat codebook path and all four vector-accumulator K-quants, and it lives in `BlockDotFacts`, *outside* `FlatBlockDotDescriptor`. The schedule dimension is unified already; neither template owns it.
- **nvfp4 IS an extension of the flat body** (§3c/§5b) — its decode is literally `emitFlatBlockDot`'s `CodebookGatherNibble`. The only place the flat template reaches cleanly into "super-block" territory.

---

## §7 — Bottom line for the decision this doc exists to make

- **Is Fork C a hard boundary or tractable? TRACTABLE.** A separate `emitSuperBlockBlockDot(descriptor)` super-block template is constructible and byte-exact-by-construction (emitter-local, à la Fork B; 4/5 K-quants already share the `aux32` core; all share `deriveWideningChain`; independent of the P1 wall). Do NOT record it as "resists mechanization" — that would wrongly tell a future session never to revisit.
- **Should a future session build it now? No — LOW-ROI.** The shareable scaffold is already mechanized (`deriveWideningChain` + shared cores), so the marginal dedup is small; the per-format decode bulk is irreducible bespoke (dispatched-to helpers, like the flat template's decode primitives); q2_K's scalar accumulator is a structural outlier that stays special-cased. The win is code-org, not perf/novelty. Against trunk-discipline (`.trellis/spec/guides/trunk-discipline.md`), it is maturity tidying, not an N1/N2/N3 main line.
- **Revisit trigger:** if the paper wants a *uniform super-block emitter* story, or K-quant coverage grows enough that five parallel monoliths become a maintenance burden, Fork C becomes worth the testing cost. Then follow §5a (q4_K→q5_K→q6_K→q3_K, q2_K special-cased last).
- **The genuinely larger, different mechanism** (composable super-block sub-op vocabulary, §5c) is Fork-A-flavored, multi-quarter, P1-wall-blocked, and buys a composability claim only. Separate decision from Fork C.
- **The tractable-now residue** is nvfp4 (§5b): a bounded, optional extension of the EXISTING flat template.
- **Honest paper framing:** "capability-facts + a block-format descriptor generate the flat contraction bucket (plain + codebook) byte-exact" is defensible and landed. The super-block K-quants would need a *second, parallel* super-block template (different invariants: element-reordered unpack + vector accumulator + per-sub-block scale decode); it is buildable but not built, and the flat descriptor does not cover them.

---

## Caveats / Not found

- **This doc DEEP-READ** the flat skeleton (`emitFlatBlockDot` + descriptor), the q4_K bit-dance core + super-block core + block-dot, the q6_K shared aux32 core, the q5_K/q2_K/q3_K method heads + their scale/accumulator regions, and the nvfp4 emitter head + UE4M3 scale. I did NOT line-by-line read the *entire* q3_K/q2_K fold tails or the iq4_xs super-block emitter (`RVVToEmitCCodebookFp4.cpp:87`) — the latter is a second codebook-super-block instance that would strengthen the "codebook super-blocks extend the flat template" point if it matters (structurally it uses the same gather + a Q8_K sub-block scale, per `:33-77`).
- **The ~25–35% shared figure** is a node-shape estimate (loop frame + `deriveWideningChain` + aux8→aux32 MAC vs the divergent unpack/scale/accumulator/fold), not a mechanical line count. It corroborates the ROI verdict (low dedup ceiling); the achievability verdict rests on the by-construction byte-exact argument + the existing q4_K/q5_K shared core, not on the percentage.
- **Byte-exact discipline (any future impl):** must hold `check-tianchenrv` 783/780/3 under a **forced clean relink** + BEFORE==AFTER equality — NOT absolute fingerprints (memory `build-incremental-unreliable`: this tree regenerates `RVVOps.cpp.inc` each build and `tcrv-opt` sometimes fails to relink; the stale fingerprints `f810ce6b`/`cb04b219` are not pass/fail targets).
- **P1 wall applies to §5c ONLY.** The emitter-local Fork C (§5a) and the nvfp4 tidy (§5b) touch no op/ABI/route and are independent of the construction-protocol wall by construction (identical to Fork B). Only the composable sub-op vocabulary (new/changed first-class ops) hits `P1-root-cause-multivalidator-FINDING.md`. Do not attach "multi-quarter / wall-blocked" to Fork C.

---

## Key file map (absolute paths)

| Path | Role |
|---|---|
| `/home/kingdom/phdworks/TianchenRV/lib/Conversion/RVV/RVVToEmitCKQuant.cpp` | The 5 K-quant super-block emitters (4736 LOC). q6_K core `:29`; q4_K bricks `:847/973/1102`; q4_K super core `:1617`, block-dot `:2743`; q5_K `:3105` (shares q4_K core `:3315`); q2_K `:3544` (scalar-accum outlier `:3646`); q3_K `:4020` |
| `/home/kingdom/phdworks/TianchenRV/lib/Conversion/RVV/RVVToEmitCCodebookFp4.cpp` | Codebook super-blocks: `emitNVFP4Q8_0BlockDot:713`, `emitUE4M3Scale:848`, `emitIQ4XSQ8KBlockDot:87` (sibling); flat codebook `emitIQ4NLQ8_0BlockDot:29` |
| `/home/kingdom/phdworks/TianchenRV/lib/Conversion/RVV/RVVToEmitCBlockQuantLinear.cpp` | The landed flat template: `emitFlatBlockDot:5334-5949`, `deriveFlatBlockDotDescriptor:5222`, the 5 thin shims `:33/70/107/5151/5189` |
| `/home/kingdom/phdworks/TianchenRV/lib/Conversion/RVV/RVVToEmitCInternal.h` | `FlatBlockDotDescriptor` struct `:117-154`; `FlatFoldModel` `:103-108` |
| `/home/kingdom/phdworks/TianchenRV/include/TianChenRV/Conversion/RVV/RVVToEmitCSupport.h` | `WideningChain` `:345`, `deriveWideningChain` decl `:353` — the shared super-block schedule knob (crosses both templates) |
| `/home/kingdom/phdworks/TianchenRV/lib/Conversion/RVV/RVVToEmitCSupport.cpp` | `deriveWideningChain` impl `:1001` |
| `/home/kingdom/phdworks/TianchenRV/lib/Conversion/RVV/RVVToEmitC.cpp` | `kBlockDotKernels` dispatch `:333-420` (the K-quant + 8-brick recognizers `:390-419`) |
| `/home/kingdom/phdworks/TianchenRV/.trellis/tasks/07-01-arch-refactor-noperand-core/research/track-b-autoconstruct-scope.md` | The earlier scope whose §5 super-block row (~30%) this doc validates/refines |
| `/home/kingdom/phdworks/TianchenRV/.trellis/tasks/07-01-arch-refactor-noperand-core/research/P1-root-cause-multivalidator-FINDING.md` | The construction-protocol wall — applies to the §5c composable sub-op path ONLY, NOT to the emitter-local Fork C |
