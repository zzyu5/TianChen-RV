# Research: K-quant scoping (first super-block kernel) — q6_K vs q4_K

- **Query**: Scope a modern super-block K-quant (q6_K / q4_K) so the implementation is clean.
  Capture exact formats + dot math, recommend the cleanest first super-block proof, map it
  to our flat block-loop template (reuse vs new), give a validatable increment plan.
- **Scope**: external (llama.cpp primary source) + internal (our lowering template)
- **Date**: 2026-06-16
- **Sources (primary)**:
  - Formats: `/home/kingdom/phdworks/llama.cpp/ggml/src/ggml-common.h`
  - Generic refs (byte-exact ground truth): `/home/kingdom/phdworks/llama.cpp/ggml/src/ggml-cpu/quants.c`
  - RVV board paths (VLEN=128): `/home/kingdom/phdworks/llama.cpp/ggml/src/ggml-cpu/arch/riscv/quants.c`
  - Our template: `/home/kingdom/phdworks/TianchenRV/lib/Conversion/RVV/RVVToEmitC.cpp`
  - Our op schema: `/home/kingdom/phdworks/TianchenRV/include/TianChenRV/Dialect/RVV/IR/RVVOps.td`

---

## TL;DR (the three asks)

1. **First K-quant target: q6_K.** One-line why: q6_K isolates the genuinely-new super-block
   scale *hierarchy* with the lowest-risk scale extraction — its per-sub-block scales are a
   plain `int8 scales[16]` direct scalar load, whereas q4_K's are a 6-bit cross-byte packed
   bit-dance (`utmp[4]`/`kmask1/2/3`: scalar uint32 `>>`/`&`/`|`/`<<` over a 4-word temp + a
   type-punned reinterpretation). q4_K's decode is **structurable but materially more awkward
   and error-prone** (the type-punning and 4-word temp are the ugly parts), not impossible —
   q6_K is chosen because its scale path is materially cleaner and lower-risk, NOT because
   q4_K's is illegal.
2. **Single hardest NEW piece: the two-level fold.** Our entire existing family has exactly
   ONE scale level (per-block fp16 `d`, folded to a single scalar `sumf` *per block*). q6_K
   needs a **per-sub-block `int8` scale applied in the i32 domain, accumulated across 16
   sub-blocks into a carried multi-lane i32 accumulator (`aux32[8]`), with the fp32 `d`
   multiply DEFERRED to the super-block boundary into a multi-lane fp32 accumulator
   (`sums[8]`) and a final sequential horizontal sum.** That deferral + multi-lane
   accumulators + ordered horizontal reduction is both the conceptual heart and the
   byte-exactness risk. The 6-bit ql+qh unpack is more ops but the SAME KIND we already
   emit, so it is not the hard part.
3. **Increment plan:** K1 = super-block unpack + per-sub-block int8-scaled i32 partial,
   byte-exact vs the generic's `aux32` integer partial at n=256. K2 = full nested loop
   (super-block × sub-block × strip) + the deferred fp32 hierarchy + ABI, byte-exact vs
   `_generic` over many n. K3 = the measurement-tuner inherits (no per-kernel hand-tuning).

---

## Findings

### Exact formats (from `ggml-common.h`)

Constants: `QK_K = 256` (super-block size); `K_SCALE_SIZE = 12`; `ggml_half` = fp16 (2 bytes).

#### `block_q6_K` (lines 352–358) — RECOMMENDED FIRST TARGET

```c
// 6-bit quantization; weight = a * q ; 16 blocks of 16 elements each; ~6.5625 bits/weight
typedef struct {
    uint8_t ql[QK_K/2];      // = ql[128]  quants, lower 4 bits
    uint8_t qh[QK_K/4];      // = qh[64]   quants, upper 2 bits
    int8_t  scales[QK_K/16]; // = scales[16]  per-sub-block scales, quantized int8
    ggml_half d;             // super-block scale (fp16)
} block_q6_K;
// sizeof = 2 + 16 + 192 = 210 bytes  (fp16 d + 16 scales + 3*64 quant bytes)
// byte offsets: ql @0 | qh @128 | scales @192 | d @208
```

- **Super-block size**: 256 elements. **Sub-block structure**: 16 sub-blocks of 16 elements.
- **Quant width / packing**: 6 bits/weight = a 4-bit LOW nibble in `ql[]` (128 bytes, 2
  nibbles/byte) PLUS a 2-bit HIGH part in `qh[]` (64 bytes, 4 two-bit fields/byte). A 6-bit
  unsigned value `0..63` is reconstructed then biased by `-32` to a signed `int8` in
  `[-32, +31]`.
- **Scale hierarchy** (TWO levels, NO min term):
  1. super-block fp16 `d` (one per 256), multiplied by the q8_K activation `d` (fp32).
  2. per-sub-block `int8 scales[16]` — a **direct scalar load**, applied in the integer
     (i32) domain. No packing, no decode. This is the cleanliness win.
- **No `dmin` / no min term** (`x = a*q`, not `a*q + b`).

#### `block_q4_K` (lines 317–328) — DEFERRED (harder)

```c
// 4-bit; weight = a*q + b ; 8 blocks of 32 elements each; ~4.5 bits/weight
typedef struct {
    union { struct { ggml_half d; ggml_half dmin; }; ggml_half2 dm; };
    uint8_t scales[K_SCALE_SIZE]; // = scales[12]  scales AND mins, 6-bit PACKED
    uint8_t qs[QK_K/2];           // = qs[128]  4-bit quants
} block_q4_K;
// sizeof = 2 + 2 + 12 + 128 = 144 bytes
// byte offsets: d @0 | dmin @2 | scales @4 | qs @16
```

- **Super-block**: 256 elements. **Sub-block structure**: 8 sub-blocks of 32 elements.
- **Quant width / packing**: plain 4-bit nibbles (low/high of each `qs[]` byte) — UNsigned
  `0..15`, no `-8` / `-32` bias (the bias is carried by the `min` term instead).
- **Scale hierarchy** (TWO levels + a min term):
  1. super-block fp16 `d` AND `dmin` (both ×q8_K `d`).
  2. per-sub-block 6-bit scale AND 6-bit min, **packed into 12 bytes** and decoded by a
     cross-byte bit-shuffle (`utmp[4]` + `kmask1=0x3f3f3f3f`, `kmask2=0x0f0f0f0f`,
     `kmask3=0x03030303`; generic `quants.c:685–690`). Structurable as scalar uint32 bitops,
     but the type-punned `(const uint8_t*)&utmp[...]` reinterpretation + 4-word temp make it
     materially more awkward/error-prone than q6_K's direct int8 load.
  3. a `min` term subtracted via `bsums` (see below) — `x = a*q + b`.

#### `block_q8_K` (the activation, lines 361–366) — shared by BOTH

```c
// intermediate quantization for dot products only
typedef struct {
    float   d;              // delta (fp32, NOT fp16 — distinct from q8_0!)
    int8_t  qs[QK_K];       // = qs[256]  full-precision int8 quants
    int16_t bsums[QK_K/16]; // = bsums[16]  sum of quants in groups of 16
} block_q8_K;
// sizeof = 4 + 256 + 32 = 292 bytes
// byte offsets: d @0 | qs @4 | bsums @260
```

- **Activation `d` is fp32** (a plain `float`), not fp16 — so NO `fcvt.s.h` on the activation
  side (q6_K reads `y[i].d` directly; only the weight `x[i].d` needs the fp16→fp32 read).
- **`qs[256]`**: full int8, contiguous, one per super-block element. No packing.
- **`bsums[16]`**: precomputed per-sub-block sum of the 16-grouped int8 quants. Used ONLY by
  formats with a min term (q2_K/q4_K/q5_K): `min_contribution = Σ_j bsums[j] * mins[j/2]`.
  **q6_K does NOT use `bsums`** (no min term) — another reason it is the clean first target.

---

### Exact dot math (nested, with the byte-exactness-critical fp32 order)

#### q6_K — `ggml_vec_dot_q6_K_q8_K_generic` (`quants.c:800–853`) — THE byte-exact target

State per call: `int8_t aux8[256]; int16_t aux16[8]; float sums[8]={0}; int32_t aux32[8];
float sumf=0;`

```
for i in 0 .. nb-1:                                  # super-block loop (nb = n / 256)
    aux32[0..7] = 0                                  # per-super-block i32 accumulator, 8 lanes

    # ---- (A) UNPACK 6-bit weights into aux8[256], element-ordered, biased -32 ----
    # done in 2 chunks of 128 elements (j = 0, 128); per chunk, for l in 0..31:
    a[l+ 0] = ((ql[l+ 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32
    a[l+32] = ((ql[l+32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32
    a[l+64] = ((ql[l+ 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32
    a[l+96] = ((ql[l+32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32
    # then a += 128; ql += 64; qh += 32   (the ql/qh→element permutation — see note below)

    # ---- (B) per-SUB-BLOCK integer dot, int8-scaled, into the 8-lane i32 accumulator ----
    a = aux8; is = 0
    for j in 0 .. 15:                                # 16 sub-blocks of 16 elements
        scale = x[i].scales[is++]                    # int8 per-sub-block scale (scalar load)
        # the sub-block of 16 is processed as TWO halves of 8:
        for l in 0..7: aux16[l]  = q8[l] * a[l]      # i8*i8 -> i16
        for l in 0..7: aux32[l] += scale * aux16[l]  # accumulate scale*product into lane l
        q8 += 8; a += 8
        for l in 0..7: aux16[l]  = q8[l] * a[l]
        for l in 0..7: aux32[l] += scale * aux16[l]
        q8 += 8; a += 8

    # ---- (C) DEFERRED super-block fp32 fold into the 8-lane fp32 accumulator ----
    d = GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d        # fp16 weight-d * fp32 activation-d
    for l in 0..7: sums[l] += d * aux32[l]            # 8 INDEPENDENT fp32 lane-accumulators

# ---- (D) final horizontal sum, SEQUENTIAL, l = 0 .. 7 ----
for l in 0..7: sumf += sums[l]
*s = sumf
```

**fp32 accumulation order (byte-exactness-critical):**
- The per-sub-block scale multiply lives entirely in **i32** (`aux32[l] += scale * aux16[l]`)
  — integer add is order-independent, so vectorizing (B) is byte-safe.
- The fp32 multiply is **deferred to the super-block boundary** and lands in **8 INDEPENDENT
  fp32 lane-accumulators `sums[0..7]`**, summed across super-blocks (`sums[l] += d*aux32[l]`).
- The final horizontal sum `sumf += sums[l]` runs **sequentially l=0..7**. A vector
  `vfredusum` may NOT match this order bit-for-bit (fp add is non-associative). **Reproduce
  the 8-lane accumulator + sequential horizontal sum exactly; do NOT reassociate.** Treat the
  horizontal reduction as a verify-on-board item (we have already been bitten by fp32
  ordering — the INC-2a FMA-contraction red).

**The ql/qh → element permutation (also byte-exactness-critical):** within each 128-element
chunk the dequantized weights must land in q8's element order. The mapping
(`a[0..31]/[32..63]/[64..95]/[96..127]` ← interleaved ql low/high nibbles of `ql[l]`/`ql[l+32]`
with the matching 2-bit field of `qh[l]`) must be preserved exactly. A vectorized unpack that
gets the grouping wrong is silently off. Capture this permutation precisely in the lowering.

#### q4_K — `ggml_vec_dot_q4_K_q8_K_generic` (`quants.c:645–718`) — for completeness

Same skeleton, but: (1) **packed-scale decode** at `utmp[4]` / `kmask1/2/3` (lines 685–690)
splits 12 bytes into 8 scales + 8 mins — structurable as scalar uint32 bitops but the awkward
part; (2) the unpack is plain 4-bit (`q4[l]&0xF` / `q4[l]>>4`, NO `-32` bias) into 8 sub-blocks
of 32 (processed as 4 halves of 8); (3) a **min term**: `sumi = Σ_{j=0..15} bsums[j] *
mins[j/2]; sumf -= dmin * sumi` where `dmin = fp16(x.dmin) * y.d`. The min is the SAME Family-B
min pattern we already emit for q4_1 — here SOURCED from `y[i].bsums` instead of computed from
the q8 stream, but conceptually reused. **q4_K's distinguishing cost is the packed-scale
extraction, NOT the min.**

#### The on-board RVV `_vl128` paths (NOT our target, but relevant for N3)

Both `ggml_vec_dot_q6_K_q8_K_vl128` (`arch/riscv/quants.c:2295`) and
`ggml_vec_dot_q4_K_q8_K_vl128` (`:1770`) are **hand-written raw inline `__asm__`**, not
intrinsics. They also use a different (per-128-chunk fp-convert + `fmadd`) accumulation order
than the generic. So: the **byte-exact validation target is `_generic`** (well-defined fp32
order); the **N3 perf comparison target is the `_vl128` hand-asm kernel** (the thing to
match/beat on the board, same as the Q4_0 story).

---

### Recommendation: q6_K first (justification)

| Criterion | q6_K | q4_K |
|---|---|---|
| **Scale extraction (the discriminator)** | **direct** — `int8 scales[16]` scalar load, applied in i32 | **awkward** — 6-bit cross-byte packed decode (`utmp`/`kmask` scalar uint32 bitops + type-punned reinterpret); structurable but error-prone |
| Scale hierarchy | super-block fp16 `d` + per-sub-block int8 scale | + 6-bit packed scale + 6-bit packed min |
| Min term | NONE (`x=a*q`) | YES (`x=a*q+b`) via `bsums` — but this is reused q4_1 Family-B math |
| Quant unpack | 6-bit (ql 4-bit + qh 2-bit), bias `-32` | 4-bit nibble, no bias |
| `bsums` consumed | no | yes |
| Sub-block geometry | 16 × 16 | 8 × 32 |

**The discriminator is NOT "6-bit vs 4-bit" or "which is fewer ops."** It is which scale path
is cleaner under the structured-emission discipline (I5). q6_K keeps every step a structured
emitc node with the LOWEST risk: the per-sub-block scale is a direct scalar `int8` load, and
the unpack is vand/vsrl/vsll/vor + a `-32` bias, structurally the same family as our q4_0 `xor
0x88` pattern. q4_K is *also* structurable, but adds a scalar packed-scale bit-shuffle (the
`utmp`/`kmask` decode + type-punning) on top of the same super-block novelty — materially more
to get right. **Prove the super-block shape on q6_K, then q4_K reuses the proven nested loop
and adds only its packed-scale decode + the already-emitted min term.**

(q6_K is also the dominant K-quant in real deployments — q6_K is the standard output-tensor /
high-importance quant and the `*_K_M` mixes lean on it heavily, so it is not a toy target.)

---

### Reuse-vs-NEW map (our flat block-loop template → q6_K's nested super-block)

Our existing family (`emitQ4_0Q8_0BlockDot` / the q8_0 Family-A and q4_1 Family-B siblings in
`RVVToEmitC.cpp` ~3781–5170) is: **outer `emitc.for` over `nb = n/QK` → per-block address
arithmetic → 2× scalar fp16→fp32 read → inner strip loop (integer decode/product/`vwredsum`
into a scalar `sumi`) → single-scalar fp32 fold `sumf += (sumi*dX)*dY`.** The shape knobs
`{integer_core_lmul}×{multi_block_factor}×{strip_elision}` already select the *how*, and the
op carries the block-format facts as typed attrs (I4 mirror): `qk`,
`weight_block_stride`, `activation_block_stride`, `quant_byte_offset`,
`activation_high_byte_offset` (`RVVOps.td:3947`).

**REUSABLE (no new mechanism):**

| Piece | Where today | How q6_K reuses it |
|---|---|---|
| Integer core: signed widening product + reduce | `emitStripReduce` (vle8 → vwmul/vwmacc → vwredsum), `RVVToEmitC.cpp:3947` | The per-sub-block 16-element i8×i8 dot is the SAME `vwmul`→`vwredsum` chain |
| Shape knobs `{lmul}×{multi_block}×{strip_elision}` | the three `emitBlockCore`/`emitFold` branches | Apply at the **sub-block-strip** level: the 16-element sub-block IS the natural strip boundary — `strip_elision=elided` (`vsetvl_e8m1(16)` caps at 16 ∀ VLEN≥128) aligns EXACTLY with the sub-block scale boundary; `multi_block_factor` unrolls **super-blocks** |
| Per-block address arithmetic | `blockBaseValue` (`:3912`) | Re-anchored at the 210-byte q6_K super-block stride |
| Scalar fp16→fp32 read | `fp16Read` (`emitc.call_opaque "(float)*(const _Float16 *)"`, `:3927`) | Reused for `x[i].d`; the activation `y[i].d` is fp32 — a plain load, even simpler |
| Quant unpack op-kind | INC-1 offset-binary decode (`vand`/`vsrl`/`vsll`/`vor` + bias) | The 6-bit ql+qh reconstruct + `-32` bias is the same op family (more ops, same kind) |
| The measurement tuner | gearbox enumerate→prune→rank→select→stamp (research/gearbox-autotuner-wiring.md) | INHERITS unchanged — q6_K just registers its candidate shapes |
| Structured-emitc discipline (I5) | every node is a typed emitc op, zero `raw()` | Must HOLD — q6_K was chosen precisely because it can with the least friction |

**GENUINELY NEW (must be built):**

| New piece | Why it does not exist today | Risk |
|---|---|---|
| **Two-level fold (THE hard piece)** | our whole family folds to ONE scalar `sumf` per block (`emitFold`, `RVVToEmitC.cpp:4116` — verified single-scalar); q6_K needs per-sub-block **int8 scale in the i32 domain** + a **carried multi-lane i32 accumulator** + the fp32 `d`-multiply **DEFERRED to the super-block boundary** into a **multi-lane fp32 accumulator** + an ordered horizontal sum | HIGH (byte-exactness: fp32 order, see above) |
| **Nested sub-block loop** | our template has exactly ONE loop level (the block loop) + an inner strip loop; q6_K is super-block × sub-block(×16) × strip | MED (structure, not an op) |
| **Per-sub-block int8 scale extract + apply** | our scale is per-BLOCK fp16, applied in fp32; q6_K's is per-SUB-block int8, applied in i32 before the deferred fp32 | MED |
| **6-bit ql+qh unpack with the exact element permutation** | INC-1 is a single 4-bit nibble split; q6_K composes a 4-bit `ql` part + a 2-bit `qh` part into element order | MED (permutation correctness) |
| **q8_K activation block** (fp32 `d`, 256 int8, bsums) | our activation is block_q8_0 (fp16 d, 32 int8) | LOW (q6_K ignores bsums) |
| `bsums` / min term | not in our q4_0/q8_0 path | DEFERRED to q4_K (and reuses q4_1 Family-B min) |

**The lowering shape** (structured emitc, mirroring the generic):
```
emitc.for ib in [0, nb):                              # super-block loop
    aux32 = {8-lane i32 accumulator}, zeroed
    <unpack 256 weights into element order, biased -32>   # 2 chunks of 128
    emitc.for js in [0, 16):                          # sub-block loop
        scale = load x[ib].scales[js]                 # int8 scalar
        sumi16 = strip dot over the 16-element sub-block   # {lmul}×{strip_elision} HERE
        aux32 += scale * sumi16                        # i32 domain, order-free
    d = fp16(x[ib].d) * y[ib].d                        # fp32
    sums[0..7] += d * aux32[0..7]                       # deferred multi-lane fp32 fold
sumf = Σ_{l=0..7} sums[l]   (sequential)               # ordered horizontal reduce
*s = sumf
```
`multi_block_factor` unrolls the OUTER (super-block) `emitc.for`; `{integer_core_lmul,
strip_elision}` apply to the INNER sub-block strip. The 16-element sub-block as the strip unit
makes `elided` align exactly with the scale boundary (the clean reuse win).

---

## Increment plan (validatable steps — break it up; agents have hit socket drops on big tasks)

> Each increment is byte-exact on `ssh rvv` against the `_generic` reference before the next
> starts. Keep every node structured emitc (I5) — flag immediately if any step seems to force
> a `raw()` string (none is expected for q6_K; the int8 scalar scales are the reason).

- **K1 — super-block integer core (one super-block, byte-exact).** Build the 6-bit ql+qh
  unpack (256 weights, element-ordered, `-32` bias) + the per-sub-block int8-scaled i32
  partial: the 16× sub-block strip dot accumulated into the 8-lane `aux32`. Validate against
  the generic's **integer partial — the `aux32[0..7]` state right before the `d`-multiply** —
  at n=256. NOTE the granularity mismatch: the unpack works on 128-element chunks while the
  scale boundary is every 16 elements, so K1 is "one super-block's unpack + scaled i32
  partial," NOT a single isolated sub-block. **Difficulty: MEDIUM** (the unpack permutation +
  the two-level i32 accumulation are new, but all integer/order-free).

- **K2 — full nested loop + scale hierarchy + ABI → deployable.** Wrap K1 in the super-block
  `emitc.for` over `nb = n/256`, add the deferred multi-lane fp32 fold (`sums[8]` +=
  `d*aux32`), the sequential horizontal sum, the fp16-`d`×fp32-`d` scale, the `*s` store, and
  the ggml call signature (`void ggml_vec_dot_q6_K_q8_K(int n, float *s, size_t bs, const void
  *vx, size_t bx, const void *vy, size_t by, int nrc)`; contract `n % 256 == 0`, `nrc == 1`).
  Validate `*s` byte-exact vs `_generic` over many n (and under `-ffp-contract`
  default/on/off/fast, as INC-2a). **Difficulty: HARD** — the fp32 ordering (deferred fold +
  sequential horizontal reduce) is THE byte-exactness risk; verify the horizontal reduction on
  the board, do not reassociate. Break K2 itself into K2a (nested loop + fp32 hierarchy,
  byte-exact `*s`) and K2b (ABI wrap + deploy) if it risks a socket drop.

- **K3 — measurement-tuner inherits.** Register q6_K's candidate shapes
  (`{integer_core_lmul}×{multi_block_factor}×{strip_elision}` at the sub-block-strip level) so
  the gearbox enumerate→prune(VLEN legality + vreg budget)→rank→select→stamp picks the q6_K
  shape with NO hand-tuning, same wiring as the Q4_0 story
  (research/gearbox-autotuner-wiring.md). **Difficulty: LOW-MEDIUM** if the tuner is already
  live-wired for Q4_0; the new work is only the candidate registration + a q6_K cost/latency
  model entry. Then N3-measure vs the `_vl128` hand-asm kernel on the board.

### Difficulty summary & sequencing
- Hardest: **K2's fp32 ordering** (deferred multi-lane fold + ordered horizontal reduce) —
  the one thing a self-test passes but the board can fail. Verify on `ssh rvv`.
- Second: **K1's unpack permutation** (ql/qh → element order) — silently wrong if grouped
  wrong; pin it against the generic's `aux8` ordering.
- Everything else (integer core, shape knobs, address arithmetic, tuner) is reused.

---

## Caveats / Not Found

- This is a **scoping** doc — no implementation, no new op defined yet. The exact attr set for
  a `q6_K_q8_K_block_dot` op (analogous to `RVVOps.td:3947`) is a K1/K2 design decision; the
  format facts above (256, 16×16, 210-byte stride, the byte offsets) are the I4 attrs it would
  carry. The byte offsets the op's address arithmetic will need:
  - weight (`block_q6_K`, stride 210): `ql @0` | `qh @128` | `scales @192` | `d @208`.
  - activation (`block_q8_K`, stride 292): `d @0` | `qs @4` | `bsums @260` (q6_K ignores bsums).
- The `_vl128` RVV paths are **raw inline asm** — we do NOT reproduce them; they are only the
  N3 perf baseline. The byte-exact oracle is `_generic` exclusively.
- q4_K's packed-scale decode is **structurable** (scalar uint32 bitops), so it does NOT force
  unstructured emission; q6_K is preferred on lower risk/cleanliness, not on q4_K being
  illegal. If a future implementer finds the q4_K `utmp`/`kmask` decode cannot be kept
  structured in practice, THAT would be the signal to re-rank — flagged but not expected.
- q5_K (`quants.c:720`) is the q4_K sibling + a 1-bit `qh` high plane — strictly harder than
  q4_K; out of scope for the first super-block proof.
- Not measured: whether `vfredusum` matches the sequential `sums[0..7]` horizontal sum on the
  board. Flagged as a K2 verify-on-board item (likely needs the sequential scalar reduce to
  stay byte-exact, as the INC-2a FMA red showed).
