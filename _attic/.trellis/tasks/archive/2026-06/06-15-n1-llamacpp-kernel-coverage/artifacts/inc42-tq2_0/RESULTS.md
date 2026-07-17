# INC-42 — tq2_0 x q8_K byte-exact: the TERNARY ({-1,0,+1}) TriLM class

The 23rd ggml dot kernel: `ggml_vec_dot_tq2_0_q8_K`. tq2_0 is the 2-bit TERNARY
TriLM/BitNet K-quant — one of the **LAST TWO** uncommon ggml dot kernels (the other is
tq1_0). This is COVERAGE toward literal 100% of the ggml dot kernels.

## What tq2_0 is (ggml-common.h + quants.c:482, primary source)

- `block_tq2_0 = { uint8_t qs[QK_K/4]; ggml_half d }` (QK_K=256, **sizeof 66**): the
  64 packed 2-bit-weight bytes LEAD the block at byte **+0**, and the SINGLE fp16
  super-block scale d is the **SUFFIX at +64**. d at the END is the genuinely-new
  layout vs every sibling (where d is near the front).
- The activation is `block_q8_K` (stride 292, fp32 d @0, int8 qs @4) — the SAME
  super-block activation as q2_K/q3_K/q4_K/q5_K/q6_K. tq2_0 uses ONLY y.d and y.qs
  (NO bsums — there is no min term).
- **Ternary decode** (quants.c:501): `t = ((x.qs[j+k] >> (l*2)) & 3) - 1` ∈ {-1,0,1,2}
  (the TriLM encoder only emits {0,1,2} → realized ternary {-1,0,1}; the structural
  decode is exact over the full 2-bit range). l (0..3) selects the 2-bit field, k (0..31)
  the byte, j (0,32) the 32-byte half.
- **The dot** (`_generic`, quants.c:495-509): `sumi += y.qs[j*4 + l*32 + k] · t` over
  j∈{0,32}, l∈0..3, k∈0..31 — ONE int32 accumulator over the whole super-block; then
  `d = y.d · fp16(x.d)`; `sumf += (float)sumi · d`. *s = sumf. (Single fp16 scale, NO
  min, NO per-sub-block scale, fp32 accumulate.)
- The byte-exactness **oracle = `_generic`** — the well-defined fp32 order our op
  mirrors. (The board ALSO has a riscv `_vl128` path (arch/riscv/quants.c:6454), but
  that is a DIFFERENT raw-RVV summation order and does NOT define the byte-exact
  reference — every sibling op targets `_generic`.)

## The implementation (STRUCTURED, raw()=0; REUSES q2_K's 2-bit unpack)

- **New op** `tcrv_rvv.tq2_0_q8_k_block_dot` (RVVOps.td) + `verify()`
  (RVVDialectWideningOps.cpp), fail-closed (I7) on kind / scale_model
  ("ternary-single-fp16-scale-i32-domain-scalar-fp32-fold") / qk (256) / weight stride
  (66) / q8_K stride (292) / weight qs offset (+0) / weight d offset (**+64**) / the two
  q8 offsets (+0/+4) / non-float-* output / non-uint8_t* inputs. The op is LEAN — it
  carries ONLY {kind, scale_model, qk, weight_block_stride, activation_block_stride,
  weight_qs_byte_offset, weight_d_byte_offset, activation_d_byte_offset,
  activation_quant_byte_offset}; NO scales/min/dmin/bsums/sub_block attrs (it has none).
- **Lowering** `emitTQ2_0Q8_KBlockDot` (RVVToEmitC.cpp) — the THREE structured pieces:
  - the SUPER-block loop over nb = n/256; the 2-bit TERNARY unpack into aux8[256]
    REUSING q2_K's verbatim (u8m2 vle8 → vsrl_vx → vand_vx 0x03 → u8→i8 vreinterpret)
    plus the per-element `-1` ternary bias (`vadd.vx -1` in the i8 domain) BEFORE the
    vse8;
  - the SINGLE per-super-block int32 accumulator sumi, fed by the q2_K-verbatim
    per-16-lane signed widening dot (vle8 i8m1 ×2 → vwmul_vv i16m2 → vwredsum_vs i32m1
    → vmv_x_s), summed (integer add is order-free; NO per-sub-block scale multiply);
  - the SINGLE-scale fp32 fold: `d = y.d · fp16(x.d)` (its OWN product) then
    `sumf += (float)sumi · d` as ONE emitc.expression → ONE C statement; then `*s = sumf`.

## The ternary `-1` decode mechanism (item 5)

After q2_K's 2-bit field is isolated (`(qs >> (l*2)) & 3` ∈ {0,1,2,3}) and reinterpreted
u8→i8, ONE `__riscv_vadd_vx_i8m2(field, -1, vl)` applies the ternary bias PER ELEMENT in
the i8 domain (field ∈ [0,3] so the i8 subtract is exact → aux8 ∈ [-1,2]). This mirrors
`_generic`'s per-element `(((qs>>shift)&3) - 1)` EXACTLY — NOT a `Σ(field·q8) − Σq8`
refactor (which would add a bias term and a new failure mode). The aux8 element ordering
pairs contiguously with q8 (aux8[i] ↔ q8[i], since `j*4 + l*32 + k = 128·chunk + 32·field
+ k` = q2_K's aux8 index). There are 8 such vadd ops per super-block (2 qs chunks × 4
fields). Omitting the `-1` → wrong ternary → diverges (negative control 1).

## Byte-exact (ssh rvv, VLEN=128, rv64gcv) — ssh_rvv_byte_exact_stdout.txt

Board: riscv64, isa rv64imafdcv...zvfh..., clang 18.1.3, -O2 -march=rv64gcv. Oracle =
`_generic`. We compare against BOTH a faithful scalar transcription of `_generic`
(quants.c:482-511) AND an INDEPENDENT RVV transcription (a DISTINCT ternary expression:
the 32-lane m2 unpack with a `vsub` against a +1 splat over a 128-lane m4 reduce — NOT
the emitter's i8m2 vadd / per-16-lane reduce), both BIT-FOR-BIT.

**`0 failures over 3550 cases` per mode × all 3 valid clang fp-contract modes
(off/on/fast, per-mode clean rebuild).** `_generic` cross-check delta = **0/3550 in
EVERY mode** (including =fast — the fold groups into one emitc.expression → one C
statement, so even fast-math forms the identical fp). Cases: random (n ∈ {256,512,...,
8192} multiples of 256, q8 over the FULL int8 range incl. -128) + the edge crosses —
the THREE realized ternary values in ISOLATION (the task's named edge set): all-0x00 qs
(→ field 0 → ternary **-1** → −Σq8), all-0x55 qs (→ field 1 → **0** → Σ=0), all-0xAA qs
(→ field 2 → **+1** → +Σq8), plus all-0xFF qs (→ field 3 → +2, OOD stress) and marching
qs — each × q8 saturation (+127 / ±128 / -127).

THREE negative controls (all diverge, ours==real in each):
- **wrong -1 offset** (bias=0 → raw 2-bit field {0,1,2,3} instead of ternary): real
  `626987.75` vs wrong `1818196.25` → the `-1` ternary bias is load-bearing.
- **wrong scale** (fp16 d × 2): real `-73487.16` vs `-146974.31` (exactly 2×) → the
  single d scale is load-bearing.
- **perturbed weight** (flip the low 2-bit field of byte 0) → diverges (non-vacuous;
  the weight bytes are consumed).

Buffer sizing is 1:1 here (one block_q8_K per tq2_0 super-block, nb each) — no q1_0-style
4× multiplier landmine.

## raw()=0 + structured (structured_proof.txt)

raw() actual calls = 0; residual `tcrv_rvv.<op>` dialect ops = 0; residual
unrealized_conversion_cast = 0; libm calls = 0. The ONLY sanctioned opaque scalar piece
is the SINGLE fp16 read (`(float)*(const _Float16 *)` at xb+64) — NO dall/dmin pair (one
read, not two). 8 ternary `vadd_vx_i8m2(...,-1,...)`. 201 tcrv_rvv mirror provenance
comments (I4). The regenerated kernel is BYTE-IDENTICAL to the shipped one (deterministic).

## Lit + reds + clean rebuild

Full CLEAN rebuild (`ninja -t clean && ninja`, 240 targets from scratch) GREEN. Lit
suite `check-tianchenrv`: **674 total, 671 pass, exactly the 3 documented pre-existing
reds** (`Scripts/rvv-generated-bundle-abi-e2e-{explicit,pre-realized}-computed-masked-
strided-input-widening-dot-reduce-add-dry-run`, `…-self-test` — unrelated to tq2_0,
Scripts widening-dot-reduce). TWO NEW tests added, both PASS:
`test/Conversion/RVV/rvv-to-emitc-tq2-0-q8-k-block-dot.mlir` (structured emission: the
e8m2 2-bit unpack, the `__riscv_vadd_vx_i8m2` -1 ternary bias, the per-16-lane vwredsum,
the single-scale fold; CHECK-NOT bitwise_and/bitwise_right_shift/const int16_t/fp32 sub
so it is NOT a q2_K/q4_K scale/min misroute) and
`test/Dialect/RVV/tq2-0-q8-k-block-dot-dataflow.mlir` (8 cases: accept ggml ABI + 7
fail-closed rejections incl. wrong kind / wrong scale_model / wrong d offset (+64) /
wrong qs offset / wrong stride / an unexpected scales attr / wrong output C type).
Additive — new op, new verifier, new recognizer + dispatch branch, new emitter; no
shared path touched. q1_0 regenerates BYTE-IDENTICAL after the tq2_0 addition, and ALL
sibling conversion lit tests stay green (the lit suite is the regeneration check for the
other kernels).

## Files

- `include/TianChenRV/Dialect/RVV/IR/RVVOps.td` — `GgmlBlockDotTQ20Q8KOp` def.
- `lib/Dialect/RVV/IR/RVVDialectWideningOps.cpp` — `GgmlBlockDotTQ20Q8KOp::verify()`.
- `lib/Conversion/RVV/RVVToEmitC.cpp` — `isTQ2_0Q8_KBlockDotBody` recognizer, the
  dispatch branch, and `emitTQ2_0Q8_KBlockDot` (the structured emitter).
- `test/Conversion/RVV/rvv-to-emitc-tq2-0-q8-k-block-dot.mlir` (new).
- `test/Dialect/RVV/tq2-0-q8-k-block-dot-dataflow.mlir` (new).
- artifacts/inc42-tq2_0/: tq2_0_input.mlir, tq2_0_lowered.mlir, tq2_0_kernel_body.c,
  inc42_validate.cpp, ssh_rvv_byte_exact_stdout.txt, structured_proof.txt, this file.

## Coverage: 23 ggml dot kernels; the LAST one (item 6)

| class | kernels | mechanism |
|---|---|---|
| block-quant linear | q4_0, q4_1, q5_0, q5_1, q8_0 | nibble unpack × i8 dot, per-block fp16 scale(+min) |
| K-quant super-block | q2_K, q3_K, q4_K, q5_K, q6_K | super-block ql/qh unpack + per-sub-block scale/min |
| CODEBOOK / FP4 | iq4_nl, iq4_xs, mxfp4, nvfp4 | non-linear int8 table / e2m1 vrgather gather |
| GRID / ternary | iq2_xxs, iq2_xs, iq2_s, iq3_xxs, iq3_s, iq1_s, iq1_m | packed grid lookup + sign plane / delta |
| BINARY | q1_0 | per-bit ±1 sign decode (kmask/vmsne/vmerge) |
| **TERNARY (TriLM)** | **tq2_0** | **2-bit unpack `((qs>>shift)&3)-1` → {-1,0,1}, single fp16 scale, ONE i32 acc** |

**The LAST remaining ggml dot kernel: `tq1_0`** (1.6875 bpw TriLM). tq1_0 is the
base-3-PACKED sibling: `block_tq1_0 = { uint8_t qs[(QK_K - 4·QK_K/64)/5]; uint8_t
qh[QK_K/64]; ggml_half d }` (5 ternary trits packed per byte, 3^5=243<256, plus a 4-trit
qh tail; QK_K=256). The decode is NOT a bit-field shift — it MULTIPLIES by a power of 3
and reads the high base-3 digit (`q = (x·3) >> 8` style, then `q - 1`), a genuinely new
base-3 unpack mechanism distinct from tq2_0's 2-bit fields. It is the final 100%-coverage
rung. (NOTED, not implemented here — tq2_0 is the deliverable.)
