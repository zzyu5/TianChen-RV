# INC-43 — tq1_0 x q8_K byte-exact: the BASE-3-PACKED TERNARY TriLM class (24/24, 100%)

The 24th and LAST ggml dot kernel: `ggml_vec_dot_tq1_0_q8_K`. tq1_0 is the 1.6875-bpw
TriLM/BitNet ternary K-quant, the BASE-3-PACKED sibling of tq2_0. With this increment
the compiler covers **all 24 `ggml_vec_dot_*_generic` dot kernels — literal 100%**.

## What tq1_0 is (ggml-common.h:266 + quants.c:430, primary source)

- `block_tq1_0 = { uint8_t qs[(QK_K-4*QK_K/64)/5]; uint8_t qh[QK_K/64]; ggml_half d }`
  (QK_K=256). The struct sizes resolve to **qs = 48 bytes** (5 trits/byte, 3^5=243<256),
  **qh = 4 bytes** (4 trits/byte), **d** the fp16 super-block scale — `sizeof
  block_tq1_0 == 54` (confirmed by the static_assert). Layout: qs @ +0, qh @ **+48**, d
  @ **+52** (the fp16 scale is the SUFFIX; the genuinely-new fact vs tq2_0 is the SECOND
  weight array qh @ +48). NOTE: the task prose said "qs=32 bytes" — the arithmetic and
  the static_assert give qs=48 / stride=54; the larger qs is what we implemented.
- The activation is `block_q8_K` (stride 292, fp32 d @0, int8 qs @4) — the SAME
  super-block activation as q2_K..q6_K / tq2_0. tq1_0 uses ONLY y.d and y.qs (NO bsums,
  NO min).
- **BASE-3 trit extract** (quants.c:452-453): `pow3[6]={1,3,9,27,81,243}`; for byte b,
  digit l: `q = (uint8_t)(b * pow3[l])` (an **8-bit multiply** — the mod-256 wrap IS the
  decode), `xi = ((uint16_t)q * 3) >> 8` (the high base-3 digit, in {0,1,2}), value
  `xi - 1` in {-1,0,1}. Verified exhaustively: over ALL 256 bytes x all 6 digits the
  structural pipeline == `_generic`'s arithmetic, 0 mismatch, xi range [0,2].
- **The dot** (`_generic`, quants.c:446-477): THREE q8 index regions feed ONE `int sum`:
  (a) qs MAIN j=0, l 0..4, m 0..31 -> q8[l*32+m], weight qs[m] (160 elems);
  (b) qs TAIL j=32, l 0..4, m 0..15 -> q8[160+l*16+m], weight qs[32+m] (80 elems);
  (c) qh        l 0..3, j 0..3      -> q8[240+l*4+j], weight qh[j] (16 elems).
  Every q8 index 0..255 is written exactly once (160+80+16=256). Then `sumf += (float)sum
  * (fp16(x.d) * y.d)`. *s = sumf. (Single fp16 scale, NO min, NO per-sub-block scale,
  fp32 accumulate.)
- The byte-exactness **oracle = `_generic`** — the well-defined fp32 order our op mirrors.

## The implementation (STRUCTURED, raw()=0)

- **New op** `tcrv_rvv.tq1_0_q8_k_block_dot` (RVVOps.td) + `verify()`
  (RVVDialectWideningOps.cpp), fail-closed (I7) on kind / scale_model
  ("ternary-base3-single-fp16-scale-i32-domain-scalar-fp32-fold") / qk (256) / weight
  stride (54) / q8_K stride (292) / qs offset (+0) / **qh offset (+48)** / d offset (+52)
  / the two q8 offsets (+0/+4) / non-float-* output / non-uint8_t* inputs. The op carries
  the NEW `weight_qh_byte_offset` attr (the second weight array) on top of the tq2_0 attr
  set; NO scales/min/dmin/bsums.
- **Lowering** `emitTQ1_0Q8_KBlockDot` (RVVToEmitC.cpp) — the decode REPLACES tq2_0's
  2-bit field shift; the dot + fold + store are tq2_0's VERBATIM (offsets changed). The
  THREE structured pieces:
  - the SUPER-block loop over nb = n/256; the BASE-3 unpack into `aux8[256]` in q8 order.
    14 decode regions (5 qs-main at e8m2 32-lane, 5 qs-tail at e8m1 16-lane, 4 qh at e8m1
    4-lane). Each region: `vmul.vx u8 (byte*pow3[l])` (the mandatory uint8 WRAP, NOT
    widened) -> `vwmulu.vx u8->u16 (q*3)` -> `vsrl u16 (>>8)` -> `vncvt u16->u8` ->
    `vreinterpret u8->i8` -> `vadd.vx i8 (-1)` (the ternary bias) -> `vse8`. The pow3
    multiply is NEVER fused into the widening multiply (fusing would skip the mod-256
    wrap and be wrong for any byte where byte*pow3 >= 256). pow3[l] is a per-l literal
    (qs main/tail = {1,3,9,27,81}; qh = {1,3,9,27}).
  - the SINGLE per-super-block int32 accumulator `sumi`, fed by tq2_0's VERBATIM per-16-
    lane signed widening dot (vle8 i8m1 x2 -> vwmul_vv i16m2 -> vwredsum_vs i32m1 ->
    vmv_x_s), summed (integer add is order-free; NO per-sub-block scale).
  - the SINGLE-scale fp32 fold: `d = fp16(x.d) * y.d` (its OWN product, fp16 weight scale
    FIRST per quants.c:476) then `sumf += (float)sum * d` as ONE emitc.expression -> ONE
    C statement; then `*s = sumf`.
- The aux8[i] <-> q8[i] pairing is contiguous because the decode writes each region at
  exactly the `_generic` q8 index (l*32+m / 160+l*16+m / 240+l*4+j). So `Σ aux8[i]*q8[i]`
  over 256 contiguous lanes == `_generic`'s triple sum (integer, order-free).

## Byte-exact (ssh rvv, VLEN=128, rv64gcv) — ssh_rvv_byte_exact_stdout.txt

Board: riscv64, isa rv64imafdcv...zvfh..., clang 18.1.3, -O2 -march=rv64gcv. The oracle
is our faithful **scalar transcription** of `ggml_vec_dot_tq1_0_q8_K_generic`
(quants.c:430-480) — the SAME well-defined fp32 order validation bar every sibling uses
(we transcribe the source body, not the linked ggml symbol; the transcription was
cross-checked for full 0..255 q8-index coverage and sizeof block_tq1_0 == 54, and == the
real `_generic` over 2000 cases off-board). This scalar leg, with a decode entirely
different from the vector emitter, is the independent oracle that carries the 0-delta. A
SECOND witness is an RVV transcription that varies only the bias form (`vsub` against a +1
splat vs the emitter's `vadd -1`) and the reduction width (the whole 32-byte main chunk in
one e8m2 strip vs the per-16-lane reduce) — it shares the base-3 extraction chain, so it
is a corroborating witness, not a second independent decode. Both legs are compared
BIT-FOR-BIT.

**`0 failures over 3550 cases` per mode x all 3 valid clang fp-contract modes
(off/on/fast, per-mode clean rebuild).** `_generic` cross-check delta = **0/3550 in EVERY
mode** (including =fast — the fold groups into one emitc.expression -> one C statement, so
even fast-math forms the identical fp). Cases: random (n in {256,512,...,8192} multiples
of 256, q8 over the FULL int8 range incl. -128) + the edge crosses — qs bytes 0x00 (every
trit -1 ISOLATED -> -Sum(q8)), 121 / 242 (the all-1 / all-2 base-3 packings, mixed trits
under ggml's >>8 reconstruction), 0xFF (high-byte wrap stress), marching bytes — each x
q8 saturation (+127 / +/-128 / -127). Every n>=256 forces the qs-MAIN + qs-TAIL + qh
paths (256 = 160 main + 80 tail + 16 qh), so the tail + qh regions are exercised on every
case.

The base-3 decode pipeline was ALSO verified in ISOLATION (off-board): over ALL 256 bytes
x 6 digits, the structural pipeline (vmul-wrap -> vwmulu*3 -> vsrl 8 -> narrow ->
reinterpret -> -1) == `_generic`'s `((uint8_t)(b*pow3[l])*3>>8)-1`, 0 mismatch. And the
harness's `_generic` transcription == ggml's REAL `ggml_vec_dot_tq1_0_q8_K_generic` body
over 2000 random cases, 0 mismatch (the index math + sizes are exact).

FOUR negative controls (all diverge, ours==real in each):
- **wrong base-3 extract** (bias=0 -> the raw high base-3 digit xi in {0,1,2} instead of
  the ternary xi-1): real `21478.76` vs wrong `-759870.88` -> the `-1` trit bias is
  load-bearing.
- **wrong scale** (fp16 d x 2): real `-8224.46` vs `-16448.91` (exactly 2x) -> the single
  d scale is load-bearing.
- **drop-qh** (omit the qh contribution, the last 16 q8 indices): real `-9181.81` vs
  `-4446.78` -> the qh region is consumed (load-bearing).
- **perturbed weight** (flip a qs byte) -> diverges (non-vacuous).

Buffer sizing is 1:1 here (one block_q8_K per tq1_0 super-block, nb each, weight nb*54,
activation nb*292, n=nb*256) — no q1_0-style 4x multiplier landmine.

## raw()=0 + structured (structured_proof.txt)

raw() actual calls = 0; residual ACTUAL `tcrv_rvv.<op>` ops in the lowered MLIR = 0;
residual unrealized_conversion_cast = 0; libm calls = 0. The ONLY sanctioned opaque
scalar piece is the SINGLE fp16 read (`(float)*(const _Float16 *)` at xb+52) — ONE read,
not a dall/dmin pair. 14 vse8 decode stores (5 qs-main e8m2 + 5 qs-tail e8m1 + 4 qh e8m1),
16 sub-block vwredsum reduces, 287 tcrv_rvv mirror provenance comments (I4). The
regenerated kernel is BYTE-IDENTICAL whether the input carries selected_path_role
"dispatch case" or "direct variant" (deterministic).

## Lit + reds + clean rebuild

Full CLEAN rebuild (`ninja -t clean && ninja`) GREEN. Lit suite `check-tianchenrv`:
**676 total, 673 pass, exactly the 3 documented pre-existing reds** (the Scripts
widening-dot-reduce-add dry-run / self-test trio — unrelated to tq1_0). TWO NEW tests
added, both PASS:
`test/Conversion/RVV/rvv-to-emitc-tq1-0-q8-k-block-dot.mlir` (structured emission: the
e8m2/e8m1 base-3 pipeline `vmul_vx_u8 / vwmulu_vx / vsrl_vx_u16 / vncvt / vadd_vx_i8 -1`,
the per-16-lane vwredsum, the single-scale fold; CHECK-NOT `vand_vx_u8m2` /
bitwise_right_shift / const int16_t so it is NOT a tq2_0 2-bit / q4_K scale/min misroute)
and `test/Dialect/RVV/tq1-0-q8-k-block-dot-dataflow.mlir` (8 cases: accept ggml ABI + 7
fail-closed rejections incl. wrong kind / scale_model / qh offset (+48, the new fact) / d
offset (+52) / stride (54) / an unexpected scales attr / wrong output C type). Additive —
new op, new verifier, new recognizer + dispatch branch, new emitter; no shared path
touched. tq2_0's emitted body regenerates BYTE-IDENTICAL after the tq1_0 addition (directly
diffed vs the committed inc42 artifact). The other siblings (q2_K..q6_K, the IQ/FP4/binary
families) rest on their green conversion lit tests, which are the structural regeneration
check (any change to a shared path would flip them).

## Files

- `include/TianChenRV/Dialect/RVV/IR/RVVOps.td` — `GgmlBlockDotTQ10Q8KOp` def (+ the new
  `weight_qh_byte_offset` attr).
- `lib/Dialect/RVV/IR/RVVDialectWideningOps.cpp` — `GgmlBlockDotTQ10Q8KOp::verify()`.
- `lib/Conversion/RVV/RVVToEmitC.cpp` — `isTQ1_0Q8_KBlockDotBody` recognizer, the dispatch
  branch, and `emitTQ1_0Q8_KBlockDot` (the structured emitter).
- `test/Conversion/RVV/rvv-to-emitc-tq1-0-q8-k-block-dot.mlir` (new).
- `test/Dialect/RVV/tq1-0-q8-k-block-dot-dataflow.mlir` (new).
- artifacts/inc43-tq1_0/: tq1_0_input.mlir, tq1_0_lowered.mlir, tq1_0_kernel_body.c,
  inc43_validate.cpp, ssh_rvv_byte_exact_stdout.txt, structured_proof.txt,
  coverage_24_kernels.txt, this file.

## Coverage: ALL 24 ggml dot kernels (literal 100%) — coverage_24_kernels.txt

Every `ggml_vec_dot_*_generic` in quants.c (24 of them) now has a covering recognizer +
structured emitter in RVVToEmitC.cpp:

| class | kernels | mechanism |
|---|---|---|
| block-quant linear | q4_0, q4_1, q5_0, q5_1, q8_0 | nibble unpack x i8 dot, per-block fp16 scale(+min) |
| K-quant super-block | q2_K, q3_K, q4_K, q5_K, q6_K | super-block ql/qh unpack + per-sub-block scale/min |
| CODEBOOK / FP4 | iq4_nl, iq4_xs, mxfp4, nvfp4 | non-linear int8 table / e2m1 vrgather gather |
| GRID / ternary | iq2_xxs, iq2_xs, iq2_s, iq3_xxs, iq3_s, iq1_s, iq1_m | packed grid lookup + sign plane / delta |
| BINARY | q1_0 | per-bit +-1 sign decode (kmask/vmsne/vmerge) |
| TERNARY (TriLM) 2-bit | tq2_0 | 2-bit unpack `((qs>>shift)&3)-1`, single fp16 scale, ONE i32 acc |
| **TERNARY (TriLM) BASE-3** | **tq1_0** | **base-3 trit `((uint8_t)(b*pow3[l])*3>>8)-1` of qs[48]+qh[4], single fp16 scale, ONE i32 acc** |

tq1_0 was the LAST remaining rung. **24/24 = literal 100% of the ggml dot kernels.**
