# INC-41 — q1_0 x q8_0 byte-exact: the BINARY ({-1,+1}) class

The 22nd ggml dot kernel: `ggml_vec_dot_q1_0_q8_0`. q1_0 is the 1-bit BINARY class
— one of the LAST THREE uncommon ggml dot kernels. The q1_0 weight is a
1-bit-per-element SIGN: a SET bit decodes to `+q8`, a CLEAR bit to `-q8`, and the q8
value IS the magnitude. This is COVERAGE toward literal 100% of the ggml dot kernels.

## What q1_0 is (ggml-common.h + quants.c:123, primary source)

- `block_q1_0 = { ggml_half d; uint8_t qs[16] }` (QK1_0 = 128, **sizeof 18**): the
  fp16 scale `d0` at byte +0, then 16 packed BIT bytes (128 element signs) at +2.
- The activation is `block_q8_0` (stride 34, quants at +2). ONE q1_0 super-block (128
  elems) spans **FOUR** block_q8_0 blocks: sub-block k (k=0..3) reads q8 block
  `4*ib + k`, and its 32 bits live at weight bytes `+2 + k*4 .. +2 + k*4 + 3`.
- Bit pairing (quants.c:152-161): within sub-block k, bit `bits[k*4 + b]` bit `m`
  (m=0..7) pairs with q8 lane `b*8 + m`, value `(bit & (1<<m)) ? +q8 : -q8`.
- Dot (`_generic`, quants.c:139-167): `sumf += d0 · Σ_{k=0..3}( d1_k · Σ_{32 elems}(
  ±q8 ) )`, with `d0 = fp16(x.d)`, `d1_k = fp16(y[4*ib+k].d)`, fp32 accumulate.
- **There is NO arch/riscv q1_0 kernel** — `ggml_vec_dot_q1_0_q8_0` IS the `_generic`
  kernel (quants.c:123), so `_generic` is the real board kernel AND the oracle. The
  validation oracle is a faithful TRANSCRIPTION of quants.c:123 (NOT a link against
  compiled ggml) — exactly the nvfp4 sibling bar.

## The implementation (STRUCTURED, raw()=0; REUSES iq2_xxs's sign plane)

- **New op** `tcrv_rvv.q1_0_q8_0_block_dot` (RVVOps.td) + `verify()`
  (RVVDialectWideningOps.cpp), fail-closed (I7) on kind / scale_model
  ("binary-sign-per-bit") / qk (128) / weight stride (18) / q8 stride (34) /
  activation_blocks_per_weight (4) / the two quant offsets (+2/+2) / non-m1 anchor /
  operand C types. NO codebook attr (the q8 value is the magnitude).
- **Lowering** `emitQ1_0Q8_0BlockDot` (RVVToEmitC.cpp) — the THREE structured pieces:
  - the SUPER-block loop over nb = n/128; the four q8 sub-blocks UNROLLED, each with
    its own d1_k fp16 read + q8-block address `(ib*4 + k)*34`;
  - the BINARY sign decode over four 8-lane groups per sub-block, REUSING the iq2_xxs
    kmask/vmsne sign-plane primitive (a `{1,2,4,...,128}` bit-selector emitted ONCE as
    a `static const uint8_t[8]` decl + vle8 load, vand, `vmsne != 0` -> mask);
  - the two-level fp32 fold `sumi += d1_k * (float)sumi_block` (per k, strict
    ascending) then `sumf += d0 * sumi` (per super-block), each ONE emitc.expression
    -> ONE C FMA; then `*s = sumf`.

## The ±1 binary decode mechanism (item 5)

Each 8-lane group: broadcast the scalar bits byte into 8 lanes (`vmv_v_x_u8`), `vand`
the `{1,2,4,8,16,32,64,128}` kmask, `vmsne != 0` -> a per-lane `vbool8_t` sign mask
(SET bit -> mask true). The q8 group is **WIDENED to i16 FIRST**
(`vwcvt_x_x_v_i16m2`), then `vmerge(vneg_i16(q8w), q8w, mask)` applies the sign IN THE
i16 DOMAIN — `vmerge` selects `q8w` (the +q8) where the mask is true and `vneg(q8w)`
(the -q8) where false, EXACTLY ggml's `bit ? +q8 : -q8`. The signed i16 group is
reduced by `vwredsum` into the carried `sumi_block`.

The i16-domain placement is LOAD-BEARING and the one genuine subtlety: ggml's
`_generic` negates in the INT domain, so `-q8` of the int8 boundary value `-128` is
`+128`. An i8 `vneg(-128)` OVERFLOWS back to `-128` (verified on the board:
`vneg(-128) as i8 = -128`). Widening i8->i16 BEFORE the negate keeps `+128`
representable, so the decode is byte-exact over the FULL int8 range — not only ggml's
quantizer domain `[-127,127]`. (The first run, decoding in i8, FAILED exactly the
`q8 = -128` cases; the i16-domain fix made it 0-failure.)

## Byte-exact (ssh rvv, VLEN=128, rv64gcv) — ssh_rvv_byte_exact_stdout.txt

Oracle = `_generic` (= the real board kernel). We compare against BOTH a faithful
scalar transcription of `_generic` (quants.c:123-171) AND an INDEPENDENT RVV
transcription (a DISTINCT sign-plane expression: a +1/-1 i8 sign vector built by
vmerge under the same mask, then `vwmul` against the whole 32-lane block — NOT the
emitter's i16 vneg/vmerge), both BIT-FOR-BIT.

**`0 failures over 3650 cases` per mode × all 3 valid clang fp-contract modes
(off/on/fast, per-mode clean rebuild).** The strongest single signal is the
`_generic` cross-check delta = 0 even at `=fast` (an INDEPENDENT scalar reference;
the folds group into emitc.expressions -> one C FMA each, so even fast-math forms the
same fp). Cases: random (n ∈ {128,256,...,8192}, q8 over the FULL int8 range incl.
-128) + the edge crosses — all-0 bits (-> all -q8), all-1 bits (-> all +q8), marching
bytes × q8 saturation (+127 / ±128 / -127).

CRITICAL buffer-sizing fix (the landmine, same shape as nvfp4's): nb = n/128
super-blocks but the q8 stream is `n/32 = 4*nb` q8_0 blocks (sub-blocks k=1,2,3 read
q8 block 4*ib+1..+3). Under-sizing the q8 buffer to nb would make OUR kernel AND the
reference read the SAME OOB addresses -> a FALSE pass that never validates the k>=1
path. The harness sizes `4*nb`.

THREE negative controls (ssh_rvv_byte_exact_stdout.txt), all diverge:
- **wrong bit polarity** (set -> -q8 / clear -> +q8): real `24305.8` vs inverted
  `-24305.8` (exact negation) -> the sign decode is load-bearing.
- **wrong scale** (d0 × 2): real `30895.0` vs `61790.1` (exactly 2×) -> the d0 scale
  is load-bearing.
- **perturbed bit** (flip one bit byte) -> diverges (non-vacuous; the bits are consumed).

## raw()=0 + structured (structured_proof.txt)

raw() actual calls = 0; residual `tcrv_rvv.<op>` dialect ops = 0; residual
unrealized_conversion_cast = 0. All 245 "tcrv_rvv" mentions are I4-mirror provenance
comments. The ONLY sanctioned opaque scalar pieces are the FIVE fp16 reads (d0 + four
d1_k), the SAME `(float)*(const _Float16 *)` node the q4_0 sibling emits — q1_0 uses
NO libm (no math.h include needed). The binary decode is fully structured: NO codebook
gather (no vrgather), NO nibble unpack (no vand 0x0F / vsrl 0x04 on the weight), NO
offset-binary -8 bias (no vxor 0x88).

## Lit + reds + clean rebuild

Full CLEAN rebuild (`ninja -t clean && ninja`, 240 targets from scratch) GREEN. Lit
suite `check-tianchenrv`: **672 total, 669 pass, exactly the 3 documented pre-existing
reds** (`Scripts/rvv-generated-bundle-abi-e2e-{explicit,pre-realized}-computed-masked-
strided-input-widening-dot-reduce-add-dry-run`, `…-self-test` — unrelated to q1_0,
Scripts widening-dot-reduce). TWO NEW tests added, both PASS:
`test/Conversion/RVV/rvv-to-emitc-q1-0-q8-0-block-dot.mlir` (structured emission: the
kmask decl, the vmv_v_x_u8/vand/vmsne sign plane, the WIDEN-FIRST i16 vneg/vmerge, the
vwredsum, the two-level fold; CHECK-NOT vxor so it is NOT a q4_0 misroute) and
`test/Dialect/RVV/q1-0-q8-0-block-dot-dataflow.mlir` (9 cases: accept ggml ABI + 8
fail-closed rejections incl. wrong scale_model / wrong qk / wrong strides / wrong q8
span / non-m1 anchor / wrong operand C type). All siblings byte-identical (additive —
new op, new verifier branch, new emitter, new dispatch branch + recognizer; nvfp4 and
iq2_xxs regenerate BYTE-IDENTICAL after the q1_0 addition).

## Files

- `include/TianChenRV/Dialect/RVV/IR/RVVOps.td` — `GgmlBlockDotQ10Q80Op` def.
- `lib/Dialect/RVV/IR/RVVDialectWideningOps.cpp` — `GgmlBlockDotQ10Q80Op::verify()`.
- `lib/Conversion/RVV/RVVToEmitC.cpp` — `isQ1_0Q8_0BlockDotBody` recognizer, the
  dispatch branch, and `emitQ1_0Q8_0BlockDot` (the structured emitter).
- `test/Conversion/RVV/rvv-to-emitc-q1-0-q8-0-block-dot.mlir` (new).
- `test/Dialect/RVV/q1-0-q8-0-block-dot-dataflow.mlir` (new).
- artifacts/inc41-q1_0/: q1_0_input.mlir, q1_0_lowered.mlir, q1_0_kernel.cpp,
  inc41_validate.cpp, ssh_rvv_byte_exact_stdout.txt, structured_proof.txt, this file.

## Coverage: 22 ggml dot kernels; remaining (item 6)

| class | kernels | mechanism |
|---|---|---|
| block-quant linear | q4_0, q4_1, q5_0, q5_1, q8_0 | nibble unpack × i8 dot, per-block fp16 scale(+min) |
| K-quant super-block | q2_K, q3_K, q4_K, q5_K, q6_K | super-block ql/qh unpack + per-sub-block scale/min |
| CODEBOOK / FP4 | iq4_nl, iq4_xs, mxfp4, nvfp4 | non-linear int8 table / e2m1 vrgather gather |
| GRID / ternary | iq2_xxs, iq2_xs, iq2_s, iq3_xxs, iq3_s, iq1_s, iq1_m | packed grid lookup + sign plane / delta |
| **BINARY** | **q1_0** | **per-bit ±1 sign decode (kmask/vmsne/vmerge, the magnitude is q8)** |

Remaining ggml dot kernels (the last 2): **tq1_0, tq2_0** (the ternary TriLM/BitNet
super-block tail — a separate uncommon rung). q1_0 closes one of the last 3.
