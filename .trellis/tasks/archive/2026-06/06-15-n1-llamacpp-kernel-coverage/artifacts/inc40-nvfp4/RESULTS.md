# INC-40 — nvfp4 x q8_0 byte-exact: the SECOND FP4-class kernel (NVIDIA FP4)

The 13th ggml dot kernel: `ggml_vec_dot_nvfp4_q8_0`. nvfp4 is NVIDIA's FP4 — the
SECOND kernel in the FP4 (e2m1) structural class mxfp4 opened. It REUSES mxfp4's e2m1
`kvalues_mxfp4` codebook gather + the asymmetric signed-widening integer core verbatim;
the genuinely-new fact is a DIFFERENT per-block scale (an unsigned UE4M3 fp8 scale, NOT
mxfp4's E8M0 shared exponent) carried on a SUPER-block that groups four such sub-blocks.
This is COVERAGE toward literal 100% of the ggml dot kernels.

## What nvfp4 is (ggml-common.h + ggml-impl.h + quants.c, primary source)

The task prd loosely assumed `block_nvfp4 = { uint8_t e; uint8_t qs[16] }` QK=32 with a
SIGNED E4M3 (1-4-3) scale and `_q8_1`. The REAL source differs in three ways — READ +
adapted:

- `block_nvfp4 = { uint8_t d[4]; uint8_t qs[32] }` (QK_NVFP4 = 64, QK_NVFP4_SUB = 16,
  **sizeof 36**). FOUR UE4M3 sub-block scales `d[0..3]` at bytes +0..3, then 32 packed
  **FP4 (e2m1)** nibble bytes at +4 (8 per 16-element sub-block, sub-block s at +4+s*8).
- The activation is `block_q8_0` (NOT q8_1). ONE nvfp4 super-block (64 elems) spans
  **TWO** q8_0 blocks: sub-block s reads q8 block `2*ib + s/2` at half-offset `(s%2)*16`.
- The 16 FP4 values are the SAME `kvalues_mxfp4[16]` codebook nvfp4 REUSES from mxfp4
  (`= 2*E2M1`). The nibble INDEXES this table (REUSES mxfp4's vrgather gather mechanism).
- The block scale is `ggml_ue4m3_to_fp32(e)` — **UE4M3 is UNSIGNED** (4 exp bits bias 7,
  3 mantissa bits, NO sign bit), NOT a signed E4M3. `e==0 || e==0x7F -> 0.0f` (two
  specials); else `exp=(e>>3)&0xF, man=e&7, raw=(exp==0)?ldexpf(man,-9):ldexpf(1+man/8,
  exp-7)`, result `= raw * 0.5f` (the HALF form matching the doubled codebook, so the dot
  stays an exact INTEGER dot).
- Dot (`_generic`, quants.c:278-312): per sub-block `d = ue4m3(x.d[s])`, `dy =
  FP16_TO_FP32(y[2ib+s/2].d)`; `sumf += dy * d * (sumi_lo + sumi_hi)` where
  `sumi = Σ q8 · kvalues_mxfp4[nibble]`.
- **There is NO arch/riscv nvfp4 kernel** — `arch-fallback.h` aliases
  `ggml_vec_dot_nvfp4_q8_0` to `..._generic`, so `_generic` **IS** the real board kernel
  AND the oracle (we do NOT claim a separate hand-written-RVV comparison).

## The implementation (STRUCTURED, raw()=0; REUSES mxfp4's codebook)

- **New op** `tcrv_rvv.nvfp4_q8_0_block_dot` (RVVOps.td) + `verify()`
  (RVVDialectWideningOps.cpp), fail-closed (I7) on kind / scale_model / qk / qk_sub /
  strides / quant offsets / codebook size (16) / non-m1 anchor / operand C types. Carries
  the FP4 codebook as a `DenseI8ArrayAttr:$codebook` (REUSED from mxfp4) + the UE4M3
  `scale_model` string + the super-block facts (qk 64, qk_sub 16, weight stride 36,
  weight quant offset +4, the per-sub-block q8 high-half offset +8).
- **Lowering** `emitNVFP4Q8_0BlockDot` (RVVToEmitC.cpp) REUSES mxfp4's codebook-gather
  integer core (`emitOffsetBinaryProductFromDecodedValue` + vrgather + vwmul/vwmacc +
  vwredsum). The THREE new structured pieces:
  - the SUPER-block double structure (outer loop over n/64; four sub-blocks unrolled);
  - the per-sub-block activation addressing (q8 block `2*ib+s/2`, half `(s%2)*16`, the
    8-lane strip with the high nibble at +8);
  - the **UE4M3 scale** as structured emitc nodes (exp/man split, the two `ldexpf`
    branches, the exp==0 conditional, the `*0.5f`, then the two specials via two cmp +
    `logical_or` + conditional) — ggml's EXACT ldexpf arithmetic (no E8M0 bit dance:
    UE4M3 scales are all normal fp32, no FTZ risk, so byte-exact by construction).

## Byte-exact (ssh rvv, VLEN=128, rv64gcv) — ssh_rvv_byte_exact_stdout.txt

The oracle is `_generic` (= the real board kernel). We compare against BOTH a faithful
scalar transcription of `_generic` (quants.c:278-312 + ggml-impl.h ggml_ue4m3_to_fp32)
AND an independent RVV transcription (the same vand/vsrl/vrgather/vwmul/vwmacc/vwredsum
codebook core), both BIT-FOR-BIT.

`0 failures over 3468 cases` per mode × **all 3 valid clang fp-contract modes**
(off/on/fast, per-mode clean rebuild). The strongest single signal is the **`_generic`
cross-check delta = 0 even at `=fast`** (an INDEPENDENT scalar reference; the fold groups
into one emitc.expression → one C FMA, so even fast-math forms the same fp). Cases:
random (n ∈ {64,128,...,8192}) + the marching-nibble pattern (ALL 16 codebook entries) ×
the **FULL 256-code UE4M3 scale range** (the two specials 0x00/0x7F → 0, the exp==0
denormals 0x01-0x07, the large normals 0x78-0x7E) × q8 ±127/±128.

CRITICAL buffer-sizing fix (the landmine the inc31 harness shape would have hidden): `nb
= n/64` super-blocks but the q8 stream is `n/32 = 2*nb` q8_0 blocks (sub-blocks s=2,3
read q8 block `2*ib+1`). Under-sizing the q8 buffer to `nb` would make OUR kernel AND the
reference read the SAME OOB addresses → a FALSE pass that never validates the s=2,3 path.
The harness sizes `2*nb`; max q8 index = `2*nb-1` is in-bounds.

THREE negative controls (ssh_rvv_byte_exact_stdout.txt), all diverge:
- **wrong FP4 codebook** (the linear nibble-8 table): real `-41576.63` vs linear
  `-7113.49` → the codebook is load-bearing.
- **wrong UE4M3 scale** (the FULL decode, drop the `*0.5f`): half `-4118.57` vs full
  `-8237.15` (exactly 2×) → the HALF convention is load-bearing.
- **perturbed nibble** → diverges (non-vacuous; the FP4 nibbles are consumed).

## raw()=0 + structured (structured_proof.txt)

raw() actual calls = 0; residual `tcrv_rvv.<op>` dialect ops = 0 (all "tcrv_rvv" mentions
are I4-mirror provenance comments); residual unrealized_conversion_cast = 0. 106 verbatim
= 105 provenance comments + 1 sanctioned structured-const codebook decl; ZERO unstructured
C blobs. The UE4M3 scale is fully structured emitc. TWO sanctioned opaque scalar pieces,
both structured `emitc.call_opaque` to external functions (NOT raw() escapes): the q8_0
fp16 read + `ldexpf` (ggml's own ue4m3 decode calls ldexpf, so a real drop-in already
links libm; the kernel TU is compiled with `-include cmath -lm`, the emission preamble
left byte-pure so all siblings stay byte-identical).

## Lit + reds

Full lit suite (`check-tianchenrv`): **670 total, 667 pass, exactly 3 documented
pre-existing reds** (`Scripts/rvv-generated-bundle-abi-e2e-{explicit,pre-realized}-
computed-masked-strided-input-widening-dot-reduce-add-dry-run`, `…-self-test` — unrelated
to nvfp4, Scripts widening-dot-reduce). TWO NEW tests added, both PASS:
`test/Conversion/RVV/rvv-to-emitc-nvfp4-q8-0-block-dot.mlir` (structured emission + the
ldexpf branches + logical_or specials + the exp/man split + `CHECK-NOT 0x00200000` so it
is NOT a mxfp4-E8M0 misroute) and `test/Dialect/RVV/nvfp4-q8-0-block-dot-dataflow.mlir`
(9 cases: accept ggml ABI + 8 fail-closed rejections incl. the wrong scale_model / wrong
stride / wrong quant offset / wrong high offset). All siblings byte-identical (additive —
new op, new verifier branch, new emitter, new dispatch branch; the mxfp4 kernel
regenerates BYTE-IDENTICAL after the nvfp4 addition, +892 lines across 3 source files).

## The E4M3 scale decode (the new piece) + how byte-exact

The "E4M3" of the task prd is in reality **UE4M3** (unsigned). The byte-exactness comes
from replicating ggml's `ggml_ue4m3_to_fp32` arithmetic NODE-for-node in structured emitc:
the same `(e>>3)&0xF` / `e&7` split, the same `ldexpf` libm calls (the board's libm, the
same the reference uses), the same exp==0 branch, the same `*0.5f`, and the same two
specials. Because every UE4M3 scale is a comfortably-normal fp32 (~[2^-10, 224], no FTZ;
max non-special normal 0x7E = 224, 0x7F's would-be 240 is overridden to 0 by the special),
ggml's exact arithmetic needs no bit-construction dodge (the reason mxfp4's E8M0 used one),
and the result is byte-identical to `_generic` by construction — verified over all 256
scale codes on ssh rvv with delta 0.

## Coverage: 13 ggml dot kernels; remaining

| class | kernels | mechanism |
|---|---|---|
| block-quant linear | q4_0, q4_1, q5_0, q5_1, q8_0 | nibble unpack × i8 dot, per-block fp16 scale(+min) |
| K-quant super-block | q2_K, q3_K, q4_K, q5_K, q6_K | super-block ql/qh unpack + per-sub-block scale/min |
| CODEBOOK | iq4_nl | 16-entry non-linear int8 table vrgather gather |
| **FP4** | **mxfp4, nvfp4** | **e2m1 codebook gather + (E8M0 / UE4M3) per-block scale** |

Remaining ggml dot kernels (the last 3): **q1_0, tq1_0, tq2_0** (the 1-bit / ternary
super-block tail — a separate uncommon rung). nvfp4 closes one of the last 4.
