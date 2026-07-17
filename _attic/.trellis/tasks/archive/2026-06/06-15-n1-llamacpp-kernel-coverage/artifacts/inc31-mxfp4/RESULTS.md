# INC-31 — mxfp4 x q8_0 byte-exact: the FP4 structural class (the LAST one)

The 12th ggml dot kernel: `ggml_vec_dot_mxfp4_q8_0`. mxfp4 opens — and CLOSES — the
**FP4 (e2m1 + E8M0 shared exponent)** structural quantization class. With it, EVERY common
ggml dot-kernel structural class is now covered: block-quant linear (q4_0..q5_1) +
K-quant super-block (q2_K..q6_K) + CODEBOOK (iq4_nl) + **FP4 (mxfp4)**. The only remaining
tail is the deep niche IQ super-block codebooks (iq1/iq2/iq3 ternary/sub-block).

## What mxfp4 is (ggml-common.h + ggml-impl.h, primary source)

- `block_mxfp4 = { uint8_t e; uint8_t qs[16] }` (QK_MXFP4 = 32, **sizeof 17**). A SINGLE
  8-bit **E8M0** shared exponent `e` at byte +0, then 16 packed **FP4 (e2m1)** nibble bytes
  at +1 (low nibble of byte j pairs with q8[j], high nibble with q8[j+16]).
- The 16 FP4 values are the FIXED `kvalues_mxfp4[16] = {0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,
  -8,-12}` int8 codebook (= **2 × E2M1_float**, i.e. the doubled e2m1 set {0,±0.5,±1,±1.5,
  ±2,±3,±4,±6}). The nibble INDEXES this table (REUSES iq4_nl's vrgather gather mechanism).
- The block scale is `GGML_E8M0_TO_FP32_HALF(e) = 2^(e-128)` (the **HALF** form, matching
  the doubled codebook so the dot stays an exact INTEGER dot). ggml's exact bit
  construction: `bits = (e < 2) ? (0x00200000u << e) : ((e-1) << 23)`, reinterpreted as
  float — the e<2 branch yields denormal scales 2^-128 / 2^-127, the e>=2 branch the
  normalized 2^(e-128).
- Dot (`_generic`, quants.c:247-276): `d = GGML_CPU_FP16_TO_FP32(y.d) * E8M0_HALF(x.e)`;
  `sumf += d * (sumi1 + sumi2)` where `sumi = Σ q8_i · kvalues_mxfp4[nibble_i]`. The RVV
  `vl128` path (arch/riscv/quants.c:6470) folds `E8M0_HALF(e) * d_y * acc`; fp multiply
  commutes so both are the same bits.

## The implementation (STRUCTURED, raw()=0; REUSES iq4_nl)

- **New op** `tcrv_rvv.mxfp4_q8_0_block_dot` (RVVOps.td) + `verify()`
  (RVVDialectWideningOps.cpp), fail-closed (I7) on kind / scale_model / strides / the dual
  quant offsets / codebook size (16) / non-m1 anchor / operand C types / shape knobs.
  Carries the FP4 codebook as a `DenseI8ArrayAttr:$codebook` + the E8M0 `scale_model`
  string + the dual `weight_quant_byte_offset` (1) / `activation_quant_byte_offset` (2)
  (the mxfp4-specific split vs iq4_nl's shared +2).
- **Lowering** `emitMXFP4Q8_0BlockDot` (RVVToEmitC.cpp) REUSES iq4_nl's codebook-gather
  integer core (`emitOffsetBinaryProductFromDecodedValue` + vrgather) and block-loop /
  unroll / tail / fold / store scaffolding verbatim. The TWO new structured pieces:
  - the dual block offsets (weight nibbles at +1, q8 at +2);
  - the **E8M0 scale** as structured emitc nodes (cmp / bitwise_and mask / bitwise_left_shift
    / sub / conditional + `*(const float *)&bits` pointer pun) — NOT scalbnf/ldexpf, so
    byte-identical on the denormal scales. The `(e & 0x1F)` mask keeps the denormal shift
    well-defined in C and is byte-exact (consumed only when e<2 where it is a no-op).
  Only ONE fp16 read survives (q8_0 `d_y`); the weight fp16 read is replaced.
- The codebook GATHER pins the m1 anchor (VLMAX >= 16 to index all 16 entries) — an N1
  Zvl128b-gated capability fact (the verifier enforces m1).

## Byte-exact (ssh rvv, VLEN=128, rv64gcv) — ssh_rvv_byte_exact_stdout.txt

The reference is a **faithful transcription** of ggml's RVV mxfp4 vl128 method (the exact
vand/vsrl nibble split → vrgather through `kvalues_mxfp4` → vwmul/vwmacc → vwredsum chain
from arch/riscv/quants.c:6470) AND an independent scalar transcription of `_generic`
(quants.c:247-276), not the linked ggml symbol (the inc30 precedent). The two references
agree bit-for-bit, and the THREE load-bearing controls below close the shared-bug gap.

`0 failures over 3520 cases` per mode × **all 4 fp-contract modes** (off/on/default/fast,
14080 total) vs BOTH references. The strongest single signal is the **`_generic`
cross-check delta = 0 even at =fast** (an INDEPENDENT scalar reference; the fold groups
into one emitc.expression → one C FMA, so even fast-math forms the same fp). Cases:
random + edge (the marching-nibble pattern exercising ALL 16 codebook entries × the FULL
E8M0 exponent range e∈{0,1,127,254,255} incl. the denormal min/max × q8 ±127/±128) at
n ∈ {32,64,...,8192}.

THREE negative controls (ssh_rvv_byte_exact_stdout.txt), all diverge:
- **wrong FP4 codebook** (the linear nibble-8 table): real `-11980.168` vs linear
  `8059.04` → the codebook is load-bearing.
- **wrong E8M0 scale** (the FULL 2^(e-127) instead of the HALF 2^(e-128)): half
  `-38723.09` vs full `-77446.19` (exactly 2×) → the HALF convention (the crux that the
  task prose's loose "2^(e-127)" gets WRONG) is load-bearing.
- **perturbed nibble** → diverges (non-vacuous).

COMPILER-KNOB control (ssh_rvv_codebook_knob_stdout.txt): two compiler-emitted kernels
differing ONLY in the MLIR `codebook=` attr (real FP4 vs linear) → the HW result flips
(real `5118.45` vs linear `1285.88`), each matching its own table's math → the codebook
attr is the live, load-bearing knob through compile→hardware.

## raw()=0 + structured (structured_proof.txt)

raw() actual calls = 0; residual `tcrv_rvv.<op>` dialect ops = 0 (all 30 "tcrv_rvv"
mentions are I4-mirror provenance comments); residual unrealized_conversion_cast = 0.
31 verbatim = 30 provenance comments + 1 sanctioned structured-const codebook decl; ZERO
unstructured C blobs. The E8M0 scale is fully structured emitc; the ONE sanctioned opaque
piece is the single q8_0 fp16 read.

## Lit + reds

Full clean rebuild green. `check-tianchenrv`: **652 total, 649 pass, exactly 3 documented
pre-existing reds** (`Scripts/rvv-generated-bundle-abi-e2e-{explicit,pre-realized}-
computed-masked-strided-input-widening-dot-reduce-add-dry-run`, `…-self-test` — unrelated
to mxfp4, last touched by b99f31a1). Two NEW tests added (the +2 over inc30's 650), both
PASS: `test/Conversion/RVV/rvv-to-emitc-mxfp4-q8-0-block-dot.mlir` (structured emission +
the E8M0 nodes + the gather + CHECK-NOT offset-binary + the SINGLE fp16 read) and
`test/Dialect/RVV/mxfp4-q8-0-block-dot-dataflow.mlir` (8 cases: accept ggml ABI, accept
mb4/elided, + 6 fail-closed rejections incl. the wrong scale_model and wrong weight quant
offset). All siblings byte-identical (additive — new op, new verifier branch, new emitter,
new dispatch branch; iq4_nl + every other kernel byte-untouched).

## Every structural class now covered

| class | kernels | mechanism |
|---|---|---|
| block-quant linear | q4_0, q4_1, q5_0, q5_1, q8_0 | nibble unpack (offset-binary / unsigned / 5-bit) × i8 dot, per-block fp16 scale(+min) |
| K-quant super-block | q2_K, q3_K, q4_K, q5_K, q6_K | super-block ql/qh unpack + per-sub-block scale/min bit-dance |
| CODEBOOK | iq4_nl | 16-entry non-linear int8 table vrgather gather |
| **FP4** | **mxfp4** | **FP4 e2m1 codebook gather + E8M0 2^(e-128) shared-exponent scale** |

Remaining: the deep niche IQ super-block codebooks (iq1/iq2/iq3 — ternary/sub-block
grids), a separate later rung.
