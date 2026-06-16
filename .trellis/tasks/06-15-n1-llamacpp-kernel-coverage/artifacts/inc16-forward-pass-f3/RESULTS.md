# INC-16 F3 — ggml rms_norm (first forward-pass REDUCTION op) — RESULTS

Date: 2026-06-16. Board: `ssh rvv` (riscv64, rv64gcv, VLEN=128, `__riscv_v_min_vlen
128`, `__riscv_zvl128b`, `__riscv_v_intrinsic 12000`, clang 18.1.3).

## What this is

The FIRST forward-pass REDUCTION op our compiler covers: a STRUCTURED, byte-exact,
compiler-emitted drop-in for ggml's `ggml_compute_forward_rms_norm_f32` (non-fused,
no weight; `ops.cpp:3758-3817`). It is F3 in `research/forward-pass-scoping.md` and
the structurally-new rung after F1 (`ggml_vec_scale_f32`, a bare elementwise scale):
rms_norm adds a CROSS-LANE fold + a scalar rsqrt before the elementwise normalize.

ggml's exact math (one row of `ne00` elements):
```
ggml_float sum = 0.0;                        // ggml_float = double (vec.h:15)
for (i = 0; i < ne00; i++) sum += (ggml_float)(x[i] * x[i]);  // SCALAR ascending
const float mean  = sum/ne00;                // divide in double, cast to f32
const float scale = 1.0f/sqrtf(mean + eps);  // f32 add / sqrtf / reciprocal
memcpy(y, x, ne00*4); ggml_vec_scale_f32(ne00, y, scale);    // y[i] = x[i]*scale
```
Confirmed from source: F3 is the non-fused path, so it does NOT include the per-row
weight multiply (that is the `GGML_RMS_NORM_FUSE_OP_MUL` branch, lines 3805-3813);
it operates on ONE row (the i01/i02/i03 loops are ggml's framework, not the kernel).

## The op + lowering — HOW ggml's double accumulation is matched

- New typed op `tcrv_rvv.ggml_rms_norm_f32` (`include/.../RVVOps.td`): ABI operands
  (read-only `const float *` x, written `float *` y, runtime `float` eps, runtime
  element count `ne00`, the active VL token), fail-closed verifier (I7) in
  `lib/Dialect/RVV/IR/RVVDialectWideningOps.cpp` (bounds `kind`; bounds the optional
  `strip_lmul` resource knob to m1|m2|m4|m8; pins x to `const float *`, y to
  `float *`, eps to `float`). The knob is `strip_lmul` (not the forbidden
  with_vl/setvl `lmul` spelling), so no SEW/LMUL/policy dataflow-parameter attr at
  the I5 boundary — the forbidden `lmul` name is rejected fail-closed.
- Structured lowering (`lib/Conversion/RVV/RVVToEmitC.cpp` `emitGgmlRmsNormF32` +
  recognizer `isGgmlRmsNormF32Body` + op-identity dispatch). THREE structured
  pieces, every value an emitc node (raw() = 0):
  1. **The scalar-double reduction (the byte-exactness crux).** A `double sum`
     `emitc.variable` lvalue (emitc.for has NO iter_args, so the loop-carried
     accumulator is variable+assign, NOT loop-carried SSA — the genuinely new
     machinery vs F1, mirrored from how the block-dot kernels carry the fp32 `*s`
     accumulator). A SCALAR `for (i=0;i<ne00;++i)` ascending fold (step is the
     literal 1, NOT a vlmax — this loop is NOT vectorized). The body is ONE
     `emitc.expression` rendering `sum = sum + (double)(x[i]*x[i])`: the f32 product
     `mul` (one f32 rounding), then a `cast` to double, then a double `add`. The
     cast SITS BETWEEN the f32 multiply and the f64 add — this is the load-bearing
     detail AND an FMA barrier (multiply and add are different types, so
     `-ffp-contract` cannot fuse them).
  2. **The scalar mean + rsqrt.** `mean = (float)(sum / (double)ne00)` (divide in
     double, cast to f32 AFTER). `scale = 1.0f / sqrtf(mean + eps)` (f32 add, the
     scalar libm `sqrtf` as a `call_opaque` — a true IEEE correctly-rounded sqrt,
     NOT a hardware fast-rsqrt7, then an f32 reciprocal divide). `sqrtf`/`/` are
     IEEE correctly-rounded, so (unlike rope's `cosf`/`sinf`) there is no
     libm-dependence — bit-identical whether clang emits `fsqrt.s` or a call.
  3. **The vectorized normalize strip.** `y[i] = x[i] * scale`: a
     `vsetvl_e32m8(ne00-i)` strip with `vle32` / `vfmul_vf` (scalar broadcast) /
     `vse32` (the F1 machinery, two-buffer x in / y out). Byte-exact at any LMUL (a
     bare per-lane multiply — no FMA, no reduction).
  A vectorized `vfredusum` would NOT be byte-exact (it folds in f32 with a tree
  order); the reduction is therefore kept scalar-double, replicating ggml exactly.
- The rms_norm body adds `<math.h>` to its TU (for `sqrtf`); this is CONDITIONAL on
  `isGgmlRmsNormF32Body`, so every other kernel keeps the original 3-header list
  byte-identical (verified: F1 `scale_kernel.cpp` re-render == committed artifact).
- Lit test `test/Conversion/RVV/rvv-to-emitc-ggml-rms-norm-f32.mlir` (PASS):
  asserts the structured node sequence (variable / scalar for-loop / mul→cast→add
  cast chain / double div + f32 cast / sqrtf / vle32·vfmul_vf·vse32) and
  `CHECK-NOT: emitc.verbatim {{.*}}__riscv` (no raw blob).

## Byte-exact HW evidence (ssh rvv) — `board_validation_stdout.txt`

Harness `inc16_validate.cpp` builds the UNMODIFIED compiler-emitted `rms_norm_kernel.cpp`
(from `tcrv-opt --tcrv-rvv-lower-to-emitc | mlir-translate --mlir-to-cpp`) against a
verbatim transcription of ggml's rms_norm row body (rms_norm has NO RVV path in ggml
— its Sum is a scalar ggml_float loop — so a faithful transcription is equivalent,
exactly as F1 transcribed ggml's RVV scale path), and `memcmp`s the float bits over:
row sizes {1,7,31,32,33,63,256,255,257,1023,1024,1025,4096,4095,11008,11007}
(incl. the real llama-2-7b dims 256/1024/4096/11008) × 5 value distributions
(normal / large / small / mixed-magnitude / raw-bits incl denormal/inf/nan) × 5 eps
{1e-5,1e-6,0,1e-3,1e-1} × 12 seeds, PLUS a named edge-case row
(inf/nan/denormal/±0/huge/FLT_MAX mixed) × all eps.

Result under -ffp-contract = on / off / fast (ALL THREE identical):

    INC-16 F3 ggml_rms_norm_f32: 4805/4805 bit-exact cases PASS
    NC1 wrong-product-type (large-magnitude): 10/10 correctly DIFFER
    NC2 wrong-accumulator-type (long mixed):  31/54 correctly DIFFER
                          (of which the DETERMINISTIC swamping subset: 6/6)
    RESULT: PASS (byte-exact vs ggml rms_norm; NC2 the scalar-double ACCUMULATION
            control discriminates -> ggml's METHOD is matched, not just tolerance)

**EXACT, not near-exact**, and identical across all three fp-contract modes — proving
the cast barrier between the f32 product and the f64 add blocks any FMA fusion (the
reduction method is reproduced bit-for-bit, not approximated). 4805/4805 bit-for-bit.

## The negative controls discriminate the CRUX *and validate the reference*

Because the reference is a transcription of ggml (not linked ggml), the obvious worry
is a SYMMETRIC bug: a mirrored mistake in both kernel and reference would pass a
byte-compare vacuously. The negative controls close that hole — they don't merely
show "the test discriminates", they PIN the two crux dimensions and rule out a
symmetric-transcription bug:

- **NC2 (the verdict gate) — wrong ACCUMULATOR type (f32 running sum).** Diverges on
  long mixed-magnitude rows (31/54 > majority; the DETERMINISTIC swamping subset is
  6/6, so the discrimination is constructed, not seed-luck). If both kernel AND
  reference had secretly used f32 accumulation, NC2 (which IS f32 accumulation) would
  catch 0/54. It catches a majority ⇒ the reference is genuinely double-accum, the
  kernel matches it everywhere (4805/4805) ⇒ the kernel is genuinely double-accum.
  This is what makes the transcription-as-reference choice rigorous: NC2 validates
  the reference is double-accum, not just that the kernel matches it. The
  deterministic subset is a constructed swamping row (x[0]=1e4 → square 1e8, ULP 8;
  tail x[i]=1.9 → square 3.61 < ULP/2, so every f32 tail add rounds away while the
  f64 sum climbs) — it MUST diverge for an f32 accumulator at every long size, and
  does (6/6).
- **NC1 (bonus rigor) — wrong PRODUCT type (`(double)x*(double)x`, no f32 round).**
  Diverges 10/10 in its targeted regime: one huge spike (5e19) among unit values in
  a long row, where the f32 product overflows to +inf (→ scale 0 → y all 0) but the
  double product stays finite (→ finite scale → y nonzero). (Honest note: an
  ALL-large row does NOT discriminate — there the double-product mean itself
  overflows the float cast to +inf, so both collapse to scale=0 identically, which
  is also exactly why our kernel matches ggml on all-large rows. NC1 therefore gates
  nothing; it is reported as targeted evidence of the product-type distinction.)

## The hardest part — and how it was resolved

The double accumulation byte-exactness. ggml folds Sum x[i]^2 in `ggml_float`
(=double), SCALAR, in strict ascending order — a vectorized `vfredusum` would fold
in f32 with a tree order and break bit-exactness. Resolved by (a) emitting the
reduction as STRUCTURED scalar emitc nodes (variable/load/mul/cast/add/assign) that
replicate ggml's scalar double fold, (b) placing the f32→double cast BETWEEN the f32
product and the double add — which is both the correct cast chain AND an FMA barrier
that makes the result fp-contract-invariant, and (c) the loop-carried accumulator as
an emitc.variable lvalue + emitc.assign (emitc.for has no iter_args) rather than a
raw string. Only the final normalize is vectorized (a bare vfmul_vf strip, byte-exact
at any LMUL). The 4805/4805 = result across on/off/fast is the proof.

## Additivity + reds

- raw() = 0 on `rms_norm_kernel.cpp` (every value an emitc node; provenance
  verbatims are comment lines only).
- Full lit suite (`ninja check-tianchenrv`): 625 total, 622 PASS, exactly the 3
  pre-existing DOCUMENTED reds (`rvv-generated-bundle-abi-e2e-self-test` + the two
  `computed_masked_strided_input_widening_dot_reduce_add` dry-runs). The count rose
  624→625: the new F3 lit test is the +1 and it PASSES. No new reds.
- Additive: F1 `ggml_vec_scale_f32` kernel C re-render is BYTE-IDENTICAL to the
  committed inc15 artifact (the conditional `math.h` does not leak into any other
  kernel); all quant-dot sibling lit tests (q4_0/q8_0/q4_1 block-dot, gemm-tile,
  deferred-wide, narrow/byte) PASS unchanged.

## What's left for the forward-pass family

- **F4** — activation quantizers `quantize_row_q8_0`/`q8_1`: reduction
  (`vfredmax`/`vfredusum`) + scale + f32→i16→i8 narrowing-convert. Closes the
  f32→quant-dot bridge. Inherits this op's reduction-emission discipline; note q8_1
  also accumulates a block sum in double.
- **F5** — `silu` + `soft_max`: the vectorized-transcendental tail
  (`ggml_v_expf_m2` minimax polynomial as a structured intrinsic chain). The
  HARDEST rung (a vectorized polynomial, data-dependent overflow fixup); soft_max
  also folds its `vfredusum` in double — F3 isolated exactly that reduction risk.
- **F6** — `rope`: f32 elementwise rotation (reuses F1/F2 lanes) + a verbatim scalar
  `cosf`/`sinf` angle cache. The composition rung; its transcendental is scalar
  libm, NOT a polynomial — a different byte-exactness axis (linked-libm-dependent).

## Files

- `include/TianChenRV/Dialect/RVV/IR/RVVOps.td` (new GgmlRmsNormF32Op)
- `lib/Dialect/RVV/IR/RVVDialectWideningOps.cpp` (new GgmlRmsNormF32Op::verify)
- `lib/Conversion/RVV/RVVToEmitC.cpp` (recognizer + dispatch + emitGgmlRmsNormF32 +
  the conditional `math.h` include)
- `test/Conversion/RVV/rvv-to-emitc-ggml-rms-norm-f32.mlir` (lit)
- `artifacts/inc16-forward-pass-f3/{rms_norm_kernel.cpp, inc16_validate.cpp,
  board_validation_stdout.txt, rvv-to-emitc-ggml-rms-norm-f32.mlir, this file}`
