# silicon-validation-batch-1 — NOTES

Discharge the silicon-validation debt (=4 constructed formats) by promoting the
front-door-CONSTRUCTED kernels from **emit-golden** to **silicon_validated** on
real RVV hardware (`ssh rvv`, VLEN128), vs the pinned **no-FMA left-assoc ggml
scalar oracle**. Session 2026-07-06.

## What "silicon_validated" means here
The kernel-under-test is the REAL exported C from the front-door construction path
(`tcrv-opt <front-door / lower-quant-contraction> --tcrv-rvv-lower-to-emitc |
mlir-translate --mlir-to-cpp`), compiled and run on the board. Not a test-authored
body. The oracle is ggml's OWN `ggml_vec_dot_<fmt>_generic`, recompiled here at
`-ffp-contract=off` so the reduction stays scalar / left-associative / non-FMA.

Identical raw block bytes are fed to BOTH sides (no quantizer in the loop), so the
test isolates the decode+dot arithmetic on the real vector unit vs the scalar
reference. fp16 conversion matches exactly (both use `_Float16` fcvt.s.h; board has
Zfh; ggml's `GGML_CPU_FP16_TO_FP32` is inline `_Float16`, no lookup table).

## Results (per format)
| format            | class                              | verdict              | worst ULP | N |
|-------------------|------------------------------------|----------------------|-----------|---|
| iq4_nl_q8_0       | flat block-dot (codebook gather)   | **BIT_EXACT**        | 0         | 256/256 x3 seeds |
| iq1_s_q8_K        | super-block (ternary grid + bsum)  | **BIT_EXACT**        | 0         | 256/256 x3 seeds |
| iq1_m_q8_K        | super-block (packed-scale grid)    | **BIT_EXACT**        | 0         | 256/256 x3 seeds |
| q4_0_q8_0_repack  | 16x1-repack GEVM (vector FMA)      | FMA_FOLD_BOUNDED_ULP | 400-1284  | 772-920/2048 |

## The q4_0 repack GEVM is honestly a different phase
The 3 flat/super-block block-dots accumulate a scalar `sumf` with explicit `+`/`*`
(non-FMA) — they match the no-FMA scalar oracle to **ULP=0**.

The q4_0 repack GEVM does NOT: it is a matrix-vector kernel that accumulates with
**vector FMA** (`__riscv_vfmacc`), pre-groups `dx*dy`, and reduces 16 rows in f32
lanes. That is a legitimately different (and, per the f64 reference, slightly MORE
accurate) fold — not a bug. So it is silicon_validated as "computes the correct
q4_0 GEVM to fp32 rounding, FMA fold", with `<1e-3` relative error vs no-FMA scalar
and mean-ULP-vs-f64 at or below ggml's own scalar reference. The tiny relative
error also confirms the repack-layout reconstruction (XOR-0x88 nibble convention,
288B/16-row block) is correct — a wrong layout would show order-1 error.

## Falsifier / negative control
Same comparator (ordered-int ULP) and same build harness produce ULP=0 256/256 for
the three scalar-fold decoders but non-zero ULP for the FMA repack — the comparator
provably discriminates (it is not comparing a computation to itself). The three
block-dots exercise three structurally distinct decoders, all landing ULP=0.

## Reproduce
`tools/e2e-harness/silicon-validation-batch-1/run_silicon_batch.sh` (harness relocated
2026-07-06 A2; kernels/ + results/ stay in this cell). HOST= ssh alias, default `rvv`.
Fingerprints in `target_profile.txt`.
