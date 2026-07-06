# Constructed q4_0 REPACK GEMM (PREFILL) — silicon validation (rvv VLEN128, 2026-07-06)

Closes the **GEMM silicon-validation debt** left open when silicon-validation-batch-1
validated the decode **GEVM** (`q4_0_repack`) but not the prefill **GEMM** (M>1). The
GEMM is the finale-constructed sibling (be66c917): the abstract `tcrv_rvv.quant_contraction`
lowered by `--tcrv-rvv-lower-quant-contraction` into the typed `tcrv_rvv.typed_repack_gemm_loop_body`
region when `m_regime == "prefill"` (the monolithic hand op is RETIRED).

## What was validated
The exported kernel (front-door → `mlir-translate --mlir-to-cpp`, VERBATIM) was compiled
on-board (clang-17, `-march=rv64gcv_zfh_zvfh...`, `-ffp-contract=off`) and each output
`out[r][c]` compared against:
- **(A)** the pinned **no-FMA left-assoc ggml scalar oracle** (`ggml_vec_dot_q4_0_q8_0_generic`
  recompiled `-ffp-contract=off`) — the same oracle class batch-1 used;
- **(B)** an **f64 high-precision reference** (board-fp16-independent) for accuracy.

Driver `gemm_verify_driver.c` builds the two repack layouts from raw q4_0/q8_0 blocks
(weight `block_q4_0x16`/288B, activation `block_q8_0x4`/136B — both **derived from the
exported kernel body**, see the header comment), runs the kernel, and scores ULP.

## Result — VERDICT = FMA_FOLD_BOUNDED_ULP (all 4 shapes)
The GEMM uses a **vector FMA fold** (`vfmacc`), so it is NOT bit-exact to the no-FMA
left-assoc scalar oracle — a **bounded** ULP divergence is EXPECTED (exactly as the GEVM
`q4_0_repack` in batch-1). What matters is accuracy vs truth:

| shape (K,nr,nc) | worst_rel vs f64 | mean-ULP-vs-f64 kernel | mean-ULP-vs-f64 ggml-scalar |
|---|---|---|---|
| 4096, 4, 128  | 7.83e-05 | 6.625 | 5.767 |
| 2048, 8, 64   | 3.89e-05 | 3.935 | 3.767 |
| 4096, 16, 256 | 1.13e-03 | **9.611** | 11.892 |
| 128, 4, 16    | 4.75e-07 | **0.406** | 0.496 |

The kernel is **as-accurate-or-better than ggml's own scalar reference** vs the f64 truth
in 2/4 shapes and same-order in the other two. At the single worst point (shape 3) the
kernel is CLOSER to f64 than the ggml scalar (|k−f64|=0.038 < |ggml−f64|=0.053). The
larger `worst_rel` at K=4096,nc=256 is dominated by fp16-scale quantization noise over the
long accumulation, not a kernel defect.

**Silicon debt: 1 → 0.** The constructed GEMM is **FMA-bounded-correct** — the honest
verdict (not ULP=0), matching the GEVM.

## Honest scope
This is the ISOLATED-kernel bit-level silicon validation (vs a pinned oracle). It is a
DIFFERENT statement from the e2e greedy-token gate (which lives in the e2e cell) and is
NOT a perf claim. Reproduce: `run_silicon_gemm.sh`.
