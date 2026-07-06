# silicon-validation-batch-2 — NOTES (iq3_xxs + iq2_xxs, debt 2 -> 0)

Closes the two remaining flip silicon-validation debts left after batch-1: the
`iq3_xxs` (C_construct 18->19, commit 8e9b54a9) and `iq2_xxs` (C_construct 19->20,
commit 342ada90) constructed super-block-grid vec_dots — promoting them from
emit-golden to **silicon_validated** on real RVV hardware (`ssh rvv`, VLEN128).
Session 2026-07-06 (D-board return, step 4).

## What "silicon_validated" means here (identical discipline to batch-1)
The kernel-under-test is the REAL exported C from the front-door construction path:
    tcrv-opt <fixture> --tcrv-rvv-materialize-<fmt>-q8-k-block-dot-source-front-door \
             --tcrv-rvv-lower-to-emitc | mlir-translate --mlir-to-cpp
compiled and run on the board (clang-17, `-march=rv64gcv_zfh_zvfh...`, `-ffp-contract=off`).
Both kernels lower from the SAME typed region op `tcrv_rvv.typed_super_block_block_dot_loop_body`.

The oracle = ggml's OWN generic scalar `ggml_vec_dot_<fmt>_q8_K_generic`, recompiled
here from the board's `ggml-cpu/quants.c` at `-ffp-contract=off` so its `sumf` reduction
stays scalar / left-associative / non-FMA. IDENTICAL raw block bytes are fed to BOTH
sides (no quantizer in the loop): random `d` (finite f16) + random `qs` bytes are valid
for these formats — the qs bytes index the 256-entry grid tables and the 7-bit sign
fields index `ksigns_iq2xs[128]`, all in range.

## Results — VERDICT = BIT_EXACT (both formats, all seeds)
| format        | class                                             | fold          | verdict   | worst ULP | N                     |
|---------------|---------------------------------------------------|---------------|-----------|-----------|-----------------------|
| iq3_xxs_q8_K  | super-block grid (uint32 grid[256] + kmask signs) | scalar (+/*)  | BIT_EXACT | 0         | 256/256 x3 seeds      |
| iq2_xxs_q8_K  | super-block grid (int64 grid[256], vluxei16)      | scalar (+/*)  | BIT_EXACT | 0         | 256/256 x3 seeds      |

Both constructed kernels accumulate the block sums with **scalar `+`/`*` (no `vfmacc`)**,
exactly like batch-1's iq1_s/iq1_m/iq4_nl scalar-fold decoders — so they match the no-FMA
scalar oracle to **ULP=0** (predicted from the exported C: 0 `vfmacc`/`vfwmul`, confirmed on
silicon). The decode grid gather is genuinely vectorized (`vluxei16` over the grid table),
so this is not a scalar re-implementation: the vector decode + scalar reduction reproduce
the reference bit-for-bit. **Silicon debt 2 -> 0.**

## Falsifier / negative control
Same comparator (ordered-int ULP) and same build harness produce non-zero ULP for an
FMA-fold kernel (the q4_0 repack GEVM/GEMM in batch-1 / the GEMM cell: 400-1284 ULP), so
ULP=0 here is a real discriminating pass, not a computation compared to itself.

## Honest scope
ISOLATED-kernel bit-level silicon validation (vs a pinned oracle). A DIFFERENT statement
from any e2e or perf claim (these two IQ formats are decode vec_dots, not wired into the
q4_0 e2e trees). Reproduce: `tools/e2e-harness/silicon-validation-batch-2/run_silicon_batch2.sh`.
Fingerprints in `target_profile.txt`.
