# T4b M2c — decisive re-emit + re-run: FRESH S6-tiled q4_K repack-GEMM through REAL ggml_mul_mat (ssh rvv)

Operational index only. Outcome: (b) — the ~5% NORM min-term defect is NOT the stale b0b5beac golden;
it is a TRUE latent defect in the CURRENT committed emitter's SHARED min-term fold. STOP + escalate.

## What (vs M2)
M2 measured the ~5% min-term error on the OLD deployed golden `golden_q4K.c` (md5 **b0b5beac**, full-unroll,
14113 lines). Leading hypothesis: b0b5beac's full-unroll min-term wiring was buggy and the S6 rewrite fixed
it. M2c is the DECISIVE bisection: re-emit the CURRENT committed emitter's q4_K repack-GEMM kernel, deploy it
through the SAME reversible M2 patch (the ONLY variable is the kernel body), re-run the SAME driver, compare.

## FRESH kernel (the re-emit)
- Host, tree quiescent: `build/bin/tcrv-opt` FORCE-rebuilt (md5 81bc24eb; ninja bin/tcrv-opt bin/tcrv-translate,
  NO source change), then:
      build/bin/tcrv-opt test/Conversion/RVV/rvv-to-emitc-repack-gemm-q4-K-q8-K.mlir --tcrv-rvv-lower-to-emitc \
        | /usr/lib/llvm-20/bin/mlir-translate --mlir-to-cpp > fresh_q4K.inc
- `fresh_q4K.inc` md5 **90d454da655f2fc1f88435d2d5826942** (19521 lines, S6-tiled) — **distinct from b0b5beac**
  (14113 lines, full-unroll). Byte-identical to the S6-tile findings kernel (`s6_q4K.c` md5 90d454da).
- Exports the same 7-arg ABI symbol the M2 adapter calls:
  `tcrv_emitc_ggml_repack_gemm_q4_K_q8_K_kernel_ggml_repack_gemm_q4_K_q8_K(size_t,float*,const uint8_t*,const uint8_t*,size_t,size_t,size_t)` (bs LAST).
- Emitter source UNCHANGED (git diff HEAD empty). NO emitter source modification. NO board-side tcrv-opt.

## Deploy (board `/home/ubuntu/tcrv-llamacpp`, build-gcc15-rv64gcv; reversible, 板末强制 restore)
- `m2c_patch.py` — a faithful copy of M2's `m2_patch.py`; ONLY the golden `#include` path changes to
  `/tmp/m2c_q4k_dispatch/fresh_q4K.inc`. Same case128 selector edit, same VLEN128 branch, same banner, same
  ABI adapter. 2 tracked files edited: `ggml/src/ggml-cpu/repack.cpp` + `.../arch/riscv/repack.cpp`.
- `m2c_board_run.sh` — baseline md5 verify (GEN deb61a29 / ARCH 99131cf7) + backup -> patch -> `cmake --build
  build-gcc15-rv64gcv --target ggml-cpu` -> compile driver (g++-15.2) -> run -> EXIT-trap forced restore
  (cp *.ORIG back, md5-verify, rebuild pristine). NO git stash/rm/mv/add/commit.
- Driver = M2's `m2_dispatch.cpp` (unchanged): real `ggml_mul_mat` over a GGML_OP_MUL_MAT node, weight in
  `ggml_backend_cpu_repack_buffer_type()` -> patched selector -> trait q4_K_16x1_q8_K -> our VLEN128 branch
  -> FRESH kernel. Compares R vs independent scalar oracle(full/main-only) + tcrv-stock, using ggml's REAL
  `ggml_quantize_mat_q8_K_4x1` activation (iscale=-127/max, bsums g*4+r) — so the min-term IS exercised.

## Result (m2c_dispatch_result.log) — BIT-IDENTICAL to M2 (b0b5beac)
- ROUTING: PROVEN. Banner `TCRV EMITTED GEMM(q4_K_16x1 VLEN128 compiler-emitted) ENGAGED` fires from inside
  the rebuilt libggml-cpu.so (918264 B patched); `repack tensor with q4_K_16x1`; ggml-repack bytes == M1 (all shapes).
- INT (dmin=0, no min-term): R-vs-oracle rel 4.457e-07 / 2.198e-05 — MAIN term byte-exact-capable (unchanged).
- NORM (dmin!=0, min-term ACTIVE): R-vs-oracle(full) rel **1.998e-01 / 3.143e+00 / 1.201e+00 — STILL ~5%/super-block**.
  Sample R/Ffull/Fmain identical to M2 to 4 decimals. Per-column min-term ratio R/correct = **0.947 / 0.637 /
  0.935 / 0.637** (alternating by weight-column parity; even ~0.94, odd ~0.64) — the SAME signature as b0b5beac.
- A-tree RESTORED byte-exact (GEN deb61a29 / ARCH 99131cf7); libggml-cpu.so rebuilt pristine (860736 B, golden
  symbol=0, banner=0).

## Verdict — OUTCOME (b): true latent emitter min-term defect (NOT stale golden)
b0b5beac (full-unroll) and 90d454da (S6-tiled) are DIFFERENT source forms (14113 vs 19521 lines, different md5)
yet produce **bit-identical dispatch output including the identical min-term error**. => the ~5% min-term defect
is in the SHARED `emitRepackKQuantGemmBodyQ4K` min fold (fold_model kquant_dmin_bsums_min), inherited by BOTH
lowerings — it is NOT b0b5beac's full-unroll wiring. The S6 rewrite preserved byte-exactness vs golden (as the
S6 §0 gate claimed) AND preserved the defect. Leading hypothesis (b0b5beac stale-buggy) is FALSIFIED.
- Signature (column-parity-alternating min error) points at the VLEN128 two-8-lane-strip (half_lanes=8) min-term
  materialization — which has NO counterpart in ggml's 16-lane VLEN256 `ggml_gemm_q4_K_16x1_q8_K`, so the M2b
  PHASE 1 source-diff-vs-ggml (which validated the ggml-shaped read) structurally could not catch it. HYPOTHESIS.
- STOP per stop-rule: do NOT speculatively modify the measured-winner kernel. Fix + re-cert awaits ruling.
- q5_K (same dmin*mn*bsum min fold / same emitter path): expected to carry the SAME defect through real dispatch;
  do NOT certify q5_K repack-GEMM byte-exact until the shared min-term fold is fixed. (Inferred, not run here.)

## Durable files
- `m2c_patch.py`, `m2c_board_run.sh` — the reversible deploy (applied on board via ssh).
- `m2c_dispatch_result.log` — captured board run (routing banner + numeric breakdown + forced-restore proof).
- fresh kernel `fresh_q4K.inc` (md5 90d454da) is regenerable (recipe above); board scratch `/tmp/m2c_q4k_dispatch`
  is ephemeral; M2's `/tmp/m2_q4k_dispatch` (b0b5beac) left untouched for the A/B record.

## Durable Files
- `m2c_patch.py`
- `m2c_board_run.sh`
- `m2c_dispatch_result.log`
