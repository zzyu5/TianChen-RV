# EXPORT RECIPE — pinned-compiler kernel export (provenance)

Compilers built by `tools/bench/byte-exact-baseline.sh` (detached-worktree cached build;
NEVER touches the main tree, NEVER `git stash`):

- **pre-[GAP-P1]** = `685ab5a1` (parent of the fix) -> `.worktrees/cache/685ab5a1…/tcrv-opt`  (CACHE HIT)
- **post-[GAP-P1]** = `828d9814` (the fix)          -> `.worktrees/cache/828d9814…/tcrv-opt`  (built this batch)
- translate: `/usr/lib/llvm-20/bin/mlir-translate --mlir-to-cpp`

## q4_0 PREFILL REPACK GEMM (the reachable [GAP-P1] cell)

    <opt> test/Target/RVV/q4-0-q8-0-repack-gemm-full-pipeline-export-e2e.mlir \
      --tcrv-rvv-lower-quant-contraction=march=rv64gcv_zvl256b --tcrv-rvv-lower-to-emitc \
      | mlir-translate --mlir-to-cpp

- pre-fix opt  -> `kernels/q4_0_gemm_pre_mf2_vlen256.kernel.c`  (8x i8mf2, f32m2 — underfed fractional core)
- post-fix opt -> `kernels/q4_0_gemm_post_m1_vlen256.kernel.c` (32x i8m1, f32m4 — whole-LMUL wide core)
- The ONLY input delta between the two kernels is the compiler (same MLIR, same march); the flip is the
  `selectsWholeLMULRepackCore` selector change in 828d9814. Byte-identical numeric result (cksum equal).

## q8_0 symmetric REPACK GEVM strip-width (task 三; post-fix opt, both marches)

    # narrow (authored half_lanes 8):
    <opt> test/Conversion/RVV/rvv-to-emitc-repack-gemv-q8-0-q8-0.mlir --tcrv-rvv-lower-to-emitc | mlir-translate --mlir-to-cpp
    # wide (half_lanes 16, VLEN256 strip-width materializer):
    <opt> …mlir --tcrv-rvv-materialize-repack-strip-width=march=rv64gcv_zvl256b --tcrv-rvv-lower-to-emitc | …
    # m1 whole-LMUL (RVV0.7-form, compiled for RVV1.0 on board):
    <opt> …mlir --tcrv-rvv-materialize-repack-strip-width=march=rv64gc_xtheadvector --tcrv-rvv-lower-to-emitc | …

-> `kernels/q8_0_gevm_{narrow_hl8,wide_hl16,m1_wholeLMUL}.kernel.c`

## board harness (tools/, in touch-set)

- `tools/e2e-harness/board/gemm_timing_driver.c` + `gemm_p1_ab.sh`  (q4_0 GEMM mf2-vs-m1 paired A/B)
- `tools/e2e-harness/board/gevm_q8_timing_driver.c` + `gevm_q8_ab.sh` (q8_0 GEVM 3-way paired A/B)
- run dir on k1: `/tmp/p1_gemm_ab` (scratch, removed on restore).

Note: q8_0 CANNOT be expressed through the q4_0 `quant_contraction` bridge (its verifier requires
weight_block_stride==18); q8_0 reaches the strip-width core via its own monolithic
`tcrv_rvv.repack_gemv_q8_0_q8_0` op (conversion fixture above).
