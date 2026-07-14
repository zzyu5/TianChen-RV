# T4b M1 — q4_K repack-GEMM min-term instrumented BISECT (ssh rvv, VLEN128, CONFIRM-ONLY)

Operational index only. Outcome: **half_lanes=8 two-8-lane-strip min-materialization hypothesis FALSIFIED
(bit-exact).** The kernel is NOT defective; the M2c "5% min-term parity bug" is a certification-harness
**activation-quantization-path artifact**. No emitter/kernel change; A-tree restored byte-exact.

## What (vs M2c)
M2c proved b0b5beac(full-unroll) and 90d454da(S6) share the same NORM min-term error through real dispatch,
and pointed (HYPOTHESIS) at the VLEN128 two-8-lane-strip (half_lanes=8) min materialization. M1 is the
instrumented bisect that reads the kernel's REAL per-(block,row,col) min-term intermediates and localizes
the divergence, verifying/falsifying that hypothesis with bit-exact integer evidence.

## Method (reversible, board)
- Host: re-emit CURRENT emitter's q4_K repack-GEMM (`build/bin/tcrv-opt … | mlir-translate --mlir-to-cpp`)
  -> `fresh_q4K.inc` md5 **90d454da** (tree quiescent, git diff HEAD empty on lib/include). NO emitter change.
- `instrument_kernel.py` — prepends pure-observation capture globals (extern "C", visibility default) +
  injects a tile0 (weight cols 0..7) capture of the integer MAIN accumulator (v47/v49/v51/v53), the integer
  MIN accumulator (v46), fp16 d (v5765) / dmin (v5762), a_d, and the raw activation bsums region (v32+1040)
  for group0 all blocks. ZERO change to the vector math (only memory reads + fprintf). Base md5 asserted 90d454da.
- `m1_patch.py` — same reversible 2-file A-tree patch as M2c (case128 selector + VLEN128 branch + banner),
  include path -> `/tmp/m1_minterm_board/instr_q4K.inc`. `m1_board_run.sh` — baseline md5 verify (GEN deb61a29 /
  ARCH 99131cf7) + backup -> patch -> rebuild ggml-cpu -> compile driver -> run -> objdump min-fold -> EXIT-trap
  forced restore (byte-exact, pristine .so 860736 B). NO git stash/rm/mv/add/commit.
- `m1_probe.cpp` — real `ggml_mul_mat` dispatch (banner nr=4 nc=64 nb=2) + THREE references computed in-driver:
  (i) independent scalar oracle (−127/max, lrintf); (ii) q8-model-FREE recompute of MIN from oracle weight-mins
  × the KERNEL's OWN captured bsums; (iii) ZERO-MODEL recompute of MAIN/MIN from ggml's OWN
  `ggml_quantize_mat_q8_K_4x1` (what the kernel eats) AND `quantize_row_q8_K` (what STOCK eats), both dynamically
  exported by libggml-cpu.so.

## Result (m1_bisect_result.log) — kernel BIT-EXACT for its input
- **q8-FREE:** `min_int(oracle-weight-min × KERNEL-bsums) vs kernel v46 = 0 mismatches` (both blocks, all rows/cols)
  -> the two-8-lane-strip min VALUE unpack + bsums pairing + accumulation are STRUCTURALLY CORRECT. d/dmin fp16
  loads = 0 mismatches.
- **ZERO-MODEL:** `kernel v47 MAIN vs recompute-from-mat-q8 = 0`, `kernel v46 MIN vs recompute-from-mat-q8 = 0`;
  `kernel captured bsums vs ggml mat-quant bsums = 0` -> the kernel computes q4_K·q8 BIT-EXACTLY for the
  dispatch-provided (generic mat-quant) activation. `recompute-from-ROW-q8` mismatches=64/63 (kernel does NOT
  use row-quant).
- **Root of the NORM "5%":** ggml's GEMM-path `ggml_quantize_mat_q8_K_4x1` (generic) ≠ scalar-path RISC-V arch
  `quantize_row_q8_K` (SIMD): qs diffs 780/2048, bsums diffs 64/128. Two components: (a) benign SIGN-FLIP for
  positive-abs-max blocks (riscv d=+1/iscale vs generic −1/iscale; q8·a_d recovers the value; cancels in the
  dot — both q4_K terms sign-flip-invariant), (b) ROUND-HALF divergence (generic `nearest_int` rounds x.5
  e.g. −63.5→−63 vs RNE −64 on ~1.5% of elements) — the real ±1 q8 noise, dominated by the MAIN term,
  amplified at cancellation-prone near-zero outputs to the ~5-20% RELATIVE figure. R-vs-F max ABS err=12.5
  at F=3662 (0.34% there).
- **objdump min-fold (instruction-level):** `vsetivli zero,8,e32,m2` (the two-8-lane strip) then
  `vfmul.vf v30,v6,fa5` (dmin·a_d) -> `vfmacc.vv v2,v4,v30` (main) -> `vfnmsac.vv v2,v30,v16`
  (−(dmin·a_d)·min_acc) — the correct min subtraction. See T8 objdump_evidence_ptr row.

## Verdict — nothing to fix in the emitter
half_lanes=8 FALSIFIED (bit-exact). The kernel == ggml's own repack GEMM for the SAME (mat-quant) q8. The M2c
"latent min-term defect" is a HARNESS artifact: oracle/STOCK quantize the activation differently (−127/max lrintf /
riscv SIMD row-quant) from the actual GEMM dispatch (generic mat-quant). M2 direction: (1) no emitter fix; re-cert
(M3/M4) must oracle against the ACTUAL dispatch q8 (mat-quant) — against which the kernel is bit-exact — or vs
ggml's own repack GEMM; (2) the ggml mat-quant-vs-riscv-row-quant divergence (sign choice + round-half) is a
ggml-side numerics question, orthogonal to our compiler. Awaits ruling.

## Durable Files
- `m1_bisect_result.log` — captured board run (routing banner + per-(block,row,col) intermediates + q8-FREE +
  ZERO-MODEL decomposition + mat-vs-row divergence dump + objdump min-fold + forced-restore proof).
- harness/驱动 relocated to `tools/e2e-harness/board/t4b-m1-minterm-bisect/` (可复演入口): `instrument_kernel.py`
  (pure-observation kernel instrumenter), `m1_probe.cpp` (bisect driver), `m1_patch.py` (reversible ggml dispatch
  patcher), `m1_board_run.sh` (board build+run+restore driver).
- Base kernel `fresh_q4K.inc` md5 90d454da regenerable from `test/Conversion/RVV/rvv-to-emitc-repack-gemm-q4-K-q8-K.mlir`.
