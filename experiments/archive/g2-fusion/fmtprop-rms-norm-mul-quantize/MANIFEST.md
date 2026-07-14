# cell MANIFEST — fmtprop-rms-norm-mul-quantize

> ★M2c min-term-bug 隔离确认(2026-07-09 G3-minterm-fix 裁决一.2):本 cell kernel(rms_norm→mul→quantize(q8_0)
> 融合)grep 无 `kquant_dmin_bsums_min` / dmin·bsums-min fold —— 无 K-quant min fold。故与 M2c q4_K/q5_K/q2_K
> repack-GEMM min-term VLEN128 bug(commit 53666846)**无共享路径、不受影响**;1.139× 数值不动。

- **campaign**: fuse / [FMT-PROP] (三 G2 铺量 phase1b — format propagation: fold activation quantize
  into the rms_norm→mul epilogue; L3 memory axis)
- **status**: ACTIVE — **BOARD-MEASURED (2026-07-08, rvv/VLEN128, core 8, 2.6 GHz)**: paired
  fused-vs-unfused cold N=12 => **1.139× wall speedup** (fused 218.18 ms vs unfused 248.50 ms; IQR
  0.03–0.08% non-overlapping; drift sentinel 1.0047 ≈0.47% floor => ~30× floor => T-N PASS) +
  **measured DRAM bytes eliminated = 255.8 MiB/iter = 99.9% of predicted 256.0 MiB** (2·n·4·rows f32
  activation round trip), DUAL-COUNTER: LLC-store-miss delta 127.97 MiB = z[] store (100.0%),
  LLC-load-miss delta 127.80 MiB = z[] reload (99.8%); cycle delta 30.01 ms/iter ≈ wall 30.32 ms/iter.
  On-hardware q8_0 **byte-exact** (ndiff 0/4,456,448). [NG-4] isolated A/B on the memory axis — NOT a
  ggml beat, NOT an e2e [PERF-1] eight-gate. Smaller than the prior rms_norm→mul leg (1.308×) because
  the FMT-PROP baseline does strictly more work (a full quantize pass) so the same 256 MiB elimination
  is a smaller traffic fraction — the load-bearing claim is the elimination magnitude, which is
  identical (one f32 activation round trip). 铺量触发 satisfied; fan-out立项 stays the user's.
- **role**: [FMT-PROP] board evidence. The rms_norm→mul→quantize(q8_0) fused chain (commit 2b814e46,
  real emitter `emitElementwiseRmsNormReduceStrip` + shared `emitQuantizeQ80BlockBody`) folds the
  downstream independent activation-quantize pass into the epilogue: the register-kept weighted-
  normalized `vz` is quantized to block_q8_0 in-place and the intermediate f32 activation `z[]` is
  NEVER stored / NEVER reloaded. The board leg measures the real DRAM benefit of that elimination.
- **layout**: org STAGE2 per-cell manifest. Fusion CODE lives outside this data cell (RVVOps.td /
  RVVToEmitCForwardElementwise.cpp / RVVToEmitC.cpp allowlist / RVVDialectWideningOps.cpp verifier +
  the tracer lit). Driver + harness live under `tools/e2e-harness/board/` (fmtprop_driver.c,
  fmtprop_paired.sh). This cell holds only data/evidence.

## durable files

- `board_measured.md`   — ★ board paired measure writeup: setup faithful to the emit census, on-hardware
  byte-exact q8_0, cold N=12 wall speedup 1.139× (T-N PASS), measured DRAM elimination 255.8 MiB =
  99.9% of the 2·n·4 prediction (dual counters), cycle cross-check, [NG-4] framing + honest delta-vs-
  prior-leg reading.
- `board_raw.txt`       — raw board output: byte-exact verify, N=12 paired rounds + median/IQR, the two
  perf-stat counter runs (cycles + LLC load/store-misses), per-iter byte-delta arithmetic, reproduce +
  restore commands.
- `byte_accounting.csv` — the eliminated f32 activation round-trip accounting (store 4n / reload 4n /
  total 2n·4) with measured-vs-predicted percentages per counter.

## note — data-only cell
Data/evidence only. The 1.139× is an **isolated A/B on the memory axis** ([NG-4]), NOT a ggml beat and
NOT an e2e eight-gate. Nothing here was committed / git add-ed.
