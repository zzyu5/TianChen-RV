# FMT-PROP board measure — rms_norm→mul→quantize(q8_0) fused vs unfused (rvv/VLEN128)

**Campaign**: 三 G2 铺量 phase1b [FMT-PROP] — format propagation: fold the downstream **activation
quantize** pass into the rms_norm→mul epilogue. **L3 memory axis, [NG-4] isolated A/B (NOT a ggml
beat, NOT an e2e [PERF-1] eight-gate).** Sibling of the prior leg `g2-fuse-rms-norm-mul` (which folded
only rms_norm→mul; 1.308× / 256 MiB). This leg extends the chain to q8_0 quantize.

## What was measured

Faithful board reconstruction of the emit census of commit **2b814e46** (real emitter
`emitElementwiseRmsNormReduceStrip` + shared `emitQuantizeQ80BlockBody`):

- **FUSED** (one pass/row): row rms scale → per 32-elem q8_0 block, in-register
  `vx→vfmul_vf(scale)→vy→load w→vfmul_vv→vz` then the q8_0 body on the register-kept `vz`
  (`vfabs→vfredmax amax→d=amax/127→vfmul_vf(id)→vfncvt_x(RMM)→vncvt→vse8_v_i8m2`) + fp16 `d`.
  The f32 activation `z[]` is **never stored, never reloaded**.
- **UNFUSED** (the pre-FMT-PROP baseline = the already-fused rms_norm→mul that lands f32 + an
  INDEPENDENT `quantize_row_q8_0` pass): pass1 writes f32 `z[]` (`vse32`); pass2 reloads `z[]`
  (`vle32`) and runs the SAME q8_0 body.

The only difference is the intermediate f32 activation `z[]` round trip. Both sides are ONE source,
ONE `clang-17 -O3 -march` (preflight(0) toolchain symmetry total, no fast-math), and the q8_0 body is
the SAME shared helper → the emitted q8_0 output is byte-identical.

## Results (rvv/VLEN128, core 8, 2.6 GHz performance gov, ROWS=8192 NEMB=4096; tensors 128 MiB each, ws 290 MiB > 3×L3)

| metric | value | verdict |
|---|---|---|
| on-hardware q8_0 value equality | **ndiff = 0 / 4,456,448 bytes (byte-exact)** | fused ≡ unfused |
| cold paired wall speedup (N=12) | **1.139×** (fused 218.18 ms vs unfused 248.50 ms) | — |
| IQR (fused / unfused) | 0.082% / 0.032% (non-overlapping) | tight |
| drift sentinel (fused-repeat / fused) | 1.0047 (≈0.47% noise floor) | ~30× floor → **T-N PASS** |
| **DRAM bytes eliminated / iter (measured)** | **255.8 MiB = 99.9% of predicted 256.0 MiB** | dual-counter |
| — LLC-store-miss delta | 127.97 MiB (= z[] store, **100.0%** of 128 MiB) | independent |
| — LLC-load-miss delta | 127.80 MiB (= z[] reload, **99.8%** of 128 MiB) | independent |
| cycle delta cross-check | 30.01 ms/iter ≈ wall delta 30.32 ms/iter (99.0%) | consistent |

**Predicted elimination** = the f32 activation round trip = `2·n·4·rows` = 2 × 4096 × 4 × 8192 =
**256.0 MiB** (store `4·n·rows` + reload `4·n·rows`). The measured DRAM traffic delta matches this to
99.9%, with the load-miss delta isolating the reload and the store-miss delta isolating the store —
two independent counters, each ≈ its predicted half.

## Honest reading ([NG-4] discipline)

- The **1.139× is an isolated A/B on the memory axis**, same category as `g2-fuse-rms-norm-mul`. It is
  **NOT** a ggml beat and **NOT** an e2e [PERF-1] eight-gate.
- It is **smaller than the prior rms_norm→mul leg (1.308×)** — expected and honest: the FMT-PROP
  UNFUSED baseline does strictly MORE work (a full quantize pass + q8_0 output write), so the same
  256 MiB elimination is a smaller fraction of total traffic, and the added quantize compute (vfredmax
  / vfncvt) dilutes the memory win. The **byte-elimination magnitude is identical** (one f32 activation
  round trip, ~256 MiB) — that is the load-bearing structural claim; the wall speedup is the second-
  order consequence at this shape.
- **铺量触发 (rollout-trigger)**: the FMT-PROP fold reproduces the prior leg's byte-elimination on
  silicon (99.9% of prediction, dual-counter) with byte-exact numerics. Fan-out立项 stays the user's.

## Provenance

- Fusion CODE (measured contract): commit 2b814e46 — `include/…/RVVOps.td` (ElementwiseMulMapOp
  `$quant_epilogue` region + ElementwiseQuantizeQ80MapOp), `lib/Conversion/RVV/RVVToEmitCForwardElementwise.cpp`
  (`emitQuantizeQ80BlockBody`), `RVVToEmitC.cpp` allowlist, `RVVDialectWideningOps.cpp` verifier +
  tracer lit `…rms-norm-mul-quantize-q8-0-fused-epilogue-loop-body.mlir`.
- Driver + harness (data-only cell holds no code): `tools/e2e-harness/board/fmtprop_driver.c` +
  `fmtprop_paired.sh`.
- Raw board output + derived arithmetic: `board_raw.txt`. Byte accounting: `byte_accounting.csv`.
- Nothing committed / git add-ed by the measure (user commits).
