# r51g B1 — [GAP-P1]-loosen repack FAMILY board-sweep (q4_0/q4_1/q5_0/q5_1)

**Verdict.** The [GAP-P1] loosen is extended from q8_0 to the DEPLOYED flat nibble
family. Board-measured @rvv (VLEN128, system clang 17.0.6, compiler-symmetric,
2-seed cold), all 3-arm byte-exact (mism=0), objdump spill/gather-checked:

| format | decode GEVM m1/mf2 | prefill GEMM m1/mf2 | m1 spill (GEVM/GEMM) | verdict |
|---|---|---|---|---|
| **q4_0** | **0.39–0.44** (~2.3–2.5× faster) | **0.78–0.82** (~1.24× faster) | 0 / 0 | **FLIP → m1** |
| **q4_1** | **0.40–0.45** (~2.3× faster) | 0.99–1.02 (parity) | 0 / 0 | **FLIP → m1** |
| q5_0 | 0.55–0.63 (~1.6–1.8× faster) | **2.27** (2.3× SLOWER) | 0 / **1** | KEEP mf2 |
| q5_1 | 0.51–0.65 (~1.5–2.0× faster) | **2.43** (2.4× SLOWER) | 0 / **2** | KEEP mf2 |

**2 formats flipped (q4_0, q4_1) / 2 kept mf2 (q5_0, q5_1).** Plus the pre-existing
q8_0 row (r51f). Only board-MEASURED, spill-free, double-regime-safe formats flip.

## What is measured (DEPLOYED, CORE==PROD)

The kernels (`kernels/*.cpp`) are the byte-for-byte front-door emit
`weft-opt <fixture> --weft-rvv-lower-quant-contraction=march=rv64gcv
--weft-rvv-lower-to-emitc | mlir-translate-20 --mlir-to-cpp`, mf2 = selector
default, m1 = the measured-table m1 whole-LMUL chain (integer_core_lmul="m1",
half_lanes=16). The post-fill front-door m1 emit is BYTE-IDENTICAL (diff=0) to the
measured m1 kernel — the kernels board-tested here are EXACTLY what the deployed
compiler now produces for q4_0/q4_1.

Only the accumulator LMUL differs: mf2 = two 8-lane strips (i8mf2→i16m1→i32m2, two
f32m2 accumulators); m1 = one 16-lane strip (i8m1→i16m2→i32m4, one f32m4). The
nibble/qh unpack + fold arithmetic is identical, so the two chains are byte-exact to
each other AND to an independent scalar ZERO-MODEL oracle (3-arm mism=0).

## Why the flat family wins in decode (resolves the W2 confound)

W2 (`../w2-widen-to-m1-board/`) measured a STANDALONE q4 chain with a SCALAR nibble
unpack and found ~parity (0.95–1.005×) — CONFOUNDED, because the scalar unpack
dominated and diluted the chain difference. The DEPLOYED front-door emit VECTORIZES
the nibble unpack. mf2 does that vectorized unpack TWICE (once per 8-lane strip);
m1 collapses it to ONE 16-lane pass. So the deployed q4_0/q4_1 decode win (~2.5×) is
LARGER than q8_0's (no unpack, 1.4–2.4×). The board refutes the W2 "q4 parity"
reading for the deployed path.

## Why q5_0/q5_1 stay mf2 — [GAP-P1] spill VINDICATED in prefill

q5_0/q5_1 add the qh 5th-bit (per-position bitmask load + masked add/sub) on top of
the nibble unpack. In the DECODE GEVM (unroll=1, one column strip) m1 is spill-free
and wins (1.5–2.0×). But the PREFILL GEMM keeps columnsPerPass=4 activation columns
live; at m1 (i16m2 + i32m4 + the qh/min working set × 4 columns) the register
pressure EXCEEDS the 32-vreg budget and the m1 GEMM emit SPILLS (objdump: q5_0 = 1,
q5_1 = 2 whole-register reloads), running 2.3–2.4× SLOWER than mf2. The selector's
spill-free precheck (rvvRegisterPressureLegal) models only the GEVM chain
(i16m2+i32m4=6 vregs), NOT the GEMM's column-amortized working set — so the board is
the only authority. Because ONE measured row flips BOTH regimes, flipping q5 would
deploy a 2.3× prefill regression: they stay mf2. This is exactly the [GAP-P1]
"regfile spill" concern, board-vindicated for the qh+min prefill core.

## q4_1 prefill = parity (still flipped)

q4_1 decode m1 wins 2.3× and its GEMM is parity (0.99–1.02, spill-free, no
regression). Since the flip causes no prefill regression and a large decode win
(the autoregressive-generation-critical regime), q4_1 is flipped. Honest note: the
q4_1 GEMM is NOT a prefill win, only neutral.

## K-quant decode scope (PRD "若走此路" — bounded continuation, NOT swept)

q2_K/q3_K/q4_K/q6_K decode repack-GEVM DO route through
`lowerToRepackGemvKQuant -> selectRepackAccumulatorLMUL` (same measured gate), so
they are eligible for the same flip. They are NOT swept here: each needs a byte-exact
super-block harness (dual d/dmin fp16, 6-bit + scale-min sub-quant layout) —
materially larger than the flat family. Prediction from the q5 result: the K-quant
super-block CORE is heavier than q5's qh core, so the PREFILL GEMM m1 chain will
spill at least as badly (columnsPerPass=4 × super-block working set ≫ 32 vregs) ⇒
K-quant most likely STAYS mf2 (prefill regression), even if the memory-bound decode
GEVM m1 wins. Confirming this predicted-null is the bounded continuation; until
swept, K-quant has no measured row ⇒ mf2 default holds (byte-exact, no regression).

## Board pin

- `ssh rvv`: openEuler riscv64, VLEN=128 (e8m1 VLMAX=16), 64 cores, `rv64...zvfh`.
- compiler: system **clang 17.0.6** BOTH arms (compiler-symmetric ratio),
  `-O3 -march=rv64gcv_zvfh`.
- weft-rv HEAD `e5befd70e` + this task's kRepackMeasuredM1FasterMeasurements rows.
- objdump: mf2 = e8,mf2 + e16,m1 + e32,m2; m1 = e8,m1 + e16,m2 + e32,m4; gather=0
  (no vlux/vrgather — clean int-widening dot, NOT a codebook-gather leaf) for all.
- raw board CSVs: `raw/gevm_q4_run_reps15.txt`, `raw/gevm_q5_run.txt`,
  `raw/gemm_run.txt`. reps=15 (GEVM) / reps=7 (GEMM), median cold, 64 MB flush/rep.

## Scope / vs-opponent accounting

The measured table encodes the INTERNAL m1-vs-mf2 LMUL choice (both ours, byte-exact).
The vs-DEPLOYED-OPPONENT ratio (ggml repack) is tracked separately for master 入账;
see `opponent/VS-OPPONENT-FINDING.md`. Key result: the ggml x16 repack
`ggml_gemv_q8_0_16x1_q8_0` is byte-INEXACT on VLEN128 (a VLEN256-tuned kernel,
vl=16 > VLMAX=8) — it is NOT the valid VLEN128 opponent, so the naive x16 projection
is REFUTED (no fabricated ratio recorded). Identifying ggml's true VLEN128 q8_0
repack variant (4x8 / SpacemiT / block-dot) is the remaining vs-opponent continuation.
mf2 is the already-shipping default, so flipping to a strictly-faster m1 can only
improve whatever the real vs-opponent ratio is.
