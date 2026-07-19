# r51f — [GAP-P1] loosen rollout: DEPLOYED q8_0 repack-GEVM m1-vs-mf2 board verify

**Verdict.** [GAP-P1] loosen for **q8_0 only** is CONFIRMED on the DEPLOYED
(CORE==PROD) kernel: the m1 whole-LMUL repack-GEVM is **byte-exact** to the mf2
default AND **1.6–2.4× faster** @rvv, robust from cache-resident THROUGH 35.6 MB
DRAM-bound (the "micro washes at e2e" concern does NOT wash out within the
deployed GEVM). The measured-table fill (`lookupRepackMeasuredM1Faster` →
q8 = m1) is KEPT.

## What is measured (DEPLOYED, not standalone)

Unlike W2 (`../w2-widen-to-m1-board/`) which measured hand-written STANDALONE
dot-reduce chains, this measures the **exact front-door emit** of the deployed
q8_0 decode repack-GEVM — the two kernels are the byte-for-byte
`weft-opt ... --weft-rvv-lower-quant-contraction=march=rv64gcv --weft-rvv-lower-to-emitc
| mlir-translate-20 --mlir-to-cpp` output of
`test/Conversion/RVV/rvv-lower-quant-contraction-q8-0-decode-repack-gevm.mlir`:

- `q8_gevm_mf2` = the selector's **mf2 default** (binary BEFORE the table fill);
  sealed CORE emit md5 `2b4cacd77c13743ee1c0998e105bb404`.
- `q8_gevm_m1`  = the selector's **m1 whole-LMUL** (binary AFTER the table fill,
  reason=measured, half_lanes=16, integer_core_lmul=m1); sealed CORE emit md5
  `3223e26fc1e9148f210d31e53f48d551`.

(The `kernels/*.cpp` in this dir are those emits with ONLY the extern-"C" symbol
renamed to `q8_gevm_{mf2,m1}` so both link into one harness; the compute body is
byte-identical to the sealed CORE emit above.)

The kernels are pure lane-wise widening int8 GEVM (block_q8_0x16 weight × plain
block_q8_0 activation, dual-fp16 scale fold). Only the accumulator LMUL differs
(mf2: two 8-lane f32m2 strips; m1: one 16-lane f32m4 strip), so the arithmetic is
identical.

## Pre-check: objdump (no wide-can-narrow / no gather)

The two `.o` emit DISTINCT vtypes (compiler-symmetric, same clang-17.0.6):
- mf2: `e8,mf2` + `e16,m1` + `e32,m2`.
- m1 : `e8,m1`  + `e16,m2` + `e32,m4`.
Neither chain uses any gather (`vlux*`/`vrgather` count = 0): this is a clean
int8 widening dot, NOT a codebook-gather leaf (unlike grid dequant / iq4_nl), so
there is no wide-can-narrow / gather-bound confound — the only lever is the
accumulator LMUL.

## Board

- `ssh rvv`: openEuler riscv64, VLEN=128 (e8m1 VLMAX=16), 64 cores,
  `rv64imafdcv_..._zvfh_...`.
- compiler: system **clang 17.0.6** for BOTH chains (compiler-symmetric ratio),
  `-O3 -march=rv64gcv_zvfh`.
- weft-rv HEAD `4deb9b33bc371ade92d9f14237ffb535c18423a8` + this task's
  `lookupRepackMeasuredM1Faster` fill.

## Data — cold GEVM, 2 seeds, median of 7 (64 MB cache flush each rep)

All rows **3-arm byte-exact PASS mism=0** (mf2 == m1 == scalar oracle, bit-exact).
Ratio = m1_ns / mf2_ns (<1 ⇒ m1 faster).

| footprint | seed | mf2 median_ns | m1 median_ns | m1/mf2 |
|---|---|---|---|---|
| M=256 K=1024 (cache-resident, 272 KB) | 0 | 104,740 | 64,980 | **0.62** |
| M=256 K=1024 | 1 | 110,820 | 46,800 | **0.42** |
| M=2048 K=4096 (~8.5 MB) | 0 | 3,820,337 | 2,137,910 | **0.56** |
| M=2048 K=4096 | 1 | 3,452,316 | 1,913,268 | **0.55** |
| M=8192 K=4096 (~35.6 MB, DRAM-bound) | 0 | 15,424,869 | 8,890,520 | **0.58** |
| M=8192 K=4096 | 1 | 15,272,148 | 8,782,520 | **0.58** |

**Deployed q8: m1 is 1.6–2.4× FASTER than mf2** (deployed win is STRONGER than
W2's ~1.41–1.46× standalone, because the deployed GEVM's mf2 default carries TWO
8-lane strips + double the vsetvl/loop overhead that the single 16-lane m1 strip
collapses). vs the mf2 baseline the m1 ratio 0.42–0.62 is ≥0.8 by a wide margin.

## e2e-wash conclusion (the [GAP-P1] live falsifier)

The m1 win **HOLDS at 35.6 MB DRAM-bound** (0.58, i.e. m1 ~1.7× faster) — it does
NOT wash out with footprint within the deployed GEVM. The isolated-kernel micro
win transfers through to the largest (DRAM-bound) single-GEVM footprint ⇒ REAL
win, kept, 入账. Honest scope: this is still a SINGLE deployed GEVM, not a full
llama decode (KV-cache/activation Amdahl dilution across a whole forward pass);
that remains only measurable by a full e2e harness. The DRAM-bound single-GEVM
result is necessary-not-sufficient for a full-decode e2e win, but it is the
strongest e2e-transfer evidence available and it refutes the "washes at large
footprint" half of the [GAP-P1] concern for this format.

## Scope discipline

- **q8_0 ONLY** (`kNibbleQ80ScaleModel`); every other format stays mf2
  (no-blind-widest preserved). q4 measured ~parity in W2 (confounded) ⇒ NOT
  flipped.
- Registration-as-DATA (one board-measured row), NOT a per-format C++ switch and
  NOT a VLEN projection.

## GEMM (prefill) arm — measured SEPARATELY (not extrapolated from GEVM)

The prefill GEMM is a DIFFERENT crossover than the decode GEVM: mf2 folds
columnsPerPass=4 (amortizing weight-strip loads across 4 activation columns) but
m1 drops to columnsPerPass=1 (one 16-lane f32m4 strip). To avoid projecting the
GEVM result across regime+structure ([GAP-P1] forbids projection), the deployed
q8_0 repack-GEMM was board-measured on its OWN:

- kernels: `weft-opt ... rvv-emit-quant-contraction-q8-0-repack-gemm-prefill-vlen128.mlir
  --weft-rvv-lower-to-emitc | mlir-translate-20 --mlir-to-cpp`. mf2 sealed emit
  md5 `453ced3ab9f10c20732bdd1e032a3eb1`; m1 sealed emit md5
  `17fcbd63da044e9b2be5df199de86a0f`.
- objdump: mf2 = `e8,mf2`+`e16,m1`+`e32,m2`; m1 = `e8,m1`+`e16,m2`+`e32,m4`.

| footprint | seed | mf2 median_ns | m1 median_ns | m1/mf2 |
|---|---|---|---|---|
| M=256 N=256 K=1024 | 0 | 14,795,367 | 10,268,906 | **0.69** |
| M=256 N=256 K=1024 | 1 | 14,801,546 | 10,537,008 | **0.71** |
| M=512 N=256 K=2048 | 0 | 61,383,035 | 43,111,793 | **0.70** |
| M=512 N=256 K=2048 | 1 | 60,897,653 | 43,315,394 | **0.71** |
| M=512 N=512 K=2048 | 0 | 122,157,948 | 87,524,772 | **0.72** |
| M=512 N=512 K=2048 | 1 | 121,965,306 | 86,862,390 | **0.71** |

**Deployed q8 GEMM prefill: m1 ~1.40x FASTER than mf2**, all 3-arm byte-exact
mism=0 (the GEMM oracle mirrors the x4-interleaved activation + x16 weight +
row-major output; its bit-exact PASS independently validates the deployed layout).
The wide 16-lane strip DOMINATES the loss of the mf2 4-column amortization ⇒ the
m1 flip is a WIN in the prefill regime TOO.

## Combined verdict

Both deployed regimes are board-proven m1-faster AND byte-exact @rvv:
- **decode GEVM**: m1 1.6–2.4x faster (holds to 35.6 MB DRAM-bound; no e2e wash).
- **prefill GEMM**: m1 ~1.40x faster.
⇒ the scale_model-keyed measured row (q8_0 → m1, flips BOTH lowerToRepackGemv and
lowerToRepackGemm) is justified by board evidence in BOTH regimes; every other
format stays mf2.
