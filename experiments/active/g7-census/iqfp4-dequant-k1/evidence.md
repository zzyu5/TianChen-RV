# G7 L1 iq/fp4 vec_dot + dequant@k1 kernel-sym census — k1-lane (VLEN256, clang-18 symmetric)

> **Line**: G7 终编成令三 · 第一段全量 kernel 冷启动普查 · Batch 3c (iq/fp4 vec_dot 4) + 3d (dequant [DEQ-AXIS] 18) · k1-lane (rvv 忙 B 类 rvv-half).
> **Board**: k1 / SpacemiT-X60 / VLEN256 (vlenb=32) / clang-18.1.8 (Bianbu) symmetric domain / pin core0 idle=100% gov=performance freq=1.6GHz.
> **Axis**: **kernel-sym micro (第二赛道)** · [NG-4] NOT e2e · NOT perf-covered 9/83 · NOT certified · NOT a sealed Win. 禁互推.
> **Casefile**: `experiments/active/g7-census/iqfp4-dequant-k1/` — kernels_vecdot/ (+GEN_SEAL), kernels_dequant/ (+GEN_SEAL), {iqfp4_vecdot,dequant_census}_driver.c, k1_build.sh, k1_measure.sh, raw/{k1_run.log, k1_build_seal.txt, k1_opponent_classification.txt}, summary_kernel_sym_k1.csv.

---

## 0. Harness (net-new; only "construct" gap was the harness — all emit already in)

- **OURS vec_dot** = weft-emitted grid/codebook **block-dot** vec_dot, regenerated from the working-tree emitter on the
  front-door fixtures (`weft-opt <fixture> --weft-rvv-materialize-<slug>-block-dot-source-front-door --weft-rvv-lower-to-emitc
  | mlir-translate-20 --mlir-to-cpp`; md5 in kernels_vecdot/GEN_SEAL.txt). iq1_s/iq1_m = super-block grid-gather (QK_K=256,
  q8_K act); iq4_nl = 16-entry integer codebook (QK4_NL=32, q8_0 act); nvfp4 = E2M1 4-bit fp codebook + UE4M3 sub-scale
  (QK_NVFP4=64, q8_0 act).
- **OURS dequant** = weft-emitted `weft_rvv.typed_dequantize_row_loop_body` streaming kernels, regenerated per-format via
  `sed 's/q8_0/<fmt>/g' rvv-dequantize-row-stream-front-door-construct.mlir | weft-opt --weft-rvv-materialize-dequantize-row-stream-front-door
  --weft-rvv-lower-to-emitc | mlir-translate-20 --mlir-to-cpp` (md5 in kernels_dequant/GEN_SEAL.txt). front-door facts
  (decode_model/qk/weight_block_stride) verified per format.
- **OPP vec_dot** = ggml **as-shipped dispatched** `ggml_vec_dot_<fmt>`, linked from stock `libggml-cpu.so` (md5 871169a0…).
- **OPP dequant** = ggml **as-shipped DEPLOYED** `dequantize_row_<fmt>`, linked from stock `libggml-base.so` (md5 00267134…).
  **★ deployed-path machine-proof**: ggml-cpu/`ops.cpp` (lines 485/596/941/4334/4756) dequantizes via
  `ggml_get_type_traits(type)->to_float`; `ggml.c` sets `.to_float = (ggml_to_float_t) dequantize_row_<fmt>` (base scalar
  reference). → the linked symbol IS the genuine deployed dequant, NOT a straw-man swap. (It is, however, the **UNVECTORIZED
  scalar reference** — objdump rvv=0 all 18 — so ours-RVV-vec-vs-opp-scalar wins are a *weaker class* than vs hand-tuned RVV;
  disclosed §4.)
- **Both sides consume IDENTICAL plain block bytes** (no repack) — fair kernel-axis A/B, same footprint.
- **Byte-exact gate (ZERO-MODEL vs independent stock-ggml oracle)**: vec_dot INT-mode (weight d = fp16 1.0 / UE4M3≈1.0,
  activation q8 payload = ±1 → decoded products exact-representable → order-independent exact fold); dequant small-byte fill
  (`&0x3F` → every fp16/E4M3/E8M0 scale finite, no NaN/Inf). Require ours[k]==opp[k] bit-for-bit. Nonzero varied samples
  (q6_K dequant 411.7, iq4_nl vec_dot −11490, q4_K dequant 192.6, …) confirm real work, not 0==0.
- **COLD protocol**: 32 MiB flush (64× k1 L2 512KiB, NO L3) before EACH timed region; paired A/B; N=12 reps; median + relIQR;
  T-N noise-floor (nf 42–250 ns). **HOT** = warmup + best-of-8. CLOCK_MONOTONIC (k1 has NO cache-miss PMU). vec_dot GEVM
  output[M][nc], M=1 + M=8 便车, K=2048, nc=512. dequant streaming K=131072 (512 KiB f32 out > L2).
- **Load-gate**: idle-core pick over 0-3 (300 ms /proc/stat delta, ≥70 % idle; picked core0 @100 %). Post-run stray check =
  0 (pgrep -x clean); gov=performance untouched; k1 restored.

---

## 1. ★ Batch 3c — iq/fp4 vec_dot four-column table (cold_med ratio = ours ÷ opp; >1 ⇒ ours faster; parity band ±5%)

| 格@板 | hot | cold (M=1) | cold (M=8) | 对手类 [机判·符号级] | 胜负 (cold) |
|---|:--:|:--:|:--:|---|---|
| **iq1_s@k1** | 0.363× | **0.367×** | 0.364× | VLEN-adaptive dispatcher → **vectorized RVV gather** (generic 149 rvv / 68 gather) | **LOSS 0.37×** (gather-trap vs vectorized opp — prediction confirmed) |
| **iq1_m@k1** | 0.241× | **0.247×** | 0.242× | VLEN-adaptive dispatcher → **vectorized RVV gather** | **LOSS 0.24×** (heaviest grid; C2-reuse of iq1_s core, most re-config) |
| **iq4_nl@k1** | 0.686× | **0.694×** | 0.685× | dispatcher → `_vl256` **RVV** (79 ins, rvv/gather-4) | **LOSS 0.69×** (vec_dot path; cf. iq4_nl gemm 0.217×) |
| **nvfp4@k1** | 1.066× | **1.064×** | 1.043× | **NEAR-SCALAR / unoptimized** (194 ins, rvv=2, no VLEN-spec) | **WIN 1.06×** ⚠ opponent-immaturity (see §4.2) — marginal, vs unvectorized opp |

**cold tally (4 格)**: **3 LOSS** (iq1_s/iq1_m/iq4_nl — gather-bound vs vectorized RVV opponents) · **1 marginal-WIN**
(nvfp4 1.06×, ⚠ opponent near-scalar). M=1/M=8 agree tightly (IQR 0.3–3.7 %). Gather-bound LOSS **prediction confirmed**
for the 3 vectorized-opponent formats; nvfp4 is the exception because its ggml opponent is itself unvectorized.

## 2. ★ Batch 3d — dequant [DEQ-AXIS] streaming census (cold_med ratio ours ÷ opp; GB/s = (in+out)/t)

> **★ 全 18 格 byte-exact (mismatch=0, worst_ulp=0).** Opponent = deployed-scalar-reference (§0). **[DEQ-AXIS] sub-account —
> does NOT enter the matmul ≥parity headline count (体例).** Ours = weft RVV-vectorized streaming (except q2_K/q6_K/nvfp4
> emit scalar, vsetvl=0).

| 格@板 | cold_med | our GB/s | opp GB/s | ours vsetvl (self-probe) | 胜负 (cold) [DEQ-AXIS] |
|---|:--:|:--:|:--:|:--:|---|
| q2_K   | 0.81× | 0.51 | 0.64 | 0 (scalar) | LOSS 0.81× (ours emit scalar too) |
| q3_K   | 1.94× | 1.03 | 0.53 | 36 | WIN 1.94× |
| q4_K   | 2.78× | 2.13 | 0.77 | 21 | WIN 2.78× |
| q5_K   | 5.14× | 2.72 | 0.53 | 24 | **WIN 5.14×** (largest) |
| q6_K   | 1.13× | 0.56 | 0.50 | 0 (scalar) | WIN 1.13× (both scalar, near-parity) |
| iq1_s  | 2.92× | 1.57 | 0.54 | 65 | WIN 2.92× |
| iq1_m  | 3.63× | 1.73 | 0.48 | 65 | WIN 3.63× |
| iq2_xxs| 1.84× | 1.06 | 0.58 | 65 | WIN 1.84× |
| iq2_xs | 0.93× | 0.48 | 0.52 | 385 gather-32 | LOSS 0.93× (heavy gather re-config) |
| iq2_s  | 1.70× | 0.99 | 0.58 | 65 | WIN 1.70× |
| iq3_xxs| 0.53× | 0.31 | 0.58 | 354 gather-64 | LOSS 0.53× (heaviest gather) |
| iq3_s  | 0.67× | 0.41 | 0.61 | 321 gather-32 | LOSS 0.67× (heavy gather) |
| iq4_nl | 3.76× | 2.23 | 0.59 | 6 | WIN 3.76× |
| iq4_xs | 3.28× | 1.97 | 0.60 | 56 gather-16 | WIN 3.28× |
| mxfp4  | 3.17× | 2.19 | 0.69 | 6 gather-2 | WIN 3.17× |
| nvfp4  | 0.74× | 0.32 | 0.43 | 0 (scalar, ldexpf) | LOSS 0.74× (ours emit scalar) |
| tq1_0  | 2.23× | 1.04 | 0.47 | 9 | WIN 2.23× |
| tq2_0  | 4.22× | 3.13 | 0.74 | 16 | WIN 4.22× |

**dequant tally (18 格·DEQ-AXIS·vs deployed-scalar-reference)**: **13 WIN** (q3_K/q4_K/q5_K/q6_K/iq1_s/iq1_m/iq2_xxs/iq2_s/
iq4_nl/iq4_xs/mxfp4/tq1_0/tq2_0) · **0 PARITY** · **5 LOSS** (q2_K/iq2_xs/iq3_xxs/iq3_s/nvfp4).

---

## 3. 数值 byte-exact 逐格 (ZERO-MODEL vs stock-ggml oracle)

All **22 formats** (4 vec_dot × {M=1,M=8}×{nc=64 verify, nc=512 census} + 18 dequant × {K=131072}): **byte_mismatch = 0,
worst_ulp = 0 → BYTE-EXACT-vs-stock-ggml OK.** Hard codebook-oracle gate PASS (all cells). Sample folded values (nonzero,
varied): iq1_s vec_dot −42.75, iq4_nl vec_dot −11490, nvfp4 vec_dot −40.0; q4_K dequant 192.6, q6_K dequant 411.7, iq4_xs
dequant −12.65, iq3_xxs dequant 1.084.

---

## 4. 净新普查发现 + 输格出口候选 (供 L1')

1. **iq/fp4 vec_dot gather-trap LOSS prediction CONFIRMED (3/4)**: iq1_s/iq1_m/iq4_nl lose 0.24–0.69× to **vectorized RVV**
   opponents (iq1_s/iq1_m dispatch → RVV-gather generic; iq4_nl dispatch → `_vl256` RVV). ours vsetvl 79–158 (super-block
   re-config storm) vs opp vectorized gather. Disposition = **stays LOSS-on-vec_dot-axis** (gather-bound); name
   `[GAP-IQ-VECDOT-GATHER-VS-RVV]`. (iq/tq have NO repack-GEMM axis to route the beat to — opponent-absent, T-CENSUS §一.A.)
2. **★ nvfp4 vec_dot WIN 1.06× — disclose opponent immaturity**: the ONE iq/fp4 format whose ggml opponent is **unvectorized**
   (194 ins, rvv=2, no VLEN specialization). Ours ≥parity here = **opponent-immaturity-driven**, marginal (5 %, barely above
   parity band), byte-exact. Directly analogous to the prior census's q5_K vec_dot WIN (same mechanism). **Do NOT over-claim**;
   recommend report-as-marginal-parity + flag for L1' (opponent-immaturity, not "ours beats a hand-tune").
3. **★★ DEQ-AXIS falsifies the "parity @roofline / mem-bound" census prediction**: the deployed dequant opponent is
   **scalar reference** (rvv=0, deployed via `to_float`) and **COMPUTE-BOUND** — opp GB/s 0.43–0.77 ≪ ~6 GB/s k1 roofline
   (B-class near-wall). Neither side is memory-bound at K=131072; the per-element decode arithmetic dominates. Where our
   emitter **vectorizes the decode** (13/18) it beats the scalar reference 1.13–5.14×; where ours emits scalar
   (q2_K/q6_K/nvfp4, vsetvl=0) or heavy gather (iq2_xs/iq3_xxs/iq3_s) it ties/loses. **This is a genuine RVV-vectorized-vs-
   deployed-scalar-reference win-class on the dequant axis — real & deployed, but weaker than beating hand-tuned RVV.**
   Naming: `[DEQ-AXIS-RVV-VEC-BEATS-SCALAR-REF]`. Two emitter-maturity targets for L1' (Exit B): (a) vectorize
   q2_K/q6_K/nvfp4 dequant (currently scalar), (b) tame iq2_xs/iq3_xxs/iq3_s gather re-config storm (vsetvl 321–385).
4. **axis hygiene**: this batch is **kernel-sym micro (第二赛道)**. vec_dot cells = matmul-axis (但全 LOSS/marginal). dequant
   cells = **[DEQ-AXIS] sub-account** (NOT the matmul ≥parity headline). Neither transplants to perf-covered 9/83 (system/e2e)
   nor to certified. 禁互推.

---

## 5. kernel-sym 计数建议 (主会裁)

- **iq/fp4 vec_dot (matmul axis)**: **0 genuine ≥parity beats.** iq1_s/iq1_m/iq4_nl = LOSS (gather-bound vs vectorized RVV;
  prediction confirmed). **nvfp4 1.06× = marginal WIN but ⚠ opponent-immaturity** (near-scalar opp) — recommend
  **report-as-marginal-parity, do NOT count as an independent matmul beat** (same class as q5_K vec_dot in prior census).
- **dequant [DEQ-AXIS] (separate sub-account, NOT matmul headline)**: **13/18 WIN vs deployed-scalar-reference**
  (q5_K 5.14× top; q4_K 2.78×, tq2_0 4.22×, iq4_nl 3.76×, iq1_m 3.63×, iq4_xs 3.28×, mxfp4 3.17×, iq1_s 2.92×, tq1_0 2.23×,
  iq2_xxs 1.84×, iq2_s 1.70×, q3_K 1.94×, q6_K 1.13×) · 5 LOSS (q2_K/iq2_xs/iq3_xxs/iq3_s/nvfp4). **These are DEQ-AXIS
  datapoints — record in a dequant sub-account, do NOT add to the matmul kernel-sym ≥parity count** (体例 [DEQ-AXIS];
  opponent = deployed-scalar-reference, not hand-tuned RVV). If 主会 opens a DEQ-AXIS ledger, the honest headline =
  "weft RVV dequant ≥parity on 13/18 formats vs ggml deployed-scalar dequant".
- **byte-exact hard gate**: 22/22 formats mismatch=0/worst_ulp=0 — all cells certified byte-exact vs stock ggml. 禁互推
  perf-covered/certified.
