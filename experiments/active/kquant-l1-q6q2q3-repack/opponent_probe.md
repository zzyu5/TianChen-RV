# [KQUANT-L1] opponent-absence + prefill probe — q6_K / q2_K / q3_K (rvv/VLEN128)

Sibling of `kquant-l1-q4k-q5k-repack-prefill`. Completes the K-quant family (q4_K, q5_K, q6_K,
q2_K, q3_K) L1 opponent probe. Board: rvv (openEuler, riscv64, LIVE `__riscv_vlenb()*8 = 128`,
64c, core 8, governor=performance, 2.6 GHz). Opponent = board's own upstream llama.cpp
`/home/ubuntu/llama.cpp-upstream-native` (commit **f3e1828**), `libggml-cpu.so`.

## 1. Board-silicon NUMERICAL confirmation (upgrades the construction oracle)

The construction oracle (`.trellis/tasks/increment1-q6k-repack/oracle_repack_q{6,2,3}K.cpp`) was a
C++ **emitter-MODEL** vs reference — never a hardware run. This cell **compiles the ACTUAL exported
`.kernel.c` with board clang-17 and runs it on rvv silicon**, comparing the emitted kernel's fp32
output to an INDEPENDENT reference decoded from the ORIGINAL (pre-repack) per-block q{6,2,3}_K
exactly as ggml's canonical `vec_dot_q{6,2,3}_K_q8_K`. Two certificates, both GEVM (plain q8_K act)
and GEMM (block_q8_Kx4 interleaved act), 8 shapes each, 2 seeds:

- **(INT)** unit fp scales (d_w=1.0 fp16, d_a=1.0 fp32; q2_K also dmin=1.0) + magnitude-bounded
  integer core ⇒ the kernel's fp32 fold is an EXACT integer == the int64 reference isum. Result:
  **int-mismatch = 0 on ALL shapes / kernels / both seeds** ⇒ the emitted kernels are
  **BYTE-EXACT-INTEGER on silicon** (the "byte-exact 整数" commit-924dc31f claim, now hardware-run).
- **(NORM)** full adversarial fp16 super-block d (+ dmin) + fp32 activation d ⇒ f64 reference;
  worst bounded-norm `maxAbsErr / RMS(output)` ≈ **1e-7 to 7e-7** across all three (fp32 FMA fold
  vs f64 — expected, not a bug). (A few `worst_ulp` outliers, e.g. q3_K 917132, are near-zero-
  crossing cancellations: the norm-to-RMS of ~1.6e-7 confirms the absolute error is negligible.)

| fmt | INT (all shapes, GEVM+GEMM, 2 seeds) | NORM worst (maxAbsErr/rms) | verdict |
|---|---|---|---|
| q6_K | byte-exact, 0 mismatch | 7.0e-07 | SILICON BYTE-EXACT-INTEGER + BOUNDED-NORM |
| q2_K | byte-exact, 0 mismatch | 6.4e-07 | SILICON BYTE-EXACT-INTEGER + BOUNDED-NORM |
| q3_K | byte-exact, 0 mismatch | 6.7e-07 | SILICON BYTE-EXACT-INTEGER + BOUNDED-NORM |

Our kernels are fp16-hardware, **libcall-free** (no `__extendhfsf2/__truncsfhf2`). Raw:
`numeric_silicon.txt` / `numeric_silicon.csv`.

## 2. Opponent-ABSENCE @ VLEN128 (L1 structural crux) — PROVEN for all three

`get_tensor_traits` (repack.cpp, commit f3e1828) selects the online-repack GEMM/GEMV trait; nullptr
⇒ no repack ⇒ prefill `mul_mat` falls back to the standard per-(row,col) block-dot. Verbatim source
+ LIVE `__riscv_vlenb()=128` runtime dispatch probe (`dispatch_probe_raw.txt`):

- **q2_K** — HAS a `ggml_cpu_has_riscv_v()` branch, but `switch(__riscv_vlenb()*8){ case 128:
  {break;} // TODO … }` ⇒ **nullptr @128** (only `case 256` returns `&q2_K_16x1_q8_K`). Same
  precedent as q4_K.
- **q6_K** — NO `ggml_cpu_has_riscv_v()` branch at all (only NEON `matmul_int8`/`dotprod`). On a
  non-NEON RISC-V host ⇒ **nullptr @every VLEN**. Same precedent as q5_K.
- **q3_K** — **NOT PRESENT in the selector at all** (no `else if (cur->type == GGML_TYPE_Q3_K)`;
  `grep GGML_TYPE_Q3_K|q3_K_16x1|repack_q3_K repack.cpp = 0`) and **no q3_K repack kernel exists
  anywhere in ggml** (arch/riscv/repack.cpp has only q4_K/q2_K 16x1). ⇒ **nullptr @every VLEN** —
  the STRONGEST absence of the family.

⇒ **L1 STRUCTURAL opponent-absence CONFIRMED for all three**: at VLEN128 none has a working online
repack; all fall back to `ggml_vec_dot_q{6,2,3}_K_q8_K`.

## 3. Prefill paired A/B (ours repack GEMM vs opponent's REAL linked block-dot)

Protocol: PAIRED interleaved A/B, N=10 reps, cold-flush 224 MiB (>3×L3=64 MiB) before each timed
region, single full-GEMM call, best+median, K=2048 nc=512 nr∈{16,64,256}, core 8. Opponent linked
from the board's own `libggml-cpu.so`. Raw: `prefill_paired.txt` / `.csv`.

| fmt | nr | OURS GMAC/s | OPP GMAC/s | ratio ours/opp |
|---|---:|---:|---:|---:|
| q6_K | 16/64/256 | 0.65 / 0.67 / 0.68 | 3.38 / 3.71 / 3.77 | **0.18–0.19** |
| q2_K | 16/64/256 | 0.92 / 0.95 / 0.96 | 4.57 / 4.46 / 4.62 | **0.20–0.21** |
| q3_K | 16/64/256 | 0.60 / 0.62 / 0.62 | 3.38 / 3.50 / 3.53 | **0.176** |

**Result: our repack GEMM is ~5–6× SLOWER than the opponent block-dot at VLEN128 — a throughput
LOSS, NOT a beat.** This is the honest, thesis-coherent finding:

- **[NG-4]** fully honored (and moot — a loss cannot be a beat). The eight [PERF-1] gates are not
  walked; single-core; opponent = single-thread block-dot loop (kernel-axis proxy).
- **Why the loss (not a cache/harness artifact — 5× is far outside any cache regime):** the q{6,2,3}
  repack GEMM kernels are FIRST-CONSTRUCTION, correctness-first, **un-pipelined** — the emitter is
  all hand-written with no auto-vectorization/scheduling. `kernels_objdump_seal.txt`: q2_K GEMM emits
  **77 vsetvli** (the min-term dual d/dmin fold), q6_K 21, q3_K 13; ours runs at ~0.2–0.3 MAC/cycle,
  ~20× off VLEN128 fp32 peak.
- **The opponent block-dot is MATURE, not naive:** `nm libggml-cpu.so` shows dedicated
  **`ggml_vec_dot_q2_K_q8_K_vl128`** and **`ggml_vec_dot_q3_K_q8_K_vl128`** hand-tuned kernels (q6_K
  falls to its main/generic). So the L1 *structural opening* (no online repack) exists, but ggml's
  VLEN128-tuned block-dot fallback is a strong opponent — mirrors the q4_K sibling (parity vs a
  hand-tuned `_vl128`), amplified here because our new kernels are less mature than q4_K's.

## Bottom line

- **Numerical (load-bearing deliverable):** SILICON byte-exact-integer + bounded-norm — CONFIRMED
  for q6_K, q2_K, q3_K (GEVM+GEMM). Model-level certificate upgraded to hardware-run. ✓
- **Opponent-absence (L1 structural):** PROVEN — no working repack @VLEN128 for any of the three
  (q3_K has zero repack support anywhere in ggml). ✓
- **Prefill:** our first-construction repack GEMM **loses ~5–6×** to ggml's VLEN128 block-dot.
  NOT a beat, NOT a candidate-parity — a documented throughput deficit (marginal-cost / maturity
  evidence: correctness first, perf later). Nothing here was committed / git add-ed.
