# G7 §3 — M-sweep batch-regime characterization + kernel-axis micro

**Axis**: kernel-axis micro M-sweep (SECOND track). **INDEPENDENT of perf-covered 9/83 — no互推.**
M=4/8 kernel win **≠ e2e** (needs batched-e2e to verify transduction; this役 gives potential判 only).
Byte-exact hard gate: **PASSED** — all fresh runs 0 mismatch / 0 ULP vs stock-ggml ZERO-MODEL oracle.
Boards: rvv (gcc-15.2, VLEN128, 2.6 GHz, core8-15, co-tenant vLLM core0,1) · k1 (clang-18, VLEN256, 1.6 GHz, core0-3). Symmetric compilers per board. Cold = 224 MiB flush / P-tile pool. Load-gated, pinned.

## 0. Headline crossover conclusion (roofline位置 M=1 vs M=8)

The compute-bound crossover is **REAL but board- and format-specific**, and it operates through
**cold weight-load amortization across the M batched rows** (the first row streams weight cold from
DRAM; rows 2..M reuse it warm from L2/L3) — *not* through any kernel-algorithm change. The batching
lever therefore fires **only** where (a) the kernel is memory-bound at M=1 AND (b) the two sides
differ in memory-vs-compute regime.

- **rvv block-dot/GEVM, weight-heavy formats (all K-quant + q8_0)**: MEMORY-bound at M=1 →
  ours GMAC/s **rises +100…+276 %** M1→M32, hitting the compute roofline by **M≈8–16**.
- **rvv block-dot, light 4-bit flat (q4_0/q4_1/q5_0/q5_1)**: already COMPUTE-bound at M=1 (weight
  footprint tiny, 18–24 B/block) → GMAC/s flat, ratio flat.
- **k1 block-dot, ALL formats**: already COMPUTE-bound at M=1 (1.6 GHz clock makes decode the
  bottleneck at every M) → GMAC/s flat, **ratio flat → the M=1 kernel verdict holds at all M**.
- **repack-GEMM (both boards)**: ours reuses weight *algorithmically* → compute-bound at M=4 already
  → GMAC/s flat; ratio-vs-hand-brick (q4_K@k1) **flat 0.622** (compute-quality gap, batching-immune).

**Verdict on the §3 hypothesis** ("M=1 memory-bound win-washout → M=4/8 compute-bound win-transduce"):
- **CONFIRMED, one clean case: q5_K@rvv** (GEVM/block-dot). ratio 0.896 (M1 LOSS) → **1.115 (M4 WIN)**
  → 1.192 (M8) → 1.253 (M16) → **1.294 (M32)**. Crosses parity between **M2 and M4**.
- **REFUTED as a universal lever**: on k1 batching changes nothing (compute-bound at M=1); for
  repack-GEMM-vs-hand-brick the loss is a fixed compute-quality gap; losing rvv K-quants only *narrow*
  the loss without crossing.

## 1. M-sweep tables (ratio ours/opp, cold-median; ours GMAC/s)

### 1a. rvv block-dot / GEVM (VLEN128, gcc-15.2 symmetric) — ours block-dot vs as-shipped ggml_vec_dot
| fmt | M1 | M2 | M4 | M8 | M16 | M32 | ours GMAC/s M1→M32 | regime@M1 |
|---|--|--|--|--|--|--|--|--|
| **q5_K** | 0.896 | 0.999 | **1.115** | **1.192** | **1.253** | **1.294** | 0.58→1.74 (+202 %) | **MEM→COMPUTE (crosses)** |
| q4_0 | 0.909 | 0.942 | 0.954 | 0.957 | 0.961 | 0.962 | 1.12→1.18 (+6 %) | compute (flat, ~parity) |
| q4_1 | 1.021 | 1.020 | 1.020 | 1.021 | 1.021 | 1.026 | 1.29→1.29 (flat) | compute (flat WIN-by-adoption) |
| q8_0 | 1.047 | 1.031 | 1.017 | 0.988 | 0.985 | 0.981 | 0.29→1.08 (+276 %) | MEM→COMPUTE, ratio ↓ (opp gains) |
| q5_0 | 0.264 | 0.266 | 0.262 | 0.262 | 0.262 | 0.263 | 0.81→0.81 (flat) | compute (flat LOSS) |
| q5_1 | 0.272 | 0.271 | 0.269 | 0.269 | 0.269 | 0.269 | 0.77→0.78 (flat) | compute (flat LOSS) |
| q4_K | 0.165 | 0.234 | 0.301 | 0.347 | 0.381 | 0.399 | 0.78→1.94 (+149 %) | MEM→COMPUTE, ratio ↑ stays LOSS |
| q2_K | 0.252 | 0.331 | 0.404 | 0.446 | 0.464 | 0.485 | 1.13→2.28 (+102 %) | MEM→COMPUTE, LOSS |
| q3_K | 0.277 | 0.367 | 0.446 | 0.502 | 0.557 | 0.574 | 0.97→2.00 (+107 %) | MEM→COMPUTE, LOSS |
| q6_K | 0.282 | 0.334 | 0.381 | 0.423 | 0.451 | 0.471 | 0.65→1.78 (+175 %) | MEM→COMPUTE, LOSS |

**Decisive mechanism proof (q5_K@rvv)**: the **HOT** ratio (weight warm in cache = compute-bound) is
**1.37 already at M=1** and stays ~1.35 for all M. The **COLD** ratio starts at 0.896 (M=1, weight
streamed cold from DRAM = memory-bound → ours' compute win masked) and **converges UP toward the HOT
compute-bound asymptote** as M amortizes the cold load: 0.896 → 0.999 → 1.115 → 1.192 → 1.253 → **1.294
(M32 ≈ HOT 1.35)**. i.e. batching is literally turning the cold/memory-bound case into the
hot/compute-bound case — the win was always there in compute, hidden only by the M=1 cold weight load.

### 1b. k1 block-dot / GEVM (VLEN256, clang-18 symmetric) — ratio FLAT in M (already compute-bound)
| fmt | M1 | M4 | M8 | M32 | ours GMAC/s M1→M32 | verdict |
|---|--|--|--|--|--|--|
| q5_K | 1.083 | 1.080 | 1.081 | 1.080 | 0.68→0.67 (flat) | **WIN @all M** (not batching-gated) |
| q4_1 | 1.010 | 1.008 | 1.009 | 1.008 | 1.14→1.13 (flat) | parity @all M |
| q8_0 | 0.981 | 0.984 | 0.983 | 0.982 | 1.09→1.09 (flat) | parity @all M |
| q4_0 | 0.984 | 0.983 | 0.982 | 0.984 | 1.10→1.11 (flat) | parity @all M |
| q4_K | 0.622 | 0.614 | 0.610 | 0.609 | 0.80→0.79 (flat) | LOSS @all M |
| q6_K | 0.559 | 0.553 | 0.556 | 0.789* | 0.68→0.64 (flat) | LOSS (*M32 = single-point co-tenant noise, disregard) |
| q2_K/q3_K/q5_0/q5_1 | 0.68/0.54/0.38/0.40 | … | … | flat | flat | LOSS @all M |

### 1c. rvv repack-GEMM (VLEN128) — ours repack-GEMM (weight reuse ∝M) vs opp block-dot [existing census nr 4/8/16/64]
| fmt | M4 | M8 | M16 | M64 | ours GMAC/s | note |
|---|--|--|--|--|--|--|
| q5_K | 0.680 | 0.759 | 0.782 | 0.767 | 1.04 flat | ratio ↑ (opp loses reuse), no cross; block-dot path (1b) is faster for batched q5_K |
| q4_K | 0.235 | 0.292 | 0.390 | 0.390 | 1.73 flat | ratio ↑, stays LOSS |
| q2_K | 0.367 | 0.354 | 0.369 | 0.389 | 1.80 flat | LOSS |
| q3_K | 0.052 | 0.052 | 0.051 | 0.050 | 0.17 flat | deep LOSS (slow emit) |
| q6_K | 0.040 | 0.039 | 0.039 | 0.038 | 0.14 flat | deep LOSS (slow emit) |

### 1d. q4_K@k1 repack-GEMM vl=8 core vs TRUE hand-brick 16x1 (VLEN256, clang-18 symmetric) — FRESH
| M(nr) | ratio ours/16x1 | ours GMAC/s | opp GMAC/s |
|---|--|--|--|
| 4 | 0.6315 | 3.67 | 5.82 |
| 8 | 0.6238 | 3.67 | 5.89 |
| 16 | 0.6220 | 3.70 | 5.95 |
| 32 | 0.6230 | 3.71 | 5.95 |

→ **FLAT 0.622** across M: both are algorithmic-weight-reuse repack-GEMMs (both compute-bound at M=1),
so batching is neutral. The 1.61× hand-brick lead is a compute-quality gap, NOT a memory-regime artifact.
(NOTE: this is the vl=8 core, kernel-sym LOSS. The sealed **vl=16** winner — the 1.197× kernel / 1.085× e2e
double-axis Win-K1-VLEN — has source `s6_q4K_vl16_sealed.c` md5 e437fd3b which was NOT available on-box;
its M-sweep is a targeted follow-up.)

## 2. Roofline classification (memory-bound vs compute-bound)

Signature used: does ours GMAC/s **rise** with M (memory-bound at low M, cold weight-load amortizes) or
stay **flat** (compute-bound at all M)?

- **MEMORY-bound at M=1 (rises to compute roofline as M↑)**: rvv block-dot q2_K,q3_K,q4_K,q5_K,q6_K,q8_0.
  Compute roofline reached by M≈8–16 (GMAC/s within ~10 % of M32 plateau).
- **COMPUTE-bound at all M (flat)**: rvv block-dot q4_0/q4_1/q5_0/q5_1; ALL k1 block-dot; ALL repack-GEMM
  (both boards); q4_K@k1 hand-brick both sides.
- Weight-GB/s in the repack-GEMM census (ours 0.13–0.15, opp 0.32–0.64 GB/s) is **far below** board DRAM
  peak → confirms repack-GEMM is compute-bound, not bandwidth-starved — so batching cannot unlock a
  memory-bound win there (kernel was never memory-bound).

## 3. e2e-transduce potential list (compute-bound-at-M≥4 ∧ win holds/grows)

| rank | cell | why | batched-e2e worth building? |
|---|---|---|---|
| **1** | **q5_K@rvv (GEVM/block-dot kernel)** | textbook M=1 LOSS (0.896) → M≥4 WIN (1.12–1.29); ours compute-superior, masked by M=1 memory-boundness | **YES — strongest** |
| 2 | q5_K@k1 (block-dot) | already WIN 1.08 flat; transduces at M=1 too but decode is mem-bound → batching de-risks | secondary |
| 3 | q4_1@rvv / q8_0@rvv / q4_0@rvv (block-dot) | flat parity; e2e transduction not batching-driven (verdict set at M=1) | low value (no batching lift) |
| — | q4_K@k1 (0.622 flat), q5_0/q5_1@rvv (0.26 flat), rvv repack-GEMM K-quants (<1) | compute-quality gap or non-crossing; batching does NOT create a win | **NO** |

**Recommendation to main会话**: if a batched-e2e harness is built, target **q5_K@rvv** first — it is the
single cell where the M-sweep predicts a memory→compute win transduction (M=1 washout → M≥4 win). All
other cells either already-win-flat (not batching-gated) or never cross.

## 4. Honesty / scope

- Kernel-axis micro M-sweep, second track, **independent of perf-covered 9/83** — no cross-inference.
- The rvv crossover is driven by the driver-loop (r-outer, c-inner) amortizing the cold weight load
  across M rows. A real batched `mul_mat` may or may not reproduce this depending on its tiling/traffic
  → this is a **potential** signal, NOT an e2e verdict. Batched-e2e is the next役 (main会话 decides harness).
- q6_K@k1 M32=0.789 is a single-point co-tenant-noise artifact (ours GMAC/s *dropped* while ratio jumped,
  loadavg 3.1) — disregarded.
- Byte-exact PASSED all fresh runs (0 mism / 0 ULP). No counts changed (perf-covered / certified /
  kernel-sym untouched). No emitter/ODS/lib edits. No git.

## 5. Board hygiene
- rvv: 0 stray benchmark procs (ps-exact), loadavg 2.05 (baseline vLLM co-tenant), ggml md5 d1adc634 test-pre==post (read-only), governor left performance, my scratch /tmp/g7_msweep_* removed.
- k1: 0 stray benchmark procs (ps-exact), loadavg 2.16, stock lib md5 871169a0 untouched (read-only), my /tmp/g7_msweep_vecdot_k1 removed (pre-existing /tmp/q4k_hb_resolve left as-found).
