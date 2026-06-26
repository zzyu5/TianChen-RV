# k1 repack fill — 自查表「表2 repack」k1 空格 — FINDING

**Date:** 2026-06-25
**Board:** `ssh k1` (Spacemit X60, RISC-V, **VLEN256**, 8 cores, ISA has `v zvfh ime`).
Grounded live: `__riscv_vlenb()=32 → VLEN=256`; `VLMAX e8m1=32 e8mf2=16`. Serial,
**k1-only** (rvv left untouched per discipline). `taskset -c 0` (within the allowed
0-3 set), best-of-reps **min** latency. **NO lib/ change, NO rebuild** — emitted with
the existing host `build/bin/tcrv-opt` (built 06-25 21:52).
**Correctness gate passed BEFORE any perf was reported, for the exact anchor whose
ratio is reported.**

**Emit method (host, x86, LLVM/MLIR 20, NO rebuild):**
`build/bin/tcrv-opt <input>.mlir --tcrv-rvv-lower-to-emitc | mlir-translate --mlir-to-cpp`.
Compiled on k1 with `clang++ -O3 -march=rv64gcv_zvfh -mabi=lp64d -ffp-contract=fast`.

## Anchor arithmetic at VLEN256 (the load-bearing correction over the rvv findings)

Lanes = VLEN/SEW · LMUL. At VLEN256, e8: **mf2 = 16 lanes**, **m1 = 32 lanes**. A
16-column repack group is **exactly one mf2 strip** (`half_lanes=16`) — the
VLEN256-NATIVE single strip. **NOTE: h16 is a MANUAL `half_lanes` stamp on the input
op** (sed-edited the attribute; the default q4_K/q5_0 emit is h8, and there is NO
VLEN→half_lanes auto-selection — consistent with the `repack-winA-always-mf2` record).
VLEN-aware auto-selection of `half_lanes` is a separate Win-A gap, NOT done here; this
task emits the native strip by hand and measures it. The rvv (VLEN128) findings used
`half_lanes=8` (mf2=8
lanes there, two strips); m1 (h16, 32 lanes) OVER-provisions on k1 (32 lanes for a
16-col group). So the k1 anchors are: **mf2/h16 = native single strip** (best
expected), **m1/h16 = over-provisioned** (expected worse). LMUL-disjointness is
SOURCE-level (emitter grep): mf2 emits `vle8_v_u8mf2`+`vwmacc_vx_i16m1`; m1 emits
`vle8_v_u8m1`+`vwmacc_vx_i16m2` (objdump can't decode RVV vtype on this part — same
caveat as all prior board findings).

---

## CELL 1 (REQUIRED, 没做 #3) — q4_K repack GEVM k1 正确性 oracle = **PASS, WORST_NORM 7.07e-7**

The q4_K repack GEVM emitter `tcrv_rvv.repack_gemv_q4_K_q8_K`, emitted at the
**VLEN256-native `half_lanes=16` single mf2 strip** (the q4_K verifier accepts
`half_lanes ∈ {8,16}`; h16 emits exactly HALF the static vle8/vwmacc of the rvv h8
two-strip path — 288 vs 576 vle8, 512 vs 1024 vwmacc — i.e. one 16-lane strip), is
**NUMERICALLY VALIDATED on k1**.

INDEPENDENT scalar q4_K dequant-matmul oracle (computes from the ORIGINAL pre-repack
`block_q4_K`, feeds the kernel the SAME blocks repacked via `make_block_q4_Kx16` —
different layouts, same values, a repack-inversion bug can't agree). Bar: norm < 1e-4.

```
# shape(nc x n)  max_abs_err    rms(ref)     norm         rel          verdict
  16x256         1.5259e-05     9.7579e+01   1.5637e-07   1.1721e-07   PASS
  16x4096        6.1035e-05     2.8748e+02   2.1231e-07   1.1542e-06   PASS
  16x11008       2.4414e-04     6.3228e+02   3.8613e-07   4.7615e-07   PASS
  32x256         7.6294e-06     5.9304e+01   1.2865e-07   1.2407e-07   PASS
  32x4096        6.1035e-05     2.5113e+02   2.4304e-07   1.0671e-06   PASS
  256x256        1.5259e-05     6.6739e+01   2.2863e-07   1.7561e-07   PASS
  256x4096       2.4414e-04     8.3691e+02   2.9172e-07   2.4680e-07   PASS
  256x11008      3.6621e-04     5.1834e+02   7.0651e-07   1.4713e-04   PASS
WORST_NORM 7.0651e-07  VERDICT PASS
```
Both negative controls fire HARD (proving the test would catch the bug class):
- **NOMIN** (zero the min/bsums term in ref): norm 1.156e-1 → min path exercised at
  **396,124×** margin over true-ref norm. bsums live: mean|bsum|=186.7, max|bsum|=716.
- **PERM** (rotate per-sub-block 6-bit scale +1 in ref): norm 6.07e-1 → distinct
  sub-block-scale pairing exercised at **2,080,930×** margin.

norm ≡ rvv (7.07e-7) because the integer dot is EXACT and only the fp16 scale-fold
rounds — VLEN-invariant. This fills 没做 #3 ("q4_K repack k1 正确性 oracle 还没跑")
**on the VLEN256-native strip** (not just a port of the rvv mf2 re-derivation).
Artifacts: `q4k/oracle_gemv_q4K.cpp`, `q4k/k_gemv_q4K_h16.cpp`, `q4k/q4_K-oracle-k1-h16.log`.

---

## CELL 2 (REQUIRED, 表2 q5_0 repack k1) — q5_0 repack GEVM k1 micro = **0.821× LOSS** (best anchor)

Baseline = ggml's REAL shipped RVV q5_0 **block-dot** (`ggml_vec_dot_q5_0_q8_0`, the
unified `__riscv_v` body; `vlenb==32` takes the VLEN256 `vslideup`+`vlmul_ext` branch
— ggml ships NO q5_0 repack, so the per-column block-dot IS the methodology-correct
baseline) run once per output column (16). Our repack GEVM does all 16 columns at
once (block-as-lane, NO per-block vredsum).

**Byte-exact gate (BEFORE perf): norm = 0.000e+00 PASS at NC∈{16,32,64,336}, BOTH
anchors** (mf2 AND m1) vs a CANONICAL ggml ref reconstructing from PLAIN pre-repacked
`block_q5_0`. SANITY ours-GEVM == ggml-per-col max_rel = 0.000e+00.

| anchor | ours (ns) | ggml block-dot ×16col (ns) | **RATIO ggml/ours** | verdict |
|---|---:|---:|---:|---|
| **mf2 (h16, native 16-lane single strip)** | 51118.9 | 41965.3 | **0.821** | LOSS (ggml ~1.22× faster) |
| m1 (h16, over-provisioned 32-lane) | 74625.6 | 41948.9 | **0.562** | LOSS (worse) |

**Best anchor (native mf2) = 0.821× LOSS.** Regime: k1 VLEN256, clang-18 -O3,
`taskset -c 0`, baseline = `ggml_vec_dot_q5_0_q8_0` per-column. The setup HANDICAPS
ggml / FAVORS ours (ggml re-streams the activation 16× per group, ours reads it once)
— so the comparison is generous to US, and we STILL lose. That makes the LOSS MORE
robust (remove ggml's handicap → ggml gets faster → we lose by more), and pins it to
the COMPUTE axis, not a memory-traffic artifact.

**Why LOSS (matches the recorded pattern):** q5_0 is COMPUTE-bound, not gather-heavy.
ggml's block-dot folds the qh in ONE wide masked `vsub` over the full 32-elem i8 + one
`vwmul`/`vwredsum`; our repack GEVM must lane-wise EXPAND the transposed qh mask
(vid/vsrl/vand/vsll/vncvt) per nibble step. This is the recorded N3 pattern —
"gather-heavy WINS (our split-gather), compute-bound LOSES (ggml wide LMUL)";
q5_0 is compute-bound. m1 (32-lane over-provision) is worse, as expected on this
microarch (m1 = more uops regardless of kernel, per the `repack-winA-always-mf2`
record). **k1 0.821× is LESS bad than rvv's 0.769×** — sensible: the native 16-lane
strip fits VLEN256 exactly, while rvv VLEN128 splits into two 8-lane strips.
Artifacts: `q5_0/winB_micro_q5_0.cpp`, `q5_0/k_gemv_q5_0_h16-{mf2,m1}.cpp`,
`q5_0/verify_q5_0_k1_{mf2,m1}.log`, `q5_0/winB_q5_0_k1_{mf2,m1}.log`.

---

## CELL 3 (若有余力, 表2 q4_0 repack prefill k1) — q4_0 repack GEMM k1 = **正确✓ + 0.997× TIE**

q4_0 IS a ggml-shipped repack GEMM, so this used ggml's OWN real vectorized GEMM as
the perf baseline (NOT scalar) — linked against k1's built `libggml-cpu.so`
(`/data/k1build/bin`).

**Correctness gate (BEFORE perf): norm = 9.06e-6 PASS** (bar 1e-4) — ours h16 vs
ggml's `ggml_gemm_q4_0_16x1_q8_0_generic` reference (5 trials × n∈{64,256,4096,11008,
14336}, max_abs_err 1.14e-4, rms 12.63, max_per_elem_rel 2.4e-3 on near-cancellation
elems). (Harness printf hard-codes the label "VLEN=128"; the run is on k1 VLEN256,
h16 emit.)

**Perf micro (prefill GEMM, nr=4 × nc=16, n=2048):** ours h16 vs ggml's REAL
vectorized `ggml_gemm_q4_0_16x1_q8_0` (SAME x16 repack layout). SANITY ours==ggml
max_rel = 0.000e+00 (byte-exact). Stable across 3 runs.

| run | ours h16 (ns) | ggml real 16x1 GEMM (ns) | **RATIO ggml/ours** | verdict |
|---|---:|---:|---:|---|
| run1 | 15692.3 | 15637.7 | **0.997** | TIE |
| run2 | 15861.6 | 15804.7 | **0.996** | TIE |
| run3 | 15690.5 | 15634.4 | **0.996** | TIE |

Stable TIE across 3 runs (0.996–0.997), byte-exact (SANITY max_rel 0) every run.

**TIE (parity).** Honest framing: our compiler-emitted q4_0 repack GEMM converges to
essentially ggml's own hand-written vectorized 16x1 GEMM at VLEN256 (byte-exact output
AND ~equal latency). This is the "parity = adopting ggml's own instructions for the
same layout" outcome — a correct, mature emit, NOT a perf win. (Distinct from the
prior rvv GEVM repack q4_0 e2e win 2.6×, which was a memory-layout-change story at
decode; this is the compute-bound prefill GEMM vs ggml's matched kernel.)
Artifacts: `q4_0/winB_micro_q4_0_gemm.cpp`, `q4_0/k_gemm_q4_0_h16.cpp`,
`q4_0/q4_0-gemm-correctness-k1.log`, `q4_0/winB_q4_0_gemm_k1.log`.

---

## Self-check-table cells filled (表2 repack, k1 columns)

| row | cell | filled value |
|---|---|---|
| q4_K repack | k1 正确性 oracle (没做 #3) | **PASS, WORST_NORM 7.07e-7** (VLEN256-native h16 strip) |
| q5_0 repack(decode) | k1 micro | **0.821× LOSS** (best anchor mf2; m1 0.562×) byte-exact norm 0 |
| q4_0 repack(prefill) | k1 GEMM | **正确✓ (norm 9.06e-6) + 0.997× TIE** vs ggml real 16x1 |

## What was NOT measured / why
- **q4_K repack k1 MICRO (vs ggml) NOT done** — out of the task's required scope
  (it asked for the q4_K *oracle*, done). FLAGGED as follow-up: at VLEN256 the ggml
  baseline FLIPS to ggml's OWN `ggml_gemv_q4_K_16x1_q8_K` repack (which only RUNS
  correctly at VLEN≥256 — at VLEN128 it returned nullptr → fell back to block-dot).
  So a k1 q4_K micro is a DIFFERENT, harder measurement (vs a stronger,
  now-actually-running competitor) than the rvv q4_K winB (0.55×/0.74× vs block-dot)
  — not silently folded in.
- **q5_0 / q4_0 e2e on k1 NOT measured** — micro/e2e are separate axes; no e2e claim.
  (Prior k1 q4_0 Win-B e2e was 0.74× — X60 autovec beats the repack at decode; this
  finding is the micro/correctness axis only.)
