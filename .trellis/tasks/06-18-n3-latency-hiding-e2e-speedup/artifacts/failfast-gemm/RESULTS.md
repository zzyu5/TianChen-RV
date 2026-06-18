# FAIL-FAST physics probe #2: q4_0 GEMM decode-amortization ceiling on `ssh rvv`

**Question:** GEMV (decode, batch=1) caps at ~1.22× (reduction-bound, prior probe).
Does the q4_0 **GEMM** (prefill / M output columns) clear ~1.5× over ggml's
`ggml_vec_dot_q4_0_q8_0`, via **DECODE AMORTIZATION** across M activation columns —
without touching the per-block reduction wall?

**Bottom line: YES. In-layer q4_0 GEMM clears 1.5× by decode amortization.**
It **crosses 1.5× at M = 8** (M=6 is just under), and **plateaus at ~1.56× (n=11008)
to ~1.59× (n=4096)** by M≈12–16. The win is real, accuracy-preserving (bit-exact to
ggml here), and robust to the fold choice. The plateau sits at ~1.56–1.59×, **not**
higher, because the per-block cross-lane `vwredsum` reduction wall (the 1.38× GEMV
ceiling) is unchanged — decode amortization buys exactly the decode fraction back and
no more.

---

## Host facts (stated exactly, this run)

- Host: `ssh rvv`, user `ubuntu`, riscv64, 64 cores. Single-thread kernels,
  `taskset -c 3`, build with low `-j`.
- ISA: `rv64imafdcv` + `zfh zfhmin zvfh zvfhmin zba zbb zbc zbs zfa zicond zca zcb
  zcd zve32f/x zve64d/f/x ...` (full `/proc/cpuinfo` isa string captured). **VLEN=128**
  (`m1` i32 = 4 lanes; `m4` f32 = 16 lanes).
- Clock: governor is **`ondemand`** and we have **NO permission** to pin
  `performance` (verified: write to `scaling_governor` → EACCES). The core idles
  at ~1.1–1.4 GHz. The harness **forces a clock-ramp warmup** (spin on the real V0
  kernel until `scaling_cur_freq` holds **2 600 000 kHz** for 8 consecutive reads)
  before any measurement, and samples the clock every rep. **Observed clock during
  the canonical run (run4): min = max = 2 600 000 kHz** for both n — fully pinned.
- **Clock honesty:** speedups are the **ratio** V0_ns/GEMM_ns, measured in the SAME
  interleaved best-of-N run, so they are **clock-invariant**. We additionally
  re-measure V0 at the END of each size's sweep (drift check): run4 V0_start vs
  V0_end = 1070.5/1069.9 (n=4096), 2856.8/2855.4 (n=11008) — **no drift**. The
  re-measured **V0 = 1070.5 ns reproduces the prior probe's 2.6 GHz V0 = 1070.8 ns**
  to 0.03%, and **V4-GEMV = 1.225× reproduces the prior 1.22×** — this board run is
  the same silicon/clock as failfast-q4_0.
- Toolchain: Ubuntu clang 18.1.3. Build: `clang -O3 -march=rv64gcv_zfh_zvfh
  -ffp-contract=fast`, kernels and harness as **separate TUs** (so -O3 cannot hoist
  the loop-invariant work out of the best-of-N loop).
- Timing: best-of-**200** min, **2000** iters/rep, warmup, `taskset -c 3`,
  n ∈ {4096, 11008} (llama-2-7B hidden / ffn). **ns reported PER OUTPUT element**
  (= total kernel ns / M for GEMM). Raw board stdout: `board_run4.log` (canonical),
  `board_run1..3.log` (earlier stages, incl. a cold-V0 run1 documented below).

Baseline **V0 = ggml's REAL RVV `ggml_vec_dot_q4_0_q8_0`**, verbatim, own TU. It is
both the perf baseline and the bit-faithful reference.

---

## Activation layout assumption (block-major pack)

The GEMM inner loop reuses one decoded weight block across M activation columns. For
that loop to be cache-friendly, column j's block-ib q8_0 record must be **contiguous**
with the other columns' block-ib records. We pack activations **block-major**:
`vyp[(ib*M + j)*YS]` holds column j's block-ib record. The GEMM then streams `vyp`
sequentially — **no strided loads** (the layout that killed TRANSPOSED at 0.14× in the
prior probe). This repack amortizes across the full weight matrix in real prefill; we
treat activations as **pre-packed** and the weight row as standard q4_0. (Empirically
confirmed no cache cliff: GEMM_M16's per-output is still improving over M12.)

---

## Probe 1 — cost-axis isolation (the decode discriminator)

These are **cost-only** probes (correctness irrelevant); they isolate where the
540-ns-ish per-output compute goes. n=4096, ns/out:

| probe | ns/out @4096 | ns/out @11008 | what it isolates |
|---|---:|---:|---|
| **decode_ONLY** | **329.3** | 861.6 | weight nibble unpack ONLY: `vle8 tx`, `vand`, `vsrl`, 2×`vsub`. No activation load, no product, no reduce. **This is the amortizable work.** |
| vwmacc_ONLY | 800.9 | 2131.6 | product+accumulate: activation loads + `vwmul`/`vwmacc` into ONE serial i16 acc, no reduce (decode hoisted out). |
| V4_GEMV (ref) | 874.6 | 2355.0 | best per-output GEMV (8 indep redsums + vector-store fold). |
| V0_ggml (ref) | 1070.5 | 2856.8 | ggml baseline per output. |

**Decode fraction of the per-output cost ≈ 37.6 %** (329 / 875 of V4). 
**Caveat (do not over-read):** decode_ONLY is an **upper bound** — it includes a
`vxor`+`vadd` anti-DCE sink that real decode does not pay; and vwmacc_ONLY is a
single-accumulator latency-bound probe, so it overstates the product cost. Use them
as *mechanism color*, not exact accounting.

**The clock-invariant discriminator (re-derived from THIS run, not the stale 714 ns):**
GEMM clears 1.5× asymptotically iff amortizable weight-decode ≥ `V4 − V0/1.5`
= 874.6 − 1070.5/1.5 = **161 ns** (≈ 18 % of V4). Measured decode (≤329 ns, ~37 %)
is **well above** the bar → the direct M-sweep should — and does — cross 1.5×.

---

## Probe 2 — q4_0 GEMM M-sweep (PRIMARY kernel: `kern_gemm`)

`kern_gemm` = V4 *intent* for GEMM: outer loop over weight blocks, **decode each
weight block's nibbles ONCE**, inner loop over M columns **reusing the decoded
`v0/v1`**, one `vwredsum` per (block,column), folded into **M independent fp32
accumulators** carried across the block loop (no serial cross-block fold). The M
reductions for a block are naturally independent (same weights, different
activations) — that *is* V4's independent-reduction trick, free from the M columns.

ns **per output**, best-of-200 @ 2.6 GHz (run4, clock pinned):

| M | ns/out @4096 | speedup @4096 | ns/out @11008 | speedup @11008 |
|---:|---:|---:|---:|---:|
| 1 | 1334.4 | 0.803 | 3506.5 | 0.815 |
| 2 | 1018.7 | 1.051 | 2701.6 | 1.058 |
| 4 |  794.2 | 1.348 | 2135.4 | 1.338 |
| 6 |  748.8 | 1.430 | 2026.8 | 1.410 |
| **8** | **685.8** | **1.562** | **1875.9** | **1.523** |
| 12 |  678.6 | 1.578 | 1836.2 | 1.556 |
| 16 |  672.5 | **1.593** | 1831.2 | **1.560** |

- **Crosses 1.5× at M = 8** (M=6 is 1.43×/1.41×, just under). 
- **Plateaus at ~1.59× (n=4096) / ~1.56× (n=11008)** by M≈12; M16 adds <1%.
- **M=1 is 0.80× (slower than V0).** Expected, not a bug: at M=1 there is no decode
  reuse *and* the kernel has a single serial reduction chain per block — neither
  column-ILP nor V4's block-batched-ILP. The M-curve is therefore **decode
  amortization PLUS ILP recovery** mixed: M=1→plateau drops ~660 ns @4096, of which
  decode is ≤329 ns; the remainder is overlapping M independent reductions/folds.

---

## Probe 3 — accuracy (fp64 truth, per output)

Ground truth = fp64 fold per output (sumi exact integer; fp16 scales widen exactly).
2000 trials × {4096, 11008} × 8 columns, realistic log-uniform fp16 scales in
[2^-12, 2^0], full-range payload bytes.

| GEMM kernel | max rel-err fp64 | mean rel-err fp64 | max rel-err vs V0 | mean rel-err vs V0 |
|---|---:|---:|---:|---:|
| `kern_gemm` (scalar fold, primary) | 8.71e-4 | 8.03e-7 | **0.000** | **0.000** |
| `kern_gemm_v4fold` (vector fold) | 8.71e-4 | 8.03e-7 | **0.000** | **0.000** |

Both GEMM kernels are **bit-exact to ggml V0** (rel-err vs V0 = 0): each column's
per-block accumulation order matches V0's serial fold. fp64 rel-err (~1e-6 mean,
8.7e-4 max) is the same magnitude as V0's own fp64 drift. **Accuracy-preserving,
drop-in.** Decode amortization is purely a *scheduling* change; it does not perturb
the numerics.

---

## Negative result — the literal V4 vector-store fold REGRESSES in GEMM

We also built `kern_gemm_v4fold`: the literal V4 fold (per-(block,col) `vwredsum` →
`vse32` lane-0 store, NO `vmv.x.s`; one `vfmacc` over an M-lane `f32m4` accumulator
per block). The prior probe found this *helped* GEMV. In GEMM it **regresses at low M**:

| M | v4fold ns/out @11008 | scalar ns/out @11008 |
|---:|---:|---:|
| 1 | 10169 (0.28×) | 3507 (0.82×) |
| 4 |  3354 (0.85×) | 2135 (1.34×) |
| 8 |  2402 (1.19×) | 1876 (1.52×) |
| 16 | 1597 (1.79×) | 1831 (1.56×) |

**Why:** the literal fold interleaves a `vl=1` lane-store with the `vl=16` reduction
*inside the inner M-loop*, forcing a `vsetvli` reconfiguration every column — measured
**15 `vsetvli` per block-iter vs 5** for the scalar kernel (objdump). On this in-order
board `vsetvli` toggling serializes; the thrash dominates at small M.

**Smoking gun for excluding the v4fold M=16 number:** v4fold's M=12 and M=16
**totals** are nearly identical — **9236 vs 9240 ns @4096** (and 25129 vs 25560
@11008). 33% more columns producing ~0% more total work is physically impossible —
M=16 is hitting a degenerate `vsetvli` fast path (`f32m4` is exactly full at the
data's natural width), **not** a real compute ceiling. By contrast the **scalar**
GEMM's M12/M16 totals scale correctly (8143 → 10759 ≈ ×1.32), independently
confirming scalar is the clean, trustworthy kernel. So v4fold's 1.79–1.86× at M=16 is
an artifact and we **do not report it as a result.**

**The lesson (a real N3 cross-kernel-shape finding):** V4's vector-store gather was a
*GEMV-specific* mechanism to kill a **serial sumf fold**. GEMM has no serial fold —
its M `acc[j]` are already independent and `vmv.x.s` keeps the inner loop at `vl=16`
(no thrash). So the *scalar* fold is the correct realization of "V4 intent" for GEMM,
and the literal V4 fold is the wrong transfer. **Decode amortization clears 1.5×
regardless of the fold choice** — the verdict is robust.

---

## Bottom line

- **Does in-layer q4_0 GEMM clear 1.5× by decode amortization? YES.**
- **At what M?** Crosses 1.5× at **M = 8** (M=6 ≈ 1.42×, just under) for both n.
- **Plateau value:** ~**1.56× (n=11008) / ~1.59× (n=4096)**, reached by M≈12; M16
  adds <1%. (Per-output ~1830 ns @11008, ~673 ns @4096.)
- **Mechanism:** decode is ~37% of per-output cost (upper bound); amortizing it
  across M columns recovers most of that, lifting 1.22× (GEMV) → ~1.56–1.59× (GEMM).
  The curve also benefits from ILP recovery (M independent reductions overlap).
- **The limiting axis at the plateau: the per-block `vwredsum` reduction wall PLUS a
  secondary per-column extract/fold floor.** Decode amortization removes the decode
  cost but cannot touch the mandatory per-block reduction (each block carries its own
  fp16 scale). Quantitatively: GEMV's reduction floor `RED_k4` = 774 ns/out @4096
  (prior probe). The GEMM plateau is **672 ns** — only ~100 ns below that floor, **not**
  the ~300 ns that fully cashing out decode (≤329 ns) would predict. The ~200 ns
  residual is the per-column cost `RED_k4` never paid: `vmv.x.s` extract + scalar fp
  fold + traffic to the runtime-M `acc[]` array (which is not register-promoted — this
  also explains scalar M=1 being 0.80×, ~25% slower than V0, which keeps `sumf` in a
  register). So the plateau is **reduction-bound with a secondary per-column-fold
  floor**, and the 1.98× raw-compute ceiling stays unreachable. (Nailing the exact GEMM
  reduction floor would need a RED-style probe with decode reused across M; beyond a
  fail-fast probe.)
- **Accuracy:** bit-exact to ggml (rel-err vs V0 = 0); fp64 drift ≈ V0's. Drop-in.

So unlike the q4_0 **GEMV** result (caps ~1.22×, fail), the q4_0 **GEMM** result is a
**pass**: prefill/multi-column decode amortization clears 1.5× at M≥8 and plateaus
~1.56–1.59× on this rvv board.

---

## Files

- `kernels.c` — V0 (ggml-real), V4-GEMV (per-output reference), `kern_gemm`
  (PRIMARY: scalar-fold GEMM, decode reuse), `kern_gemm_v4fold` (literal V4 fold,
  documented vsetvli-thrash negative), `kern_decode_only`, `kern_vwmacc_only`.
- `harness.c` — clock-ramp warmup + V0 drift check + best-of-N timing (ns/out) +
  fp64-anchored per-output accuracy + block-major activation packer.
- `board_run4.log` — **canonical** clean run (clock pinned 2.6 GHz both n, V0 no drift).
- `board_run2.log` — earlier clean run (scalar GEMM only), agrees with run4.
- `board_run3.log` — first run with both GEMM kernels; @4096 V0 was cold (1134 vs
  drift-end 1070) — superseded by run4's stronger warmup. Kept for provenance.
- `board_run1.log` — first run, **cold-V0 denominator** (V0=1952 ns, ramping from
  idle) → @4096 speedups inflated ~80%; this is the bug that motivated the warmup +
  drift-check. Kept only to document the methodology fix.
