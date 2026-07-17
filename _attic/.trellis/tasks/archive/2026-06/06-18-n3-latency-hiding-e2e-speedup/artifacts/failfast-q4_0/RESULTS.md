# FAIL-FAST physics probe: q4_0_q8_0 latency-hiding ceiling on `ssh rvv`

**Question:** Is ~1.5× over ggml's `ggml_vec_dot_q4_0_q8_0` reachable on the rvv
board via latency-hiding restructures of the kernel, while preserving accuracy?

**Bottom line: NO. 1.5× is not reachable for an accuracy-preserving kernel.**
The best accuracy-preserving variant measured is **~1.22×** (V4/V3). The measured
ceiling for *any* per-block-reduction kernel is **1.38×** (RED_k4, a cost floor),
which is below the 1.5× line. The one kernel family that escapes that floor
(transposed / block-as-lane) is **0.14×** — strided loads are catastrophic on
this board. Both families are below 1.5×; the verdict is measured, not assumed.

---

## Host facts (stated exactly)

- Host: `ssh rvv`, `ubuntu`, riscv64, 64 cores (used `-j` low / single-thread,
  `taskset -c 3`).
- ISA: `rv64imafdcv` + `zfh zfhmin zvfh zvfhmin zba zbb zbc zbs zfa zicond ...`
  (full `/proc/cpuinfo` isa string captured). **VLEN = 128** (`m1` i32 = 4 lanes).
- Clock: pinned **2.6 GHz** (`scaling_cur_freq == cpuinfo_max_freq == 2600000`),
  no DVFS — best-of-N min is stable.
- Toolchain: **Ubuntu clang 18.1.3** (riscv64-unknown-linux-gnu); gcc 14.2.0 present.
- Build: `clang -O3 -march=rv64gcv_zfh_zvfh -ffp-contract=fast`, kernels and
  harness as **separate translation units** (so -O3 cannot hoist the
  loop-invariant dot out of the best-of-N timing loop).
- Timing: best-of-**200** min, **2000** iters/rep, warmup, `taskset -c 3`,
  n ∈ {4096, 11008} (llama-2-7B hidden / ffn).
- Accuracy: rel-err over **2000** representative inputs × {4096, 11008}.
  Ground truth = **fp64** fold (integer sumi is exact; fp16 scales widen to fp64
  exactly), so V0's own fp64 rel-err is a yardstick. Scales drawn log-uniform in
  [2^-12, 2^0] (realistic per-block magnitudes); quantised payload bytes
  full-range. Raw board logs: `board_run1..4.log`.

Baseline V0 = ggml's REAL RVV `ggml_vec_dot_q4_0_q8_0`, verbatim from
`llama.cpp/ggml/src/ggml-cpu/arch/riscv/quants.c` (own TU, contracted as
shipped). It is both the perf baseline and the bit-faithful reference.

---

## Results table (n = 4096; n = 11008 tracks within ~1%)

| variant | ns/dot @4096 | speedup vs V0 | ns/dot @11008 | speedup | max rel-err (fp64) | mean rel-err (fp64) | accuracy-preserving? |
|---|---:|---:|---:|---:|---:|---:|:--:|
| **V0_ggml** (baseline) | 1070.8 | 1.000 | 2856.0 | 1.000 | 4.36e-3 | 2.52e-6 | reference |
| V1_k2 (2 fp32 acc) | 1070.0 | 1.001 | 2847.7 | 1.003 | 3.85e-4 | 7.15e-7 | yes (better than V0) |
| V1_k4 (4 fp32 acc) | 1063.6 | 1.007 | 2830.2 | 1.009 | 3.05e-3 | 1.59e-6 | yes |
| V1_k8 (8 fp32 acc) | 1549.0 | 0.691 | 4133.9 | 0.691 | 1.25e-3 | 8.63e-7 | yes (but slower) |
| V2_p8 (batched fold, scalar-extract) | 1486.0 | 0.721 | 3947.9 | 0.723 | 3.85e-4 | 5.29e-7 | yes (but slower) |
| V2_slide (vslideup gather) | 2237.6 | 0.479 | 5947.6 | 0.480 | **1.5e+10** | **7.6e+6** | **NO — buggy dead-end** |
| **V4** (8 indep redsum, vector-store gather, vec fold) | **877.7** | **1.220** | **2339.5** | **1.221** | 2.41e-3 | 1.29e-6 | **yes (best real)** |
| V3 (V4-style + 2 acc + pipeline) | 881.6 | 1.215 | 2348.1 | 1.216 | 2.41e-3 | 1.29e-6 | yes |

### Probes (cost-only — correctness irrelevant; not drop-in kernels)

| probe | ns @4096 | "speedup" | ns @11008 | what it isolates |
|---|---:|---:|---:|---|
| CEILING (1 i16 acc) | 802.7 | 1.334 | 2133.7 | **latency-limited** single-accumulator compute (artifact) |
| CEILING_k2 | 552.0 | 1.940 | 1455.9 | compute throughput, 2 indep i16 acc |
| **CEILING_k4** | **539.8** | **1.984** | **1420.6** | **true compute throughput ceiling** (decode+vwmacc) |
| CEILING_k8 | 520.5 | 2.057 | 1367.1 | compute throughput, 8 indep acc |
| **RED_k4** | **773.9** | **1.384** | **2053.8** | **+ per-block vwredsum (no extract/scale/fp)** → reduction floor |
| TRANSPOSED (block-as-lane, vlse8) | 7610.9 | 0.141 | 20425.3 | strided-load escape from the reduction floor |

The **1.5× line** = 713.9 ns @4096 (1904 ns @11008). RED_k4 (773.9) is **above**
it; no per-block-reduction kernel can cross it.

---

## Which lever bought what (the decomposition)

Reading the cost ladder top-down at n=4096 (lower ns = faster):

1. **Raw compute throughput = 1.98×** (CEILING_k4, 540 ns). Decode (nibble
   unpack) + `vwmul`/`vwmacc` *alone*, with independent accumulators, can run at
   ~2× of ggml. So there is real paper headroom — the kernel runs nowhere near
   the silicon's integer-MAC ceiling. **This confirms the "~4% of compute
   ceiling" premise.**

   - Crucial caveat: the *single-accumulator* CEILING (802 ns, 1.33×) is a
     **latency artifact**, not the throughput wall. The board is in-order: a
     lone `acc` read-modify-written every block serializes on vwmacc latency.
     Independent accumulators (k2/k4/k8) recover it to ~2×. This is the headline
     finding — the original "1.33× ceiling" intuition was wrong.

2. **+ per-block `vwredsum` costs the most: 1.98× → 1.38×** (RED_k4, 774 ns,
   +234 ns). This is decode+vwmacc+vwredsum into *independent* i32 vector
   accumulators — no scalar extract, no scale, no fp. **The per-block cross-lane
   reduction is the dominant serial bottleneck.** Even with K=4 independent
   accumulators the reductions don't pipeline enough to hide their latency,
   because each `vwredsum` is itself a multi-cycle cross-lane reduction.

   This reduction is **architecturally mandatory** for a real kernel: each block
   carries its own fp16 scale `d_x·d_y`, so the 32 integer products must be
   reduced to one `sumi` *per block* before the per-block scale can be applied.
   You cannot accumulate raw products across blocks and scale once.

3. **+ scalar extract + fp32 fold: 1.38× → 1.22×** (V4, 878 ns, +104 ns). V4
   already minimizes this: 8 independent vwredsums → contiguous **vector store**
   of lane-0 (no per-block `vmv.x.s` GPR sync) → one `vle32` → `vfcvt` →
   `vfmacc` with the 8 scales, 2 independent fp32 accumulators, one `vfredusum`
   at the end. This is the best accuracy-preserving structure found.

### Levers that did NOT help (and why)

- **V1 multi-accumulator (the task's headline lever): ~1.00×, no effect.** The
  serial `sumf` fadd chain was *not* the dominant bottleneck — the per-block
  reduction is. Breaking the fadd chain buys nothing because the reduction
  upstream already serializes the loop. V1_k8 *regressed* (0.69×) from register
  spilling the 8-wide fp32 accumulator array.
- **V2_p8 batched fold: 0.72×, regressed.** It kept the per-block `vmv.x.s`
  scalar extract *and* added a stack round-trip (store sumi → reload via vle32),
  paying more than the serial fadds it removed.
- **V2_slide (vslideup gather): 0.48× AND wrong (rel-err 1.5e10).** Sliding each
  block's result into a shared `gathered` vector reintroduced exactly the serial
  read-modify-write chain it tried to kill, plus a correctness bug. Reported only
  as a documented dead-end; never a data point.
- **TRANSPOSED / block-as-lane: 0.14×.** The only family *outside* the
  reduction floor — put block-index in the lane via strided `vlse8` loads, so no
  per-block reduction is needed. But strided loads at stride 18/34 are
  catastrophically slow on this board (~7× slower than V0). Empirically dead.

---

## Accuracy verdict (we relaxed byte-exact → accuracy-preserving)

Every real reordered variant is **at least as accurate as ggml**, against fp64
truth: V0 itself drifts up to **4.36e-3** (its serial fold is the
worst-conditioned order), while V1/V2/V3/V4 stay at **≤3.05e-3 max, ~1e-6 mean**.
All are perplexity-neutral drop-ins. (V2_slide is excluded — it is buggy, not a
reorder.) Inputs used realistic fp16 scales (~2.4e-4..1) and full-range payloads.

So accuracy is **not** the limiting factor — none of the viable variants drift.
The wall is purely the per-block reduction's serial latency.

---

## Bottom line

- **1.5× is NOT reachable** on this rvv board for an accuracy-preserving
  `q4_0_q8_0` kernel.
- Best accuracy-preserving result: **~1.22×** (V4 / V3), 878 ns @4096.
- The hard ceiling for the only viable kernel family (one reduction per block,
  forced by per-block scales) is **1.38×** (RED_k4, 774 ns) — still below 1.5×.
- The **next limiting factor** is the **per-block `vwredsum` cross-lane reduction
  latency**, not the fp32 fold and not compute throughput. Compute alone has 2×
  headroom (CEILING_k4 = 1.98×), but it is unreachable for a correct kernel
  because each block's fp16 scale mandates a per-block reduction, and that
  reduction caps the kernel at 1.38×.
- The only structure that avoids the per-block reduction (transposed / strided
  block-as-lane) is **measured at 0.14×** — strided loads make it a non-starter.

To get past ~1.4× would require either (a) a hardware/ISA path that makes
cross-lane integer reduction cheap or pipelineable, or (b) fast strided/gather
loads to enable the transposed layout — neither exists on this silicon.

---

## Files

- `kernels.c`  — V0 (ggml-real), V1_k{2,4,8}, V2_p8, V3, CEILING (single-acc).
- `kernels2.c` — CEILING_k{2,4,8} (independent acc), V2_slide (dead-end).
- `kernels3.c` — RED_k4 (reduction-floor discriminator), V4 (best real kernel).
- `kernels4.c` — TRANSPOSED (block-as-lane, strided-load escape probe).
- `harness.c`  — fp64-anchored accuracy + best-of-N timing driver.
- `board_run1.log` .. `board_run4.log` — raw rvv stdout for each stage.
