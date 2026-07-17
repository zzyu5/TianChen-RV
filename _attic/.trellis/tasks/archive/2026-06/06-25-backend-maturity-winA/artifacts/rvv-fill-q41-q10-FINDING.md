# rvv fill: q4_1 repack GEMM (prefill) Win-A knob + q1_0 block-dot micro — FINDING

**Date:** 2026-06-25
**Board:** `ssh rvv` (Sophgo SG2044, RVV1.0, VLEN128, 64 cores, zvfh). Serial, rvv-only
(k1 left untouched per discipline). `taskset -c 0`, best-of-reps min latency.
**Tooling:** existing `build/bin/tcrv-opt` (built 06-25 21:52) — **NO rebuild**. Emit
host-side: `tcrv-opt … --tcrv-rvv-lower-to-emitc | /usr/lib/llvm-20/bin/mlir-translate
--mlir-to-cpp`. Compiled on the board with `clang18 -O3 -march=rv64gcv_zvfh -mabi=lp64d
-ffp-contract=fast`. **byte-exact gate passed before any perf was reported.**

Two self-check-table cells filled:
1. Table 2 row `q4_1 repack(prefill)`, `Win-A·rvv` — was 空(注4 "最容易补的一格").
2. Table 1 row `q1_0`, `vs-ggml·rvv` — was empty ("没测").

---

## Cell 1 — q4_1 repack GEMM (prefill) Win-A·rvv = ~1.24–1.31× (WIDE/NARROW), byte-exact

**Filled value: ~1.24–1.31× (clean floor) — WIDE (m1) beats NARROW (mf2).**

Both arms compiler-emitted from the SAME op `tcrv_rvv.repack_gemm_q4_1_q8_1`
(test `rvv-to-emitc-repack-gemm-q4-1-q8-1.mlir`). The ONLY difference is the
strip-width gearbox:
- **NARROW** (default emit): `{i8mf2, u8mf2, i16m1, i32m2, f32m2, f16m1}`, 2× vle8.
- **WIDE** (stamp `--tcrv-rvv-materialize-repack-strip-width=march=rv64gc_xtheadvector`):
  `{i8m1, u8m1, i16m2, i32m4, f32m4, f16m2}`, 8× vle8.

LMUL-disjointness is SOURCE-level (grep on the emitted `.cpp` — SG2044 binutils
`objdump` cannot decode RVV vtype mnemonics, same caveat as the q4_0/q4_1-GEVM
findings). **The knob is REAL, not N/A** — the two arms emit cleanly distinct LMUL
families AND agree byte-exact (norm 0), so the difference is pure *how*, not *what*.

**Byte-exact gate (WIDE↔NARROW): norm 0.000e+00 PASS** at n=4096 nr=16 nc∈{4096,11008}.

**Ratio NARROW/WIDE (Win-A·micro, prefill):**

| shape (n × nr × nc) | NARROW | WIDE | ratio NARROW/WIDE | gate |
|---|---|---|---|---|
| 4096 × 16 × 4096 (square, run1 clean) | 36.9M ns | 29.8M ns | **1.239×** | 0.000e+00 |
| 4096 × 16 × 11008 (mlp, fresh) | 105.5M ns | 80.4M ns | **1.313×** | 0.000e+00 |
| 4096 × 16 × 4096 (later runs ×4) | 40–48M ns | 29.9–30.3M ns | 1.35–1.58× | 0.000e+00 |

**Honesty on the ratio range (1.24–1.58×) — anchor on the min:** best-of-reps **min**
latency is the correct estimator (noise only ADDS time, so the lowest observed value is
closest to true). The **lowest observed NARROW=36.9M / WIDE=29.8M → 1.24×** is the best
estimate; the higher ratios (up to 1.58×) come from NARROW's min rising under accumulating
board load (WIDE's min stayed 29.8–30.3M across all 6 runs — already near-floor), i.e. they
are upper noise on NARROW, not a larger true win. **Reported value: ~1.24–1.31×** (1.24×
square min, 1.31× mlp distinct-shape fresh min). Same direction as the already-done q4_1
GEVM Win-A (1.80×, archive) — same op family, WIDE/m1 faster.

**Mechanism NOT isolated (honest — the data refutes a strip-count story):** for the GEMM,
the WIDE arm emits MORE static vle8 (8 vs NARROW's 2) — the OPPOSITE of the GEVM, where
WIDE halved the strip count. So the GEMM win is NOT strip-count halving; it is more likely
the wider fold arithmetic (i16m2/i32m4/f32m4 vs i16m1/i32m2/f32m2 — fewer fold passes per
column). What was MEASURED is the byte-exact WIDE/NARROW ratio (WIDE/m1 faster); the
mechanism is not isolated and is not claimed.

**Framing label (per "Repack Win-A always mf2" memory):** WIDE = the manually
xtheadvector-stamp-triggered m1/RVV0.7-form arm; RVV1.0 auto-selects mf2 (NARROW).
This is a **knob ON/OFF ablation**, NOT an RVV1.0 auto-tune win. e2e remains
upstream-BLOCKED (ggml ships no q8_1x4 mat-quantizer / no q4_1-repack mul_mat routing)
— only the pure-kernel micro knob is measurable, which is exactly what 注4 asked for.
GEMM correctness vs the scalar dequant-matmul oracle was ALREADY validated (archive
GAP 2, norm ≤7.6e-6) — cited, not redone; the new deliverable is just the WIDE/NARROW
ratio.

---

## Cell 2 — q1_0 block-dot vs-ggml·rvv = 0.033× (LOSS, ~30×), byte-exact

**Filled value: 0.033× — LOSS (ggml ~30× faster). NOT N/A.**

q1_0 IS a real ggml type (`GGML_TYPE_Q1_0`=41; `block_q1_0`={fp16 d; uint8 qs[16]},
QK1_0=128, 18B; binary {−1,+1} sign decode). **ggml ships a RISC-V-vectorized
`ggml_vec_dot_q1_0_q8_0_vl128`** (arch/riscv/quants.c:523), dispatched on VLEN128 —
so there IS a legitimate vectorized competitor (unlike nvfp4 = N/A scalar-only). Our
emit = `tcrv_rvv.q1_0_q8_0_block_dot` (test `rvv-to-emitc-q1-0-q8-0-block-dot.mlir`).

**Byte-exact gate: max_rel_norm 0.000e+00 PASS** (8 seeds × 6 sizes), nonzero=1,
nonfinite=0. Both kernels read identical super-block bytes.

| | ours | ggml (real, vl128) | RATIO ggml/ours | gate |
|---|---|---|---|---|
| run1 | 44877 ns | 1492 ns | **0.033×** | 0.000e+00 |
| run2 | 44887 ns | 1497 ns | **0.033×** | 0.000e+00 |
| run3 | 44887 ns | 1502 ns | **0.033×** | 0.000e+00 |

**Why the ~30× LOSS (root-caused, honest):** ggml `_vl128` does ONE `vwredsum` over a
full **32-lane** e8m2 register per q8 sub-block (4 reductions per 128-elem super-block).
OUR emit processes the binary sign-decode in **8-lane groups** (`vsetvl_e8m1(8)`): 32
`vwredsum` + 32 `vmerge` per super-block — **8× more, far narrower reductions, fully
unrolled**. This is the SAME "narrower strip, correct-but-not-LMUL/shape-tuned" maturity
gap the IQ block-dot kernels showed; q1_0's tiny 8-lane group shape makes it the worst
case. The loss is a real emitter-maturity gap (wide-LMUL block-as-32-lane emit), NOT a
correctness or algorithm-choice issue.

**−128 byte-exact note (maturity, NOT a gate failure):** ggml `_vl128` negates in the
i8 domain (`vneg_i8m2` → vneg(−128)=−128, overflow); OUR emit widens to i16 FIRST then
negates (correct +128 over the FULL int8 range). They diverge ONLY when a q8 quant byte
== −128, which NEVER occurs in genuine q8_0 (quant domain [−127,127]). The harness
constrains the q8 fill to [−127,127] (the real domain) → byte-exact agreement. At the
−128 super-boundary **we are MORE correct than ggml** — recorded as a maturity note.

---

## Headline (both cells, rvv provenance)

- **Cell 1 — q4_1 GEMM (prefill) Win-A·rvv ≈ 1.24–1.31× (WIDE/NARROW), byte-exact
  (norm 0).** Knob is real (LMUL-disjoint mf2↔m1, source-grep proven). WIDE/m1 wins;
  manual-stamp m1 vs default mf2 (NOT RVV1.0 auto-tune). e2e upstream-BLOCKED.
- **Cell 2 — q1_0 block-dot vs-ggml·rvv = 0.033× (LOSS ~30×), byte-exact (norm 0).**
  NOT N/A — ggml has a real `_vl128` q1_0 kernel. LOSS root-caused: our 8-lane-group
  emit vs ggml's 32-lane single-vwredsum. Emitter-maturity gap (wide-LMUL emit target).

## Not measured / why
- q4_1 GEMM e2e: upstream-BLOCKED (no ggml q8_1x4 mat-quantizer) — only micro knob
  measurable (as scoped by 注4).
- q4_1 GEMM correctness vs scalar oracle: already done in archive (GAP 2, norm ≤7.6e-6)
  — cited, not redone.
- Both kernels emitted with existing tcrv-opt; no lib/ change, no rebuild.

## Artifacts
- `rvv-fill-q41-q10/q10.log` — q1_0 byte-exact + ratio raw log.
- `rvv-fill-q41-q10/gemm-winA.log` — q4_1 GEMM Win-A byte-exact + ratio raw log (all runs).
- `rvv-fill-q41-q10/{q10_harness.cpp, q10_ggml_ref.cpp, ablation_gemm_q41.cpp}` — harness
  sources. On rvv: `~/fill-q41-q10/` (emitted kernels q10_ours.cpp / k_gemm_{NARROW,WIDE}.cpp,
  binaries q10_bench / ablation_gemm_q41). q1_0 ggml_ref lifted verbatim from
  llama.cpp quants.c:523 (`_vl128` body). GEMM harness adapts the archived GAP-2/GAP-4
  q4_1 oracle+ablation (repack/fill reused; already oracle-validated).
