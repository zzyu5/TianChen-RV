# q4_1 repack — Win-A·micro (GEVM LMUL ablation) + GEMM scalar oracle — FINDING

**Date:** 2026-06-24
**Scope:** two QUICK matrix-coverage gaps for q4_1, mirroring the already-done q4_0 work.
Measured on `ssh rvv` (Sophgo SG2044, RVV1.0, VLEN128, clang18). `taskset -c 0`, best-of-reps min.
Methodology per `N3-METHODOLOGY.md`: Win-A = tune ON(WIDE) vs OFF(NARROW), BOTH compiler-emitted &
vectorized — NO scalar-contribution multiples; the GEMM oracle diff is a CORRECTNESS check (norm),
NOT a speedup.

- **GAP 4** — q4_1 repack GEVM Win-A (LMUL/strip) ablation: WIDE (`m1`, 1×16 strip) vs NARROW
  (`mf2`, 2×8 strips), both emitted from the SAME `tcrv_rvv.repack_gemv_q4_1_q8_1` op via local
  `build/bin/tcrv-opt`. Deliverable = ns/call + WIDE/NARROW ratio. (Win-A·micro cell; e2e
  upstream-blocked — no q8_1x4 quantizer / no q4_1-repack mul_mat routing in mainline ggml.)
- **GAP 2** — q4_1 repack GEMM (`tcrv_rvv.repack_gemm_q4_1_q8_1`) numeric oracle: emit the GEMM,
  diff per-element against a scalar q4_1 dequant-matmul reference (unsigned nibble decode + per-block
  d/min, q8_1x4 activation with the s[4] scaled-sum field), confirm norm~0. Upgrades the q4_1 GEMM
  from "code-complete, oracle-deferred" to a measured Win-B·micro correctness result. (No upstream
  ggml q4_1-repack GEMM exists → oracle is a scalar dequant-matmul, clearly labeled.)

## How the kernels were produced (both compiler-emitted, host tool)
Local host (x86, LLVM/MLIR 20): `build/bin/tcrv-opt … --tcrv-rvv-lower-to-emitc |
/usr/lib/llvm-20/bin/mlir-translate --mlir-to-cpp`. Emitted `.cpp` scp'd to rvv, compiled there with
clang18 `-O3 -march=rv64gcv_zvfh -mabi=lp64d` (zvfh for the f16 scale strips). The
`xtheadvector` march in the WIDE pipeline is STAMP-TIME only (it flips the strip-width pass's
`integer_core_lmul=m1` branch); the actual SG2044 compile target is `rv64gcv_zvfh` for BOTH arms.

LMUL spellings host-verified disjoint (grep on emitted `.cpp`):
- NARROW (mf2): {i8mf2, i16m1, i32m2, f32m2, f16m1}, **4× vle8 (2 strips)**.
- WIDE (m1):    {i8m1, i16m2, i32m4, f32m4, f16m2}, **2× vle8 (1 strip)**.

## GAP 4 — q4_1 GEVM Win-A WIDE/NARROW ablation — CLEAN POSITIVE (rvv, measured 2026-06-24)

WIDE (m1, 1×16) beats NARROW (mf2, 2×8) **~1.80×**, byte-exact (norm 0.000e+00). The 16-way
interleaved repack reads the IDENTICAL bytes either LMUL — LMUL is the *how*, not the *what* — so
the two arms agree to the bit; the speedup is the strip-count halving (1 strip vs 2). q4_1 GEVM, like
q4_0 GEVM, accumulates LANE-WISE via `vwmacc` (NO cross-lane `vredsum`), so widening LMUL halves the
strip count cleanly with no per-block-reduce wall. Best-of-reps min latency (`taskset -c 0`):

| shape (n × nc) | NARROW (mf2, 2×8) | WIDE (m1, 1×16) | **ratio NARROW/WIDE** | WIDE↔NARROW norm |
|---|---|---|---|---|
| 4096 × 4096 (run1) | 3,163,930 ns | 1,756,186 ns | **1.802×** | 0.000e+00 PASS |
| 4096 × 11008 (mlp) | 8,525,749 ns | 4,725,796 ns | **1.804×** | 0.000e+00 PASS |
| 4096 × 4096 (run3, stability) | 3,161,870 ns | 1,757,266 ns | **1.799×** | 0.000e+00 PASS |

**Headline: q4_1 GEVM Win-A = ~1.80× (WIDE/NARROW), byte-exact.** Slightly below q4_0 GEVM's ~2.1×.
LIKELY because q4_1's per-block fold carries the extra MIN term (vle16 the m strip + `vfwmul(m_x, s_y)`
+ `vfadd`), non-vwmacc FP work the LMUL-width knob does not accelerate, which would dilute the
strip-count win — but note this is an inference about the *mechanism*; what was measured is the ratio
(clean, 3-run consistent at ~1.80×), not an isolation of the min-fold's contribution.
Both arms compiler-emitted from the SAME `tcrv_rvv.repack_gemv_q4_1_q8_1` op (only `integer_core_lmul`
+ the strip count differ) → a clean Win-A·micro ablation per N3-METHODOLOGY (tune ON vs OFF, both
vectorized, NO scalar-contribution multiple).

## GAP 2 — q4_1 GEMM scalar-oracle verdict — PASS (rvv, measured 2026-06-24)

The compiler-emitted `tcrv_rvv.repack_gemm_q4_1_q8_1` (prefill, block-as-lane, lane-wise `vwmacc`,
NO `vredsum`) matches a hand-derived SCALAR q4_1 dequant-matmul oracle within fp32 rounding across all
four shapes. **VERDICT: PASS**, norm ~5.2e-6–7.6e-6 (max over shapes 7.634e-6), bar = norm < 1e-4.
This upgrades the q4_1 GEMM from "code-complete, oracle-deferred" to a numerically VALIDATED kernel —
a CORRECTNESS result only (norm), NOT a measured speedup: there is no q4_1-repack-GEMM baseline and a
scalar oracle is not a methodology-valid speed baseline. `out_mine` is zero-init per trial, so any
output element the kernel failed to write would read 0 against a ref ~22 and spike the error — norm at
~6e-6 therefore confirms the FULL nr×nc output is correct, not just spot-checked elements.
`taskset -c 0`, trials=5, n ∈ {64,256,4096,11008}:

| shape (nr × nc) | max_abs_err | rms(ref) | **norm** | rel | verdict |
|---|---|---|---|---|---|
| 16 × 16  | 1.144e-04 | 2.171e+01 | **5.272e-06** | 1.22e-02 | PASS |
| 16 × 32  | 1.221e-04 | 2.043e+01 | **5.975e-06** | 6.95e-04 | PASS |
| 16 × 64  | 1.068e-04 | 2.065e+01 | **5.173e-06** | 8.57e-04 | PASS |
| 16 × 336 | 1.678e-04 | 2.199e+01 | **7.634e-06** | 3.67e-02 | PASS |

### Oracle framing (HONEST — norm-based, NOT byte-exact ggml parity)
There is **no upstream ggml q4_1 *repack* GEMM** to diff against (mainline ships no `ggml_gemm_q4_1` /
`block_q8_1x4` / `quantize_mat_q8_1`; the only `block_q4_1x16` upstream is spacemit's INTEGER-zp model,
a different ABI). So the oracle is a hand-derived **scalar q4_1 dequant-matmul** that consumes the SAME
repacked `block_q4_1x16` (320B) weights and `block_q8_1x4` (144B) activations the kernel reads,
dotting the QUANTIZED q8 (NOT the pre-quantization fp), with the dual-scale fold
`sumf[r][col] += (d_x·d_y_c)·sumi + m_x·s_y_c` (`s_y_c = d_y_c·Σq8_c`). Because the oracle dots the q8
quants, norm reflects fp-ROUNDING (the kernel groups `(sumf + scale·sumi) + min`, the ref groups
`sumf + (scale·sumi + min)` — equal in exact arithmetic, ~fp apart), NOT q8 quantization error — which
is why norm sits at ~5–8e-6 (the SAME magnitude the q4_1 GEVM oracle sat at, 6e-6), not literally 0.
Per N3-METHODOLOGY this is a CORRECTNESS check (norm), not a speedup. (No `GgmlBlockDotQ41Q81Op`
cross-check was run for the GEMM — out of scope; the scalar oracle is the requested deliverable.)
The `rel` column (up to 3.67e-2) is a near-zero-denominator artifact, NOT a real error: those elements
are near-cancellation outputs (true value ~e-3), so a ~1e-4 fp-rounding error reads as a few percent
relative. norm (max_abs_err / rms(ref)) is the correct aggregate and it is clean (≤7.6e-6) — same
treatment the q4_1 GEVM oracle gave its `rel`.

## Headline (both gaps closed, rvv provenance)
- **GAP 4 — q4_1 GEVM Win-A WIDE/NARROW ratio = ~1.80× (ns/call), byte-exact (norm 0).** Clean
  positive Win-A·micro (1.802× / 1.804× / 1.799× across three runs). Both arms compiler-emitted from
  the same op, only the LMUL-width knob + strip count differ.
- **GAP 2 — q4_1 GEMM scalar-oracle verdict = PASS, norm ≤ 7.634e-6** (fp-rounding agreement, bar
  1e-4) across nr=16 × nc∈{16,32,64,336}, n∈{64,256,4096,11008}. Measured Win-B·micro correctness.

## Artifacts
- `q4_1-winA-gevm-ablation.log` — GAP 4 raw rvv log (3 runs, AGREE + RESULT lines).
- `q4_1-winA-gemm-oracle.log` — GAP 2 raw rvv log (4-shape norm table + VERDICT).
- On rvv under `~/q41-gap-agent/`: `ablation_micro_q41.cpp` (GAP 4 harness), `k_gemv_{WIDE,NARROW}.cpp`
  (the two compiler-emitted GEVM arms, disjoint LMUL), `oracle_gemm_q41.cpp` (GAP 2 harness +
  scalar dequant-matmul oracle), `k_gemm_q41.cpp` (compiler-emitted GEMM), binaries `ablation_q41` /
  `oracle_gemm_q41`. Emitted host-side via `build/bin/tcrv-opt … --tcrv-rvv-lower-to-emitc |
  /usr/lib/llvm-20/bin/mlir-translate --mlir-to-cpp`; the WIDE GEVM arm via
  `--tcrv-rvv-materialize-repack-strip-width=march=rv64gc_xtheadvector` (stamp-time m1 trigger), all
  compiled on the SG2044 with clang18 `-O3 -march=rv64gcv_zvfh -mabi=lp64d -ffp-contract=fast`.
- Caveat on objdump: the SG2044 box's binutils `objdump` does not decode RVV vtype/vector mnemonics
  (emits `.word`), so the binary-level vsetvli histogram from the q4_0 FINDING could not be reproduced
  here. The disjoint-LMUL evidence is therefore SOURCE-level (grep on the emitted `.cpp`: WIDE =
  {i8m1,i16m2,i32m4,f32m4,f16m2} with 2× vle8 / 1 strip; NARROW = {i8mf2,i16m1,i32m2,f32m2,f16m1} with
  4× vle8 / 2 strips) plus the runtime byte-exact WIDE↔NARROW agreement (norm 0) — conclusive that the
  two arms ran distinct vectorized code.
